//! `AudioProjModel` port (model_memory.py lines 558-621) plus the per-frame
//! audio window arrangement from `WanModel.forward` (lines 1022-1037).
//!
//! The wav2vec2 analysis produces one embedding per PIXEL frame:
//! `[frames, 5, 12, 768]` (5-frame sliding window, 12 hidden layers).
//! The adapter converts these into `context_tokens = 32` audio tokens of
//! `output_dim = 768` per LATENT frame:
//!
//! - anchor (first latent frame): its pixel frame's full 5-window
//! - every other latent frame k (4 pixel frames 4k..4k+3): 8 windows =
//!   first 3 windows of pixel frame 0, the centre window of frames 1 and 2,
//!   last 3 windows of frame 3  (`seq_len_vf = audio_window + vae_scale - 1`)
//!
//! then proj1 / proj1_vf -> ReLU -> concat -> proj2 -> ReLU -> proj3
//! -> LayerNorm.

use crate::config::LiveActConfig;
use crate::tensor::{layer_norm, relu_inplace, Linear, Rng};

/// `[audio_frames, audio_ctx, audio_dim]` per latent frame.
pub struct AudioTokens {
    pub frames: usize,
    pub data: Vec<f32>,
    pub ctx_tokens: usize,
    pub dim: usize,
}

pub struct AudioProj {
    pub proj1: Linear,    // seq_len * blocks * channels -> intermediate
    pub proj1_vf: Linear, // seq_len_vf * blocks * channels -> intermediate
    pub proj2: Linear,    // intermediate -> intermediate
    pub proj3: Linear,    // intermediate -> context_tokens * output_dim
    pub norm_w: Vec<f32>, // LayerNorm(output_dim), affine
    pub norm_b: Vec<f32>,
    seq_len: usize,
    seq_len_vf: usize,
    blocks: usize,
    channels: usize,
    context_tokens: usize,
    output_dim: usize,
}

impl AudioProj {
    pub fn new(cfg: &LiveActConfig) -> Self {
        let seq_len = cfg.audio_window;
        let seq_len_vf = cfg.audio_window + cfg.vae_scale - 1;
        let input_dim = seq_len * cfg.audio_blocks * cfg.audio_channels;
        let input_dim_vf = seq_len_vf * cfg.audio_blocks * cfg.audio_channels;
        AudioProj {
            proj1: Linear::new(input_dim, cfg.audio_intermediate, true),
            proj1_vf: Linear::new(input_dim_vf, cfg.audio_intermediate, true),
            proj2: Linear::new(cfg.audio_intermediate, cfg.audio_intermediate, true),
            proj3: Linear::new(
                cfg.audio_intermediate,
                cfg.audio_context_tokens * cfg.audio_channels,
                true,
            ),
            norm_w: vec![1.0; cfg.audio_channels],
            norm_b: vec![0.0; cfg.audio_channels],
            seq_len,
            seq_len_vf,
            blocks: cfg.audio_blocks,
            channels: cfg.audio_channels,
            context_tokens: cfg.audio_context_tokens,
            output_dim: cfg.audio_channels,
        }
    }

    /// Build the `(first_frame, latter_frames)` inputs from pixel-frame
    /// wav2vec2 embeddings `[pixel_frames, window, blocks, channels]`.
    pub fn arrange(&self, pixel_audio: &[f32], pixel_frames: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let (w, s, c) = (self.seq_len, self.blocks, self.channels);
        let middle = w / 2;
        debug_assert_eq!(pixel_audio.len(), pixel_frames * w * s * c);
        debug_assert_eq!((pixel_frames - 1) % 4, 0, "1 anchor + 4k pixel frames");
        let latent_frames = (pixel_frames - 1) / 4 + 1;

        // first frame input: [1, w*s*c]
        let first = pixel_audio[..w * s * c].to_vec();

        // latter frames input: [latent_frames-1, seq_len_vf*s*c]
        let vf = self.seq_len_vf;
        let mut latter = vec![0.0f32; (latent_frames - 1) * vf * s * c];
        for lf in 0..latent_frames - 1 {
            // 4 pixel frames of this latent frame
            let mut dst = vec![0.0f32; vf * s * c];
            let mut filled = 0usize; // windows filled so far
            for n in 0..4 {
                let pf = 1 + lf * 4 + n; // skip anchor
                let src = &pixel_audio[pf * w * s * c..(pf + 1) * w * s * c];
                let (a, b) = match n {
                    0 => (0, middle + 1),      // first 3 windows
                    1 | 2 => (middle, middle + 1), // centre window
                    _ => (middle, w),          // last 3 windows
                };
                for wi in a..b {
                    let chunk = &src[wi * s * c..(wi + 1) * s * c];
                    dst[filled * s * c..(filled + 1) * s * c].copy_from_slice(chunk);
                    filled += 1;
                }
            }
            debug_assert_eq!(filled, vf);
            latter[lf * vf * s * c..(lf + 1) * vf * s * c].copy_from_slice(&dst);
        }
        (first, latter, latent_frames)
    }

    /// Full AudioProjModel.forward: returns `[latent_frames, ctx, out_dim]`.
    pub fn forward(&self, first: &[f32], latter: &[f32], latent_frames: usize) -> AudioTokens {
        let _vf = self.seq_len_vf;
        let _s = self.blocks;
        let _c = self.channels;
        // first frame
        let mut a = self.proj1.forward(first, 1);
        relu_inplace(&mut a);
        // latter frames
        let n_lat = latent_frames - 1;
        let mut b = self.proj1_vf.forward(latter, n_lat);
        relu_inplace(&mut b);
        // concat along frames -> [latent_frames, intermediate]
        let inter = self.proj1.out_dim;
        let mut cat = vec![0.0f32; latent_frames * inter];
        cat[..inter].copy_from_slice(&a);
        cat[inter..].copy_from_slice(&b);
        // proj2 (per frame) + ReLU
        let mut z = self.proj2.forward(&cat, latent_frames);
        relu_inplace(&mut z);
        // proj3 -> [latent_frames * ctx, output_dim]
        let flat = self.proj3.forward(&z, latent_frames);
        // LayerNorm(output_dim) over each token
        let normed = layer_norm(&flat, self.output_dim, Some(&self.norm_w), Some(&self.norm_b), 1e-5);
        AudioTokens {
            frames: latent_frames,
            data: normed,
            ctx_tokens: self.context_tokens,
            dim: self.output_dim,
        }
    }

    pub fn init_weights(&mut self, rng: &mut Rng) {
        self.proj1.xavier(rng);
        self.proj1_vf.xavier(rng);
        self.proj2.xavier(rng);
        self.proj3.xavier(rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrangement_counts() {
        let cfg = LiveActConfig::tiny();
        let proj = AudioProj::new(&cfg);
        assert_eq!(proj.seq_len, 5);
        assert_eq!(proj.seq_len_vf, 8);
        let pixel_frames = 53;
        let (w, s, c) = (5, 12, 768);
        let mut pixel = vec![0.0f32; pixel_frames * w * s * c];
        for (i, v) in pixel.iter_mut().enumerate() {
            *v = (i % 97) as f32 * 0.01;
        }
        let (first, latter, latent_frames) = proj.arrange(&pixel, pixel_frames);
        assert_eq!(latent_frames, 14);
        assert_eq!(first.len(), 5 * 12 * 768);
        assert_eq!(latter.len(), 13 * 8 * 12 * 768);
        // first input must be pixel frame 0 verbatim
        assert_eq!(&first[..], &pixel[..first.len()]);
        // window content check: latent frame 0 (pixel frames 1..5):
        // windows = pf1 w0-2, pf2 w2, pf3 w2, pf4 w2-4
        let mut expect: Vec<f32> = vec![];
        for (pf, ws) in [(1usize, 0..3), (2, 2..3), (3, 2..3), (4, 2..5)] {
            for wi in ws {
                expect.extend_from_slice(&pixel[pf * w * s * c + wi * s * c..pf * w * s * c + (wi + 1) * s * c]);
            }
        }
        assert_eq!(&latter[..expect.len()], &expect[..]);
    }

    #[test]
    fn forward_shape() {
        let cfg = LiveActConfig::tiny();
        let mut proj = AudioProj::new(&cfg);
        let mut rng = Rng::new(1);
        proj.init_weights(&mut rng);
        let (w, s, c) = (5, 12, 768);
        let pixel: Vec<f32> = (0..53 * w * s * c).map(|i| (i as f32 * 0.001).sin()).collect();
        let (first, latter, latent_frames) = proj.arrange(&pixel, 53);
        let out = proj.forward(&first, &latter, latent_frames);
        assert_eq!(out.frames, 14);
        assert_eq!(out.ctx_tokens, 32);
        assert_eq!(out.dim, 768);
        assert_eq!(out.data.len(), 14 * 32 * 768);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }
}
