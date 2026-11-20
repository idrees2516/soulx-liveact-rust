//! Model configuration.
//!
//! `LiveActConfig` mirrors the constructor arguments of
//! `model_liveact/model_memory.py::WanModel` plus the audio-conditioning and
//! VAE settings used by `generate.py`. Two presets ship:
//!
//! - [`LiveActConfig::full_14b`] — the exact hyper-parameters of the released
//!   SoulX-LiveAct checkpoint (Wan2.1-I2V 14B backbone + 4B audio adapter:
//!   dim 5120, 40 heads x 128, 40 layers, 18B params total). Every field is
//!   honored by the Rust modules; the configuration is fully constructible,
//!   but inference at this scale requires the released weights and a GPU.
//! - [`LiveActConfig::tiny`] — an architecturally identical compact instance
//!   (same block wiring, same ConvKV cache layout, same Neighbor-Forcing
//!   schedule) that runs end-to-end on CPU in real time, used by the shipped
//!   demo pipeline and tests.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveActConfig {
    // ---- WanModel backbone (model_liveact/model_memory.py) ----
    pub patch: (usize, usize, usize), // (1, 2, 2)
    pub text_len: usize,              // padded T5 token count
    pub in_dim: usize,                // 36 = 16 noisy latent + 16 y latent + 4 mask
    pub dim: usize,                   // transformer width
    pub ffn_dim: usize,
    pub freq_dim: usize,              // timestep sinusoid dim
    pub text_dim: usize,              // T5 encoder output dim (umt5-xxl: 4096)
    pub out_dim: usize,               // 16 = VAE latent channels
    pub num_heads: usize,
    pub num_layers: usize,
    pub eps: f32,
    pub cross_attn_norm: bool,

    // ---- audio adapter (AudioProjModel) ----
    pub audio_window: usize,        // 5 frames of wav2vec2 context
    pub vae_scale: usize,           // 4 pixel frames per latent frame
    pub audio_blocks: usize,        // wav2vec2 hidden layers used (12)
    pub audio_channels: usize,      // wav2vec2 width (768)
    pub audio_intermediate: usize,  // AudioProj intermediate dim
    pub audio_context_tokens: usize, // 32 tokens per latent frame

    // ---- CLIP visual conditioning ----
    pub clip_dim: usize,   // 1280 for open-clip xlm-roberta vit-huge-14
    pub clip_tokens: usize, // 257 (CLS + 16x16 patches)

    // ---- causal 3D VAE (wan/modules/vae.py) ----
    pub vae_dim: usize,
    pub vae_z: usize, // 16
    pub vae_dim_mult: Vec<usize>,
    pub vae_num_res_blocks: usize,

    // ---- Neighbor-Forcing schedule (generate.py) ----
    pub timesteps: Vec<f32>,   // [1000, 937.5, 833.333, 0]
    pub chunk_sizes: Vec<usize>, // [6, 8] latent frames (first chunk, then)
    pub audio_steps: Vec<usize>, // diffusion steps that apply audio (1, 2)

    // ---- video geometry ----
    pub height: usize,
    pub width: usize,
    pub fps: usize, // internal model fps (25 in the original)

    pub seed: u64,
}

impl LiveActConfig {
    /// The released 18B configuration (14B Wan2.1-I2V backbone + 4B audio
    /// adapter, 40 layers x 5120 dim, heads 40 x 128, see generate.py cache
    /// shapes `[1, tokens, 40, 128]`).
    pub fn full_14b() -> Self {
        LiveActConfig {
            patch: (1, 2, 2),
            text_len: 512,
            in_dim: 36,
            dim: 5120,
            ffn_dim: 13824,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 40,
            num_layers: 40,
            eps: 1e-6,
            cross_attn_norm: true,
            audio_window: 5,
            vae_scale: 4,
            audio_blocks: 12,
            audio_channels: 768,
            audio_intermediate: 512,
            audio_context_tokens: 32,
            clip_dim: 1280,
            clip_tokens: 257,
            vae_dim: 96,
            vae_z: 16,
            vae_dim_mult: vec![1, 2, 4, 4],
            vae_num_res_blocks: 2,
            timesteps: vec![1000.0, 937.5, 833.33333333, 0.0],
            chunk_sizes: vec![6, 8],
            audio_steps: vec![1, 2],
            height: 416,
            width: 720,
            fps: 25,
            seed: 42,
        }
    }

    /// Compact CPU-runnable instance. Same wiring, cache layout and schedule
    /// as the 14B model; every module is instantiated and exercised.
    pub fn tiny() -> Self {
        LiveActConfig {
            patch: (1, 2, 2),
            text_len: 64,
            in_dim: 36,
            dim: 384,
            ffn_dim: 1536,
            freq_dim: 256,
            text_dim: 512,
            out_dim: 16,
            num_heads: 8,
            num_layers: 6,
            eps: 1e-6,
            cross_attn_norm: true,
            audio_window: 5,
            vae_scale: 4,
            audio_blocks: 12,
            audio_channels: 768,
            audio_intermediate: 128,
            audio_context_tokens: 32,
            clip_dim: 1280,
            clip_tokens: 257,
            vae_dim: 32,
            vae_z: 16,
            vae_dim_mult: vec![1, 2, 4, 4],
            vae_num_res_blocks: 1,
            timesteps: vec![1000.0, 937.5, 833.33333333, 0.0],
            chunk_sizes: vec![6, 8],
            audio_steps: vec![1, 2],
            height: 64,
            width: 64,
            fps: 25,
            seed: 42,
        }
    }

    /// Fast CPU preset for the interactive pipeline / examples
    /// (same wiring; 32x32 video -> 2x2 token grid).
    pub fn demo() -> Self {
        let mut c = Self::tiny();
        c.height = 32;
        c.width = 32;
        c
    }

    /// VAE strides (4, 8, 8) as in generate.py.
    pub fn vae_stride(&self) -> (usize, usize, usize) {
        (4, 8, 8)
    }

    /// Latent tokens per latent frame: (H / (patch_h * 8)) * (W / (patch_w * 8)).
    pub fn frame_len(&self) -> usize {
        let (ph, pw) = (self.patch.1, self.patch.2);
        (self.height / (ph * 8)) * (self.width / (pw * 8))
    }

    /// Total latent frames covered by the KV cache window: sum(chunk_sizes).
    pub fn cache_frames(&self) -> usize {
        self.chunk_sizes.iter().sum()
    }

    /// Pixel frames encoded for the y conditioning video:
    /// (sum(chunks) - 1) * 4 + 1.
    pub fn cond_frame_num(&self) -> usize {
        (self.cache_frames() - 1) * 4 + 1
    }

    /// Latent height / width.
    pub fn lat_hw(&self) -> (usize, usize) {
        (self.height / 8, self.width / 8)
    }

    /// Patched latent grid (tokens per frame are grid_h * grid_w).
    pub fn grid_hw(&self) -> (usize, usize) {
        let (lh, lw) = self.lat_hw();
        (lh / self.patch.1, lw / self.patch.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_math_matches_generate_py() {
        // generate.py: frame_num = (sum(blksz)-1)*4 + 1 = 53, frame_len uses
        // H/(patch*8). For 416x720: (416/16)*(720/16) = 26*45 = 1170.
        let c = LiveActConfig::full_14b();
        assert_eq!(c.cond_frame_num(), 53);
        assert_eq!(c.cache_frames(), 14);
        assert_eq!(c.frame_len(), 1170);
        assert_eq!(c.lat_hw(), (52, 90));
        assert_eq!(c.grid_hw(), (26, 45));
    }

    #[test]
    fn tiny_geometry() {
        let c = LiveActConfig::tiny();
        assert_eq!(c.frame_len(), 16);
        assert_eq!(c.grid_hw(), (4, 4));
        assert_eq!(c.lat_hw(), (8, 8));
    }
}
