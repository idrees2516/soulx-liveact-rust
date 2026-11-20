//! `WanModel` port (model_memory.py lines 805-1139): the SoulX-LiveAct
//! diffusion backbone.
//!
//! Layout notes kept bit-faithful to the PyTorch modules so released
//! checkpoints could be loaded without reordering:
//! - `patch_embedding` Conv3d(in_dim, dim, (1,2,2), stride=(1,2,2)) is
//!   implemented as patch-gather + linear with the weight flattened to
//!   `[dim, in_dim * 4]` in (c, kh, kw) order.
//! - `Head` emits `out_dim * 4` per token in (pt, ph, pw, c) order; the
//!   unpatchify scatter mirrors the `einsum('fhwpqrc->cfphqwr')`.

use crate::attention::KvCache;
use crate::audio_proj::AudioProj;
use crate::block::AttentionBlock;
use crate::config::LiveActConfig;
use crate::rope::Rope3d;
use crate::tensor::{gelu_erf, gelu_tanh, layer_norm, silu, Linear, Rng};

/// `sinusoidal_embedding_1d(dim, position)` with position as one scalar.
pub fn sinusoidal_embedding_1d(dim: usize, pos: f64) -> Vec<f32> {
    assert!(dim % 2 == 0);
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for j in 0..half {
        let freq = pos * 10000f64.powf(-(j as f64) / half as f64);
        out[j] = freq.cos() as f32;
        out[half + j] = freq.sin() as f32;
    }
    out
}

pub struct MlpProj {
    // MLPProj(1280, dim): LN(1280) -> Linear(1280,1280) -> GELU -> Linear(1280,dim) -> LN(dim)
    pub norm_in_w: Vec<f32>,
    pub norm_in_b: Vec<f32>,
    pub fc1: Linear,
    pub fc2: Linear,
    pub norm_out_w: Vec<f32>,
    pub norm_out_b: Vec<f32>,
}

impl MlpProj {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        MlpProj {
            norm_in_w: vec![1.0; in_dim],
            norm_in_b: vec![0.0; in_dim],
            fc1: Linear::new(in_dim, in_dim, true),
            fc2: Linear::new(in_dim, out_dim, true),
            norm_out_w: vec![1.0; out_dim],
            norm_out_b: vec![0.0; out_dim],
        }
    }

    pub fn forward(&self, x: &[f32], tokens: usize) -> Vec<f32> {
        let in_dim = self.fc1.in_dim;
        let out_dim = self.fc2.out_dim;
        let h = layer_norm(x, in_dim, Some(&self.norm_in_w), Some(&self.norm_in_b), 1e-5);
        let h = self.fc1.forward(&h, tokens);
        let h: Vec<f32> = h.iter().map(|v| gelu_erf(*v)).collect();
        let h = self.fc2.forward(&h, tokens);
        layer_norm(&h, out_dim, Some(&self.norm_out_w), Some(&self.norm_out_b), 1e-5)
    }

    pub fn init_weights(&mut self, rng: &mut Rng) {
        self.fc1.xavier(rng);
        self.fc2.xavier(rng);
    }
}

/// One chunk forward through the backbone.
pub struct ForwardArgs<'a> {
    /// `[out_dim, frames, lh, lw]` noisy latent of the current chunk
    pub latent: &'a [f32],
    /// `[out_dim + 4, frames, lh, lw]` VAE-encoded conditioning + mask
    pub y: &'a [f32],
    /// diffusion timestep in [0, 1000]
    pub t: f32,
    /// `[tokens, text_dim]` T5 text conditioning (will be padded to text_len)
    pub text_tokens: usize,
    pub text_ctx: &'a [f32],
    /// `[clip_tokens, clip_dim]` CLIP visual features of the anchor image
    pub clip_fea: &'a [f32],
    /// `[pixel_frames, window, blocks, channels]` windowed wav2vec2 features
    /// covering this chunk's whole 53-frame y window
    pub audio_pixel: &'a [f32],
    pub audio_pixel_frames: usize,
    /// KV caches for the current diffusion step, one per layer
    pub kv_caches: &'a mut [KvCache],
    pub start_idx: usize,
    pub end_idx: usize,
    pub update_cache: bool,
    pub skip_audio: bool,
    pub mean_memory: bool,
}

pub struct WanModel {
    pub patch_w: Vec<f32>, // [dim, in_dim * ph * pw]
    pub patch_b: Vec<f32>, // [dim]
    pub text_fc1: Linear,
    pub text_fc2: Linear,
    pub time_fc1: Linear,
    pub time_fc2: Linear,
    pub time_proj: Linear, // dim -> 6*dim (SiLU before)
    pub img_emb: MlpProj,
    pub audio_proj: AudioProj,
    pub blocks: Vec<AttentionBlock>,
    pub head_w: Linear, // dim -> out_dim * ph * pw
    pub head_modulation: Vec<f32>, // [2, dim]
    pub rope: Rope3d,
    pub cfg: LiveActConfig,
}

impl WanModel {
    pub fn new(cfg: &LiveActConfig) -> Self {
        let ph = cfg.patch.1 * cfg.patch.2;
        WanModel {
            patch_w: vec![0.0; cfg.dim * cfg.in_dim * ph],
            patch_b: vec![0.0; cfg.dim],
            text_fc1: Linear::new(cfg.text_dim, cfg.dim, true),
            text_fc2: Linear::new(cfg.dim, cfg.dim, true),
            time_fc1: Linear::new(cfg.freq_dim, cfg.dim, true),
            time_fc2: Linear::new(cfg.dim, cfg.dim, true),
            time_proj: Linear::new(cfg.dim, 6 * cfg.dim, true),
            img_emb: MlpProj::new(cfg.clip_dim, cfg.dim),
            audio_proj: AudioProj::new(cfg),
            blocks: (0..cfg.num_layers).map(|_| AttentionBlock::new(cfg)).collect(),
            head_w: Linear::new(cfg.dim, cfg.out_dim * ph, true),
            head_modulation: vec![0.0; 2 * cfg.dim],
            rope: Rope3d::new(cfg),
            cfg: cfg.clone(),
        }
    }

    /// Deterministic initialization following `WanModel.init_weights`
    /// (xavier linears, normal(0.02) text/time, zero output head).
    pub fn init_weights(&mut self, rng: &mut Rng) {
        let (dim, idim, ph) = (
            self.cfg.dim,
            self.cfg.in_dim,
            self.cfg.patch.1 * self.cfg.patch.2,
        );
        // patch embedding: xavier on the flattened weight, zeros bias
        let flat = dim * idim * ph;
        let bound = (6.0 / (flat + dim) as f32).sqrt();
        for v in self.patch_w.iter_mut() {
            *v = (rng.f32() * 2.0 - 1.0) * bound;
        }
        self.patch_b.fill(0.0);
        // text / time embeddings: normal(0.02)
        for l in [&mut self.text_fc1, &mut self.text_fc2] {
            normal_init(l, rng, 0.02);
        }
        for l in [&mut self.time_fc1, &mut self.time_fc2] {
            normal_init(l, rng, 0.02);
        }
        self.time_proj.xavier(rng);
        self.img_emb.init_weights(rng);
        self.audio_proj.init_weights(rng);
        for b in &mut self.blocks {
            b.init_weights(rng);
        }
        // zero-init output head (init_weights zeroes head.head)
        self.head_w.w.fill(0.0);
        if let Some(b) = &mut self.head_w.b {
            b.fill(0.0);
        }
        let std = (dim as f32).sqrt().recip();
        for v in self.head_modulation.iter_mut() {
            *v = rng.normal() * std;
        }
    }

    /// Allocate per-step KV caches: `[step][layer]`.
    pub fn new_kv_caches(&self, fp8: bool) -> Vec<Vec<KvCache>> {
        let fs = self.cfg.frame_len();
        (0..self.cfg.timesteps.len() - 1)
            .map(|_| {
                (0..self.cfg.num_layers)
                    .map(|_| {
                        KvCache::new(
                            self.cfg.cache_frames(),
                            fs,
                            self.cfg.num_heads,
                            self.cfg.dim / self.cfg.num_heads,
                            fp8,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// Patchify `[in_c, T, H, W]` -> tokens `[T*gh*gw, dim]`.
    fn patchify(&self, x: &[f32], in_c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (_pt, ph, pw) = cfg.patch;
        let (gh, gw) = (h / ph, w / pw);
        let tokens = t * gh * gw;
        let feat = in_c * ph * pw;
        let mut gathered = vec![0.0f32; tokens * feat];
        for ti in 0..t {
            for gy in 0..gh {
                for gx in 0..gw {
                    let tok = ti * gh * gw + gy * gw + gx;
                    // gather in (c, kh, kw) order matching Conv3d weight layout
                    for c in 0..in_c {
                        for kh in 0..ph {
                            for kw in 0..pw {
                                let src = ((c * t + ti) * h + (gy * ph + kh)) * w + (gx * pw + kw);
                                let dst = tok * feat + c * (ph * pw) + kh * pw + kw;
                                gathered[dst] = x[src];
                            }
                        }
                    }
                }
            }
        }
        let mut out = vec![0.0f32; tokens * cfg.dim];
        crate::tensor::matmul_at_bt(
            &gathered,
            &self.patch_w,
            tokens,
            feat,
            cfg.dim,
            &mut out,
            0.0,
        );
        for tok in 0..tokens {
            for (d, b) in out[tok * cfg.dim..(tok + 1) * cfg.dim]
                .iter_mut()
                .zip(&self.patch_b)
            {
                *d += b;
            }
        }
        out
    }

    /// Unpatchify head output `[tokens, out_dim*ph*pw]` -> `[out_dim, T, H, W]`.
    fn unpatchify(&self, x: &[f32], t: usize, gh: usize, gw: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (ph, pw) = (cfg.patch.1, cfg.patch.2);
        let (h, w) = (gh * ph, gw * pw);
        let c_out = cfg.out_dim;
        let per_tok = c_out * ph * pw;
        let mut out = vec![0.0f32; c_out * t * h * w];
        for ti in 0..t {
            for gy in 0..gh {
                for gx in 0..gw {
                    let tok = ti * gh * gw + gy * gw + gx;
                    let row = &x[tok * per_tok..(tok + 1) * per_tok];
                    // linear output layout (pt, ph, pw, c): idx = ph*32? no:
                    // strides: pt = c_out*ph*pw, ph = pw*c_out, pw = c_out, c = 1
                    for q in 0..ph {
                        for r in 0..pw {
                            for c in 0..c_out {
                                let idx = q * (pw * c_out) + r * c_out + c;
                                let dst = ((c * t + ti) * h + (gy * ph + q)) * w + (gx * pw + r);
                                out[dst] = row[idx];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Velocity prediction for one chunk at timestep `t`.
    pub fn forward(&mut self, args: ForwardArgs<'_>) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let (lz, ly) = self.cfg.lat_hw();
        let frames = args.latent.len() / (cfg.out_dim * lz * ly);
        debug_assert_eq!(frames * cfg.out_dim * lz * ly, args.latent.len());
        debug_assert_eq!(args.y.len(), (cfg.out_dim + 4) * frames * lz * ly);

        // ---- assemble input channels: [in_dim, frames, lz, ly] = cat(latent, y)
        let (gh, gw) = self.cfg.grid_hw();
        let frame_len = gh * gw;
        let mut input = vec![0.0f32; cfg.in_dim * frames * lz * ly];
        input[..cfg.out_dim * frames * lz * ly].copy_from_slice(args.latent);
        let y_len = (cfg.out_dim + 4) * frames * lz * ly;
        input[cfg.in_dim * frames * lz * ly - y_len..].copy_from_slice(args.y);

        // ---- patchify ----
        let tokens = frames * frame_len;
        let mut x = self.patchify(&input, cfg.in_dim, frames, lz, ly);

        // ---- timestep embedding ----
        let sin = sinusoidal_embedding_1d(cfg.freq_dim, args.t as f64);
        let mut e = self.time_fc1.forward(&sin, 1);
        for v in e.iter_mut() {
            *v = silu(*v);
        }
        let e = self.time_fc2.forward(&e, 1); // [dim]
        let mut e_pre = vec![0.0f32; cfg.dim];
        for (d, s) in e_pre.iter_mut().zip(&e) {
            *d = silu(*s);
        }
        let e0 = self.time_proj.forward(&e_pre, 1); // [6*dim] -> [6, dim]

        // ---- text embedding: pad to text_len ----
        let n_txt = args.text_tokens.min(cfg.text_len);
        let mut txt_in = vec![0.0f32; cfg.text_len * cfg.text_dim];
        txt_in[..n_txt * cfg.text_dim].copy_from_slice(&args.text_ctx[..n_txt * cfg.text_dim]);
        let mut ctx = self.text_fc1.forward(&txt_in, cfg.text_len);
        for v in ctx.iter_mut() {
            *v = gelu_tanh(*v);
        }
        let ctx = self.text_fc2.forward(&ctx, cfg.text_len); // [text_len, dim]

        // ---- CLIP embedding ----
        let clip_out = self
            .img_emb
            .forward(args.clip_fea, cfg.clip_tokens); // [257, dim]
        let ctx_tokens = cfg.clip_tokens + cfg.text_len;
        let mut context = vec![0.0f32; ctx_tokens * cfg.dim];
        context[..clip_out.len()].copy_from_slice(&clip_out);
        context[clip_out.len()..].copy_from_slice(&ctx);

        // ---- audio conditioning (AudioProj + window arrangement) ----
        let (first, latter, audio_frames) =
            self.audio_proj
                .arrange(args.audio_pixel, args.audio_pixel_frames);
        let audio = self.audio_proj.forward(&first, &latter, audio_frames);

        // ---- blocks ----
        let start_f = args.start_idx / frame_len;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                &e0,
                &mut args.kv_caches[i],
                &self.rope,
                gh,
                gw,
                &context,
                ctx_tokens,
                cfg.clip_tokens,
                &audio.data,
                start_f,
                frames,
                frame_len,
                args.start_idx,
                args.end_idx,
                args.update_cache,
                args.skip_audio,
                args.mean_memory,
            );
        }

        // ---- head: modulation = [2, dim] + e ----
        let mut head_e = vec![0.0f32; 2 * cfg.dim];
        for i in 0..2 {
            for j in 0..cfg.dim {
                head_e[i * cfg.dim + j] =
                    self.head_modulation[i * cfg.dim + j] + e[j];
            }
        }
        let normed = layer_norm(&x, cfg.dim, None, None, cfg.eps);
        let mut h = normed;
        for (i, v) in h.iter_mut().enumerate() {
            let j = i % cfg.dim; // per-token broadcast of [dim] vectors
            let scale = head_e[cfg.dim + j]; // e[1]
            let shift = head_e[j]; // e[0]
            *v = *v * (1.0 + scale) + shift;
        }
        let head_out = self.head_w.forward(&h, tokens); // [tokens, out*ph*pw]

        self.unpatchify(&head_out, frames, gh, gw)
    }

    pub fn param_count(&self) -> usize {
        let cfg = &self.cfg;
        let mut n = self.patch_w.len() + self.patch_b.len();
        for l in [&self.text_fc1, &self.text_fc2, &self.time_fc1, &self.time_fc2, &self.time_proj] {
            n += l.w.len() + l.b.as_ref().map(|b| b.len()).unwrap_or(0);
        }
        n += self.img_emb.fc1.w.len() * 2; // approx (fc1+fc2)
        n += self.audio_proj.proj1.w.len()
            + self.audio_proj.proj1_vf.w.len()
            + self.audio_proj.proj2.w.len()
            + self.audio_proj.proj3.w.len();
        for b in &self.blocks {
            let linears = [
                &b.self_attn.q, &b.self_attn.k, &b.self_attn.v, &b.self_attn.o,
                &b.cross_attn.q, &b.cross_attn.k, &b.cross_attn.v, &b.cross_attn.o,
                &b.cross_attn.k_img, &b.cross_attn.v_img,
                &b.audio_attn.q_linear, &b.audio_attn.kv_linear, &b.audio_attn.proj,
                &b.ffn_up, &b.ffn_down,
            ];
            for l in linears {
                n += l.w.len() + l.b.as_ref().map(|b| b.len()).unwrap_or(0);
            }
            n += b.modulation.len();
            n += b.self_attn.memory_proj_k.len() + b.self_attn.memory_proj_v.len();
        }
        n += self.head_w.w.len() + self.head_w.b.as_ref().map(|b| b.len()).unwrap_or(0);
        n += self.head_modulation.len();
        let _ = cfg;
        n
    }
}

fn normal_init(l: &mut Linear, rng: &mut Rng, std: f32) {
    for v in l.w.iter_mut() {
        *v = rng.normal() * std;
    }
    if let Some(b) = &mut l.b {
        b.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_unpatchify_inverse_shapes() {
        let cfg = LiveActConfig::tiny();
        let mut model = WanModel::new(&cfg);
        let mut rng = Rng::new(5);
        model.init_weights(&mut rng);
        let (lz, ly) = cfg.lat_hw();
        let frames = 6;
        let mut input = vec![0.0f32; cfg.in_dim * frames * lz * ly];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i * 31) % 97) as f32 * 0.01;
        }
        let tokens = frames * cfg.frame_len();
        let x = model.patchify(&input, cfg.in_dim, frames, lz, ly);
        assert_eq!(x.len(), tokens * cfg.dim);
        // head output with identity-ish shape check via unpatchify
        let per = cfg.out_dim * 4;
        let head: Vec<f32> = (0..tokens * per).map(|i| (i as f32 * 0.001).sin()).collect();
        let out = model.unpatchify(&head, frames, cfg.grid_hw().0, cfg.grid_hw().1);
        assert_eq!(out.len(), cfg.out_dim * frames * lz * ly);
    }

    #[test]
    fn sinusoidal_matches_reference_layout() {
        // torch: cat([cos(outer(p, 10000^-j/half)), sin(...)], dim=1)
        let e = sinusoidal_embedding_1d(8, 1000.0);
        assert_eq!(e.len(), 8);
        assert!((e[0] - 1000f64.cos() as f32).abs() < 1e-4);
        assert!((e[4] - 1000f64.sin() as f32).abs() < 1e-4);
        let j = 1;
        let freq = 1000.0 * 10000f64.powf(-(j as f64) / 4.0);
        assert!((e[1] - freq.cos() as f32).abs() < 1e-4);
        assert!((e[5] - freq.sin() as f32).abs() < 1e-4);
    }

    #[test]
    fn full_forward_runs() {
        let cfg = LiveActConfig::tiny();
        let mut model = WanModel::new(&cfg);
        let mut rng = Rng::new(9);
        model.init_weights(&mut rng);
        let (lz, ly) = cfg.lat_hw();
        let frame_len = cfg.frame_len();
        let mut caches = model.new_kv_caches(false);
        // chunk 0: 6 frames, start 0
        let frames = 6;
        let latent: Vec<f32> = (0..cfg.out_dim * frames * lz * ly)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let y: Vec<f32> = vec![0.0; (cfg.out_dim + 4) * frames * lz * ly];
        let text: Vec<f32> = vec![0.01; 12 * cfg.text_dim];
        let clip: Vec<f32> = vec![0.02; cfg.clip_tokens * cfg.clip_dim];
        let (w, s, c) = (cfg.audio_window, cfg.audio_blocks, cfg.audio_channels);
        let audio: Vec<f32> = vec![0.03; cfg.cond_frame_num() * w * s * c];
        let out = model.forward(ForwardArgs {
            latent: &latent,
            y: &y,
            t: 1000.0,
            text_tokens: 12,
            text_ctx: &text,
            clip_fea: &clip,
            audio_pixel: &audio,
            audio_pixel_frames: cfg.cond_frame_num(),
            kv_caches: &mut caches[0],
            start_idx: 0,
            end_idx: 6 * frame_len,
            update_cache: false,
            skip_audio: true,
            mean_memory: false,
        });
        assert_eq!(out.len(), cfg.out_dim * frames * lz * ly);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
