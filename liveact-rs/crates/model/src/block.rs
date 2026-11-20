//! `WanAttentionBlock` port (model_memory.py lines 395-510): adaLN-modulated
//! self-attention with ConvKV cache, text/CLIP cross-attention, audio
//! cross-attention, gated FFN.

use crate::attention::{AudioAttention, CrossAttention, KvCache, SelfAttention};
use crate::config::LiveActConfig;
use crate::tensor::{gelu_tanh, layer_norm, Linear, Rng};

pub struct AttentionBlock {
    // LayerNorms. norm1/norm2: no affine (WanLayerNorm default);
    // norm3: affine when cross_attn_norm; norm_x: affine.
    pub norm3_w: Option<Vec<f32>>,
    pub norm3_b: Option<Vec<f32>>,
    pub norm_x_w: Option<Vec<f32>>,
    pub norm_x_b: Option<Vec<f32>>,

    pub self_attn: SelfAttention,
    pub cross_attn: CrossAttention,
    pub audio_attn: AudioAttention,

    // FFN: Linear(dim, ffn) + gelu(tanh) + Linear(ffn, dim)
    pub ffn_up: Linear,
    pub ffn_down: Linear,

    // modulation: [6, dim] — (shift_msa, scale_msa, gate_msa,
    //                         shift_mlp, scale_mlp, gate_mlp)
    pub modulation: Vec<f32>,

    dim: usize,
}

impl AttentionBlock {
    pub fn new(cfg: &LiveActConfig) -> Self {
        let dim = cfg.dim;
        AttentionBlock {
            norm3_w: if cfg.cross_attn_norm { Some(vec![1.0; dim]) } else { None },
            norm3_b: if cfg.cross_attn_norm { Some(vec![0.0; dim]) } else { None },
            norm_x_w: Some(vec![1.0; dim]),
            norm_x_b: Some(vec![0.0; dim]),
            self_attn: SelfAttention::new(cfg),
            cross_attn: CrossAttention::new(cfg),
            audio_attn: AudioAttention::new(cfg),
            ffn_up: Linear::new(dim, cfg.ffn_dim, true),
            ffn_down: Linear::new(cfg.ffn_dim, dim, true),
            modulation: vec![0.0; 6 * dim],
            dim,
        }
    }

    /// One block forward over the current chunk.
    ///
    /// * `x`        — `[tokens, dim]` current-chunk tokens
    /// * `e0`       — `[6, dim]` timestep projection (shared across blocks)
    /// * `context`  — `[clip_tokens + text_tokens, dim]` conditioning
    /// * `audio`    — `[audio_frames, audio_ctx, audio_dim]` per latent frame
    /// * `start_idx`/`end_idx`/`update_cache` — ConvKV cache window control
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        e0: &[f32],
        cache: &mut KvCache,
        rope: &crate::rope::Rope3d,
        gh: usize,
        gw: usize,
        context: &[f32],
        ctx_tokens: usize,
        clip_tokens: usize,
        audio: &[f32],
        start_f: usize,
        frames: usize,
        frame_len: usize,
        start_idx: usize,
        end_idx: usize,
        update_cache: bool,
        skip_audio: bool,
        mean_memory: bool,
    ) -> Vec<f32> {
        let tokens = x.len() / self.dim;
        let dim = self.dim;
        debug_assert_eq!(e0.len(), 6 * dim);

        // e = (modulation + e0).chunk(6) — six per-dim vectors
        let e: Vec<Vec<f32>> = (0..6)
            .map(|i| {
                (0..dim)
                    .map(|j| self.modulation[i * dim + j] + e0[i * dim + j])
                    .collect()
            })
            .collect();

        // ---- self-attention: x = x + gate * self_attn(norm1(x)*(1+scale)+shift)
        let normed = layer_norm(x, dim, None, None, 1e-6);
        let mut h = normed;
        for (v, (s, sh)) in h.iter_mut().zip(e[1].iter().zip(&e[0])) {
            *v = *v * (1.0 + s) + sh;
        }
        let y = self.self_attn.forward(
            &h,
            cache,
            rope,
            gh,
            gw,
            start_idx,
            end_idx,
            update_cache,
            mean_memory,
        );
        let mut x: Vec<f32> = x.to_vec();
        for t in 0..tokens {
            for j in 0..dim {
                x[t * dim + j] += y[t * dim + j] * e[2][j];
            }
        }

        // ---- text + CLIP cross-attention (no gate, no adaLN)
        let normed = layer_norm(&x, dim, self.norm3_w.as_deref(), self.norm3_b.as_deref(), 1e-6);
        let y = self
            .cross_attn
            .forward(&normed, context, ctx_tokens, clip_tokens);
        for (xv, yv) in x.iter_mut().zip(&y) {
            *xv += yv;
        }

        // ---- audio cross-attention (per-frame, anchor frame zeroed)
        if !skip_audio {
            let normed = layer_norm(&x, dim, self.norm_x_w.as_deref(), self.norm_x_b.as_deref(), 1e-6);
            let mut x_a = self
                .audio_attn
                .forward(&normed, audio, start_f, frames, frame_len);
            if start_f == 0 {
                for v in x_a[..frame_len * dim].iter_mut() {
                    *v = 0.0;
                }
            }
            for (xv, yv) in x.iter_mut().zip(&x_a) {
                *xv += yv;
            }
        }

        // ---- FFN: x = x + gate * ffn(norm2(x)*(1+scale)+shift)
        let normed = layer_norm(&x, dim, None, None, 1e-6);
        let mut h = normed;
        for (v, (s, sh)) in h.iter_mut().zip(e[4].iter().zip(&e[3])) {
            *v = *v * (1.0 + s) + sh;
        }
        let up = self.ffn_up.forward(&h, tokens);
        let act: Vec<f32> = up.iter().map(|v| gelu_tanh(*v)).collect();
        let down = self.ffn_down.forward(&act, tokens);
        let mut out = x;
        for (o, (d, g)) in out
            .iter_mut()
            .zip(down.iter().zip(&e[5]))
        {
            *o += d * g;
        }
        out
    }

    pub fn init_weights(&mut self, rng: &mut Rng) {
        self.self_attn.q.xavier(rng);
        self.self_attn.k.xavier(rng);
        self.self_attn.v.xavier(rng);
        self.self_attn.o.xavier(rng);
        self.cross_attn.q.xavier(rng);
        self.cross_attn.k.xavier(rng);
        self.cross_attn.v.xavier(rng);
        self.cross_attn.o.xavier(rng);
        self.cross_attn.k_img.xavier(rng);
        self.cross_attn.v_img.xavier(rng);
        self.audio_attn.q_linear.xavier(rng);
        self.audio_attn.kv_linear.xavier(rng);
        self.audio_attn.proj.xavier(rng);
        self.ffn_up.xavier(rng);
        self.ffn_down.xavier(rng);
        let std = (self.dim as f32).sqrt().recip();
        for v in self.modulation.iter_mut() {
            *v = rng.normal() * std;
        }
    }
}
