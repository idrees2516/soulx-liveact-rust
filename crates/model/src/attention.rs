//! Attention modules ported from `model_liveact/model_memory.py` and
//! `model_liveact/attention.py`:
//!
//! - [`SelfAttention`] — `WanSelfAttention`: QK-normed self-attention over the
//!   rolling 14-frame KV window with **ConvKV Memory** (depthwise 5:1 conv
//!   compression of history frames, constant memory for hour-scale streams)
//!   and optional fp8-e4m3 KV caching (`--fp8_kv_cache`).
//! - [`CrossAttention`] — `WanI2VCrossAttention`: text (T5) + image (CLIP)
//!   conditioning.
//! - [`AudioAttention`] — `SingleStreamAttention`: per-frame audio
//!   cross-attention over the 32 audio context tokens of each latent frame.

use rayon::prelude::*;

use crate::config::LiveActConfig;
use crate::rope::Rope3d;
use crate::tensor::{fp8, rms_norm, softmax_inplace, Linear};

// ---------------------------------------------------------------------------
// Attention kernel (NHD layout, batch = 1 sequence of tokens)
// ---------------------------------------------------------------------------

/// q: `[s, heads, d]`, k/v: `[t, heads, d]` -> out `[s, heads, d]`.
/// Parallel over heads; one scratch scores row per query token.
pub fn attn_nhd(q: &[f32], k: &[f32], v: &[f32], s: usize, t: usize, heads: usize, d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    // one buffer per head (parallel over heads), merged afterwards
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let mut oh = vec![0.0f32; s * d];
            let mut row = vec![0.0f32; t];
            for i in 0..s {
                let qv = &q[i * heads * d + h * d..i * heads * d + (h + 1) * d];
                for (j, r) in row.iter_mut().enumerate() {
                    let kv = &k[j * heads * d + h * d..j * heads * d + (h + 1) * d];
                    let mut acc = 0.0;
                    for (a, b) in qv.iter().zip(kv) {
                        acc += a * b;
                    }
                    *r = acc * scale;
                }
                softmax_inplace(&mut row);
                let ov = &mut oh[i * d..(i + 1) * d];
                for (o, w) in ov.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for (j, wgt) in row.iter().enumerate() {
                        acc += wgt * v[j * heads * d + h * d + o];
                    }
                    *w = acc;
                }
            }
            oh
        })
        .collect();
    let mut out = vec![0.0f32; s * heads * d];
    for (h, oh) in per_head.iter().enumerate() {
        for i in 0..s {
            out[i * heads * d + h * d..i * heads * d + (h + 1) * d]
                .copy_from_slice(&oh[i * d..(i + 1) * d]);
        }
    }
    out
}

/// Batched attention over `b` independent frames: q `[b, s, h, d]`,
/// k/v `[b, t, h, d]` -> out `[b, s, h, d]` (SingleStreamAttention layout).
pub fn attn_batched(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    t: usize,
    heads: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let stride_q = s * heads * d;
    let stride_kv = t * heads * d;
    // parallel over (frame, head) pairs
    let parts: Vec<(usize, usize, Vec<f32>)> = (0..b * heads)
        .into_par_iter()
        .map(|bh| {
            let (bi, h) = (bh / heads, bh % heads);
            let qb = &q[bi * stride_q..(bi + 1) * stride_q];
            let kb = &k[bi * stride_kv..(bi + 1) * stride_kv];
            let vb = &v[bi * stride_kv..(bi + 1) * stride_kv];
            let mut oh = vec![0.0f32; s * d];
            let mut row = vec![0.0f32; t];
            for i in 0..s {
                let qv = &qb[i * heads * d + h * d..i * heads * d + (h + 1) * d];
                for (j, r) in row.iter_mut().enumerate() {
                    let kv = &kb[j * heads * d + h * d..j * heads * d + (h + 1) * d];
                    let mut acc = 0.0;
                    for (a, c) in qv.iter().zip(kv) {
                        acc += a * c;
                    }
                    *r = acc * scale;
                }
                softmax_inplace(&mut row);
                let ov = &mut oh[i * d..(i + 1) * d];
                for (o, w) in ov.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for (j, wgt) in row.iter().enumerate() {
                        acc += wgt * vb[j * heads * d + h * d + o];
                    }
                    *w = acc;
                }
            }
            (bi, h, oh)
        })
        .collect();
    let mut out = vec![0.0f32; b * stride_q];
    for (bi, h, oh) in parts {
        for i in 0..s {
            out[bi * stride_q + i * heads * d + h * d
                ..bi * stride_q + i * heads * d + (h + 1) * d]
                .copy_from_slice(&oh[i * d..(i + 1) * d]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ConvKV memory cache
// ---------------------------------------------------------------------------

/// Rolling KV cache of one layer at one Neighbor-Forcing diffusion step.
///
/// Frame-slot layout (from `WanSelfAttention.forward`), `F = frame tokens`:
/// - slots `[0, 2)`  — first two frames of the whole stream (identity anchor)
/// - slots `[2, 4)`  — compressed memory: slot 2 = conv(frames 2..7),
///                     slot 3 = conv(frames 7..12); re-compressed each chunk,
///                     so the memory compounds hierarchically (log-scale past)
/// - slots `[4, 6)`  — last two frames of the previous chunk
/// - slots `[6, 14)` — the current 8-frame chunk (first chunk writes its 6
///                     frames into slots `[0, 6)`)
///
/// Every chunk therefore attends over exactly 14 frame slots regardless of
/// stream position: constant memory, the ConvKV property.
pub struct KvCache {
    pub k: Vec<f32>, // [frames_total, frame_len, heads, head_dim]
    pub v: Vec<f32>,
    frames_total: usize,
    frame_len: usize,
    heads: usize,
    head_dim: usize,
    /// fp8 e4m3 storage (`--fp8_kv_cache`): quantized bytes + per-token scales
    pub fp8: bool,
    pub q8: Vec<u8>,
    pub q8_v: Vec<u8>,
    pub q8_scale: Vec<f32>,
    pub q8_scale_v: Vec<f32>,
}

impl KvCache {
    pub fn new(frames_total: usize, frame_len: usize, heads: usize, head_dim: usize, fp8: bool) -> Self {
        let n = frames_total * frame_len * heads * head_dim;
        KvCache {
            k: vec![0.0; n],
            v: vec![0.0; n],
            frames_total,
            frame_len,
            heads,
            head_dim,
            fp8,
            q8: if fp8 { vec![0; n] } else { vec![] },
            q8_v: if fp8 { vec![0; n] } else { vec![] },
            q8_scale: if fp8 { vec![0.0; frames_total * frame_len * heads] } else { vec![] },
            q8_scale_v: if fp8 { vec![0.0; frames_total * frame_len * heads] } else { vec![] },
        }
    }

    pub fn bytes_allocated(&self) -> usize {
        let f32_bytes = (self.k.len() + self.v.len()) * 4;
        if self.fp8 {
            self.q8.len() * 2 + (self.q8_scale.len() + self.q8_scale_v.len()) * 4
                + f32_bytes // transient host buffers still allocated
        } else {
            f32_bytes
        }
    }

    /// Persist the current f32 working set into the (possibly fp8) store.
    fn write_store(&mut self) {
        if !self.fp8 {
            return; // f32 buffers are the store
        }
        let d = self.heads * self.head_dim;
        let rows = self.k.len() / d;
        let (qk, sk) = fp8::quantize_rows(&self.k, d, rows);
        let (qv, sv) = fp8::quantize_rows(&self.v, d, rows);
        self.q8 = qk;
        self.q8_v = qv;
        self.q8_scale = sk;
        self.q8_scale_v = sv;
    }

    /// Materialize the f32 working set from the store (fp8 dequantize).
    fn load_store(&mut self) {
        if !self.fp8 {
            return;
        }
        let d = self.heads * self.head_dim;
        let rows = self.q8.len() / d;
        self.k = fp8::dequantize_rows(&self.q8, &self.q8_scale, d, rows);
        self.v = fp8::dequantize_rows(&self.q8_v, &self.q8_scale_v, d, rows);
    }
}

// ---------------------------------------------------------------------------
// WanSelfAttention
// ---------------------------------------------------------------------------

pub struct SelfAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub norm_q_w: Vec<f32>,
    pub norm_k_w: Vec<f32>,
    /// depthwise Conv1d(dim, dim, k=5, s=5, groups=dim, bias=False):
    /// weights `[dim, 5]`, initialized to 1/5 (moving average) as in
    /// `WanSelfAttention.post_init`.
    pub memory_proj_k: Vec<f32>, // [dim, 5]
    pub memory_proj_v: Vec<f32>, // [dim, 5]
    dim: usize,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    pub fn new(cfg: &LiveActConfig) -> Self {
        let dim = cfg.dim;
        SelfAttention {
            q: Linear::new(dim, dim, true),
            k: Linear::new(dim, dim, true),
            v: Linear::new(dim, dim, true),
            o: Linear::new(dim, dim, true),
            norm_q_w: vec![1.0; dim],
            norm_k_w: vec![1.0; dim],
            memory_proj_k: vec![1.0 / 5.0; dim * 5],
            memory_proj_v: vec![1.0 / 5.0; dim * 5],
            dim,
            heads: cfg.num_heads,
            head_dim: dim / cfg.num_heads,
        }
    }

    /// `WanSelfAttention.k_compress`: frames `[B, N, H, C]` -> `[B, N/5, H, C]`
    /// via the grouped depthwise conv along the token axis.
    fn compress(&self, src: &[f32], frames: usize, weight: &[f32]) -> Vec<f32> {
        let dim = self.dim;
        let per_frame = dim; // heads*head_dim
        let out_frames = frames / 5;
        let mut out = vec![0.0f32; out_frames * per_frame];
        for (fo, chunk) in out.chunks_mut(per_frame).enumerate() {
            for ch in 0..dim {
                let w = &weight[ch * 5..ch * 5 + 5];
                let mut acc = 0.0;
                for (wi, wv) in w.iter().enumerate() {
                    let src_frame = fo * 5 + wi;
                    acc += wv * src[src_frame * per_frame + ch];
                }
                chunk[ch] = acc;
            }
        }
        out
    }

    /// `WanSelfAttention.kv_mean`: mean-pool every 5 frames
    /// (`--mean_memory`).
    fn mean_pool(src: &[f32], frames: usize, dim: usize) -> Vec<f32> {
        assert_eq!(frames % 5, 0);
        let out_frames = frames / 5;
        let mut out = vec![0.0f32; out_frames * dim];
        for fo in 0..out_frames {
            for ch in 0..dim {
                let mut acc = 0.0;
                for wi in 0..5 {
                    acc += src[(fo * 5 + wi) * dim + ch];
                }
                out[fo * dim + ch] = acc / 5.0;
            }
        }
        out
    }

    /// One forward over the current chunk.
    ///
    /// * `x`        — `[tokens, dim]` of the CURRENT chunk (6 or 8 frames)
    /// * `start_idx`— token index of the chunk start inside the 14-frame
    ///                window (0 for the first chunk, `6 * frame_len` after)
    /// * `end_idx`  — tokens to attend over (6*fs first chunk, 14*fs after)
    /// * `update_cache` — run the ConvKV compaction (chunks >= 2, matching
    ///                `update_cache = _ > 1` in generate.py)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        cache: &mut KvCache,
        rope: &Rope3d,
        gh: usize,
        gw: usize,
        start_idx: usize,
        end_idx: usize,
        update_cache: bool,
        mean_memory: bool,
    ) -> Vec<f32> {
        let tokens = x.len() / self.dim;
        let (heads, hd, dim) = (self.heads, self.head_dim, self.dim);
        let fs = cache.frame_len;
        debug_assert_eq!(tokens % fs, 0);

        cache.load_store(); // fp8: materialize f32 working set

        // q/k/v + RMS norm (norm applied over the full dim, then head-split,
        // exactly as `qkv_fn`)
        let q_raw = self.q.forward(x, tokens);
        let k_raw = self.k.forward(x, tokens);
        let v_raw = self.v.forward(x, tokens);
        let mut q = rms_norm(&q_raw, &self.norm_q_w, 1e-6);
        let k = rms_norm(&k_raw, &self.norm_k_w, 1e-6);
        let v = v_raw;

        // ---- ConvKV compaction (chunk boundaries) ----
        if update_cache {
            let dim = self.dim;
            let take = |buf: &[f32], a: usize, b: usize| {
                let per_frame = dim;
                let start = a * per_frame;
                let end = b * per_frame;
                buf[start..end].to_vec()
            };
            let k_a = take(&cache.k, 2, 7);
            let k_b = take(&cache.k, 7, 12);
            let v_a = take(&cache.v, 2, 7);
            let v_b = take(&cache.v, 7, 12);
            let (kc_a, vc_a, kc_b, vc_b) = if mean_memory {
                (
                    SelfAttention::mean_pool(&k_a, 5, dim),
                    SelfAttention::mean_pool(&v_a, 5, dim),
                    SelfAttention::mean_pool(&k_b, 5, dim),
                    SelfAttention::mean_pool(&v_b, 5, dim),
                )
            } else {
                (
                    self.compress(&k_a, 5, &self.memory_proj_k),
                    // NOTE: the released model compresses V with
                    // `memory_proj_k` (v_compress uses memory_proj_k in
                    // model_memory.py); ported faithfully.
                    self.compress(&v_a, 5, &self.memory_proj_k),
                    self.compress(&k_b, 5, &self.memory_proj_k),
                    self.compress(&v_b, 5, &self.memory_proj_k),
                )
            };
            let per_frame = dim;
            // slot 2 <- conv(frames 2..7), slot 3 <- conv(frames 7..12)
            cache.k[2 * per_frame..3 * per_frame].copy_from_slice(&kc_a);
            cache.v[2 * per_frame..3 * per_frame].copy_from_slice(&vc_a);
            cache.k[3 * per_frame..4 * per_frame].copy_from_slice(&kc_b);
            cache.v[3 * per_frame..4 * per_frame].copy_from_slice(&vc_b);
            // slots 4,5 <- previous chunk's last two frames (12,13)
            let (k44, v44) = (cache.k[12 * per_frame..13 * per_frame].to_vec(), cache.v[12 * per_frame..13 * per_frame].to_vec());
            let (k45, v45) = (cache.k[13 * per_frame..14 * per_frame].to_vec(), cache.v[13 * per_frame..14 * per_frame].to_vec());
            cache.k[4 * per_frame..5 * per_frame].copy_from_slice(&k44);
            cache.v[4 * per_frame..5 * per_frame].copy_from_slice(&v44);
            cache.k[5 * per_frame..6 * per_frame].copy_from_slice(&k45);
            cache.v[5 * per_frame..6 * per_frame].copy_from_slice(&v45);
        }

        // ---- write current chunk ----
        let per_frame = dim;
        if start_idx != 0 {
            debug_assert_eq!(tokens, (cache.frames_total - 6) * fs);
            cache.k[6 * per_frame..6 * per_frame + tokens * dim].copy_from_slice(&k);
            cache.v[6 * per_frame..6 * per_frame + tokens * dim].copy_from_slice(&v);
        } else {
            debug_assert_eq!(tokens, 6 * fs);
            cache.k[..tokens * dim].copy_from_slice(&k);
            cache.v[..tokens * dim].copy_from_slice(&v);
        }

        // ---- attention over the cache window ----
        let current_start_frame = start_idx / fs;
        rope.apply(&mut q, tokens, heads, gh, gw, current_start_frame);
        // the full 14-frame cache is roped with absolute frame ids from 0
        let cache_frames = cache.frames_total;
        let mut roped_key = cache.k.clone();
        rope.apply(&mut roped_key, cache_frames * fs, heads, gh, gw, 0);

        let attend_tokens = end_idx;
        let k_win = &roped_key[..attend_tokens * dim];
        let v_win = &cache.v[..attend_tokens * dim];
        // `attn_nhd` output `[tokens, heads, hd]` is already laid out
        // contiguously as `[tokens, dim]` (head-major concat), so it feeds the
        // output projection directly.
        let out = attn_nhd(&q, k_win, v_win, tokens, attend_tokens, heads, hd);

        cache.write_store(); // fp8: re-quantize
        self.o.forward(&out, tokens)
    }
}

// ---------------------------------------------------------------------------
// WanI2VCrossAttention (text + CLIP image)
// ---------------------------------------------------------------------------

pub struct CrossAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub norm_q_w: Vec<f32>,
    pub norm_k_w: Vec<f32>,
    pub k_img: Linear,
    pub v_img: Linear,
    pub norm_k_img_w: Vec<f32>,
    dim: usize,
    heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    pub fn new(cfg: &LiveActConfig) -> Self {
        let dim = cfg.dim;
        CrossAttention {
            q: Linear::new(dim, dim, true),
            k: Linear::new(dim, dim, true),
            v: Linear::new(dim, dim, true),
            o: Linear::new(dim, dim, true),
            norm_q_w: vec![1.0; dim],
            norm_k_w: vec![1.0; dim],
            k_img: Linear::new(dim, dim, true),
            v_img: Linear::new(dim, dim, true),
            norm_k_img_w: vec![1.0; dim],
            dim,
            heads: cfg.num_heads,
            head_dim: dim / cfg.num_heads,
        }
    }

    /// `context`: `[ctx_tokens, dim]` = [clip_tokens(257) + text tokens] after
    /// the model-level embedding; the first `clip_tokens` are CLIP features.
    pub fn forward(&self, x: &[f32], context: &[f32], ctx_tokens: usize, clip_tokens: usize) -> Vec<f32> {
        let tokens = x.len() / self.dim;
        let (heads, hd, dim) = (self.heads, self.head_dim, self.dim);
        let q = rms_norm(&self.q.forward(x, tokens), &self.norm_q_w, 1e-6);

        let img_ctx = &context[..clip_tokens * dim];
        let txt_ctx = &context[clip_tokens * dim..];

        let k_img = rms_norm(&self.k_img.forward(img_ctx, clip_tokens), &self.norm_k_img_w, 1e-6);
        let v_img = self.v_img.forward(img_ctx, clip_tokens);
        let k = rms_norm(&self.k.forward(txt_ctx, ctx_tokens - clip_tokens), &self.norm_k_w, 1e-6);
        let v = self.v.forward(txt_ctx, ctx_tokens - clip_tokens);

        let img_x = attn_nhd(&q, &k_img, &v_img, tokens, clip_tokens, heads, hd);
        let txt_x = attn_nhd(&q, &k, &v, tokens, ctx_tokens - clip_tokens, heads, hd);

        // both outputs share the [tokens, heads*hd] flat layout
        let mut fused = vec![0.0; tokens * dim];
        for (f, (a, b)) in fused.iter_mut().zip(img_x.iter().zip(&txt_x)) {
            *f = a + b;
        }
        self.o.forward(&fused, tokens)
    }
}

// ---------------------------------------------------------------------------
// SingleStreamAttention (audio conditioning)
// ---------------------------------------------------------------------------

pub struct AudioAttention {
    pub q_linear: Linear,           // dim -> dim, bias
    pub kv_linear: Linear,          // audio_dim -> 2*dim, bias
    pub proj: Linear,               // dim -> dim, bias
    dim: usize,
    heads: usize,
    head_dim: usize,
    audio_dim: usize,
    audio_context_tokens: usize,
}

impl AudioAttention {
    pub fn new(cfg: &LiveActConfig) -> Self {
        assert!(cfg.dim % cfg.num_heads == 0, "dim must be divisible by num_heads");
        AudioAttention {
            q_linear: Linear::new(cfg.dim, cfg.dim, true),
            kv_linear: Linear::new(cfg.audio_channels, 2 * cfg.dim, true),
            proj: Linear::new(cfg.dim, cfg.dim, true),
            dim: cfg.dim,
            heads: cfg.num_heads,
            head_dim: cfg.dim / cfg.num_heads,
            audio_dim: cfg.audio_channels,
            audio_context_tokens: cfg.audio_context_tokens,
        }
    }

    /// * `x` — `[tokens, dim]` of the current chunk (frames * gh * gw tokens)
    /// * `audio` — `[total_frames, audio_ctx_tokens, audio_dim]` (per latent
    ///   frame 32 audio context tokens)
    /// * `start_f` — first frame index of this chunk in `audio`
    /// * `frames` — latent frames in this chunk
    /// * `frame_len` — tokens per frame
    ///
    /// Per frame: spatial queries attend that frame's 32 audio tokens.
    pub fn forward(
        &self,
        x: &[f32],
        audio: &[f32],
        start_f: usize,
        frames: usize,
        frame_len: usize,
    ) -> Vec<f32> {
        let tokens = frames * frame_len;
        debug_assert_eq!(x.len(), tokens * self.dim);
        let (heads, hd, dim) = (self.heads, self.head_dim, self.dim);
        let m = self.audio_context_tokens;

        // per-frame queries: [frames, frame_len, heads, hd]
        let q = self.q_linear.forward(x, tokens);

        // slice audio K/V for this chunk's frames: [frames, m, ...]
        let kv = self.kv_linear.forward(
            &audio[start_f * m * self.audio_dim..(start_f + frames) * m * self.audio_dim],
            frames * m,
        ); // [frames*m, 2*dim]
        let mut k = vec![0.0f32; frames * m * dim];
        let mut v = vec![0.0f32; frames * m * dim];
        for r in 0..frames * m {
            for i in 0..dim {
                k[r * dim + i] = kv[r * 2 * dim + i];
                v[r * dim + i] = kv[r * 2 * dim + dim + i];
            }
        }

        // frame f queries: rows [f*frame_len, (f+1)*frame_len)
        let mut out = vec![0.0f32; tokens * dim];
        for f in 0..frames {
            let qs = f * frame_len;
            let qe = (f + 1) * frame_len;
            // extract q rows for this frame into contiguous [frame_len, dim]
            let mut qf = vec![0.0; frame_len * dim];
            qf.copy_from_slice(&q[qs * dim..qe * dim]);
            let kf = &k[f * m * dim..(f + 1) * m * dim];
            let vf = &v[f * m * dim..(f + 1) * m * dim];
            let of = attn_batched(&qf, kf, vf, 1, frame_len, m, heads, hd);
            out[qs * dim..qe * dim].copy_from_slice(&of);
        }
        self.proj.forward(&out, tokens)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_compress_is_average_when_init() {
        // With the 1/5 constant init, k_compress == mean over 5-frame groups.
        let cfg = LiveActConfig::tiny();
        let attn = SelfAttention::new(&cfg);
        let frames = 10;
        let mut src = vec![0.0f32; frames * cfg.dim];
        for (i, v) in src.iter_mut().enumerate() {
            *v = i as f32 * 0.01;
        }
        let comp = attn.compress(&src, frames, &attn.memory_proj_k);
        let mean = SelfAttention::mean_pool(&src, frames, cfg.dim);
        assert_eq!(comp.len(), mean.len());
        for (a, b) in comp.iter().zip(&mean) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn cache_layout_constant() {
        let cfg = LiveActConfig::tiny();
        let fs = cfg.frame_len();
        let cache = KvCache::new(14, fs, cfg.num_heads, cfg.dim / cfg.num_heads, false);
        assert_eq!(cache.k.len(), 14 * fs * cfg.dim);
        // bytes: 14*16 tokens * 384 dim * 2 (k+v) * 4B
        assert_eq!(cache.bytes_allocated(), 14 * fs * 384 * 2 * 4);
    }

    #[test]
    fn fp8_cache_roundtrip_shape() {
        let cfg = LiveActConfig::tiny();
        let fs = cfg.frame_len();
        let mut cache = KvCache::new(14, fs, cfg.num_heads, cfg.dim / cfg.num_heads, true);
        for (i, v) in cache.k.iter_mut().enumerate() {
            *v = (i as f32 * 0.03).sin() * 2.0;
        }
        cache.v.copy_from_slice(&cache.k);
        cache.write_store();
        cache.k.fill(0.0);
        cache.v.fill(0.0);
        cache.load_store();
        for (a, b) in cache.k.iter().zip(&cache.v) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn self_attention_runs_first_chunk() {
        let cfg = LiveActConfig::tiny();
        let attn = SelfAttention::new(&cfg);
        let rope = Rope3d::new(&cfg);
        let (gh, gw) = cfg.grid_hw();
        let fs = cfg.frame_len();
        let mut cache = KvCache::new(14, fs, cfg.num_heads, cfg.dim / cfg.num_heads, false);
        let tokens = 6 * fs;
        let x: Vec<f32> = (0..tokens * cfg.dim)
            .map(|i| ((i * 1103515245 + 12345) as i32 as f32 * 1e-4).sin())
            .collect();
        let out = attn.forward(&x, &mut cache, &rope, gh, gw, 0, 6 * fs, false, false);
        assert_eq!(out.len(), tokens * cfg.dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn audio_attention_shapes() {
        let cfg = LiveActConfig::tiny();
        let attn = AudioAttention::new(&cfg);
        let (gh, gw) = cfg.grid_hw();
        let fs = cfg.frame_len();
        let frames = 8;
        let tokens = frames * fs;
        let x: Vec<f32> = (0..tokens * cfg.dim)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let m = cfg.audio_context_tokens;
        let audio: Vec<f32> = vec![0.1; 14 * m * cfg.audio_channels];
        let out = attn.forward(&x, &audio, 6, frames, fs);
        assert_eq!(out.len(), tokens * cfg.dim);
        assert!(out.iter().all(|v| v.is_finite()));
        let _ = (gh, gw);
    }
}
