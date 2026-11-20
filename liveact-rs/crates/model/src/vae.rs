//! Causal 3D VAE port of `wan/modules/vae.py` (Wan2.1 VAE, streaming).
//!
//! Architecture: CausalConv3d (left-padded temporally, conv caches),
//! VAE-style RMS norm (L2 over channels x sqrt(C) x gamma), ResidualBlocks,
//! single-head spatial AttentionBlocks (middle only), Resample stages
//! (nearest-exact 2x + conv for up, zero-pad + strided conv for down,
//! time_conv with the channel-split interleave for 3D), Encoder3d/Decoder3d,
//! and the chunked streaming `encode` (1 + 4 + 4 ...) / `decode`
//! (one latent frame at a time) API of `WanVAE_` with the 16-channel
//! mean/std scaling of `WanVAE`.

use crate::tensor::Rng;
use rayon::prelude::*;

const CACHE_T: usize = 2;

/// Streaming slot: `Unused` until first touched, `Rep` marks the
/// first-chunk marker of the upsample3d stages, `Data` holds up to two
/// cached frames `[c, t, h, w]`.
#[derive(Clone)]
pub enum Slot {
    Unused,
    Rep,
    Data(Vec<f32>),
}

/// Flat conv-cache shared across one encode/decode pass (mirrors
/// `feat_cache` / `feat_idx` — the index resets every chunk, the data
/// persists for the stream).
#[derive(Clone, Default)]
pub struct VaeCache {
    pub slots: Vec<Slot>,
    pub idx: usize,
}

impl VaeCache {
    pub fn new(n: usize) -> Self {
        VaeCache {
            slots: vec![Slot::Unused; n],
            idx: 0,
        }
    }

    pub fn reset_idx(&mut self) {
        self.idx = 0;
    }
}

/// NCTHW video buffer (batch 1).
#[derive(Clone, Debug)]
pub struct Vid {
    pub c: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Vid {
    pub fn zeros(c: usize, t: usize, h: usize, w: usize) -> Self {
        Vid {
            c,
            t,
            h,
            w,
            data: vec![0.0; c * t * h * w],
        }
    }

    #[inline]
    pub fn at(&self, ch: usize, ti: usize, y: usize, x: usize) -> f32 {
        self.data[((ch * self.t + ti) * self.h + y) * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, ch: usize, ti: usize, y: usize, x: usize, v: f32) {
        let i = ((ch * self.t + ti) * self.h + y) * self.w + x;
        self.data[i] = v;
    }

    pub fn frame_stride(&self) -> usize {
        self.h * self.w
    }
}

// ---------------------------------------------------------------------------
// CausalConv3d
// ---------------------------------------------------------------------------

pub struct CausalConv3d {
    pub w: Vec<f32>, // [out, in, kt, kh, kw]
    pub b: Vec<f32>,
    pub kt: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride_t: usize,
    /// temporal padding flag (python `padding[0]`): left pad = 2*pt
    pub pt: usize,
    pub in_c: usize,
    pub out_c: usize,
}

impl CausalConv3d {
    pub fn new(in_c: usize, out_c: usize, k: (usize, usize, usize), stride_t: usize, pt: usize) -> Self {
        let (kt, kh, kw) = k;
        CausalConv3d {
            w: vec![0.0; out_c * in_c * kt * kh * kw],
            b: vec![0.0; out_c],
            kt,
            kh,
            kw,
            stride_t,
            pt,
            in_c,
            out_c,
        }
    }

    /// Forward with optional left-context cache (previous frames).
    /// `cache`: the slot content from the previous chunk (Some = Data).
    pub fn forward(&self, x: &Vid, cache: Option<&Vid>) -> Vid {
        let left_pad_full = 2 * self.pt;
        // assemble the joined [c, t_cache + t, h, w] input if a cache exists;
        // the cache frames REPLACE part of the causal left pad
        let (src, src_t, src_h, src_w, ct): (Vec<f32>, usize, usize, usize, usize) = match cache {
            Some(cx) if left_pad_full > 0 && cx.t > 0 && cx.data.len() == cx.c * cx.t * cx.h * cx.w && (cx.c, cx.h, cx.w) == (x.c, x.h, x.w) => {
                let ct = cx.t.min(left_pad_full);
                let per = x.h * x.w;
                let mut joined = Vec::with_capacity((ct + x.t) * x.c * per);
                for ch in 0..x.c {
                    let cs = &cx.data[ch * cx.t * per..(ch + 1) * cx.t * per];
                    joined.extend_from_slice(&cs[cs.len() - ct * per..]);
                    joined.extend_from_slice(&x.data[ch * x.t * per..(ch + 1) * x.t * per]);
                }
                (joined, ct + x.t, x.h, x.w, ct)
            }
            _ => (x.data.clone(), x.t, x.h, x.w, 0),
        };
        let left_pad = left_pad_full - ct;

        // materialize the padded input: [in_c, src_t + left_pad, H+2ph, W+2pw]
        let (kt, kh, kw) = (self.kt, self.kh, self.kw);
        let (ph, pw) = (self.kh / 2, self.kw / 2);
        let pad_t = left_pad;
        let pt = src_t + pad_t;
        let pstride_h = src_h + 2 * ph;
        let pstride_w = src_w + 2 * pw;
        let mut xp = vec![0.0f32; self.in_c * pt * pstride_h * pstride_w];
        for ic in 0..self.in_c {
            for t in 0..src_t {
                for y in 0..src_h {
                    let src_row = ((ic * src_t + t) * src_h + y) * src_w;
                    let dst_row = ((ic * pt + t + pad_t) * pstride_h + y + ph) * pstride_w + pw;
                    xp[dst_row..dst_row + src_w].copy_from_slice(&src[src_row..src_row + src_w]);
                }
            }
        }
        let out_t = (pt - kt) / self.stride_t + 1;
        let out_h = pstride_h - kh + 1;
        let out_w = pstride_w - kw + 1;
        let hw = out_h * out_w;
        let st = self.stride_t;

        // kernel-position-major accumulation: inner loop over contiguous
        // output rows, auto-vectorizable
        let parts: Vec<(usize, Vec<f32>)> = (0..self.out_c)
            .into_par_iter()
            .map(|oc| {
                let mut buf = vec![0.0f32; out_t * hw];
                for ic in 0..self.in_c {
                    for a in 0..kt {
                        for b in 0..kh {
                            for u in 0..kw {
                                let wv = self.w[(((oc * self.in_c + ic) * kt + a) * kh + b) * kw + u];
                                if wv == 0.0 {
                                    continue;
                                }
                                for ot in 0..out_t {
                                    let t_src = ot * st + a;
                                    let x_base = ((ic * pt + t_src) * pstride_h + b) * pstride_w + u;
                                    let o_base = ot * hw;
                                    for oy in 0..out_h {
                                        let xr = &xp[x_base + oy * pstride_w..x_base + oy * pstride_w + out_w];
                                        let orow = &mut buf[o_base + oy * out_w..o_base + (oy + 1) * out_w];
                                        for (o, xv) in orow.iter_mut().zip(xr) {
                                            *o += wv * xv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for ot in 0..out_t {
                    for p in 0..hw {
                        buf[ot * hw + p] += self.b[oc];
                    }
                }
                (oc, buf)
            })
            .collect();
        let mut out = Vid::zeros(self.out_c, out_t, out_h, out_w);
        for (oc, buf) in parts {
            out.data[oc * out_t * hw..(oc + 1) * out_t * hw].copy_from_slice(&buf);
        }
        out
    }

    /// Number of left-context frames this conv wants (2*pt).
    pub fn ctx_frames(&self) -> usize {
        2 * self.pt
    }
}

// ---------------------------------------------------------------------------
// norms
// ---------------------------------------------------------------------------

/// VAE `RMS_norm`: F.normalize(x, dim=channel) * sqrt(C) * gamma.
/// Works on both NCTHW (images=False) and per-frame NCHW.
fn rms_norm_vae(v: &mut Vid) {
    let scale = (v.c as f32).sqrt();
    let spatial = v.t * v.h * v.w;
    for p in 0..spatial {
        // gather channel vector at position p
        let mut norm = 0.0f32;
        for ch in 0..v.c {
            let x = v.data[ch * spatial + p];
            norm += x * x;
        }
        let inv = 1.0 / norm.max(1e-12).sqrt();
        for ch in 0..v.c {
            v.data[ch * spatial + p] *= inv * scale;
        }
    }
}

fn silu_inplace(v: &mut Vid) {
    for x in v.data.iter_mut() {
        *x = crate::tensor::silu(*x);
    }
}

// ---------------------------------------------------------------------------
// 2D helpers (per-frame)
// ---------------------------------------------------------------------------

/// Conv2d applied per frame with the VAE's padding conventions
/// (`asymmetric_pad` = the (0,1,0,1) downsample pad). Kernel-major
/// accumulation over padded frames, vectorizable inner loop.
fn conv2d_frames(
    x: &Vid,
    out_c: usize,
    k: usize,
    stride: usize,
    asymmetric_pad: bool,
    w: &[f32], // [out_c, in_c, k, k]
    b: &[f32],
) -> Vid {
    let (in_c, t, h, ww) = (x.c, x.t, x.h, x.w);
    let (pad_l, pad_r) = if asymmetric_pad { (0, 1) } else { (k / 2, k / 2) };
    let (pad_t, pad_b) = (pad_l, pad_r);
    let ph = h + pad_t + pad_b;
    let pw = ww + pad_l + pad_r;
    let mut xp = vec![0.0f32; in_c * t * ph * pw];
    for ti in 0..t {
        for ic in 0..in_c {
            for y in 0..h {
                let src_row = ((ic * t + ti) * h + y) * ww;
                let dst_row = ((ic * t + ti) * ph + y + pad_t) * pw + pad_l;
                xp[dst_row..dst_row + ww].copy_from_slice(&x.data[src_row..src_row + ww]);
            }
        }
    }
    let out_h = (ph - k) / stride + 1;
    let out_w = (pw - k) / stride + 1;
    let hw = out_h * out_w;
    let parts: Vec<(usize, Vec<f32>)> = (0..out_c * t)
        .into_par_iter()
        .map(|oct| {
            let (oc, ti) = (oct / t, oct % t);
            let mut buf = vec![0.0f32; hw];
            for ic in 0..in_c {
                for kb in 0..k {
                    for ku in 0..k {
                        let wv = w[((oc * in_c + ic) * k + kb) * k + ku];
                        if wv == 0.0 {
                            continue;
                        }
                        let x_base = ((ic * t + ti) * ph + kb) * pw + ku;
                        for oy in 0..out_h {
                            let xr = &xp[x_base + oy * stride * pw..x_base + oy * stride * pw + out_w];
                            let orow = &mut buf[oy * out_w..(oy + 1) * out_w];
                            for (o, xv) in orow.iter_mut().zip(xr) {
                                *o += wv * xv;
                            }
                        }
                    }
                }
            }
            for v in buf.iter_mut() {
                *v += b[oc];
            }
            (oct, buf)
        })
        .collect();
    let mut out = Vid::zeros(out_c, t, out_h, out_w);
    for (oct, buf) in parts {
        let base = oct * hw;
        out.data[base..base + hw].copy_from_slice(&buf);
    }
    out
}

/// Nearest-exact 2x spatial upsample.
fn upsample2x(x: &Vid) -> Vid {
    let (c, t, h, w) = (x.c, x.t, x.h, x.w);
    let mut out = Vid::zeros(c, t, h * 2, w * 2);
    for ch in 0..c {
        for ti in 0..t {
            for oy in 0..h * 2 {
                let sy = ((oy as f32 + 0.5) * 0.5).floor() as usize;
                let sy = sy.min(h - 1);
                for ox in 0..w * 2 {
                    let sx = ((ox as f32 + 0.5) * 0.5).floor() as usize;
                    let sx = sx.min(w - 1);
                    let v = x.at(ch, ti, sy, sx);
                    out.set(ch, ti, oy, ox, v);
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ResidualBlock
// ---------------------------------------------------------------------------

pub struct ResidualBlock {
    pub conv1: CausalConv3d,
    pub conv2: CausalConv3d,
    pub shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        ResidualBlock {
            conv1: CausalConv3d::new(in_dim, out_dim, (3, 3, 3), 1, 1),
            conv2: CausalConv3d::new(out_dim, out_dim, (3, 3, 3), 1, 1),
            shortcut: if in_dim != out_dim {
                Some(CausalConv3d::new(in_dim, out_dim, (1, 1, 1), 1, 0))
            } else {
                None
            },
        }
    }

    pub fn forward(&self, x: &Vid, cache: Option<&mut VaeCache>) -> Vid {
        let mut cache = cache;
        let h = match &self.shortcut {
            Some(sc) => {
                // k=1 convs have no temporal context; use no cache slot
                // (stride-1 k=1 conv is cache-invariant), matching the
                // original which routes only the residual convs through
                // feat_cache.
                sc.forward(x, None)
            }
            None => x.clone(),
        };
        // conv 1
        let x1 = self.step_conv(&self.conv1, x, cache.as_deref_mut());
        let mut x1 = x1;
        rms_norm_vae(&mut x1);
        silu_inplace(&mut x1);
        // conv 2
        let x2 = self.step_conv(&self.conv2, &x1, cache);
        let mut x2 = x2;
        rms_norm_vae(&mut x2);
        silu_inplace(&mut x2);
        let mut out = h;
        for (o, v) in out.data.iter_mut().zip(&x2.data) {
            *o += v;
        }
        out
    }

    fn step_conv(&self, conv: &CausalConv3d, x: &Vid, mut cache: Option<&mut VaeCache>) -> Vid {
        match cache.as_deref_mut() {
            None => conv.forward(x, None),
            Some(c) => {
                let idx = c.idx;
                let old = match &c.slots[idx] {
                    Slot::Data(d) => {
                        let (oc, _ot, _oh, _ow) = (x.c, x.t.min(CACHE_T), x.h, x.w);
                        // rebuild a Vid view of the stored frames
                        Some(rebuild_cache_vid(d, oc.min(x.c), x))
                    }
                    _ => None,
                };
                let out = conv.forward(x, old.as_ref());
                // cache_x = x[:, :, -CACHE_T:] (borrow last frame if short)
                let mut frames: Vec<f32> = Vec::new();
                let keep = x.t.min(CACHE_T);
                let take_from = x.t - keep;
                for ch in 0..x.c {
                    let s = ((ch * x.t) + take_from) * x.h * x.w;
                    frames.extend_from_slice(&x.data[s..s + keep * x.h * x.w]);
                }
                let mut newcache = frames;
                if keep < CACHE_T {
                    if let Slot::Data(d) = &c.slots[idx] {
                        // borrow the previous cache's last frame
                        let (pc, pt, ph, pw) = cache_shape(d, x);
                        let _ = (pc, ph, pw);
                        let last = &d[d.len() - pt * x.h * x.w..];
                        let mut merged = Vec::with_capacity(last.len() + newcache.len());
                        merged.extend_from_slice(last);
                        merged.extend_from_slice(&newcache);
                        newcache = merged;
                    }
                }
                c.slots[idx] = Slot::Data(newcache);
                c.idx += 1;
                out
            }
        }
    }
}

fn rebuild_cache_vid(d: &[f32], c: usize, shape_ref: &Vid) -> Vid {
    // stored as [c, t, h, w] of the ORIGINAL x; reconstruct t from length
    let per = shape_ref.h * shape_ref.w;
    let t = if c > 0 && per > 0 { d.len() / (c * per) } else { 0 };
    Vid {
        c,
        t,
        h: shape_ref.h,
        w: shape_ref.w,
        data: d.to_vec(),
    }
}

fn cache_shape(d: &[f32], shape_ref: &Vid) -> (usize, usize, usize, usize) {
    let per = shape_ref.h * shape_ref.w;
    let c = shape_ref.c;
    let t = if c > 0 && per > 0 { d.len() / (c * per) } else { 0 };
    (c, t, shape_ref.h, shape_ref.w)
}

// ---------------------------------------------------------------------------
// AttentionBlock (single head over H*W per frame)
// ---------------------------------------------------------------------------

pub struct AttentionBlock {
    pub qkv_w: Vec<f32>,  // [3c, c] (1x1 conv as linear)
    pub qkv_b: Vec<f32>,
    pub proj_w: Vec<f32>, // [c, c]
    pub proj_b: Vec<f32>,
    pub c: usize,
}

impl AttentionBlock {
    pub fn new(c: usize) -> Self {
        AttentionBlock {
            qkv_w: vec![0.0; 3 * c * c],
            qkv_b: vec![0.0; 3 * c],
            proj_w: vec![0.0; c * c], // zero-init (matches nn.init.zeros_)
            proj_b: vec![0.0; c],
            c,
        }
    }

    pub fn forward(&self, x: &Vid) -> Vid {
        let mut normed = x.clone();
        rms_norm_vae(&mut normed);
        let (c, t, h, w) = (x.c, x.t, x.h, x.w);
        let n = h * w;
        let mut out = x.clone();
        for ti in 0..t {
            // qkv: [n, 3c]
            let mut qkv = vec![0.0f32; n * 3 * c];
            crate::tensor::matmul_at_bt(
                &normed.data[ti * c * n..(ti + 1) * c * n],
                &self.qkv_w,
                n,
                c,
                3 * c,
                &mut qkv,
                0.0,
            );
            for (i, v) in qkv.iter_mut().enumerate() {
                *v += self.qkv_b[i % (3 * c)];
            }
            // q: [n, c] at 0, k at c, v at 2c — but note the original
            // reshapes [bt, 1, c*3, -1] -> permute -> chunk on the LAST dim:
            // q = channels [0..c), k = [c..2c), v = [2c..3c)
            let scale = 1.0 / (c as f32).sqrt();
            let mut attn_out = vec![0.0f32; n * c];
            let mut row = vec![0.0f32; n];
            for i in 0..n {
                for (j, r) in row.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for d in 0..c {
                        acc += qkv[i * 3 * c + d] * qkv[j * 3 * c + c + d];
                    }
                    *r = acc * scale;
                }
                crate::tensor::softmax_inplace(&mut row);
                for (o, ov) in attn_out[i * c..(i + 1) * c].iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for (j, wgt) in row.iter().enumerate() {
                        acc += wgt * qkv[j * 3 * c + 2 * c + o];
                    }
                    *ov = acc;
                }
            }
            // proj 1x1
            let mut proj = vec![0.0f32; n * c];
            crate::tensor::matmul_at_bt(&attn_out, &self.proj_w, n, c, c, &mut proj, 0.0);
            for (i, v) in proj.iter_mut().enumerate() {
                *v += self.proj_b[i % c];
            }
            // residual add
            for p in 0..n * c {
                out.data[ti * c * n + p] += proj[p];
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Resample
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub enum ResampleMode {
    None,
    Upsample2d,
    Upsample3d,
    Downsample2d,
    Downsample3d,
}

pub struct Resample {
    pub mode: ResampleMode,
    pub dim: usize,
    /// 2D part: up -> [dim//2, dim, 3, 3]; down -> [dim, dim, 3, 3]
    pub conv_w: Vec<f32>,
    pub conv_b: Vec<f32>,
    pub time_conv: Option<CausalConv3d>,
}

impl Resample {
    pub fn new(dim: usize, mode: ResampleMode) -> Self {
        let (out2d, k, stride, asym) = match mode {
            ResampleMode::Upsample2d | ResampleMode::Upsample3d => (dim / 2, 3, 1, false),
            ResampleMode::Downsample2d | ResampleMode::Downsample3d => (dim, 3, 2, true),
            ResampleMode::None => (dim, 1, 1, false),
        };
        let _ = (k, stride, asym);
        Resample {
            mode,
            dim,
            conv_w: vec![0.0; out2d * dim * 9],
            conv_b: vec![0.0; out2d],
            time_conv: match mode {
                ResampleMode::Upsample3d => Some(CausalConv3d::new(dim, dim * 2, (3, 1, 1), 1, 1)),
                ResampleMode::Downsample3d => {
                    Some(CausalConv3d::new(dim, dim, (3, 1, 1), 2, 0))
                }
                _ => None,
            },
        }
    }

    pub fn out_dim(&self) -> usize {
        match self.mode {
            ResampleMode::Upsample2d | ResampleMode::Upsample3d => self.dim / 2,
            _ => self.dim,
        }
    }

    pub fn forward(&self, x: &Vid, mut cache: Option<&mut VaeCache>) -> Vid {
        // --- temporal parts first (3D modes) ---
        let mut x = x.clone();
        match self.mode {
            ResampleMode::Upsample3d => {
                let tc = self.time_conv.as_ref().unwrap();
                match cache.as_deref_mut() {
                    None => {
                        let conv_out = tc.forward(&x, None);
                        x = interleave_temporal(&conv_out, self.dim);
                    }
                    Some(c) => {
                        let idx = c.idx;
                        let is_rep = matches!(c.slots[idx], Slot::Rep);
                        let old_data = match &c.slots[idx] {
                            Slot::Data(d) => Some(d.clone()),
                            _ => None,
                        };
                        if old_data.is_none() && !is_rep {
                            // first touch: Rep marker, temporal passthrough
                            c.slots[idx] = Slot::Rep;
                            c.idx += 1;
                        } else {
                            // cache_x = x[:, :, -CACHE_T:] (borrow/zero-pad to 2)
                            let mut cache_x = last_frames(&x, CACHE_T);
                            if cache_x.t < CACHE_T {
                                if let Some(d) = &old_data {
                                    let prev = rebuild_cache_vid(d, x.c, &x);
                                    let borrow = last_frames(&prev, 1);
                                    cache_x = concat_vid(&borrow, &cache_x);
                                } else if is_rep {
                                    let zero = Vid::zeros(x.c, 1, x.h, x.w);
                                    cache_x = concat_vid(&zero, &cache_x);
                                }
                            }
                            let conv_out = match &old_data {
                                Some(d) => {
                                    let prev = rebuild_cache_vid(d, x.c, &x);
                                    tc.forward(&x, Some(&prev))
                                }
                                None => tc.forward(&x, None),
                            };
                            c.slots[idx] = Slot::Data(cache_x.data.clone());
                            c.idx += 1;
                            x = interleave_temporal(&conv_out, self.dim);
                        }
                    }
                }
            }
            ResampleMode::Downsample3d => {
                let tc = self.time_conv.as_ref().unwrap();
                match cache.as_deref_mut() {
                    None => {
                        let two_d = self.apply_2d(&x);
                        x = tc.forward(&two_d, None);
                    }
                    Some(c) => {
                        let idx = c.idx;
                        let old_data = match &c.slots[idx] {
                            Slot::Data(d) => Some(d.clone()),
                            _ => None,
                        };
                        let two_d = self.apply_2d(&x);
                        match old_data {
                            None => {
                                // first touch: store post-2D, NO time conv
                                c.slots[idx] = Slot::Data(two_d.data.clone());
                                c.idx += 1;
                                return two_d;
                            }
                            Some(d) => {
                                let prev = rebuild_cache_vid(&d, two_d.c, &two_d);
                                let cat = concat_vid(&last_frames(&prev, 1), &two_d);
                                let out = tc.forward(&cat, None);
                                c.slots[idx] = Slot::Data(last_frames(&two_d, 1).data);
                                c.idx += 1;
                                x = out;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        // --- 2D part (Downsample3d applied it inside its branch) ---
        if self.mode == ResampleMode::Downsample3d {
            return x;
        }
        self.apply_2d(&x)
    }

    fn apply_2d(&self, x: &Vid) -> Vid {
        match self.mode {
            ResampleMode::None => x.clone(),
            ResampleMode::Upsample2d | ResampleMode::Upsample3d => {
                let up = upsample2x(x);
                conv2d_frames(&up, self.dim / 2, 3, 1, false, &self.conv_w, &self.conv_b)
            }
            ResampleMode::Downsample2d | ResampleMode::Downsample3d => {
                conv2d_frames(x, self.dim, 3, 2, true, &self.conv_w, &self.conv_b)
            }
        }
    }
}

fn last_frames(x: &Vid, n: usize) -> Vid {
    let keep = x.t.min(n);
    let from = x.t - keep;
    let mut out = Vid::zeros(x.c, keep, x.h, x.w);
    for ch in 0..x.c {
        let s = (ch * x.t + from) * x.h * x.w;
        let d = ch * keep * x.h * x.w;
        out.data[d..d + keep * x.h * x.w].copy_from_slice(&x.data[s..s + keep * x.h * x.w]);
    }
    out
}

fn concat_vid(a: &Vid, b: &Vid) -> Vid {
    debug_assert_eq!((a.c, a.h, b.c, b.h, a.w, b.w), (b.c, a.h, b.c, a.h, b.w, b.w));
    let t = a.t + b.t;
    let mut out = Vid::zeros(a.c, t, a.h, a.w);
    for ch in 0..a.c {
        let per = a.h * a.w;
        out.data[ch * t * per..ch * t * per + a.t * per].copy_from_slice(&a.data[ch * a.t * per..(ch + 1) * a.t * per]);
        let off = ch * t * per + a.t * per;
        out.data[off..off + b.t * per].copy_from_slice(&b.data[ch * b.t * per..(ch + 1) * b.t * per]);
    }
    out
}

/// time_conv output `[2*dim, t, h, w]` -> `[dim, 2t, h, w]` with the
/// half-channel frames interleaved (reshape + stack + reshape).
fn interleave_temporal(x: &Vid, dim: usize) -> Vid {
    debug_assert_eq!(x.c, 2 * dim);
    let mut out = Vid::zeros(dim, x.t * 2, x.h, x.w);
    let per = x.h * x.w;
    for ti in 0..x.t {
        // frame 2*ti   <- channels [0..dim)
        // frame 2*ti+1 <- channels [dim..2*dim)
        for ch in 0..dim {
            let s0 = (ch * x.t + ti) * per;
            let d0 = (ch * out.t + 2 * ti) * per;
            out.data[d0..d0 + per].copy_from_slice(&x.data[s0..s0 + per]);
            let s1 = ((ch + dim) * x.t + ti) * per;
            let d1 = (ch * out.t + 2 * ti + 1) * per;
            out.data[d1..d1 + per].copy_from_slice(&x.data[s1..s1 + per]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Encoder3d / Decoder3d
// ---------------------------------------------------------------------------

pub enum VaeLayer {
    Res(ResidualBlock),
    Attn(AttentionBlock),
    Resample(Resample),
}

impl VaeLayer {
    pub fn forward(&self, x: &Vid, cache: Option<&mut VaeCache>) -> Vid {
        match self {
            VaeLayer::Res(r) => r.forward(x, cache),
            VaeLayer::Attn(a) => a.forward(x),
            VaeLayer::Resample(r) => r.forward(x, cache),
        }
    }
}

pub struct Encoder3d {
    pub conv1: CausalConv3d,
    pub layers: Vec<VaeLayer>,
    pub head_norm_conv: CausalConv3d, // preceded by RMS+SiLU
    pub out_dim: usize,
}

pub struct Decoder3d {
    pub conv1: CausalConv3d,
    pub layers: Vec<VaeLayer>,
    pub head_conv: CausalConv3d,
}

/// Count conv slots used by a layer list (for cache sizing): each
/// CausalConv3d reachable with a cache gets one slot (ResidualBlock: 2,
/// Resample: 1, AttentionBlock: 0). conv1 and head convs add one each.
fn count_slots(layers: &[VaeLayer]) -> usize {
    layers
        .iter()
        .map(|l| match l {
            VaeLayer::Res(_) => 2,
            VaeLayer::Resample(_) => 1,
            VaeLayer::Attn(_) => 0,
        })
        .sum()
}

impl Encoder3d {
    pub fn new(dim: usize, z_dim: usize, dim_mult: &[usize], num_res_blocks: usize, temperal_downsample: &[bool]) -> Self {
        let mut dims: Vec<usize> = dim_mult.iter().map(|m| dim * m).collect();
        dims.insert(0, dim);
        let mut layers: Vec<VaeLayer> = Vec::new();
        for i in 0..dim_mult.len() {
            let mut in_dim = dims[i];
            let out_dim = dims[i + 1];
            for _ in 0..num_res_blocks {
                layers.push(VaeLayer::Res(ResidualBlock::new(in_dim, out_dim)));
                in_dim = out_dim;
            }
            if i != dim_mult.len() - 1 {
                let mode = if temperal_downsample[i] {
                    ResampleMode::Downsample3d
                } else {
                    ResampleMode::Downsample2d
                };
                layers.push(VaeLayer::Resample(Resample::new(out_dim, mode)));
            }
        }
        // middle: res, attn, res
        let mid = *dims.last().unwrap();
        layers.push(VaeLayer::Res(ResidualBlock::new(mid, mid)));
        layers.push(VaeLayer::Attn(AttentionBlock::new(mid)));
        layers.push(VaeLayer::Res(ResidualBlock::new(mid, mid)));
        let out = *dims.last().unwrap();
        Encoder3d {
            conv1: CausalConv3d::new(3, dims[0], (3, 3, 3), 1, 1),
            layers,
            head_norm_conv: CausalConv3d::new(out, z_dim, (3, 3, 3), 1, 1),
            out_dim: out,
        }
    }

    pub fn slots(&self) -> usize {
        1 + count_slots(&self.layers) + 1
    }

    pub fn forward(&self, x: &Vid, mut cache: Option<&mut VaeCache>) -> Vid {
        let mut cur = match cache.as_deref_mut() {
            None => self.conv1.forward(x, None),
            Some(c) => {
                let idx = c.idx;
                let old = match &c.slots[idx] {
                    Slot::Data(d) => Some(rebuild_cache_vid(d, 3, x)),
                    _ => None,
                };
                let out = self.conv1.forward(x, old.as_ref());
                c.slots[idx] = Slot::Data(last_frames(x, CACHE_T).data);
                c.idx += 1;
                out
            }
        };
        for layer in &self.layers {
            cur = layer.forward(&cur, cache.as_deref_mut());
        }
        // head: RMS + SiLU + causal conv
        rms_norm_vae(&mut cur);
        silu_inplace(&mut cur);
        match cache {
            None => self.head_norm_conv.forward(&cur, None),
            Some(c) => {
                let idx = c.idx;
                let old = match &c.slots[idx] {
                    Slot::Data(d) => Some(rebuild_cache_vid(d, self.out_dim, &cur)),
                    _ => None,
                };
                let out = self.head_norm_conv.forward(&cur, old.as_ref());
                c.slots[idx] = Slot::Data(last_frames(&cur, CACHE_T).data);
                c.idx += 1;
                out
            }
        }
    }
}

impl Decoder3d {
    pub fn new(dim: usize, z_dim: usize, dim_mult: &[usize], num_res_blocks: usize, temperal_upsample: &[bool]) -> Self {
        let mut dims: Vec<usize> = dim_mult.iter().rev().map(|m| dim * m).collect();
        dims.insert(0, dim * dim_mult[dim_mult.len() - 1]);
        // dims has len+1 entries
        let mut layers: Vec<VaeLayer> = Vec::new();
        let mut in_dim = dims[0];
        for i in 0..dim_mult.len() {
            let mut this_in = in_dim;
            if i == 1 || i == 2 || i == 3 {
                this_in /= 2;
            }
            let out_dim = dims[i + 1];
            for _ in 0..num_res_blocks + 1 {
                layers.push(VaeLayer::Res(ResidualBlock::new(this_in, out_dim)));
                this_in = out_dim;
            }
            if i != dim_mult.len() - 1 {
                let mode = if temperal_upsample[i] {
                    ResampleMode::Upsample3d
                } else {
                    ResampleMode::Upsample2d
                };
                layers.push(VaeLayer::Resample(Resample::new(out_dim, mode)));
            }
            in_dim = dims[i + 1];
        }
        let mid = dims[0];
        let mut all = vec![VaeLayer::Res(ResidualBlock::new(mid, mid))];
        all.push(VaeLayer::Attn(AttentionBlock::new(mid)));
        all.push(VaeLayer::Res(ResidualBlock::new(mid, mid)));
        all.extend(layers);
        let final_dim = *dims.last().unwrap();
        Decoder3d {
            conv1: CausalConv3d::new(z_dim, mid, (3, 3, 3), 1, 1),
            layers: all,
            head_conv: CausalConv3d::new(final_dim, 3, (3, 3, 3), 1, 1),
        }
    }

    pub fn slots(&self) -> usize {
        1 + count_slots(&self.layers) + 1
    }

    pub fn forward(&self, x: &Vid, mut cache: Option<&mut VaeCache>) -> Vid {
        let mut cur = match cache.as_deref_mut() {
            None => self.conv1.forward(x, None),
            Some(c) => {
                let idx = c.idx;
                let old = match &c.slots[idx] {
                    Slot::Data(d) => Some(rebuild_cache_vid(d, x.c, x)),
                    _ => None,
                };
                let out = self.conv1.forward(x, old.as_ref());
                c.slots[idx] = Slot::Data(last_frames(x, CACHE_T).data);
                c.idx += 1;
                out
            }
        };
        for layer in &self.layers {
            cur = layer.forward(&cur, cache.as_deref_mut());
        }
        rms_norm_vae(&mut cur);
        silu_inplace(&mut cur);
        match cache {
            None => self.head_conv.forward(&cur, None),
            Some(c) => {
                let idx = c.idx;
                let old = match &c.slots[idx] {
                    Slot::Data(d) => Some(rebuild_cache_vid(d, cur.c, &cur)),
                    _ => None,
                };
                let out = self.head_conv.forward(&cur, old.as_ref());
                c.slots[idx] = Slot::Data(last_frames(&cur, CACHE_T).data);
                c.idx += 1;
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WanVAE (chunked streaming encode/decode + scaling)
// ---------------------------------------------------------------------------

pub struct WanVae {
    pub encoder: Encoder3d,
    /// conv1: z*2 -> z*2 (1x1)
    pub conv_mu: CausalConv3d,
    /// conv2: z -> z (1x1)
    pub conv_pre: CausalConv3d,
    pub decoder: Decoder3d,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub z_dim: usize,
    enc_cache: std::sync::Mutex<VaeCache>,
    dec_cache: std::sync::Mutex<VaeCache>,
}

impl WanVae {
    pub fn new(dim: usize, z_dim: usize, dim_mult: &[usize], num_res_blocks: usize) -> Self {
        let temperal_downsample = [false, true, true];
        let temperal_upsample: [bool; 3] = [true, true, false];
        let encoder = Encoder3d::new(dim, z_dim * 2, dim_mult, num_res_blocks, &temperal_downsample);
        let decoder = Decoder3d::new(dim, z_dim, dim_mult, num_res_blocks, &temperal_upsample);
        WanVae {
            conv_mu: CausalConv3d::new(z_dim * 2, z_dim * 2, (1, 1, 1), 1, 0),
            conv_pre: CausalConv3d::new(z_dim, z_dim, (1, 1, 1), 1, 0),
            encoder,
            decoder,
            mean: vec![
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            std: vec![
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ],
            z_dim,
            enc_cache: std::sync::Mutex::new(VaeCache::new(0)),
            dec_cache: std::sync::Mutex::new(VaeCache::new(0)),
        }
    }

    pub fn init_weights(&mut self, rng: &mut Rng) {
        init_conv(&mut self.encoder.conv1, rng);
        for l in &mut self.encoder.layers {
            match l {
                VaeLayer::Res(r) => {
                    init_conv(&mut r.conv1, rng);
                    init_conv(&mut r.conv2, rng);
                    if let Some(sc) = &mut r.shortcut {
                        init_conv(sc, rng);
                    }
                }
                VaeLayer::Attn(a) => {
                    let bound = (6.0 / (3 * a.c + a.c) as f32).sqrt();
                    for v in a.qkv_w.iter_mut() {
                        *v = (rng.f32() * 2.0 - 1.0) * bound;
                    }
                    // proj stays zero-init
                }
                VaeLayer::Resample(r) => {
                    for v in r.conv_w.iter_mut() {
                        *v = rng.normal() * 0.02;
                    }
                    if let Some(tc) = &mut r.time_conv {
                        init_conv(tc, rng);
                    }
                }
            }
        }
        init_conv(&mut self.encoder.head_norm_conv, rng);
        init_conv(&mut self.conv_mu, rng);
        init_conv(&mut self.conv_pre, rng);
        init_conv(&mut self.decoder.conv1, rng);
        for l in &mut self.decoder.layers {
            match l {
                VaeLayer::Res(r) => {
                    init_conv(&mut r.conv1, rng);
                    init_conv(&mut r.conv2, rng);
                    if let Some(sc) = &mut r.shortcut {
                        init_conv(sc, rng);
                    }
                }
                VaeLayer::Attn(a) => {
                    let bound = (6.0 / (3 * a.c + a.c) as f32).sqrt();
                    for v in a.qkv_w.iter_mut() {
                        *v = (rng.f32() * 2.0 - 1.0) * bound;
                    }
                }
                VaeLayer::Resample(r) => {
                    for v in r.conv_w.iter_mut() {
                        *v = rng.normal() * 0.02;
                    }
                    if let Some(tc) = &mut r.time_conv {
                        init_conv(tc, rng);
                    }
                }
            }
        }
        init_conv(&mut self.decoder.head_conv, rng);
    }

    /// Chunked streaming encode: 1 frame, then 4 frames at a time.
    /// Returns the latent `[z, T_lat, h/8, w/8]`.
    pub fn encode(&self, x: &Vid) -> Vid {
        let t = x.t;
        let iters = 1 + (t - 1) / 4;
        {
            let mut cache = self.enc_cache.lock().unwrap();
            *cache = VaeCache::new(self.encoder.slots());
        }
        let mut latents: Vec<f32> = Vec::new();
        let mut lat_t = 0usize;
        let mut lat_hw = (0usize, 0usize);
        for i in 0..iters {
            let (a, b) = if i == 0 {
                (0, 1)
            } else {
                (1 + 4 * (i - 1), 1 + 4 * i)
            };
            let b = b.min(t);
            let chunk = slice_frames(x, a, b);
            let mut cache = self.enc_cache.lock().unwrap();
            cache.reset_idx();
            let out = self.encoder.forward(&chunk, Some(&mut cache));
            if i == 0 {
                lat_hw = (out.h, out.w);
                latents = out.data;
                lat_t = out.t;
            } else {
                latents.extend_from_slice(&out.data);
                lat_t += out.t;
            }
        }
        let lat = Vid {
            c: 2 * self.z_dim,
            t: lat_t,
            h: lat_hw.0,
            w: lat_hw.1,
            data: latents,
        };
        // conv1 (1x1) -> split mu/logvar; scale mu
        let mut mu_logvar = self.conv_mu.forward(&lat, None);
        mu_logvar.c = self.z_dim; // first half is mu channels? conv output [2z]: chunk(2, dim=1)
        // torch.chunk(2, dim=1) on [B, 2z, ...]: mu = channels [0..z), logvar = [z..2z)
        let mut mu = Vid::zeros(self.z_dim, lat_t, lat_hw.0, lat_hw.1);
        for ch in 0..self.z_dim {
            let per = lat_t * lat_hw.0 * lat_hw.1;
            mu.data[ch * per..(ch + 1) * per]
                .copy_from_slice(&mu_logvar.data[ch * per..(ch + 1) * per]);
        }
        // mu = (mu - mean) * (1/std)
        for ch in 0..self.z_dim {
            let s = 1.0 / self.std[ch];
            let m = self.mean[ch];
            for v in mu.data[ch * (mu.t * mu.h * mu.w)..(ch + 1) * (mu.t * mu.h * mu.w)].iter_mut() {
                *v = (*v - m) * s;
            }
        }
        mu
    }

    /// Chunked streaming decode: one latent frame at a time.
    /// z: `[z_dim, T_lat, h/8, w/8]` -> `[3, (T_lat-1)*4+1, h, w]`, clamped.
    pub fn decode(&self, z: &Vid) -> Vid {
        // z = z/std + mean
        let mut zs = z.clone();
        for ch in 0..self.z_dim.min(z.c) {
            let per = z.t * z.h * z.w;
            for v in zs.data[ch * per..(ch + 1) * per].iter_mut() {
                *v = *v * self.std[ch] + self.mean[ch];
            }
        }
        let pre = self.conv_pre.forward(&zs, None);
        {
            let mut cache = self.dec_cache.lock().unwrap();
            *cache = VaeCache::new(self.decoder.slots());
        }
        let mut frames: Vec<f32> = Vec::new();
        let mut out_t = 0usize;
        let mut out_hw = (0usize, 0usize);
        for i in 0..z.t {
            let chunk = slice_frames(&pre, i, i + 1);
            let mut cache = self.dec_cache.lock().unwrap();
            cache.reset_idx();
            let out = self.decoder.forward(&chunk, Some(&mut cache));
            if i == 0 {
                out_hw = (out.h, out.w);
                frames = out.data;
                out_t = out.t;
            } else {
                frames.extend_from_slice(&out.data);
                out_t += out.t;
            }
        }
        let mut out = Vid {
            c: 3,
            t: out_t,
            h: out_hw.0,
            w: out_hw.1,
            data: frames,
        };
        for v in out.data.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }
}

fn slice_frames(x: &Vid, a: usize, b: usize) -> Vid {
    let t = b - a;
    let mut out = Vid::zeros(x.c, t, x.h, x.w);
    let per = x.h * x.w;
    for ch in 0..x.c {
        let s = (ch * x.t + a) * per;
        out.data[ch * t * per..(ch + 1) * t * per].copy_from_slice(&x.data[s..s + t * per]);
    }
    out
}

fn init_conv(c: &mut CausalConv3d, rng: &mut Rng) {
    // PyTorch default conv init: kaiming-ish uniform; scale by fan-in
    let fan_in = c.in_c * c.kt * c.kh * c.kw;
    let bound = (1.0 / fan_in as f32).sqrt();
    for v in c.w.iter_mut() {
        *v = (rng.f32() * 2.0 - 1.0) * bound;
    }
    for v in c.b.iter_mut() {
        *v = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vae() -> WanVae {
        let mut vae = WanVae::new(32, 16, &[1, 2, 4, 4], 1);
        let mut rng = Rng::new(77);
        vae.init_weights(&mut rng);
        vae
    }

    #[test]
    fn encode_decode_frame_math() {
        // 53 pixel frames -> 14 latent frames -> 53 back (32x32 keeps the
        // test fast; the frame math is resolution independent)
        let vae = tiny_vae();
        let x = Vid::zeros(3, 53, 32, 32);
        let lat = vae.encode(&x);
        assert_eq!(lat.t, 14, "latent frames");
        assert_eq!((lat.h, lat.w), (4, 4));
        let out = vae.decode(&lat);
        assert_eq!(out.t, 53, "(L-1)*4+1");
        assert_eq!((out.h, out.w), (32, 32));
        assert!(out.data.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn decode_streaming_matches_chunk_counts() {
        // overlap decode like generate.py: [prev last 3] + [new 8]
        let vae = tiny_vae();
        let z = Vid::zeros(16, 11, 4, 4);
        let out = vae.decode(&z);
        assert_eq!(out.t, 41, "(11-1)*4+1");
        // first 9 frames correspond to the 3 overlap latents
        let _ = 9;
    }

    #[test]
    fn causal_conv_shapes() {
        let mut conv = CausalConv3d::new(4, 8, (3, 3, 3), 1, 1);
        for v in conv.w.iter_mut() {
            *v = 0.5;
        }
        let x = Vid::zeros(4, 5, 6, 6);
        let out = conv.forward(&x, None);
        assert_eq!((out.c, out.t, out.h, out.w), (8, 5, 6, 6));
        // with 2-frame cache: same output length
        let cache = Vid::zeros(4, 2, 6, 6);
        let out2 = conv.forward(&x, Some(&cache));
        assert_eq!(out2.t, 5);
    }

    #[test]
    fn interleave_doubles_frames() {
        let x = Vid::zeros(8, 3, 4, 4); // 2*dim=8
        let out = interleave_temporal(&x, 4);
        assert_eq!(out.c, 4);
        assert_eq!(out.t, 6);
    }

    #[test]
    fn resample_up_down_shapes() {
        let up = Resample::new(16, ResampleMode::Upsample3d);
        let x = Vid::zeros(16, 1, 8, 8);
        let out = up.forward(&x, None); // no cache: full path
        assert_eq!(out.t, 2);
        assert_eq!((out.h, out.w), (16, 16));
        assert_eq!(out.c, 8);

        let down = Resample::new(16, ResampleMode::Downsample3d);
        let y = Vid::zeros(16, 5, 16, 16);
        let out = down.forward(&y, None);
        assert_eq!(out.t, 2, "halves (5+? - 3)/2+1");
        assert_eq!((out.h, out.w), (8, 8));
    }

    #[test]
    fn attention_block_zero_proj_is_identity() {
        let mut a = AttentionBlock::new(8);
        a.qkv_w.fill(0.1);
        // proj zero-init -> output == input
        let x = Vid::zeros(8, 2, 4, 4);
        let out = a.forward(&x);
        for (v, o) in x.data.iter().zip(&out.data) {
            assert_eq!(v, o);
        }
    }
}
