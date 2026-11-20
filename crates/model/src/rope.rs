//! 3D RoPE (t/h/w axis-split) port of `model_liveact/model_memory.py`
//! (`rope_params`, `causal_rope_apply`).
//!
//! The original builds one frequency table per axis:
//! `cat([rope_params(1024, d - 4*(d//6)), rope_params(1024, 2*(d//6)),
//!       rope_params(1024, 2*(d//6))], dim=1)`
//! i.e. REAL dims `[d - 4*(d//6), 2*(d//6), 2*(d//6)]`, then `causal_rope_apply`
//! splits the COMPLEX dims as `[c - 2*(c//3), c//3, c//3]` with `c = d/2`.
//! Complex pairing is over consecutive element pairs across the whole head
//! dim, so axis splits may individually be odd — the port replicates exactly
//! that layout.

use crate::config::LiveActConfig;

/// cos/sin tables for one axis: `[max_seq, real_dims/2]` pairs.
#[derive(Clone, Debug)]
pub struct AxisFreq {
    pub cos: Vec<f32>, // [max_seq][pairs]
    pub sin: Vec<f32>,
    pub pairs: usize,
    pub max_seq: usize,
}

/// `rope_params(max_seq_len, dim, theta=10000)`:
/// freqs[p][j] = exp(i * p / theta^(2j/dim)), j = 0..dim/2.
pub fn rope_params(max_seq: usize, dim: usize, theta: f64) -> AxisFreq {
    let pairs = dim / 2;
    let mut cos = vec![0.0f32; max_seq * pairs];
    let mut sin = vec![0.0f32; max_seq * pairs];
    for p in 0..max_seq {
        for j in 0..pairs {
            let freq = (p as f64)
                * (1.0 / theta.powf((2 * j) as f64 / dim as f64));
            let (c, s) = (freq.cos(), freq.sin());
            cos[p * pairs + j] = c as f32;
            sin[p * pairs + j] = s as f32;
        }
    }
    AxisFreq {
        cos,
        sin,
        pairs,
        max_seq,
    }
}

/// The concatenated t/h/w frequency table for a config.
pub struct Rope3d {
    pub t: AxisFreq,
    pub h: AxisFreq,
    pub w: AxisFreq,
    /// complex-pair counts per axis: [c - 2*(c//3), c//3, c//3], c = head_dim/2
    pub split: [usize; 3],
    pub head_dim: usize,
}

impl Rope3d {
    pub fn new(cfg: &LiveActConfig) -> Self {
        let d = cfg.dim / cfg.num_heads;
        assert!(d % 2 == 0, "head dim must be even (is {d})");
        let t = rope_params(1024, d - 4 * (d / 6), 10000.0);
        let h = rope_params(1024, 2 * (d / 6), 10000.0);
        let w = rope_params(1024, 2 * (d / 6), 10000.0);
        let c = d / 2;
        Rope3d {
            t,
            h,
            w,
            split: [c - 2 * (c / 3), c / 3, c / 3],
            head_dim: d,
        }
    }

    /// Apply RoPE in place to `x` laid out `[seq][heads][head_dim]`.
    ///
    /// Mirrors `causal_rope_apply(x, grid_sizes, freqs, start_frame)` for the
    /// single-video batch used throughout the pipeline: the sequence is
    /// `frames * gh * gw` tokens of ONE video; token (f, y, xw) uses
    /// t-freq at absolute frame `start_frame + f`, h-freq at row y and
    /// w-freq at column xw. Complex pairs are consecutive dims across the
    /// whole head, with axis ranges following `self.split`.
    pub fn apply(&self, x: &mut [f32], seq: usize, heads: usize, gh: usize, gw: usize, start_frame: usize) {
        let d = self.head_dim;
        debug_assert_eq!(x.len(), seq * heads * d);
        let frames = if gh * gw > 0 { seq / (gh * gw) } else { 0 };
        // Precompute per-axis pair angle tables for this call.
        let (tp, hp, wp) = (self.t.pairs, self.h.pairs, self.w.pairs);
        let (tc, ts) = (&self.t.cos, &self.t.sin);
        let (hc, hs) = (&self.h.cos, &self.h.sin);
        let (wc, ws) = (&self.w.cos, &self.w.sin);
        let tmax = self.t.max_seq;
        let hmax = self.h.max_seq;
        let wmax = self.w.max_seq;

        for f in 0..frames {
            let tf = (start_frame + f).min(tmax - 1);
            for y in 0..gh {
                for xw in 0..gw {
                    let tok = f * gh * gw + y * gw + xw;
                    let xt = &mut x[tok * heads * d..(tok + 1) * heads * d];
                    for h in 0..heads {
                        let row = &mut xt[h * d..(h + 1) * d];
                        // complex dim index -> (axis, pair index)
                        let mut cd = 0usize;
                        for (axis, pairs) in [(0usize, tp), (1, hp), (2, wp)] {
                            for j in 0..pairs {
                                let (cos, sin) = match axis {
                                    0 => (
                                        tc[tf * tp + j],
                                        ts[tf * tp + j],
                                    ),
                                    1 => (hc[y.min(hmax - 1) * hp + j], hs[y.min(hmax - 1) * hp + j]),
                                    _ => (wc[xw.min(wmax - 1) * wp + j], ws[xw.min(wmax - 1) * wp + j]),
                                };
                                // consecutive pair (2*cd, 2*cd+1)
                                let re = row[2 * cd];
                                let im = row[2 * cd + 1];
                                row[2 * cd] = re * cos - im * sin;
                                row[2 * cd + 1] = re * sin + im * cos;
                                cd += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LiveActConfig;

    #[test]
    fn split_dims_match_original() {
        let cfg = LiveActConfig::full_14b();
        let rope = Rope3d::new(&cfg);
        // d=128 -> c=64 -> [64-2*21, 21, 21] = [22, 21, 21]
        assert_eq!(rope.split, [22, 21, 21]);
        assert_eq!(rope.t.pairs, 22); // (128 - 4*21)/2
        assert_eq!(rope.h.pairs, 21); // 2*21/2
        assert_eq!(rope.w.pairs, 21);
        let total: usize = rope.split.iter().sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn identity_at_origin() {
        // At frame 0 / row 0 / col 0 all angles are 0 => no rotation.
        let cfg = LiveActConfig::tiny();
        let rope = Rope3d::new(&cfg);
        let (gh, gw) = cfg.grid_hw();
        let heads = cfg.num_heads;
        let d = rope.head_dim;
        let seq = gh * gw; // single frame: every axis at origin
        let mut x = vec![0.0; seq * heads * d];
        for v in x.iter_mut() {
            *v = rand_like();
        }
        let orig = x.clone();
        rope.apply(&mut x, seq, heads, gh, gw, 0);
        // Only the (f=0, y=0, xw=0) token sits at the origin: it must be
        // unchanged. Every other token rotates its h/w (and t) axes.
        let d2 = heads * d;
        for i in 0..d2 {
            assert!(
                (x[i] - orig[i]).abs() < 1e-5,
                "origin token changed at dim {i}"
            );
        }
        let changed = x[d2..]
            .iter()
            .zip(&orig[d2..])
            .filter(|(a, b)| (**a - **b).abs() > 1e-6)
            .count();
        assert!(changed > 0, "other tokens should rotate");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rotation_preserves_norms() {
        let cfg = LiveActConfig::tiny();
        let rope = Rope3d::new(&cfg);
        let (gh, gw) = cfg.grid_hw();
        let heads = cfg.num_heads;
        let d = rope.head_dim;
        let seq = 5 * gh * gw;
        let mut x = Vec::with_capacity(seq * heads * d);
        let mut s = 12345u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u32 << 16) as f32) - 16.0
        };
        for _ in 0..seq * heads * d {
            x.push(next());
        }
        let norms: Vec<f32> = (0..seq * heads)
            .map(|i| x[i * d..(i + 1) * d].iter().map(|v| v * v).sum())
            .collect();
        rope.apply(&mut x, seq, heads, gh, gw, 7);
        for (i, n) in norms.iter().enumerate() {
            let m: f32 = x[i * d..(i + 1) * d].iter().map(|v| v * v).sum();
            assert!((m - n).abs() / n.max(1e-6) < 1e-3, "token {i}");
        }
    }

    fn rand_like() -> f32 {
        use std::cell::Cell;
        thread_local!(static S: Cell<u64> = Cell::new(99));
        S.with(|s| {
            let mut v = s.get();
            v = v.wrapping_mul(6364136223846793005).wrapping_add(1);
            s.set(v);
            ((v >> 40) as f32) / (1u32 << 24) as f32 - 0.5
        })
    }
}
