//! Minimal dense-tensor core for the SoulX-LiveAct model port.
//!
//! Design goals (mirroring the project's "efficient, performant, simple"):
//! - Row-major `f32` storage owned in `Vec<f32>`; no borrowing graph.
//! - A small number of high-throughput kernels (matmul with rayon blocking,
//!   vectorized elementwise ops); everything else is expressed over slices
//!   with explicit shapes so the compiler can keep codegen tight.
//! - A deterministic RNG (xoshiro256**) so every run of the untrained
//!   compact configuration is bit-reproducible across machines.

use rayon::prelude::*;

/// Row-major f32 tensor.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: vec![0.0; shape.iter().product()],
            shape: shape.to_vec(),
        }
    }

    pub fn filled(shape: &[usize], v: f32) -> Self {
        Self {
            data: vec![v; shape.iter().product()],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), self.data.len());
        self.shape = shape;
        self
    }
}

/// Deterministic xoshiro256** PRNG (splitmix64 seeded).
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    pub fn u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn f32(&mut self) -> f32 {
        // 24 random mantissa bits in [1, 2) minus 1 -> [0, 1).
        ((self.u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Standard normal via Box-Muller (paired).
    pub fn normal(&mut self) -> f32 {
        let (mut a, mut b) = (0.0f32, 0.0f32);
        loop {
            a = self.f32();
            if a > 1e-9 {
                b = self.f32();
                break;
            }
        }
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }

    pub fn fill_normal(&mut self, dst: &mut [f32]) {
        for v in dst.iter_mut() {
            *v = self.normal();
        }
    }

    pub fn normal_vec(&mut self, n: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        self.fill_normal(&mut v);
        v
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// C[m, n] = A[m, k] @ B[k, n]  (all row-major).
///
/// Cache-blocked and rayon-parallel over row panels; the inner loop keeps both
/// operands streaming contiguous memory. `beta` scales the initial C
/// (0.0 overwrites, 1.0 accumulates).
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, c: &mut [f32], beta: f32) {
    debug_assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    const BM: usize = 48;
    let panel_rows = BM.min(m.max(1));
    let panel_len = panel_rows * n;
    let par = m * n >= 64 * 64;
    let run = |(pi, cpanel): (usize, &mut [f32])| {
        let mi = pi * panel_rows;
        let mj = (mi + panel_rows).min(m);
        for (i, crow) in cpanel.chunks_mut(n).enumerate() {
            let i = mi + i;
            if i >= mj { break; }
            let arow = &a[i * k..(i + 1) * k];
            for (p, av) in arow.iter().enumerate() {
                if beta == 0.0 && p == 0 {
                    crow.fill(0.0);
                }
                if *av == 0.0 { continue; }
                let brow = &b[p * n..(p + 1) * n];
                for (cv, bv) in crow.iter_mut().zip(brow) {
                    *cv += av * bv;
                }
            }
        }
    };
    if par {
        c.par_chunks_mut(panel_len).enumerate().for_each(run);
    } else {
        for (pi, cpanel) in c.chunks_mut(panel_len).enumerate() {
            run((pi, cpanel));
        }
    }
}

/// C[m, n] = A[m, k] @ B^T where B is stored row-major as [n, k] (PyTorch
/// `Linear` weight layout). This avoids transposing weights at inference.
pub fn matmul_at_bt(
    a: &[f32],
    b_t: &[f32],
    m: usize,
    k: usize,
    n: usize,
    c: &mut [f32],
    beta: f32,
) {
    debug_assert!(a.len() >= m * k && b_t.len() >= n * k && c.len() >= m * n);
    const BR: usize = 32;
    let par = m * n >= 48 * 48;
    let run = |(ri, crows): (usize, &mut [f32])| {
        let start = ri * BR;
        for (ii, crow) in crows.chunks_mut(n).enumerate() {
            let i = start + ii;
            let arow = &a[i * k..(i + 1) * k];
            for (j, cij) in crow.iter_mut().enumerate() {
                let brow = &b_t[j * k..(j + 1) * k];
                let mut acc = if beta != 0.0 { *cij * beta } else { 0.0 };
                let mut t = 0usize;
                while t + 4 <= k {
                    acc += arow[t] * brow[t]
                        + arow[t + 1] * brow[t + 1]
                        + arow[t + 2] * brow[t + 2]
                        + arow[t + 3] * brow[t + 3];
                    t += 4;
                }
                while t < k {
                    acc += arow[t] * brow[t];
                    t += 1;
                }
                *cij = acc;
            }
        }
    };
    if par {
        c.par_chunks_mut(BR * n).enumerate().for_each(run);
    } else {
        for (ri, crows) in c.chunks_mut(BR * n).enumerate() {
            run((ri, crows));
        }
    }
}

// ---------------------------------------------------------------------------
// Layer primitives
// ---------------------------------------------------------------------------

/// PyTorch-style `nn.Linear`: weight `[out, in]` row-major, optional bias.
#[derive(Clone, Debug)]
pub struct Linear {
    pub w: Vec<f32>, // [out, in]
    pub b: Option<Vec<f32>>,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, bias: bool) -> Self {
        Linear {
            w: vec![0.0; out_dim * in_dim],
            b: if bias { Some(vec![0.0; out_dim]) } else { None },
            in_dim,
            out_dim,
        }
    }

    /// `x`: `[m, in_dim]` row-major -> `[m, out_dim]`.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        debug_assert!(x.len() >= m * self.in_dim);
        let mut y = vec![0.0; m * self.out_dim];
        matmul_at_bt(x, &self.w, m, self.in_dim, self.out_dim, &mut y, 0.0);
        if let Some(b) = &self.b {
            for r in 0..m {
                let row = &mut y[r * self.out_dim..(r + 1) * self.out_dim];
                for (v, bv) in row.iter_mut().zip(b) {
                    *v += bv;
                }
            }
        }
        y
    }

    pub fn xavier(&mut self, rng: &mut Rng) {
        let bound = (6.0 / (self.in_dim + self.out_dim) as f32).sqrt();
        for v in self.w.iter_mut() {
            *v = (rng.f32() * 2.0 - 1.0) * bound;
        }
        if let Some(b) = &mut self.b {
            b.fill(0.0);
        }
    }
}

/// RMS norm over the last dimension with learnable scale (WanRMSNorm).
#[inline]
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    debug_assert!(x.len() % d == 0);
    let mut out = vec![0.0; x.len()];
    for (o, chunk) in out.chunks_exact_mut(d).zip(x.chunks_exact(d)) {
        let ms = chunk.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for ((o, v), s) in o.iter_mut().zip(chunk).zip(w) {
            *o = v * inv * s;
        }
    }
    out
}

/// LayerNorm over the last dim, optional affine.
#[inline]
pub fn layer_norm(x: &[f32], d: usize, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (o, chunk) in out.chunks_exact_mut(d).zip(x.chunks_exact(d)) {
        let mean = chunk.iter().sum::<f32>() / d as f32;
        let var = chunk.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in chunk.iter().enumerate() {
            let mut y = (v - mean) * inv;
            if let Some(w) = w {
                y *= w[i];
            }
            if let Some(b) = b {
                y += b[i];
            }
            o[i] = y;
        }
    }
    out
}

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    // tanh approximation, matches nn.GELU(approximate='tanh')
    0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Abramowitz-Stegun style erf, adequate f32 accuracy.
#[inline]
pub fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    s * y
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax over `buf`.
pub fn softmax_inplace(buf: &mut [f32]) {
    let max = buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        buf.fill(1.0 / buf.len() as f32);
        return;
    }
    let mut sum = 0.0;
    for v in buf.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-30);
    for v in buf.iter_mut() {
        *v *= inv;
    }
}

/// fp8 e4m3fn quantization used by the `--fp8_kv_cache` option.
pub mod fp8 {
    /// Quantize with per-row (last-dim) absmax scaling, returns (bytes, scales).
    pub fn quantize_rows(x: &[f32], d: usize, rows: usize) -> (Vec<u8>, Vec<f32>) {
        const F8_MAX: f32 = 448.0;
        let mut q = vec![0u8; x.len()];
        let mut scales = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &x[r * d..(r + 1) * d];
            let amax = row.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-12);
            let scale = amax / F8_MAX;
            scales[r] = scale;
            let inv = 1.0 / scale;
            for (qb, v) in q[r * d..(r + 1) * d].iter_mut().zip(row) {
                *qb = f32_to_e4m3(v * inv);
            }
        }
        (q, scales)
    }

    pub fn dequantize_rows(q: &[u8], scales: &[f32], d: usize, rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; q.len()];
        for r in 0..rows {
            let s = scales[r];
            for (o, b) in out[r * d..(r + 1) * d]
                .iter_mut()
                .zip(&q[r * d..(r + 1) * d])
            {
                *o = e4m3_to_f32(*b) * s;
            }
        }
        out
    }

    /// Round float to IEEE-754 binary32 e4m3fn bit pattern (no inf, NaN
    /// payload 0x7f), saturating to +-448.
    pub fn f32_to_e4m3(v: f32) -> u8 {
        let bits = v.to_bits();
        let sign = (((bits >> 31) & 1) << 7) as u8; // e4m3 sign bit is 0x80
        let abs = f32::from_bits(bits & 0x7fff_ffff);
        if abs.is_nan() {
            return sign | 0x7f;
        }
        if abs > 448.0 {
            return sign | 0x7e; // +-448 (e4m3fn max, no infinity)
        }
        let pow2 = |p: i32| (2.0f32).powi(p);
        if abs >= pow2(-6) {
            // normal binade: value = (1 + m/8) * 2^(e-7), e in 1..=15
            let mut e = (abs.log2().floor() as i32 + 7).clamp(1, 15);
            let step = pow2(e - 7 - 3);
            let base = pow2(e - 7);
            let mut m = ((abs - base) / step).round() as i32;
            if m > 7 {
                // rounded into the next binade
                e += 1;
                m = 0;
                if e > 15 || (e == 15 && m == 7) {
                    return sign | 0x7e;
                }
            }
            if e == 15 && m == 7 {
                return sign | 0x7e; // 480 is out of range -> clamp to 448
            }
            sign | (((e as u32) << 3) | m as u32) as u8
        } else {
            // subnormal: value = m/8 * 2^-6, m in 0..=7 (step 2^-9)
            let m = (abs / pow2(-9)).round() as i32;
            if m >= 8 {
                sign | 0x08
            } else {
                sign | m.max(0) as u8
            }
        }
    }

    pub fn e4m3_to_f32(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let e = ((b & 0x7f) >> 3) as u32; // 4-bit exponent field (mask sign)
        let m = (b & 0x7) as u32;
        if e == 0 {
            if m == 0 {
                return 0.0;
            }
            // subnormal: m/8 * 2^-6
            sign * (m as f32 / 8.0) * (2.0f32).powi(-6)
        } else {
            sign * (1.0 + m as f32 / 8.0) * (2.0f32).powi(e as i32 - 7)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_naive() {
        let (m, k, n) = (17, 33, 21);
        let mut rng = Rng::new(7);
        let a = rng.normal_vec(m * k);
        let b = rng.normal_vec(k * n);
        let mut c = vec![f32::NAN; m * n];
        matmul(&a, &b, m, k, n, &mut c, 0.0);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                assert!((acc - c[i * n + j]).abs() < 1e-3, "({i},{j}) {acc} vs {}", c[i * n + j]);
            }
        }
    }

    #[test]
    fn matmul_at_bt_matches_linear() {
        let (m, k, n) = (5, 64, 33);
        let mut rng = Rng::new(11);
        let x = rng.normal_vec(m * k);
        let w = rng.normal_vec(n * k);
        let bias = rng.normal_vec(n);
        let mut c = vec![0.0; m * n];
        matmul_at_bt(&x, &w, m, k, n, &mut c, 0.0);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += x[i * k + p] * w[j * k + p];
                }
                assert!((acc - c[i * n + j]).abs() < 1e-3);
            }
        }
        // full Linear path (with bias) sanity
        let lin = Linear { w: w.clone(), b: Some(bias.clone()), in_dim: k, out_dim: n };
        let y = lin.forward(&x, m);
        assert_eq!(y.len(), m * n);
        for (i, (yv, cv)) in y.iter().zip(&c).enumerate() {
            let j = i % n;
            assert!((yv - cv - bias[j]).abs() < 1e-3, "linear must add bias");
        }
    }

    #[test]
    fn rms_norm_unit_scale() {
        let x = [3.0, 4.0];
        let w = [1.0, 1.0];
        let y = rms_norm(&x, &w, 1e-6);
        // RMS norm: mean square == 1 -> for d=2, sum of squares == 2
        assert!((y[0] * y[0] + y[1] * y[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn fp8_roundtrip_precision() {
        // subnormals: absolute tolerance of one step (2^-9); normals: relative
        for &v in &[0.0, 1.0, -1.0, 0.25, 448.0, -448.0, 512.0, 0.001953125, 0.0009765625, 0.001] {
            let q = fp8::f32_to_e4m3(v);
            let d = fp8::e4m3_to_f32(q);
            if v.abs() <= 448.0 {
                let sub = v.abs() < (2.0f32).powi(-6);
                let err = (d - v).abs();
                assert!(
                    if sub { err <= (2.0f32).powi(-9) + 1e-9 } else { err / v.abs() < 0.2 },
                    "v={v} d={d}"
                );
            } else {
                assert_eq!(d.abs(), 448.0);
            }
        }
    }

    #[test]
    fn fp8_row_roundtrip() {
        let d = 48;
        let mut rng = Rng::new(3);
        let x = rng.normal_vec(d * 4);
        let (q, s) = fp8::quantize_rows(&x, d, 4);
        let y = fp8::dequantize_rows(&q, &s, d, 4);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() / a.abs().max(0.05) < 0.15, "{a} vs {b}");
        }
    }

    #[test]
    fn deterministic_rng() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.u64(), b.u64());
        }
        let xs = a.normal_vec(1000);
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        assert!(mean.abs() < 0.1, "normal mean {mean}");
    }
}
