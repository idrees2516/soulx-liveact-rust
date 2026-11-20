//! Wav2Vec2 audio encoder port (`src/audio_analysis/wav2vec2.py` +
//! `torch_utils.py` from SoulX-LiveAct, architecture from HuggingFace
//! `transformers` `Wav2Vec2Model`).
//!
//! The SoulX-LiveAct audio path:
//! 1. waveform (16 kHz, x3.0 gain) -> feature-extractor conv stack
//! 2. `linear_interpolation` of conv features to the video frame count
//!    (`seq_len = duration * 25`)
//! 3. feature projection -> 12-layer transformer encoder (post-LN)
//! 4. stack hidden states of layers 1..12 -> `[frames, 12, 768]`
//!
//! Real pretrained weights (`TencentGameMate/chinese-wav2vec2-base`,
//! 94M params) are loaded from the flat blob produced by
//! `scripts/convert_wav2vec2.py`; when absent, a deterministic fallback
//! initialization is used so the pipeline still runs end-to-end.

use crate::tensor::{gelu_erf, layer_norm, softmax_inplace, Linear, Rng};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Wav2Vec2Config {
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub conv_bias: bool,
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub num_conv_pos: usize,
    pub num_conv_pos_groups: usize,
    pub eps: f32,
}

impl Default for Wav2Vec2Config {
    fn default() -> Self {
        // TencentGameMate/chinese-wav2vec2-base config.json
        Wav2Vec2Config {
            conv_dim: vec![512; 7],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_bias: false,
            hidden: 768,
            heads: 12,
            layers: 12,
            intermediate: 3072,
            num_conv_pos: 128,
            num_conv_pos_groups: 16,
            eps: 1e-5,
        }
    }
}

struct Conv1d {
    w: Vec<f32>, // [out, in, k]
    b: Option<Vec<f32>>,
    stride: usize,
    k: usize,
    in_c: usize,
    out_c: usize,
}

impl Conv1d {
    fn new(in_c: usize, out_c: usize, k: usize, stride: usize, bias: bool) -> Self {
        Conv1d {
            w: vec![0.0; out_c * in_c * k],
            b: if bias { Some(vec![0.0; out_c]) } else { None },
            stride,
            k,
            in_c,
            out_c,
        }
    }

    /// x: `[in_c, t]` -> `[out_c, (t - k)/stride + 1]` (no padding, as HF).
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, usize) {
        let out_t = if t >= self.k { (t - self.k) / self.stride + 1 } else { 0 };
        if out_t == 0 {
            return (vec![0.0; self.out_c], 0);
        }
        // im2col: [out_t, in_c * k] then GEMM with w viewed as [out_c, in_c*k]
        let feat = self.in_c * self.k;
        let mut cols = vec![0.0f32; out_t * feat];
        for ot in 0..out_t {
            let base = ot * self.stride;
            let row = &mut cols[ot * feat..(ot + 1) * feat];
            for ic in 0..self.in_c {
                for kk in 0..self.k {
                    row[ic * self.k + kk] = x[ic * t + base + kk];
                }
            }
        }
        let mut out = vec![0.0f32; out_t * self.out_c];
        crate::tensor::matmul_at_bt(&cols, &self.w, out_t, feat, self.out_c, &mut out, 0.0);
        if let Some(b) = &self.b {
            for ot in 0..out_t {
                for c in 0..self.out_c {
                    out[ot * self.out_c + c] += b[c];
                }
            }
        }
        // transpose to [out_c, out_t]
        let mut ch = vec![0.0f32; self.out_c * out_t];
        for ot in 0..out_t {
            for c in 0..self.out_c {
                ch[c * out_t + ot] = out[ot * self.out_c + c];
            }
        }
        (ch, out_t)
    }
}

struct EncoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ffn1: Linear,
    ffn2: Linear,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
}

pub struct Wav2Vec2 {
    cfg: Wav2Vec2Config,
    convs: Vec<Conv1d>,
    // layer-0 GroupNorm(512, eps 1e-5), affine
    gn_w: Vec<f32>,
    gn_b: Vec<f32>,
    fp_ln_w: Vec<f32>,
    fp_ln_b: Vec<f32>,
    fp_proj: Linear, // 512 -> hidden
    // positional conv (weight-normed, grouped)
    pos_v: Vec<f32>, // [hidden, hidden/groups, k]
    pos_g: Vec<f32>, // [k]
    pos_b: Vec<f32>, // [hidden]
    enc_ln_w: Vec<f32>,
    enc_ln_b: Vec<f32>,
    layers: Vec<EncoderLayer>,
    /// true when real pretrained weights were loaded
    pub pretrained: bool,
}

#[derive(Debug)]
pub enum LoadError {
    Missing(String),
    Io(std::io::Error),
    BadManifest(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Missing(m) => write!(f, "missing tensor: {m}"),
            LoadError::Io(e) => write!(f, "io error: {e}"),
            LoadError::BadManifest(m) => write!(f, "bad manifest: {m}"),
        }
    }
}

#[derive(Deserialize)]
struct ManifestEntry {
    shape: Vec<usize>,
    offset: u64,
}

impl Wav2Vec2 {
    /// Load pretrained weights from `models_dir` (wav2vec2-weights.bin +
    /// wav2vec2-manifest.json). Falls back to deterministic init when the
    /// files are absent.
    pub fn load(models_dir: &Path) -> Result<Wav2Vec2, LoadError> {
        let bin = models_dir.join("wav2vec2-weights.bin");
        let json = models_dir.join("wav2vec2-manifest.json");
        if !bin.exists() || !json.exists() {
            return Ok(Wav2Vec2::deterministic());
        }
        let data = std::fs::read(&bin).map_err(LoadError::Io)?;
        let manifest: HashMap<String, ManifestEntry> =
            serde_json::from_str(&std::fs::read_to_string(&json).map_err(LoadError::Io)?)
                .map_err(|e| LoadError::BadManifest(e.to_string()))?;

        let mut model = Wav2Vec2::deterministic();
        let cfg = model.cfg.clone();

        let get = |name: &str| -> Result<Vec<f32>, LoadError> {
            let e = manifest
                .get(name)
                .ok_or_else(|| LoadError::Missing(name.to_string()))?;
            let n: usize = e.shape.iter().product();
            let start = e.offset as usize;
            let end = start + n * 4;
            if end > data.len() {
                return Err(LoadError::BadManifest(format!("offset out of range for {name}")));
            }
            let mut v = vec![0.0f32; n];
            let bytes = &data[start..end];
            for (i, chunk) in v.iter_mut().enumerate() {
                *chunk = f32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
            }
            Ok(v)
        };

        // conv stack
        for (i, conv) in model.convs.iter_mut().enumerate() {
            conv.w = get(&format!("feature_extractor.conv_layers.{i}.conv.weight"))?;
            if cfg.conv_bias {
                conv.b = Some(get(&format!("feature_extractor.conv_layers.{i}.conv.bias"))?);
            }
        }
        model.gn_w = get("feature_extractor.conv_layers.0.layer_norm.weight")?;
        model.gn_b = get("feature_extractor.conv_layers.0.layer_norm.bias")?;
        model.fp_ln_w = get("feature_projection.layer_norm.weight")?;
        model.fp_ln_b = get("feature_projection.layer_norm.bias")?;
        model.fp_proj.w = get("feature_projection.projection.weight")?;
        model.fp_proj.b = Some(get("feature_projection.projection.bias")?);
        model.pos_v = get("encoder.pos_conv_embed.conv.weight_v")?;
        model.pos_g = get("encoder.pos_conv_embed.conv.weight_g")?;
        model.pos_b = get("encoder.pos_conv_embed.conv.bias")?;
        model.enc_ln_w = get("encoder.layer_norm.weight")?;
        model.enc_ln_b = get("encoder.layer_norm.bias")?;
        for (i, l) in model.layers.iter_mut().enumerate() {
            l.q.w = get(&format!("encoder.layers.{i}.attention.q_proj.weight"))?;
            l.q.b = Some(get(&format!("encoder.layers.{i}.attention.q_proj.bias"))?);
            l.k.w = get(&format!("encoder.layers.{i}.attention.k_proj.weight"))?;
            l.k.b = Some(get(&format!("encoder.layers.{i}.attention.k_proj.bias"))?);
            l.v.w = get(&format!("encoder.layers.{i}.attention.v_proj.weight"))?;
            l.v.b = Some(get(&format!("encoder.layers.{i}.attention.v_proj.bias"))?);
            l.o.w = get(&format!("encoder.layers.{i}.attention.out_proj.weight"))?;
            l.o.b = Some(get(&format!("encoder.layers.{i}.attention.out_proj.bias"))?);
            l.ln1_w = get(&format!("encoder.layers.{i}.layer_norm.weight"))?;
            l.ln1_b = get(&format!("encoder.layers.{i}.layer_norm.bias"))?;
            l.ffn1.w = get(&format!("encoder.layers.{i}.feed_forward.intermediate_dense.weight"))?;
            l.ffn1.b = Some(get(&format!("encoder.layers.{i}.feed_forward.intermediate_dense.bias"))?);
            l.ffn2.w = get(&format!("encoder.layers.{i}.feed_forward.output_dense.weight"))?;
            l.ffn2.b = Some(get(&format!("encoder.layers.{i}.feed_forward.output_dense.bias"))?);
            l.ln2_w = get(&format!("encoder.layers.{i}.final_layer_norm.weight"))?;
            l.ln2_b = get(&format!("encoder.layers.{i}.final_layer_norm.bias"))?;
        }
        model.pretrained = true;
        Ok(model)
    }

    /// Deterministic random init (no pretrained weights available).
    pub fn deterministic() -> Wav2Vec2 {
        let cfg = Wav2Vec2Config::default();
        
        let mut w2v = Wav2Vec2 {
            cfg: cfg.clone(),
            convs: Vec::new(),
            gn_w: vec![1.0; cfg.conv_dim[0]],
            gn_b: vec![0.0; cfg.conv_dim[0]],
            fp_ln_w: vec![1.0; cfg.conv_dim[0]],
            fp_ln_b: vec![0.0; cfg.conv_dim[0]],
            fp_proj: Linear::new(cfg.conv_dim[0], cfg.hidden, true),
            pos_v: vec![0.0; cfg.hidden * (cfg.hidden / cfg.num_conv_pos_groups) * cfg.num_conv_pos],
            pos_g: vec![1.0; cfg.num_conv_pos],
            pos_b: vec![0.0; cfg.hidden],
            enc_ln_w: vec![1.0; cfg.hidden],
            enc_ln_b: vec![0.0; cfg.hidden],
            layers: Vec::new(),
            pretrained: false,
        };
        let mut rng = Rng::new(88172645463325252);
        for (i, k) in cfg.conv_kernel.iter().enumerate() {
            let in_c = if i == 0 { 1 } else { cfg.conv_dim[i - 1] };
            let mut c = Conv1d::new(in_c, cfg.conv_dim[i], *k, cfg.conv_stride[i], cfg.conv_bias);
            let bound = (6.0 / ((in_c * k + cfg.conv_dim[i]) as f32)).sqrt();
            for v in c.w.iter_mut() {
                *v = (rng.f32() * 2.0 - 1.0) * bound;
            }
            w2v.convs.push(c);
        }
        w2v.fp_proj.xavier(&mut rng);
        for v in w2v.pos_v.iter_mut() {
            *v = rng.normal() * 0.02;
        }
        w2v.pos_g.fill(1.0);
        for _ in 0..cfg.layers {
            let mut l = EncoderLayer {
                q: Linear::new(cfg.hidden, cfg.hidden, true),
                k: Linear::new(cfg.hidden, cfg.hidden, true),
                v: Linear::new(cfg.hidden, cfg.hidden, true),
                o: Linear::new(cfg.hidden, cfg.hidden, true),
                ln1_w: vec![1.0; cfg.hidden],
                ln1_b: vec![0.0; cfg.hidden],
                ffn1: Linear::new(cfg.hidden, cfg.intermediate, true),
                ffn2: Linear::new(cfg.intermediate, cfg.hidden, true),
                ln2_w: vec![1.0; cfg.hidden],
                ln2_b: vec![0.0; cfg.hidden],
            };
            l.q.xavier(&mut rng);
            l.k.xavier(&mut rng);
            l.v.xavier(&mut rng);
            l.o.xavier(&mut rng);
            l.ffn1.xavier(&mut rng);
            l.ffn2.xavier(&mut rng);
            w2v.layers.push(l);
        }
        w2v
    }

    /// Normalize the waveform (zero mean / unit variance, like the HF
    /// feature extractor with do_normalize=True) and apply the x3.0 gain
    /// from `resample_audio` in generate.py.
    pub fn preprocess_wave(wave: &[f32]) -> Vec<f32> {
        if wave.is_empty() {
            return vec![];
        }
        let mean = wave.iter().sum::<f32>() / wave.len() as f32;
        let var = wave.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / wave.len() as f32;
        let inv = 1.0 / (var + 1e-7).sqrt();
        wave.iter().map(|v| (v - mean) * inv * 3.0).collect()
    }

    /// Full analysis: waveform (16 kHz, already preprocessed) -> stacked
    /// hidden states `[frames, layers, hidden]` (layers 1..=12).
    pub fn analyze(&self, wave: &[f32], frames: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        // ---- conv stack: [1, t] -> [512, t']
        let mut feat: Vec<f32> = wave.to_vec();
        let mut t = wave.len();
        let mut channels = 1usize;
        for (i, conv) in self.convs.iter().enumerate() {
            let (out, out_t) = conv.forward(&feat, t);
            feat = out;
            t = out_t;
            channels = conv.out_c;
            if i == 0 {
                // GroupNorm over time per channel + GELU
                group_norm_time(&mut feat, channels, t, &self.gn_w, &self.gn_b, cfg.eps);
                for v in feat.iter_mut() {
                    *v = gelu_erf(*v);
                }
            } else {
                for v in feat.iter_mut() {
                    *v = gelu_erf(*v);
                }
            }
        }
        // ---- linear interpolation to `frames` (align_corners=True) ----
        let conv_dim = channels;
        let interp = linear_interpolate(&feat, conv_dim, t, frames);

        // ---- feature projection: LN(conv_dim) + Linear(conv_dim -> hidden) ----
        let normed = layer_norm(&interp, conv_dim, Some(&self.fp_ln_w), Some(&self.fp_ln_b), cfg.eps);
        let mut h = self.fp_proj.forward(&normed, frames); // [frames, hidden]

        // ---- positional conv (weight-normed grouped conv) + encoder LN ----
        let pos = self.pos_conv(&h, frames);
        for (hv, pv) in h.iter_mut().zip(&pos) {
            *hv += pv;
        }
        let mut h = layer_norm(&h, cfg.hidden, Some(&self.enc_ln_w), Some(&self.enc_ln_b), cfg.eps);

        // ---- transformer layers (post-LN); collect outputs 1..=12 ----
        let layers = cfg.layers;
        let mut stacked = vec![0.0f32; frames * layers * cfg.hidden];
        for (li, l) in self.layers.iter().enumerate() {
            let attn_out = self.layer_attention(l, &h, frames);
            // h = LN(h + attn)
            let mut res = vec![0.0f32; h.len()];
            for ((r, a), hv) in res.iter_mut().zip(&attn_out).zip(&h) {
                *r = hv + a;
            }
            h = layer_norm(&res, cfg.hidden, Some(&l.ln1_w), Some(&l.ln1_b), cfg.eps);
            // ffn: h = LN(h + ffn(h))
            let ff = l.ffn1.forward(&h, frames);
            let ff: Vec<f32> = ff.iter().map(|v| gelu_erf(*v)).collect();
            let ff = l.ffn2.forward(&ff, frames);
            let mut res = vec![0.0f32; h.len()];
            for ((r, a), hv) in res.iter_mut().zip(&ff).zip(&h) {
                *r = hv + a;
            }
            h = layer_norm(&res, cfg.hidden, Some(&l.ln2_w), Some(&l.ln2_b), cfg.eps);
            // record hidden state AFTER this layer
            let dst = &mut stacked[li * frames * cfg.hidden..(li + 1) * frames * cfg.hidden];
            dst.copy_from_slice(&h);
        }
        stacked
    }

    fn layer_attention(&self, l: &EncoderLayer, h: &[f32], frames: usize) -> Vec<f32> {
        let (hidden, heads) = (self.cfg.hidden, self.cfg.heads);
        let hd = hidden / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = l.q.forward(h, frames);
        let k = l.k.forward(h, frames);
        let v = l.v.forward(h, frames);
        // [frames, heads, hd] layouts
        let parts: Vec<(usize, Vec<f32>)> = (0..heads)
            .into_par_iter()
            .map(|hh| {
                let mut oh = vec![0.0f32; frames * hd];
                let mut row = vec![0.0f32; frames];
                for i in 0..frames {
                    for (j, r) in row.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for d in 0..hd {
                            acc += q[i * hidden + hh * hd + d] * k[j * hidden + hh * hd + d];
                        }
                        *r = acc * scale;
                    }
                    softmax_inplace(&mut row);
                    let ov = &mut oh[i * hd..(i + 1) * hd];
                    for (o, w) in ov.iter_mut().enumerate() {
                        let mut acc = 0.0;
                        for (j, wgt) in row.iter().enumerate() {
                            acc += wgt * v[j * hidden + hh * hd + o];
                        }
                        *w = acc;
                    }
                }
                (hh, oh)
            })
            .collect();
        let mut attn = vec![0.0f32; frames * hidden];
        for (hh, oh) in parts {
            for i in 0..frames {
                attn[i * hidden + hh * hd..i * hidden + (hh + 1) * hd]
                    .copy_from_slice(&oh[i * hd..(i + 1) * hd]);
            }
        }
        l.o.forward(&attn, frames)
    }

    /// Grouped causal-ish conv with symmetric padding k/2, weight-norm
    /// recombination, right-trim of 1 (Wav2Vec2SamePadLayer).
    fn pos_conv(&self, h: &[f32], frames: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (hidden, k, groups) = (cfg.hidden, cfg.num_conv_pos, cfg.num_conv_pos_groups);
        let per_group_in = hidden / groups; // 48
        let per_group_out = hidden / groups; // 48
        let pad = k / 2;
        // effective weight [hidden, 48, k] = v * g[k] / ||v[:, :, k]||
        let mut w = self.pos_v.clone();
        for kk in 0..k {
            let mut norm = 0.0f32;
            for v in self.pos_v.iter().skip(kk).step_by(k) {
                norm += v * v;
            }
            norm = norm.sqrt().max(1e-8);
            let g = self.pos_g[kk] / norm;
            for idx in (kk..w.len()).step_by(k) {
                w[idx] *= g;
            }
        }
        // symmetric padding k/2 on both sides: out_len = frames + 1,
        // then Wav2Vec2SamePadLayer drops the last position (even kernel)
        let out_len = frames + 1;
        let mut out = vec![0.0f32; out_len * hidden];
        // parallel over output channels
        let results: Vec<(usize, Vec<f32>)> = (0..hidden)
            .into_par_iter()
            .map(|oc| {
                let group = oc / per_group_out;
                let mut col = vec![0.0f32; out_len];
                for t in 0..out_len {
                    let mut acc = self.pos_b[oc];
                    let wrow = &w[oc * per_group_in * k..(oc + 1) * per_group_in * k];
                    for ic in 0..per_group_in {
                        let ch = group * per_group_in + ic;
                        for kk in 0..k {
                            let src = t as isize - pad as isize + kk as isize;
                            if src >= 0 && (src as usize) < frames {
                                acc += wrow[ic * k + kk] * h[src as usize * hidden + ch];
                            }
                        }
                    }
                    col[t] = acc;
                }
                (oc, col)
            })
            .collect();
        for (oc, col) in results {
            for t in 0..out_len {
                out[t * hidden + oc] = col[t];
            }
        }
        // trim last position -> [frames, hidden], then GELU
        let mut with_pad = out[..frames * hidden].to_vec();
        for v in with_pad.iter_mut() {
            *v = gelu_erf(*v);
        }
        with_pad
    }
}

/// GroupNorm(num_groups=C, num_channels=C): normalize each channel over time.
fn group_norm_time(x: &mut [f32], c: usize, t: usize, w: &[f32], b: &[f32], eps: f32) {
    for ch in 0..c {
        let slice = &mut x[ch * t..(ch + 1) * t];
        let mean = slice.iter().sum::<f32>() / t as f32;
        let var = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / t as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for v in slice.iter_mut() {
            *v = (*v - mean) * inv;
        }
        for v in slice.iter_mut() {
            *v = *v * w[ch] + b[ch];
        }
    }
}

/// `linear_interpolation` (torch_utils.py): F.interpolate(..., size=seq_len,
/// mode='linear', align_corners=True) on [C, T] -> [C, frames].
pub fn linear_interpolate(x: &[f32], c: usize, t: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * c];
    if frames == 0 || t == 0 {
        return out;
    }
    for ch in 0..c {
        let src = &x[ch * t..(ch + 1) * t];
        let dst = &mut out[ch * frames..(ch + 1) * frames];
        if t == 1 || frames == 1 {
            for (d, s) in dst.iter_mut().enumerate() {
                let idx = if t == 1 { 0 } else { (d * (t - 1)).min(t - 1) / (frames - 1).max(1) };
                *s = src[idx.min(t - 1)];
            }
            continue;
        }
        let scale = (t - 1) as f32 / (frames - 1) as f32;
        for (d, v) in dst.iter_mut().enumerate() {
            let pos = d as f32 * scale;
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(t - 1);
            let frac = pos - i0 as f32;
            *v = src[i0] * (1.0 - frac) + src[i1] * frac;
        }
    }
    out
}

/// Simple linear resampler (arbitrary src/dst rates) for the 24 kHz TTS
/// output -> 16 kHz wav2vec2 input path. The original uses sox `tempo`, which
/// is the identity at fps=25 (rate = 25/fps = 1.0); only the sample-rate
/// conversion remains.
pub fn resample_linear(wave: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || wave.is_empty() {
        return wave.to_vec();
    }
    let n_out = (wave.len() as f64 * to_hz as f64 / from_hz as f64).round() as usize;
    let mut out = vec![0.0f32; n_out];
    let step = (wave.len() - 1) as f64 / (n_out.max(1) - 1).max(1) as f64;
    for (i, v) in out.iter_mut().enumerate() {
        let pos = i as f64 * step;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(wave.len() - 1);
        let frac = (pos - i0 as f64) as f32;
        *v = wave[i0] * (1.0 - frac) + wave[i1] * frac;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_stack_lengths() {
        // 33920 samples (2.12 s @16k): floor-division through the stack
        // (t-k)/s+1 seven times lands on 105
        let cfg = Wav2Vec2Config::default();
        let model = Wav2Vec2::deterministic();
        let t = 33920usize;
        let mut cur = t;
        for (i, conv) in model.convs.iter().enumerate() {
            cur = if cur >= conv.k { (cur - conv.k) / conv.stride + 1 } else { 0 };
            let _ = i;
        }
        assert_eq!(cur, 105);
        let _ = cfg;
    }

    #[test]
    fn deterministic_analyze_shapes() {
        let model = Wav2Vec2::deterministic();
        let wave: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
        let frames = 55;
        let out = model.analyze(&wave, frames);
        assert_eq!(out.len(), frames * 12 * 768);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn pretrained_load_when_present() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        if !dir.join("wav2vec2-weights.bin").exists() {
            eprintln!("skipping: no pretrained weights");
            return;
        }
        let model = Wav2Vec2::load(&dir).expect("load pretrained");
        assert!(model.pretrained);
        // real weights: run a short analysis and check statistics are sane
        let wave: Vec<f32> = (0..8000)
            .map(|i| (i as f32 * 0.02).sin() * 0.3 + (i as f32 * 0.005).cos() * 0.1)
            .collect();
        let out = model.analyze(&wave, 25);
        assert_eq!(out.len(), 25 * 12 * 768);
        let finite = out.iter().all(|v| v.is_finite());
        assert!(finite);
        let abs_mean = out.iter().map(|v| v.abs()).sum::<f32>() / out.len() as f32;
        eprintln!("pretrained abs mean activation: {abs_mean:.4}");
        assert!(abs_mean > 0.001 && abs_mean < 100.0, "activation scale {abs_mean}");
    }

    #[test]
    fn resample_lengths() {
        let x = vec![0.5f32; 2400];
        let y = resample_linear(&x, 24000, 16000);
        assert_eq!(y.len(), 1600);
        assert!(y.iter().all(|v| (*v - 0.5).abs() < 1e-4));
    }
}
