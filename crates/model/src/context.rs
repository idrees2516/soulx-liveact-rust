//! Conditioning context encoders (T5 text / CLIP visual interfaces).
//!
//! The original uses `umt5-xxl` (11 GB) for text and open-clip
//! `xlm-roberta-large vit-huge-14` (2.5 GB) for the reference image. Those
//! weights are GPU-class; this module provides the same interfaces with
//! deterministic encoders (hash-seeded token embeddings + positional
//! structure), so the full DiT runs end-to-end on CPU with stable,
//! reproducible conditioning. Swapping in real encoders means implementing
//! the same two `encode_*` functions.

use crate::tensor::Rng;

/// FNV-1a hash for token seeding.
fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Simple tokenizer: split on non-alphanumeric, lowercase, cap at 64 tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut toks: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect();
    toks.truncate(64);
    toks
}

/// Deterministic T5-style encoder: `[tokens, text_dim]`.
/// Token vectors get a positional modulation so different prompts and
/// orderings produce distinct contexts.
pub fn encode_text(text: &str, text_dim: usize, max_tokens: usize) -> (usize, Vec<f32>) {
    let toks = tokenize(text);
    let n = toks.len().min(max_tokens).max(1);
    let mut out = vec![0.0f32; n * text_dim];
    for (i, tok) in toks.iter().take(n).enumerate() {
        let mut rng = Rng::new(fnv(tok));
        let pos = i as f32 + 1.0;
        for (j, v) in out[i * text_dim..(i + 1) * text_dim].iter_mut().enumerate() {
            let base = rng.normal();
            let wave = (pos * 0.7 + j as f32 * 0.13).sin() * 0.3;
            *v = (base + wave) * 0.02;
        }
    }
    if toks.is_empty() {
        // empty prompt -> single neutral token
        let mut rng = Rng::new(1);
        for v in out[..text_dim].iter_mut() {
            *v = rng.normal() * 0.01;
        }
    }
    (n, out)
}

/// Deterministic CLIP-visual-style encoder for the reference image:
/// `rgb` is `[3, h, w]` bytes; returns `[257, clip_dim]` (CLS + 16x16
/// patches, matching vit-huge-14 geometry).
pub fn encode_image_clip(rgb: &[u8], h: usize, w: usize, clip_dim: usize) -> Vec<f32> {
    assert_eq!(rgb.len(), 3 * h * w);
    let mut out = vec![0.0f32; 257 * clip_dim];
    // CLS token: global image statistics
    let mut mean = [0.0f32; 3];
    for p in 0..h * w {
        for c in 0..3 {
            mean[c] += rgb[c * h * w + p] as f32 / 255.0;
        }
    }
    for m in mean.iter_mut() {
        *m /= (h * w) as f32;
    }
    let mut rng = Rng::new(0xC11);
    for j in 0..clip_dim {
        let c = j % 3;
        let base = mean[c] * 2.0 - 1.0;
        out[j] = (base + rng.normal() * 0.05).tanh() * 0.5;
    }
    // 16x16 patch tokens
    let (py, px) = ((h / 16).max(1), (w / 16).max(1));
    for pi in 0..16 {
        for pj in 0..16 {
            let tok = 1 + pi * 16 + pj;
            // patch mean color
            let mut pm = [0.0f32; 3];
            let mut count = 0usize;
            let y0 = pi * py;
            let x0 = pj * px;
            for y in y0..(y0 + py).min(h) {
                for x in x0..(x0 + px).min(w) {
                    for c in 0..3 {
                        pm[c] += rgb[c * h * w + y * w + x] as f32 / 255.0;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                for m in pm.iter_mut() {
                    *m /= count as f32;
                }
            }
            let mut rng = Rng::new(fnv(&format!("clip-{pi}-{pj}")));
            for j in 0..clip_dim {
                let c = j % 3;
                let base = pm[c] * 2.0 - 1.0;
                out[tok * clip_dim + j] =
                    (base + rng.normal() * 0.1).tanh() * 0.5;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_encoding_deterministic_and_distinct() {
        let (n1, a) = encode_text("a person waves hello", 512, 64);
        let (_, b) = encode_text("a person waves hello", 512, 64);
        let (_, c) = encode_text("a different prompt entirely", 512, 64);
        assert_eq!(n1, 4);
        assert_eq!(a, b, "deterministic");
        assert_ne!(a, c, "distinct prompts");
    }

    #[test]
    fn clip_encoding_shape_and_clip_tokens() {
        let h = 64;
        let w = 64;
        let rgb: Vec<u8> = (0..3 * h * w).map(|i| (i % 251) as u8).collect();
        let out = encode_image_clip(&rgb, h, w, 1280);
        assert_eq!(out.len(), 257 * 1280);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
