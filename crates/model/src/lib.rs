//! # liveact-model — the SoulX-LiveAct video model, in Rust
//!
//! System-level port of every neural component of
//! [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct):
//!
//! | Original (Python) | This crate |
//! | --- | --- |
//! | `src/audio_analysis/wav2vec2.py` | [`wav2vec2`] |
//! | `model_liveact/model_memory.py` (`WanModel`, ConvKV, blocks, head) | [`model`], [`attention`], [`block`] |
//! | `model_liveact/attention.py` (`SingleStreamAttention`) | [`attention::AudioAttention`] |
//! | `AudioProjModel` + audio windowing | [`audio_proj`] |
//! | `wan/modules/vae.py` (causal 3D VAE, streaming caches) | [`vae`] |
//! | Neighbor-Forcing loop in `generate.py` | [`scheduler`], [`pipeline`] |
//! | `util_liveact.py` (get_msk, get_audio_emb, resample) | [`pipeline`] |
//!
//! Everything runs on CPU with `rayon` parallel kernels; see [`config`].

pub mod attention;
pub mod context;
pub mod pipeline;
pub mod scheduler;
pub mod audio_proj;
pub mod block;
pub mod config;
pub mod model;
pub mod rope;
pub mod tensor;
pub mod vae;
pub mod wav2vec2;
