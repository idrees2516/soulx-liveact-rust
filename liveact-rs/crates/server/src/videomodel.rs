//! Video-model endpoints: the complete SoulX-LiveAct neural pipeline exposed
//! as `/api/model/status` and a streaming `/api/model/generate` (NDJSON).
//!
//! The generate endpoint chains every part: text -> Kokoro-82M TTS ->
//! wav2vec2 (pretrained) -> AudioProj windowing -> WanModel DiT with ConvKV
//! caches + Neighbor Forcing -> causal 3D VAE -> RGB frames, all in Rust.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use liveact_model::config::LiveActConfig;
use liveact_model::pipeline::{
    GenerateParams, Pipeline, PipelineEvent, PipelineOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::state::App;

static BUSY: AtomicBool = AtomicBool::new(false);
static PIPELINE: OnceLock<Mutex<Pipeline>> = OnceLock::new();

fn get_pipeline() -> &'static Mutex<Pipeline> {
    PIPELINE.get_or_init(|| {
        Mutex::new(Pipeline::new(PipelineOptions {
            cfg: LiveActConfig::demo(),
            ..Default::default()
        }))
    })
}

#[derive(Deserialize)]
pub struct GenerateRequest {
    /// text spoken by the TTS (conditioning audio source)
    pub text: String,
    /// text-prompt conditioning for the DiT (defaults to `text`)
    pub prompt: Option<String>,
    /// optional (start_chunk, end_chunk, prompt) overrides
    pub edit_prompts: Option<Vec<(usize, usize, String)>>,
    pub voice: Option<String>,
    pub speed: Option<f32>,
    /// number of 8-latent chunks to generate (>= 1)
    pub chunks: Option<usize>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    Status {
        config: serde_json::Value,
        params: usize,
        wav2vec2_pretrained: bool,
    },
    ChunkStart {
        chunk: usize,
        total: usize,
    },
    Step {
        chunk: usize,
        step: usize,
        t: f32,
        x0_rms: f32,
        latent_rms: f32,
        audio_on: bool,
    },
    Chunk {
        #[serde(flatten)]
        report: serde_json::Value,
    },
    Frames {
        chunk: usize,
        first_frame: usize,
        w: usize,
        h: usize,
        n_frames: usize,
        rgb: String, // base64, n_frames * w * h * 3 bytes
    },
    Done {
        #[serde(flatten)]
        summary: serde_json::Value,
    },
    Error {
        message: String,
    },
}

fn event_line(ev: &Event) -> Vec<u8> {
    let mut line = serde_json::to_vec(ev).unwrap_or_else(|_| b"{}".to_vec());
    line.push(b'\n');
    line
}

/// tiny base64 (standard alphabet, padded)
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// The same deterministic reference face used by `nf_demo` (32x32).
pub fn procedural_face(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; 3 * w * h];
    let (cy, cx) = (h as f32 * 0.47, w as f32 * 0.5);
    let (ry, rx) = (h as f32 * 0.45, w as f32 * 0.42);
    for y in 0..h {
        for x in 0..w {
            let dy = (y as f32 - cy) / ry;
            let dx = (x as f32 - cx) / rx;
            let d = (dy * dy + dx * dx).sqrt();
            let face = (1.0 - d).clamp(0.0, 1.0);
            let face = if d < 1.0 { face } else { 0.0 };
            let eye_y = (h as f32 * 0.40) as usize;
            let eye_l = (w as f32 * 0.32) as usize;
            let eye_r = (w as f32 * 0.65) as usize;
            let eye = if y >= eye_y && y <= eye_y + 2 && (x == eye_l || x == eye_r) { 1.0 } else { 0.0 };
            let mouth = if y >= (h as f32 * 0.66) as usize
                && y <= (h as f32 * 0.70) as usize
                && x >= w / 3
                && x <= 2 * w / 3
            {
                1.0
            } else {
                0.0
            };
            let r = 0.55 * face + 0.10 - mouth * 0.25;
            let g = 0.42 * face + 0.12 - mouth * 0.25;
            let b = 0.38 * face + 0.22 - eye * 0.18;
            img[0 * w * h + y * w + x] = (r * 255.0) as u8;
            img[1 * w * h + y * w + x] = (g * 255.0) as u8;
            img[2 * w * h + y * w + x] = (b * 255.0) as u8;
        }
    }
    img
}

fn config_json(cfg: &LiveActConfig) -> serde_json::Value {
    json!({
        "preset": "liveact-demo",
        "width": cfg.width,
        "height": cfg.height,
        "fps": cfg.fps,
        "dim": cfg.dim,
        "layers": cfg.num_layers,
        "heads": cfg.num_heads,
        "ffn": cfg.ffn_dim,
        "latent_ch": cfg.out_dim,
        "chunk_sizes": cfg.chunk_sizes,
        "cache_frames": cfg.cache_frames(),
        "timesteps": cfg.timesteps,
        "audio": {
            "window": cfg.audio_window,
            "blocks": cfg.audio_blocks,
            "channels": cfg.audio_channels,
            "context_tokens": cfg.audio_context_tokens,
        },
        "vae": {"dim": cfg.vae_dim, "z": cfg.vae_z, "mult": cfg.vae_dim_mult},
        "full_scale": {
            "dim": 5120, "layers": 40, "heads": 40, "ffn": 13824,
            "params": "18B", "checkpoint": "SoulX-LiveAct (Wan2.1-I2V 14B + 4B audio)"
        }
    })
}

/// `GET /api/model/status` — implemented parts + configuration.
pub async fn status(State(app): State<Arc<App>>) -> Response {
    let cfg = LiveActConfig::demo();
    let initialized = PIPELINE.get().is_some();
    let parts = json!([
        {"part": "wav2vec2 audio encoder", "rust": "liveact_model::wav2vec2", "original": "src/audio_analysis/wav2vec2.py + torch_utils.py", "weights": "TencentGameMate/chinese-wav2vec2-base (377 MB, loaded when present)"},
        {"part": "audio windowing (get_audio_emb ±2 frames)", "rust": "liveact_model::pipeline::audio_window", "original": "util_liveact.py"},
        {"part": "AudioProjModel adapter (5/8-window -> 32 tokens)", "rust": "liveact_model::audio_proj", "original": "model_memory.py::AudioProjModel"},
        {"part": "WanModel DiT backbone (patchify, adaLN, head)", "rust": "liveact_model::model", "original": "model_memory.py::WanModel"},
        {"part": "self-attention with ConvKV memory (14-frame rolling cache, 5:1 depthwise compression, fp8 option)", "rust": "liveact_model::attention::SelfAttention + KvCache", "original": "model_memory.py::WanSelfAttention"},
        {"part": "text + CLIP cross-attention", "rust": "liveact_model::attention::CrossAttention", "original": "model_memory.py::WanI2VCrossAttention"},
        {"part": "audio cross-attention (SingleStreamAttention, per-frame 32 tokens)", "rust": "liveact_model::attention::AudioAttention", "original": "model_liveact/attention.py::SingleStreamAttention"},
        {"part": "3D RoPE (t/h/w split [22,21,21] @ head_dim 128)", "rust": "liveact_model::rope", "original": "model_memory.py::causal_rope_apply"},
        {"part": "Neighbor-Forcing scheduler (3 steps, per-step caches, x0 re-noise)", "rust": "liveact_model::scheduler", "original": "generate.py 347-374"},
        {"part": "causal 3D VAE (streaming encode/decode, ConvKV-style frame caches)", "rust": "liveact_model::vae", "original": "wan/modules/vae.py"},
        {"part": "streaming AR chunk pipeline (6+8 latents, 3-frame overlap decode)", "rust": "liveact_model::pipeline", "original": "generate.py::generate"},
        {"part": "Kokoro-82M TTS conditioning source", "rust": "liveact_server::tts", "original": "kokoro/ (pipeline.py)"},
        {"part": "T5/CLIP context encoders (deterministic stand-ins for the 11 GB umt5 / 2.5 GB CLIP)", "rust": "liveact_model::context", "original": "wan/modules/t5.py, wan/modules/clip.py"},
        {"part": "audio CFG (null-audio second pass)", "rust": "liveact_model::pipeline (audio_cfg)", "original": "generate.py 360-368"}
    ]);
    let (params, kv_bytes, w2v) = if initialized {
        let p = get_pipeline().lock().unwrap();
        (p.params, p.model.new_kv_caches(false).iter().map(|s| s.iter().map(|c| c.bytes_allocated()).sum::<usize>()).sum::<usize>(), p.wav2vec2_pretrained())
    } else {
        (0, 0, false)
    };
    let _ = app;
    axum::Json(json!({
        "status": "ok",
        "initialized": initialized,
        "busy": BUSY.load(Ordering::SeqCst),
        "parts": parts,
        "config": config_json(&cfg),
        "params": params,
        "kv_cache_bytes": kv_bytes,
        "wav2vec2_pretrained": w2v,
    }))
    .into_response()
}

/// `POST /api/model/generate` — NDJSON event stream.
pub async fn generate(State(app): State<Arc<App>>, axum::Json(req): axum::Json<GenerateRequest>) -> Response {
    if BUSY.swap(true, Ordering::SeqCst) {
        return (
            StatusCode::CONFLICT,
            [(header::CONTENT_TYPE, "application/json")],
            json!({"error": "a generation is already running"}).to_string(),
        )
            .into_response();
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
    let engine = app.engine.clone();
    let voice = req.voice.unwrap_or_else(|| "af_bella".into());
    let speed = req.speed.unwrap_or(1.0).clamp(0.5, 2.0);

    let text = req.text.trim().to_string();
    if text.is_empty() {
        BUSY.store(false, Ordering::SeqCst);
        return (StatusCode::BAD_REQUEST, "text must not be empty").into_response();
    }

    tokio::task::spawn_blocking(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // 1) conditioning audio via Kokoro TTS (24 kHz)
            let audio_f32: Vec<f32> = match &engine {
                Some(e) if e.has_voice(&voice) => match e.synthesize(&text, &voice, speed) {
                    Ok(s) => s.pcm_i16.iter().map(|s| *s as f32 / 32768.0).collect(),
                    Err(err) => {
                        let _ = tx.send(event_line(&Event::Error { message: format!("tts: {err}") }));
                        return;
                    }
                },
                _ => {
                    // no TTS engine: deterministic synthetic speech envelope
                    let n = 16000 * 3;
                    (0..n)
                        .map(|i| {
                            let t = i as f32 / 16000.0;
                            ((t * 3.5).sin() * 0.5 + 0.5) * (i as f32 * 0.07).sin() * 0.4
                        })
                        .collect()
                }
            };

            let pipeline = get_pipeline();
            let mut p = match pipeline.lock() {
                Ok(p) => p,
                Err(_) => {
                    let _ = tx.send(event_line(&Event::Error { message: "pipeline lock poisoned".into() }));
                    return;
                }
            };
            let cfg = p.cfg.clone();
            let chunks = req
                .chunks
                .unwrap_or_else(|| p.chunks_for_audio(&audio_f32, 24000))
                .clamp(1, 12);

            let _ = tx.send(event_line(&Event::Status {
                config: config_json(&cfg),
                params: p.params,
                wav2vec2_pretrained: p.wav2vec2_pretrained(),
            }));

            let cond_image = procedural_face(cfg.width, cfg.height);
            let prompt = req.prompt.unwrap_or_else(|| text.clone());
            let edits = req.edit_prompts.unwrap_or_default();
            let tx2 = tx.clone();
            let cur_chunk = std::cell::Cell::new(0usize);
            p.generate(
                GenerateParams {
                    prompt,
                    edit_prompts: edits,
                    audio: audio_f32,
                    audio_hz: 24000,
                    cond_image,
                    chunks,
                },
                &mut |ev| {
                    let line = match &ev {
                        PipelineEvent::ChunkStart { chunk, total } => {
                            cur_chunk.set(*chunk);
                            event_line(&Event::ChunkStart {
                                chunk: *chunk,
                                total: *total,
                            })
                        }
                        PipelineEvent::Step(s) => {
                            event_line(&Event::Step {
                                chunk: cur_chunk.get(),
                                step: s.step,
                                t: s.t,
                                x0_rms: s.x0_rms,
                                latent_rms: s.latent_rms,
                                audio_on: s.audio_on,
                            })
                        }
                        PipelineEvent::Chunk(c) => event_line(&Event::Chunk {
                            report: serde_json::to_value(c).unwrap_or(json!({})),
                        }),
                        PipelineEvent::Frames(f) => event_line(&Event::Frames {
                            chunk: f.chunk,
                            first_frame: f.first_frame,
                            w: f.w,
                            h: f.h,
                            n_frames: f.frames.len() / (f.w * f.h * 3),
                            rgb: base64(&f.frames),
                        }),
                        PipelineEvent::Done(s) => event_line(&Event::Done {
                            summary: serde_json::to_value(s).unwrap_or(json!({})),
                        }),
                    };
                    let _ = tx2.send(line);
                },
            );
        }));
        if result.is_err() {
            let _ = tx.send(event_line(&Event::Error { message: "pipeline panicked".into() }));
        }
        BUSY.store(false, Ordering::SeqCst);
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|bytes| (Ok::<_, std::io::Error>(bytes), rx))
    });
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "application/x-ndjson; charset=utf-8",
        )],
        Body::from_stream(stream),
    )
        .into_response()
}
