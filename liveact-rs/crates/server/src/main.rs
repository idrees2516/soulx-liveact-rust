//! LiveAct-RS: real-time streaming talking-avatar system in Rust.
//!
//! A system-level port of the SoulX-LiveAct architecture:
//! - Kokoro-82M TTS (ONNX, CPU) with per-token durations
//! - WebSocket streaming of audio + viseme timelines per chunk
//! - bounded-look-ahead pipeline (O(1) session memory, hour-scale safe)
//! - Rust/WASM frontend rendering a lip-synced avatar

use std::path::PathBuf;
use std::sync::Arc;

use liveact_server::state::App;
use liveact_server::tts;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,liveact_server=debug".into()),
        )
        .init();

    let models_dir = PathBuf::from(
        std::env::var("LIVEACT_MODELS").unwrap_or_else(|_| "models".to_string()),
    );
    let engine = match tts::Kokoro::load(&models_dir) {
        Ok(e) => {
            tracing::info!(
                model = e.model_name(),
                voices = e.voice_ids().len(),
                "TTS engine ready"
            );
            Some(Arc::new(e))
        }
        Err(e) => {
            tracing::error!(err = %e, "TTS engine unavailable; server starts degraded");
            None
        }
    };

    let app = Arc::new(App {
        engine,
        started_at: std::time::Instant::now(),
    });

    let addr = std::env::var("LIVEACT_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.block_on(async move {
        let listener = match tokio::net::TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(addr = %addr, err = %e, "failed to bind");
                std::process::exit(1);
            }
        };
        tracing::info!(addr = %addr, "LiveAct-RS server listening");
        axum::serve(listener, liveact_server::http::router(app))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                tracing::info!("shutting down");
            })
            .await
            .expect("server error");
    });
}
