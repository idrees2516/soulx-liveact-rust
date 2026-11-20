//! Per-session streaming pipeline: synthesizes chunks ahead of playback
//! with a bounded look-ahead window, so memory stays O(1) for any
//! session length (the ConvKV-Memory property of the original system,
//! recreated on the system level).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use liveact_proto::{encode_audio_frame, ServerMsg};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Notify;

use crate::tts::Kokoro;
use crate::ws::WsOut;

/// Public wrapper used by the websocket layer.
pub fn split_script_public(script: &str) -> Vec<String> {
    crate::tts::split::split_script(script, MAX_CHUNK_CHARS)
}

/// How far ahead of real-time playback we synthesize (ms).
const LOOKAHEAD_MS: f32 = 14_000.0;
/// Maximum characters per chunk fed to the model.
pub const MAX_CHUNK_CHARS: usize = 300;

pub struct Queue {
    chunks: Mutex<VecDeque<String>>,
    pub total: AtomicUsize,
    pub generated: AtomicUsize,
    notify: Notify,
}

impl Queue {
    fn new_inner() -> Self {
        Self {
            chunks: Mutex::new(VecDeque::new()),
            total: AtomicUsize::new(0),
            generated: AtomicUsize::new(0),
            notify: Notify::new(),
        }
    }

    pub fn new() -> Arc<Self> {
        Arc::new(Self::new_inner())
    }

    pub fn push(&self, chunks: Vec<String>) -> usize {
        let n = chunks.len();
        {
            let mut q = self.chunks.lock().unwrap();
            for c in chunks {
                q.push_back(c);
            }
        }
        self.total.fetch_add(n, Ordering::SeqCst);
        self.notify.notify_one();
        n
    }

    pub fn clear(&self) -> usize {
        let mut q = self.chunks.lock().unwrap();
        let n = q.len();
        q.clear();
        n
    }

    fn pop(&self) -> Option<String> {
        self.chunks.lock().unwrap().pop_front()
    }

    fn is_empty(&self) -> bool {
        self.chunks.lock().unwrap().is_empty()
    }

    /// Waits until new chunks are pushed (or a permit is stored).
    pub async fn notified(&self) {
        self.notify.notified().await
    }
}

impl Default for Queue {
    fn default() -> Self {
        Self::new_inner()
    }
}

pub struct PipelineCtx {
    pub engine: Arc<Kokoro>,
    pub queue: Arc<Queue>,
    pub stop: Arc<AtomicBool>,
    pub tx: UnboundedSender<WsOut>,
    pub voice: String,
    pub speed: f32,
}

pub async fn run_pipeline(ctx: PipelineCtx) {
    let PipelineCtx {
        engine,
        queue,
        stop,
        tx,
        voice,
        speed,
    } = ctx;

    let mut scheduled_ms: f32 = 0.0;
    let mut audio_ms_total: u64 = 0;
    let mut synth_ms_total: u64 = 0;
    let started = Instant::now();
    let mut sent_done = false;

    loop {
        if stop.load(Ordering::SeqCst) {
            let _ = tx.send(WsOut::json(ServerMsg::Progress {
                state: "stopped".into(),
                generated: queue.generated.load(Ordering::SeqCst),
                total: queue.total.load(Ordering::SeqCst),
                message: "stopped".into(),
            }));
            break;
        }

        match queue.pop() {
            Some(text) => {
                // Pace: don't synthesize far ahead of playback.
                let elapsed_ms = started.elapsed().as_millis() as f32;
                let ahead = scheduled_ms - elapsed_ms;
                if ahead > LOOKAHEAD_MS {
                    let wait = Duration::from_millis((ahead - LOOKAHEAD_MS) as u64);
                    tokio::select! {
                        _ = tokio::time::sleep(wait) => {}
                        _ = queue.notified() => {}
                        _ = await_stop(&stop) => {}
                    }
                    if stop.load(Ordering::SeqCst) {
                        continue;
                    }
                }

                // Synthesize on a blocking thread (CPU-bound ONNX work).
                let engine = engine.clone();
                let voice2 = voice.clone();
                let text_for_synth = text.clone();
                let t0 = Instant::now();
                let result = tokio::task::spawn_blocking(move || {
                    engine.synthesize(&text_for_synth, &voice2, speed)
                })
                .await;
                let synth_ms = t0.elapsed().as_millis() as u64;
                synth_ms_total += synth_ms;

                match result {
                    Ok(Ok(synth)) => {
                        let idx = queue.generated.fetch_add(1, Ordering::SeqCst);
                        scheduled_ms += synth.audio_ms as f32 + synth.gap_ms as f32;
                        audio_ms_total += synth.audio_ms as u64;

                        let _ = tx.send(WsOut::json(ServerMsg::ChunkStart {
                            idx,
                            text: text.clone(),
                            audio_ms: synth.audio_ms,
                            gap_ms: synth.gap_ms,
                            visemes: synth.visemes,
                        }));
                        let mut frame = Vec::with_capacity(8 + synth.pcm_i16.len() * 2);
                        encode_audio_frame(idx as u32, &synth.pcm_i16, &mut frame);
                        let _ = tx.send(WsOut::Binary(frame));
                        let _ = tx.send(WsOut::json(ServerMsg::ChunkEnd { idx }));
                        let _ = tx.send(WsOut::json(ServerMsg::Progress {
                            state: "running".into(),
                            generated: idx + 1,
                            total: queue.total.load(Ordering::SeqCst),
                            message: format!("chunk {} synthesized in {synth_ms} ms", idx + 1),
                        }));
                        sent_done = false;
                    }
                    Ok(Err(e)) => {
                        let _ = tx.send(WsOut::json(ServerMsg::Error {
                            message: format!("chunk failed: {e}"),
                        }));
                        let _ = queue.generated.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(e) => {
                        let _ = tx.send(WsOut::json(ServerMsg::Error {
                            message: format!("synthesis task panicked: {e}"),
                        }));
                        break;
                    }
                }
            }
            None => {
                // Queue drained.
                if !sent_done {
                    let generated = queue.generated.load(Ordering::SeqCst);
                    let total = queue.total.load(Ordering::SeqCst);
                    if generated >= total && total > 0 {
                        let rtf = if audio_ms_total > 0 {
                            synth_ms_total as f64 / audio_ms_total as f64
                        } else {
                            0.0
                        };
                        let _ = tx.send(WsOut::json(ServerMsg::Done {
                            audio_ms: audio_ms_total as u32,
                            synth_ms: synth_ms_total as u32,
                            rtf: ((rtf * 1000.0).round() / 1000.0) as f32,
                        }));
                        let _ = tx.send(WsOut::json(ServerMsg::Progress {
                            state: "done".into(),
                            generated,
                            total,
                            message: "all chunks streamed".into(),
                        }));
                        sent_done = true;
                    }
                }
                // Wait for more text (Enqueue) or stop, with a periodic
                // re-check of the stop flag.
                tokio::select! {
                    _ = queue.notified() => {}
                    _ = tokio::time::sleep(Duration::from_millis(250)) => {}
                }
            }
        }
    }
}

async fn await_stop(stop: &Arc<AtomicBool>) {
    loop {
        if stop.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
