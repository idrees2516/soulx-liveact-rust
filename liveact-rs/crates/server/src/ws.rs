//! WebSocket session handling.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use liveact_proto::{ClientMsg, ServerMsg, PROTOCOL_VERSION, SAMPLE_RATE};
use tokio::sync::mpsc;

use crate::pipeline::{self, Queue};
use crate::state::App;

pub enum WsOut {
    Text(String),
    Binary(Vec<u8>),
}

impl WsOut {
    pub fn json(msg: ServerMsg) -> Self {
        WsOut::Text(serde_json::to_string(&msg).unwrap_or_else(|_| "{}".into()))
    }
}

pub async fn ws_handler(ws: WebSocketUpgrade, State(app): State<Arc<App>>) -> Response {
    ws.on_upgrade(move |socket| run_socket(socket, app))
}

struct ActiveSession {
    queue: Arc<Queue>,
    stop: Arc<AtomicBool>,
}

impl ActiveSession {
    fn shutdown(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.queue.clear();
    }
}

async fn run_socket(socket: WebSocket, app: Arc<App>) {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<WsOut>();

    // Writer task
    let writer = tokio::spawn(async move {
        while let Some(out) = rx.recv().await {
            let msg = match out {
                WsOut::Text(s) => Message::Text(s.into()),
                WsOut::Binary(b) => Message::Binary(b.into()),
            };
            if sender.send(msg).await.is_err() {
                break;
            }
        }
        let _ = sender.send(Message::Close(None)).await;
    });

    let hello = ServerMsg::Hello {
        sample_rate: SAMPLE_RATE,
        protocol: PROTOCOL_VERSION,
        voices: app.available_voices(),
        model: app
            .engine
            .as_ref()
            .map(|e| e.model_name().to_string())
            .unwrap_or_else(|| "unavailable".into()),
    };
    let _ = tx.send(WsOut::json(hello));

    let mut active: Option<ActiveSession> = None;

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                let parsed: Result<ClientMsg, _> = serde_json::from_str(&text);
                match parsed {
                    Ok(ClientMsg::Start {
                        script,
                        voice,
                        speed,
                        segments: _,
                    }) => {
                        if let Some(sess) = active.take() {
                            sess.shutdown();
                        }
                        let Some(engine) = app.engine.clone() else {
                            let _ = tx.send(WsOut::json(ServerMsg::Error {
                                message: "TTS engine not loaded (models missing)".into(),
                            }));
                            continue;
                        };
                        if !engine.has_voice(&voice) {
                            let _ = tx.send(WsOut::json(ServerMsg::Error {
                                message: format!("unknown voice '{voice}'"),
                            }));
                            continue;
                        }
                        let chunks = pipeline::split_script_public(&script);
                        if chunks.is_empty() {
                            let _ = tx.send(WsOut::json(ServerMsg::Error {
                                message: "script is empty".into(),
                            }));
                            continue;
                        }
                        let queue = Queue::new();
                        let n = queue.push(chunks);
                        let stop = Arc::new(AtomicBool::new(false));
                        let _ = tx.send(WsOut::json(ServerMsg::Started { total_chunks: n }));

                        let ctx = pipeline::PipelineCtx {
                            engine,
                            queue: queue.clone(),
                            stop: stop.clone(),
                            tx: tx.clone(),
                            voice,
                            speed: speed.clamp(0.5, 2.0),
                        };
                        tokio::spawn(pipeline::run_pipeline(ctx));
                        active = Some(ActiveSession { queue, stop });
                    }
                    Ok(ClientMsg::Enqueue { text }) => match &active {
                        Some(sess) => {
                            let chunks = pipeline::split_script_public(&text);
                            if chunks.is_empty() {
                                let _ = tx.send(WsOut::json(ServerMsg::Error {
                                    message: "nothing to enqueue".into(),
                                }));
                            } else {
                                sess.queue.push(chunks);
                            }
                        }
                        None => {
                            let _ = tx.send(WsOut::json(ServerMsg::Error {
                                message: "no active session: send start first".into(),
                            }));
                        }
                    },
                    Ok(ClientMsg::Stop) => {
                        if let Some(sess) = active.take() {
                            sess.shutdown();
                        }
                    }
                    Ok(ClientMsg::Ping { t }) => {
                        let _ = tx.send(WsOut::json(ServerMsg::Pong { t }));
                    }
                    Err(e) => {
                        let _ = tx.send(WsOut::json(ServerMsg::Error {
                            message: format!("bad message: {e}"),
                        }));
                    }
                }
            }
            Message::Binary(_) => {
                let _ = tx.send(WsOut::json(ServerMsg::Error {
                    message: "binary frames from client are not used".into(),
                }));
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    if let Some(sess) = active.take() {
        sess.shutdown();
    }
    drop(tx);
    writer.abort();
}
