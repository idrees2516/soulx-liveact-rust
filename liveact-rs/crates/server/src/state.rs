//! Shared application state.

use std::sync::Arc;
use std::time::Instant;

use liveact_proto::VoiceInfo;

use crate::tts::Kokoro;

pub struct App {
    /// None when the TTS models failed to load (degraded mode).
    pub engine: Option<Arc<Kokoro>>,
    pub started_at: Instant,
}

impl App {
    pub fn available_voices(&self) -> Vec<VoiceInfo> {
        let catalog = liveact_proto::voice_catalog();
        match &self.engine {
            Some(engine) => catalog
                .into_iter()
                .filter(|v| engine.has_voice(&v.id))
                .collect(),
            None => vec![],
        }
    }
}
