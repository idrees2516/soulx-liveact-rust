//! Audio output: schedules 24 kHz PCM chunks on a precise Web Audio
//! timeline and exposes the stream position used to drive the avatar.
//!
//! The stream clock is `performance.now()` (monotonic, immune to the
//! suspended-context clock jumps some browsers show); the AudioContext
//! clock is only used *relatively* (`now + delta`) for start times.

use wasm_bindgen::{JsCast, UnwrapThrowExt};
use web_sys::{AudioBufferSourceNode, AudioContext, AudioContextOptions, BaseAudioContext};

pub const SAMPLE_RATE: f32 = liveact_proto::SAMPLE_RATE as f32;

pub fn perf_now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or_else(|| js_sys::Date::now())
}

pub struct AudioOut {
    ctx: AudioContext,
    /// Stream origin in performance-now ms (set when the first chunk is
    /// scheduled).
    origin: Option<f64>,
    /// Next chunk start in performance-now ms (includes gaps).
    cursor: f64,
    live: Vec<(AudioBufferSourceNode, f64)>,
    /// Total stream ms scheduled (audio + gaps).
    scheduled_ms: f64,
}

impl AudioOut {
    pub fn new() -> Result<Self, String> {
        let mut opts = AudioContextOptions::new();
        opts.set_sample_rate(SAMPLE_RATE);
        let ctx = AudioContext::new_with_context_options(&opts)
            .map_err(|e| format!("audio context: {e:?}"))?;
        Ok(Self {
            ctx,
            origin: None,
            cursor: 0.0,
            live: Vec::new(),
            scheduled_ms: 0.0,
        })
    }

    pub fn reset(&mut self) {
        for (s, _) in self.live.drain(..) {
            let _ = s.stop_with_when(0.0);
        }
        self.origin = None;
        self.cursor = 0.0;
        self.scheduled_ms = 0.0;
    }

    pub fn wake(&mut self) {
        let _ = self.ctx.resume();
    }

    /// Schedule one chunk of i16 PCM. `gap_ms` is the pause inserted
    /// before this chunk. Returns the stream-ms at which this chunk
    /// actually starts (late chunks are clamped to "now" so the viseme
    /// timeline matches what is audible).
    pub fn schedule_chunk(&mut self, pcm_i16: &[i16], gap_ms: u32) -> u32 {
        if pcm_i16.is_empty() {
            return self.scheduled_ms as u32;
        }
        let now_p = perf_now();
        if self.origin.is_none() {
            self.origin = Some(now_p + 80.0);
            self.cursor = now_p + 80.0;
        }
        let origin = self.origin.unwrap();

        let desired = self.cursor + gap_ms as f64;
        let start_perf = desired.max(now_p + 40.0); // never in the past

        let n = pcm_i16.len() as u32;
        let buffer = self.ctx.create_buffer(1, n, SAMPLE_RATE).unwrap_throw();
        let mut samples: Vec<f32> = Vec::with_capacity(pcm_i16.len());
        for s in pcm_i16 {
            samples.push(*s as f32 / 32768.0);
        }
        buffer.copy_to_channel(&mut samples, 0).unwrap_throw();

        // Map the perf-time start onto the ctx clock *relatively*.
        let when = self.ctx.current_time() + (start_perf - now_p) / 1000.0;
        let src = self.ctx.create_buffer_source().unwrap_throw();
        src.set_buffer(Some(&buffer));
        let _ = src.connect_with_audio_node(&self.ctx.destination());
        let _ = src.start_with_when(when);

        self.live.push((src, start_perf));
        self.prune(now_p);

        let chunk_ms = samples.len() as f64 * 1000.0 / SAMPLE_RATE as f64;
        let chunk_start_ms = start_perf - origin;
        self.cursor = start_perf + chunk_ms;
        self.scheduled_ms = chunk_start_ms + chunk_ms;
        chunk_start_ms.max(0.0) as u32
    }

    fn prune(&mut self, now_p: f64) {
        self.live.retain(|(_, t)| *t > now_p);
        if self.live.len() > 32 {
            let cut = self.live.len() - 32;
            self.live.drain(..cut);
        }
    }

    /// Current playback position within the stream (ms).
    pub fn stream_pos_ms(&self) -> f64 {
        match self.origin {
            Some(o) => (perf_now() - o).max(0.0),
            None => 0.0,
        }
    }

    /// How much audio is scheduled but not yet played (ms).
    pub fn buffered_ahead_ms(&self) -> f64 {
        (self.scheduled_ms - self.stream_pos_ms()).max(0.0)
    }

    /// Total stream ms scheduled so far (audio + gaps).
    pub fn scheduled_ms(&self) -> f64 {
        self.scheduled_ms
    }

    pub fn is_active(&self) -> bool {
        self.origin.is_some() && self.stream_pos_ms() < self.scheduled_ms + 500.0
    }

    pub fn ctx(&self) -> &AudioContext {
        &self.ctx
    }
}
