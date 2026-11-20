//! Shared wire protocol between the LiveAct-RS server (native) and the
//! LiveAct-RS web client (wasm). Both sides depend on this crate.
//!
//! The protocol mirrors the streaming architecture of SoulX-LiveAct:
//! the server streams, per chunk: a viseme timeline (from Kokoro's
//! per-token durations) followed by the PCM audio for that chunk.

use serde::{Deserialize, Serialize};

pub const SAMPLE_RATE: u32 = 24000;
pub const PROTOCOL_VERSION: u32 = 1;

/// Viseme classes used by the avatar renderer.
/// u8 codes travel on the wire; keep in sync with the client renderer.
pub mod viseme {
    pub const REST: u8 = 0; // silence / closed
    pub const MBP: u8 = 1; // p, b, m (lips pressed)
    pub const FV: u8 = 2; // f, v
    pub const AA: u8 = 3; // wide open (a, ah)
    pub const AH: u8 = 4; // mid open (schwa)
    pub const E: u8 = 5; // e, eh
    pub const IH: u8 = 6; // i, ee, y
    pub const OH: u8 = 7; // o
    pub const UW: u8 = 8; // u, oo, w
    pub const SS: u8 = 9; // s, z, sh, ch, j
    pub const TT: u8 = 10; // t, d, n, th
    pub const LL: u8 = 11; // l
    pub const RR: u8 = 12; // r
    pub const KK: u8 = 13; // k, g, ng, h
    pub const COUNT: u8 = 14;

    pub const NAMES: [&str; 14 as usize] = [
        "rest", "mbp", "fv", "aa", "ah", "e", "ih", "oh", "uw", "ss", "tt", "ll", "rr", "kk",
    ];
}

/// A timed viseme span relative to the start of a chunk's audio.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VisemeSpan {
    pub v: u8,
    #[serde(rename = "t0")]
    pub t0_ms: u32,
    #[serde(rename = "t1")]
    pub t1_ms: u32,
    /// True when the underlying phoneme carries lexical stress.
    #[serde(default)]
    pub stress: bool,
}

/// An action/emotion editing segment, mirroring the original demo's
/// timed "Edit Prompt" segments.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Segment {
    #[serde(rename = "from")]
    pub from_s: f32,
    #[serde(rename = "to")]
    pub to_s: f32,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub language: String,
    pub gender: String,
}

/// Messages the browser sends to the server.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    /// Begin a streaming session from a script.
    Start {
        script: String,
        voice: String,
        speed: f32,
        #[serde(default)]
        segments: Vec<Segment>,
    },
    /// Queue more text while the session is live.
    Enqueue { text: String },
    /// Stop the current stream.
    Stop,
    Ping { t: u64 },
}

/// Messages the server sends to the browser.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    Hello {
        sample_rate: u32,
        protocol: u32,
        voices: Vec<VoiceInfo>,
        model: String,
    },
    Started {
        total_chunks: usize,
    },
    /// Sent right before the chunk's audio binary frame.
    ChunkStart {
        idx: usize,
        text: String,
        #[serde(rename = "audioMs")]
        audio_ms: u32,
        #[serde(rename = "gapMs")]
        gap_ms: u32,
        visemes: Vec<VisemeSpan>,
    },
    ChunkEnd {
        idx: usize,
    },
    Progress {
        state: String, // running | stopped | done
        generated: usize,
        total: usize,
        message: String,
    },
    Done {
        #[serde(rename = "audioMs")]
        audio_ms: u32,
        #[serde(rename = "synthMs")]
        synth_ms: u32,
        rtf: f32,
    },
    Error {
        message: String,
    },
    Pong {
        t: u64,
    },
}

/// Binary audio frame layout (little-endian), sent right after ChunkStart:
///   [0..4]  u32 chunk index
///   [4..8]  u32 sample count N
///   [8..]   N * i16 PCM mono @ SAMPLE_RATE
pub fn encode_audio_frame(chunk_idx: u32, pcm_i16: &[i16], out: &mut Vec<u8>) {
    out.clear();
    out.reserve_exact(8 + pcm_i16.len() * 2);
    out.extend_from_slice(&chunk_idx.to_le_bytes());
    out.extend_from_slice(&(pcm_i16.len() as u32).to_le_bytes());
    for s in pcm_i16 {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

/// Metadata for the voices we ship. Grades/names follow the model card.
pub fn voice_catalog() -> Vec<VoiceInfo> {
    let mk = |id: &str, name: &str, language: &str, gender: &str| VoiceInfo {
        id: id.to_string(),
        name: name.to_string(),
        language: language.to_string(),
        gender: gender.to_string(),
    };
    vec![
        mk("af_sky", "Sky", "en-us", "female"),
        mk("af_bella", "Bella", "en-us", "female"),
        mk("af_nicole", "Nicole", "en-us", "female"),
        mk("af_nova", "Nova", "en-us", "female"),
        mk("af_sarah", "Sarah", "en-us", "female"),
        mk("af_heart", "Heart", "en-us", "female"),
        mk("am_adam", "Adam", "en-us", "male"),
        mk("am_michael", "Michael", "en-us", "male"),
        mk("am_onyx", "Onyx", "en-us", "male"),
        mk("am_echo", "Echo", "en-us", "male"),
        mk("bf_emma", "Emma", "en-gb", "female"),
        mk("bf_isabella", "Isabella", "en-gb", "female"),
        mk("bm_george", "George", "en-gb", "male"),
        mk("ef_dora", "Dora", "es", "female"),
        mk("ff_siwis", "Siwis", "fr", "female"),
        mk("hf_alpha", "Alpha", "hi", "female"),
        mk("if_sara", "Sara", "it", "female"),
        mk("im_nicola", "Nicola", "it", "male"),
        mk("pf_dora", "Dora", "pt", "female"),
        mk("pm_alex", "Alex", "pt", "male"),
    ]
}
