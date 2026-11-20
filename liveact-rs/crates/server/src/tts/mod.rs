//! Kokoro-82M TTS engine: text -> (24 kHz PCM, viseme timeline),
//! running fully on CPU via ONNX Runtime, with espeak-ng phonemization.
//!
//! This mirrors the reference Kokoro pipeline:
//! normalize -> split punctuation -> espeak IPA per segment -> restore
//! punctuation -> kokoro post-fixes -> vocab tokens -> ONNX (with the
//! timestamped graph that also returns per-token durations).

pub mod espeak;
pub mod normalize;
pub mod split;
pub mod visemes;
pub mod vocab;

use std::collections::HashMap;
use std::path::Path;

use liveact_proto::{viseme as vis, VisemeSpan};
use ort::session::{Session, SessionInputValue, SessionInputs, SessionOutputs};
use ort::value::Value;

use crate::tts::vocab::vocab;

/// Each voice pack is a [510 x 256] f32 matrix; the style vector is row
/// `min(token_count, 509)` (token_count excludes the $ pads).
pub struct VoicePack {
    data: Vec<f32>,
}

impl VoicePack {
    pub fn from_bytes(raw: &[u8]) -> Option<Self> {
        if raw.len() != 510 * 256 * 4 {
            return None;
        }
        let mut data = vec![0f32; 510 * 256];
        for (i, chunk) in raw.chunks_exact(4).enumerate() {
            data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Some(VoicePack { data })
    }

    fn style_row(&self, token_count: usize) -> &[f32] {
        let row = token_count.min(509);
        &self.data[row * 256..(row + 1) * 256]
    }
}

pub struct Kokoro {
    session: std::sync::Mutex<Session>,
    voices: HashMap<String, VoicePack>,
    model_name: String,
}

pub struct Synthesis {
    pub pcm_i16: Vec<i16>,
    pub visemes: Vec<VisemeSpan>,
    pub audio_ms: u32,
    pub gap_ms: u32,
    pub phonemes: String,
}

/// Frames per second of the duration output (hop 600 @ 24 kHz).
const DUR_FRAMES_PER_SEC: f32 = 40.0;

impl Kokoro {
    /// Load the engine from a models directory containing
    /// `kokoro-q8f16.onnx` (or $LIVEACT_MODEL) and `voices/*.bin`.
    pub fn load(models_dir: &Path) -> Result<Self, String> {
        let model_path = std::env::var("LIVEACT_MODEL")
            .map(PathBuf_to_string)
            .unwrap_or_else(|_| models_dir.join("kokoro-q8f16.onnx"));
        if !model_path.exists() {
            // try any .onnx in the dir
            let mut found = None;
            if let Ok(rd) = std::fs::read_dir(models_dir) {
                for e in rd.flatten() {
                    if e.path().extension().is_some_and(|x| x == "onnx") {
                        found = Some(e.path());
                        break;
                    }
                }
            }
            let model_path = found.ok_or_else(|| {
                format!(
                    "TTS model not found at {} (set LIVEACT_MODEL or place kokoro-q8f16.onnx there)",
                    model_path.display()
                )
            })?;
            return Self::load_inner(model_path, models_dir);
        }
        Self::load_inner(model_path, models_dir)
    }

    fn load_inner(model_path: std::path::PathBuf, models_dir: &Path) -> Result<Self, String> {
        tracing::info!(path = %model_path.display(), "loading Kokoro model");
        let started = std::time::Instant::now();
        // NOTE: ONNX Runtime 1.23.2 segfaults on this model during the
        // level-3 graph-transformer pass; since every transformer reports
        // "modified: 0" for this graph anyway, we disable optimization by
        // default (LIVEACT_ORT_OPT=1 re-enables it for experimentation).
        let opt_level = if std::env::var("LIVEACT_ORT_OPT").is_ok() {
            ort::session::builder::GraphOptimizationLevel::Level3
        } else {
            ort::session::builder::GraphOptimizationLevel::Disable
        };
        let session = ort::session::builder::SessionBuilder::new()
            .map_err(|e| format!("session builder: {e}"))?
            .with_optimization_level(opt_level)
            .map_err(|e| format!("opt level: {e}"))?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(|e| format!("execution providers: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("failed to load ONNX model: {e}"))?;
        tracing::info!(ms = started.elapsed().as_millis() as u64, "model loaded");

        let mut voices = HashMap::new();
        let voices_dir = models_dir.join("voices");
        if voices_dir.is_dir() {
            for entry in std::fs::read_dir(&voices_dir).map_err(|e| e.to_string())?.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|x| x == "bin") {
                    let Some(name) = p.file_stem().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    match std::fs::read(&p) {
                        Ok(raw) => match VoicePack::from_bytes(&raw) {
                            Some(pack) => {
                                voices.insert(name.to_string(), pack);
                            }
                            None => {
                                tracing::warn!(voice = name, "unexpected voice size, skipped");
                            }
                        },
                        Err(e) => tracing::warn!(voice = name, err = %e, "voice read failed"),
                    }
                }
            }
        }
        tracing::info!(count = voices.len(), "voices loaded");
        let model_name = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("kokoro")
            .to_string();
        Ok(Kokoro {
            session: std::sync::Mutex::new(session),
            voices,
            model_name,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn voice_ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self.voices.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn has_voice(&self, id: &str) -> bool {
        self.voices.contains_key(id)
    }

    /// Full kokoro phonemization of a chunk of text.
    pub fn phonemize(&self, text: &str, voice: &str) -> Result<String, String> {
        let norm = normalize::normalize(text);
        let espeak_voice = espeak::espeak_voice_for(voice);
        let mut joined = String::new();
        for (seg, punct) in split::split_punctuation(&norm) {
            let ph = espeak::phonemize_segment(&seg, espeak_voice)?;
            if !ph.is_empty() {
                joined.push_str(ph.trim());
                joined.push(' ');
            }
            if !punct.is_empty() {
                joined.push_str(&punct);
                joined.push(' ');
            }
        }

        // kokoro post-fixes (reference order)
        let mut ps = joined
            .replace("k\u{259}k\u{2c8}o\u{2d0}\u{279}o\u{28a}", "k\u{2c8}o\u{28a}k\u{259}\u{279}o\u{28a}")
            .replace("k\u{259}k\u{2c8}\u{254}\u{2d0}\u{279}\u{259}\u{28a}", "k\u{2c8}\u{259}\u{28a}k\u{259}\u{279}\u{259}\u{28a}")
            .replace('\u{2b2}', "j")
            .replace('r', "\u{279}")
            .replace('x', "k")
            .replace('\u{26c}', "l");
        ps = insert_space_before_hundred(&ps);
        ps = join_z_before_punct(&ps);
        if voice.starts_with('a') {
            ps = ninety_fix(&ps);
        }

        // Keep only vocabulary characters, collapse whitespace.
        let v = vocab();
        let mut filtered = String::with_capacity(ps.len());
        let mut last_space = false;
        for c in ps.chars() {
            if c == ' ' {
                if !last_space {
                    filtered.push(' ');
                }
                last_space = true;
                continue;
            }
            last_space = false;
            if v.forward.contains_key(&c) {
                filtered.push(c);
            }
        }
        Ok(filtered.trim().to_string())
    }

    /// Synthesize one chunk: phonemize, run the model, build PCM + the
    /// viseme timeline from the per-token durations.
    pub fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Synthesis, String> {
        let pack = self
            .voices
            .get(voice)
            .ok_or_else(|| format!("unknown voice '{voice}'"))?;
        let speed = speed.clamp(0.5, 2.0);

        let phonemes = self.phonemize(text, voice)?;
        if phonemes.is_empty() {
            return Err("nothing to speak after normalization".into());
        }

        let tokens = vocab::tokenize(&phonemes);
        if tokens.is_empty() {
            return Err("no tokens after phonemization".into());
        }
        let mut padded: Vec<i64> = Vec::with_capacity(tokens.len() + 2);
        padded.push(0);
        padded.extend_from_slice(&tokens);
        padded.push(0);

        let style = pack.style_row(tokens.len());

        // ---- ONNX inference ----
        let tokens_len = padded.len();
        let tokens_flat = padded.clone();
        let style_vec = style.to_vec();
        let speed_vec = vec![speed];

        let t_tokens: Value = Value::from(
            ort::value::Tensor::from_array(([1usize, tokens_len], tokens_flat))
                .map_err(|e| e.to_string())?,
        );
        let t_style: Value = Value::from(
            ort::value::Tensor::from_array(([1usize, 256usize], style_vec))
                .map_err(|e| e.to_string())?,
        );
        let t_speed: Value = Value::from(
            ort::value::Tensor::from_array(([1usize], speed_vec))
                .map_err(|e| e.to_string())?,
        );

        let inputs = vec![
            (
                std::borrow::Cow::Borrowed("input_ids"),
                SessionInputValue::Owned(t_tokens),
            ),
            (
                std::borrow::Cow::Borrowed("style"),
                SessionInputValue::Owned(t_style),
            ),
            (
                std::borrow::Cow::Borrowed("speed"),
                SessionInputValue::Owned(t_speed),
            ),
        ];
        // Run and extract inside the lock scope: `SessionOutputs` borrows
        // the session, so we copy the tensors out before releasing it.
        let (wave_data, durs) = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| "session mutex poisoned".to_string())?;
            let outputs: SessionOutputs = session
                .run(SessionInputs::from(inputs))
                .map_err(|e| format!("inference failed: {e}"))?;

            let wave = outputs["waveform"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("no waveform output: {e}"))?;
            let wave_data: Vec<f32> = wave.1.to_vec();

            let durs = outputs["durations"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("no durations output: {e}"))?;
            let durs: Vec<f32> = durs.1.to_vec();
            (wave_data, durs)
        };

        // PCM: f32 -> i16
        let mut pcm_i16 = Vec::with_capacity(wave_data.len());
        for s in &wave_data {
            let s = s.clamp(-1.0, 1.0);
            pcm_i16.push((s * 32767.0) as i16);
        }
        let audio_ms = (pcm_i16.len() as f64 * 1000.0 / liveact_proto::SAMPLE_RATE as f64) as u32;

        // ---- viseme timeline ----
        let visemes = build_viseme_timeline(&padded, &durs, audio_ms);

        // gap after this chunk, from its trailing punctuation
        let trimmed = text.trim_end();
        let last = trimmed.chars().last().unwrap_or('.');
        let gap_ms = match last {
            '.' | '!' | '?' | '\u{2026}' | '\u{3002}' | '\u{ff01}' | '\u{ff1f}' => 420,
            ',' | ';' | ':' => 260,
            _ => 180,
        };

        Ok(Synthesis {
            pcm_i16,
            visemes,
            audio_ms,
            gap_ms,
            phonemes,
        })
    }
}

/// Insert a space before "hˈʌndɹɪd" when it follows a letter/ɹ/ː
/// (reference lookbehind rule, hand-coded).
fn insert_space_before_hundred(s: &str) -> String {
    const TARGET: &str = "h\u{2c8}\u{28c}nd\u{279}\u{26a}d";
    if !s.contains(TARGET) {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let tchars: Vec<char> = TARGET.chars().collect();
    let mut out = String::with_capacity(s.len() + 4);
    let mut i = 0;
    while i < chars.len() {
        if i > 0
            && chars.len() >= i + tchars.len()
            && chars[i..i + tchars.len()] == tchars[..]
        {
            let prev = chars[i - 1];
            if prev.is_ascii_lowercase() || prev == '\u{279}' || prev == '\u{2d0}' {
                out.push(' ');
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// " z" before punctuation/end -> "z" (reference lookahead rule, hand-coded).
fn join_z_before_punct(s: &str) -> String {
    const PUNCT: &str = ";:,.!?\u{a1}\u{bf}\u{2014}\u{2026}\"\u{ab}\u{bb}\u{201c}\u{201d} ";
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' '
            && chars.get(i + 1) == Some(&'z')
            && chars
                .get(i + 2)
                .is_none_or(|n| PUNCT.contains(*n))
        {
            // skip the space, keep the z
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// nˈaɪnti -> nˈaɪndi (en-us only), unless followed by ː.
fn ninety_fix(s: &str) -> String {
    const A: &str = "n\u{2c8}a\u{26a}nti";
    const B: &str = "n\u{2c8}a\u{26a}ndi";
    if !s.contains(A) {
        return s.to_string();
    }
    let mut out = s.to_string();
    while let Some(pos) = out.find(A) {
        let after = &out[pos + A.len()..];
        if after.starts_with('\u{2d0}') {
            break;
        }
        out = format!("{}{}{}", &out[..pos], B, after);
    }
    out
}

/// Build the viseme timeline from per-token durations. If the model's
/// duration timeline does not exactly match the audio length, the spans
/// are rescaled to the real audio duration (keeps lips in sync).
fn build_viseme_timeline(padded: &[i64], durations: &[f32], audio_ms: u32) -> Vec<VisemeSpan> {
    let v = vocab();
    let rev = &v.reverse;
    // durations may or may not include the $ pads; find the offset.
    let (offset, count) = if durations.len() == padded.len() {
        (0usize, padded.len())
    } else if durations.len() + 2 == padded.len() {
        (1usize, durations.len())
    } else {
        (1usize, durations.len().min(padded.len().saturating_sub(1)))
    };

    let mut spans: Vec<VisemeSpan> = Vec::new();
    let mut t_sec: f32 = 0.0;
    let mut pending_stress = false;

    let mut push = |spans: &mut Vec<VisemeSpan>, vcode: u8, t0: f32, t1: f32, stress: bool| {
        // merge with previous when identical class and adjacent
        if let Some(last) = spans.last_mut() {
            if last.v == vcode && last.stress == stress && (last.t1_ms as f32 / 1000.0 - t0).abs() < 0.002 {
                last.t1_ms = (t1 * 1000.0).round() as u32;
                return;
            }
        }
        spans.push(VisemeSpan {
            v: vcode,
            t0_ms: (t0 * 1000.0).round() as u32,
            t1_ms: (t1 * 1000.0).round() as u32,
            stress,
        });
    };

    for k in 0..count {
        let tok = padded[offset + k];
        let ch = rev.get(&(tok as usize)).copied().unwrap_or(' ');
        let d = durations.get(k).copied().unwrap_or(0.0) / DUR_FRAMES_PER_SEC;
        let t0 = t_sec;
        let t1 = t_sec + d;
        if visemes::is_length_mark(ch) {
            // extend the previous span visually
            if let Some(last) = spans.last_mut() {
                last.t1_ms = ((t1 * 1000.0).round() as u32).max(last.t1_ms);
            }
        } else if visemes::is_stress_mark(ch) {
            pending_stress = true;
            push(&mut spans, vis::REST, t0, t1, false);
        } else if let Some(vcode) = visemes::viseme_for(ch) {
            let stress = pending_stress;
            pending_stress = false;
            push(&mut spans, vcode, t0, t1, stress);
        } else {
            pending_stress = false;
            push(&mut spans, vis::REST, t0, t1, false);
        }
        t_sec = t1;
    }

    // rescale to real audio length when they drift apart
    let audio_sec = audio_ms as f32 / 1000.0;
    if t_sec > 0.15 && audio_sec > 0.15 {
        let ratio = audio_sec / t_sec;
        if (ratio - 1.0).abs() > 0.02 {
            for s in &mut spans {
                s.t0_ms = ((s.t0_ms as f32) * ratio).round() as u32;
                s.t1_ms = ((s.t1_ms as f32) * ratio).round() as u32;
            }
        }
    }
    spans
}

// Helper to keep PathBuf usage tidy above.
fn PathBuf_to_string(s: String) -> std::path::PathBuf {
    std::path::PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hundred_z_ninety_fixes() {
        assert_eq!(
            insert_space_before_hundred("əwʌnh\u{2c8}\u{28c}nd\u{279}\u{26a}d"),
            "əwʌn h\u{2c8}\u{28c}nd\u{279}\u{26a}d"
        );
        assert_eq!(join_z_before_punct("apple z."), "apple z.");
        assert_eq!(join_z_before_punct("apples z,"), "apples z,");
        assert_eq!(join_z_before_punct("apples z more"), "apples z more");
        assert_eq!(
            ninety_fix("n\u{2c8}a\u{26a}nti w\u{28a}n"),
            "n\u{2c8}a\u{26a}ndi w\u{28a}n"
        );
    }
}
