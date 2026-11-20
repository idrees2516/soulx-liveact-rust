//! Minimal, direct FFI to the system `libespeak-ng` for text -> IPA
//! phonemes, following the same call pattern as the reference
//! implementations (espeak_Initialize / espeak_SetVoiceByName /
//! espeak_TextToPhonemes).
//!
//! espeak-ng keeps global state, so every call is serialized behind a
//! static mutex. Initialization happens once per process.

use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::{Mutex, OnceLock};

mod ffi {
    use std::ffi::{c_char, c_int};
    #[link(name = "espeak-ng")]
    extern "C" {
        pub fn espeak_Initialize(
            output: c_int,
            buflength: c_int,
            path: *const c_char,
            options: c_int,
        ) -> c_int;
        pub fn espeak_SetVoiceByName(name: *const c_char) -> c_int;
        pub fn espeak_TextToPhonemes(
            textptr: *mut *const c_char,
            textmode: c_int,
            phonememode: c_int,
        ) -> *mut c_char;
    }
}

const AUDIO_OUTPUT_RETRIEVAL: c_int = 1;
const CHARS_UTF8: c_int = 1;
const PHONEMES_IPA: c_int = 0x02;
const INITIALIZE_DONT_EXIT: c_int = 0x0001_0000;

static INIT: OnceLock<Result<(), String>> = OnceLock::new();
static LOCK: Mutex<()> = Mutex::new(());

fn find_data_path() -> Option<CString> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("LIVEACT_ESPEAK_DATA") {
        candidates.push(p);
    }
    candidates.extend(
        [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib",
            "/usr/share/espeak-ng",
            "/usr/local/lib",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    for c in candidates {
        let dir = std::path::Path::new(&c).join("espeak-ng-data");
        if dir.is_dir() {
            return CString::new(dir.to_string_lossy().as_bytes().to_vec()).ok();
        }
    }
    None
}

fn ensure_init() -> Result<(), String> {
    INIT.get_or_init(|| {
        let path = find_data_path();
        let ptr = path.as_ref().map(|p| p.as_ptr()).unwrap_or(std::ptr::null());
        let r = unsafe { ffi::espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, ptr, INITIALIZE_DONT_EXIT) };
        if r <= 0 {
            Err(format!("espeak_Initialize failed with code {r}"))
        } else {
            tracing::debug!(rate = r, "espeak-ng initialized");
            Ok(())
        }
    })
    .clone()
}

/// Espeak voice name for a Kokoro voice id, derived from the language
/// prefix (`af_sky` -> `en-us`, `bf_emma` -> `en-gb`, ...).
pub fn espeak_voice_for(kokoro_voice: &str) -> &'static str {
    match kokoro_voice.get(0..1) {
        Some("a") => "en-us",
        Some("b") => "en-gb",
        Some("e") => "es",
        Some("f") => "fr",
        Some("h") => "hi",
        Some("i") => "it",
        Some("p") => "pt",
        _ => "en-us",
    }
}

/// Phonemize one punctuation-free text segment into IPA (with stress
/// marks). The espeak API returns one clause per call; segments here
/// contain no clause-ending punctuation, but we still loop defensively
/// and join any extra clauses with a space.
pub fn phonemize_segment(text: &str, espeak_voice: &str) -> Result<String, String> {
    ensure_init()?;
    let _guard = LOCK.lock().unwrap();

    let name = CString::new(espeak_voice).map_err(|e| e.to_string())?;
    let rc = unsafe { ffi::espeak_SetVoiceByName(name.as_ptr()) };
    if rc != 0 {
        return Err(format!("espeak_SetVoiceByName({espeak_voice}) -> {rc}"));
    }

    let clean = text.replace(['\n', '\r'], " ");
    let ctext = CString::new(clean).map_err(|e| e.to_string())?;
    let mut ptr: *const c_char = ctext.as_ptr();
    let mut parts: Vec<String> = Vec::new();
    while !ptr.is_null() {
        // SAFETY: espeak_TextToPhonemes advances *textptr past the clause it
        // phonemized (or sets it to NULL) and returns a pointer to its own
        // buffer holding the clause phonemes. We copy out before the next call.
        let ph = unsafe {
            ffi::espeak_TextToPhonemes(
                &mut ptr as *mut *const c_char,
                CHARS_UTF8,
                PHONEMES_IPA,
            )
        };
        if ph.is_null() {
            break;
        }
        let s = unsafe { CStr::from_ptr(ph) }.to_string_lossy().into_owned();
        if !s.is_empty() {
            parts.push(s);
        }
    }
    Ok(parts.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phonemize_hello() {
        let ph = phonemize_segment("Hello world", "en-us").unwrap();
        assert!(ph.contains("l"), "unexpected phonemes: {ph}");
        assert!(!ph.contains("world"));
    }
}
