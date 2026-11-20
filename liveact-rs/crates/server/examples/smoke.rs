//! Offline TTS smoke test: synthesizes a few sentences, reports timing,
//! audio stats and duration-timeline sanity, and writes a WAV file.

use liveact_server::tts::Kokoro;

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("trace")
        .try_init();
    let dir = std::path::PathBuf::from(
        std::env::var("LIVEACT_MODELS").unwrap_or_else(|_| "models".into()),
    );
    eprintln!("[smoke] loading engine from {}...", dir.display());
    let engine = match Kokoro::load(&dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("[smoke] engine loaded");
    println!("model: {}", engine.model_name());
    println!("voices: {:?}", engine.voice_ids());

    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello world! This is LiveAct, a real-time talking avatar system. It cost Mr. Smith $5.50 to run, in 1984, at 3.5 frames per second.".into());
    let voice = std::env::args().nth(2).unwrap_or_else(|| "af_sky".into());
    let speed: f32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    // phonemes first
    let t0 = std::time::Instant::now();
    let ph = engine.phonemize(&text, &voice).expect("phonemize");
    let ph_ms = t0.elapsed().as_millis();
    println!("phonemes ({ph_ms} ms): {ph}");

    let t0 = std::time::Instant::now();
    let out = engine
        .synthesize(&text, &voice, speed)
        .unwrap_or_else(|e| panic!("synthesize failed: {e}"));
    let synth_ms = t0.elapsed().as_millis() as u64;

    let peak = out.pcm_i16.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    let rms = (out
        .pcm_i16
        .iter()
        .map(|s| (*s as f64 / 32768.0).powi(2))
        .sum::<f64>()
        / out.pcm_i16.len().max(1) as f64)
        .sqrt();
    let rtf = synth_ms as f64 / out.audio_ms as f64;
    println!("audio: {} ms, synth {} ms, RTF {rtf:.3}", out.audio_ms, synth_ms);
    println!("peak: {peak}, rms: {rms:.4}, samples: {}", out.pcm_i16.len());
    println!("viseme spans: {}", out.visemes.len());
    let last = out.visemes.last().map(|v| v.t1_ms).unwrap_or(0);
    println!("viseme timeline end: {last} ms (audio {} ms)", out.audio_ms);

    // write wav
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: liveact_proto::SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create("smoke.wav", spec).expect("wav create");
    for s in &out.pcm_i16 {
        w.write_sample(*s).unwrap();
    }
    w.finalize().unwrap();
    println!("wrote smoke.wav");
}
