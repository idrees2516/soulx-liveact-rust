//! Offline Neighbor-Forcing demo: runs the complete SoulX-LiveAct video
//! model pipeline (wav2vec2 -> AudioProj -> DiT + ConvKV + NF -> causal VAE)
//! on synthetic audio and prints per-chunk metrics, writing decoded frames
//! as PPM files.
//!
//! ```text
//! cargo run -p liveact-model --release --example nf_demo [-- frames out_dir]
//! ```

use liveact_model::config::LiveActConfig;
use liveact_model::pipeline::{GenerateParams, Pipeline, PipelineEvent, PipelineOptions};
use liveact_model::tensor::Rng;

fn main() {
    let frames: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(25 * 4); // 4 seconds @ 25 fps
    let out_dir = std::env::args().nth(2).unwrap_or_else(|| "nf_frames".into());
    let chunks = 1 + (frames - 1) / 32;

    println!("LiveAct-RS :: SoulX-LiveAct video model (all parts, pure Rust)");
    let cfg = if std::env::var("TINY").is_ok() {
        LiveActConfig::tiny()
    } else {
        LiveActConfig::demo()
    };
    println!(
        "config: {}x{} | dim {} | layers {} | heads {}",
        cfg.width, cfg.height, cfg.dim, cfg.num_layers, cfg.num_heads
    );
    let t0 = std::time::Instant::now();
    let mut pipeline = Pipeline::new(PipelineOptions {
        cfg,
        ..Default::default()
    });
    println!(
        "init: {:.0} ms | DiT+VAE params: {:.1} M | wav2vec2: {}",
        t0.elapsed().as_millis(),
        pipeline.params as f64 / 1e6,
        if pipeline.wav2vec2_pretrained() {
            "pretrained TencentGameMate/chinese-wav2vec2-base"
        } else {
            "deterministic fallback"
        }
    );

    // synthetic "speech": 1 kHz-ish bursts with envelope, 16 kHz mono
    let mut rng = Rng::new(1234);
    let sr = 16000u32;
    let samples = frames * (sr as usize) / 25;
    let mut audio = Vec::with_capacity(samples);
    let mut phase = 0.0f32;
    for i in 0..samples {
        let t = i as f32 / sr as f32;
        let voiced = ((t * 3.5).sin() + 1.0) * 0.5; // speech-like envelope
        phase += 110.0 + 60.0 * (t * 2.0).sin();
        let f = phase * std::f32::consts::TAU / sr as f32;
        let noise = rng.normal() * 0.1;
        audio.push(((f.sin() * 0.6 + noise) * voiced * 0.5).clamp(-1.0, 1.0));
    }

    // procedural reference image (a simple face on a gradient)
    let (h, w) = (32usize, 32usize);
    let mut img = vec![0u8; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let (dy, dx) = ((y as f32 - 15.0) / 15.0, (x as f32 - 15.5) / 13.0);
            let d = (dy * dy + dx * dx).sqrt();
            let face = (1.0 - d).clamp(0.0, 1.0);
            let face = if d < 0.95 { face } else { 0.0 };
            let eye = if (x == 10 || x == 21) && (12..=14).contains(&y) { 1.0 } else { 0.0 };
            let mouth = if y >= 21 && y <= 22 && x >= 12 && x <= 20 { 1.0 } else { 0.0 };
            let r = 0.55 * face + 0.1 - mouth * 0.2;
            let g = 0.42 * face + 0.12 - mouth * 0.2;
            let b = 0.38 * face + 0.22 - eye * 0.15;
            img[0 * h * w + y * w + x] = (r * 255.0) as u8;
            img[1 * h * w + y * w + x] = (g * 255.0) as u8;
            img[2 * h * w + y * w + x] = (b * 255.0) as u8;
        }
    }

    std::fs::create_dir_all(&out_dir).ok();
    let mut frame_files = 0usize;

    println!(
        "\nchunks: {chunks} (audio {:.1}s) — window 14 latents, 3 NF steps, per-step ConvKV",
        samples as f32 / sr as f32
    );
    println!("{:-<96}", "");

    pipeline.generate(
        GenerateParams {
            prompt: "a person gives a short greeting".into(),
            edit_prompts: vec![(
                chunks.saturating_sub(1),
                chunks,
                "the person waves goodbye".into(),
            )],
            audio,
            audio_hz: sr,
            cond_image: img,
            chunks,
        },
        &mut |ev| match ev {
            PipelineEvent::Step(s) => {
                println!(
                    "  step {} t={:>7.1} x0_rms={:.4} latent_rms={:.4} audio={}",
                    s.step,
                    s.t,
                    s.x0_rms,
                    s.latent_rms,
                    if s.audio_on { "on" } else { "off" }
                );
            }
            PipelineEvent::Chunk(c) => {
                println!(
                    "chunk {:>2}: {:>2} latents -> {:>2} frames | {:>7.1} ms | RTF {:.2} | KV {:.1} KiB | audio_feat_rms {:.3}",
                    c.chunk,
                    c.latents,
                    c.frames_out,
                    c.gen_ms,
                    c.rtf,
                    c.kv_cache_bytes as f64 / 1024.0,
                    c.audio_feat_rms
                );
            }
            PipelineEvent::Frames(f) => {
                // write PPM filmstrip (every 4th frame)
                let n_frames = f.frames.len() / (f.w * f.h * 3);
                let mut idx = 0usize;
                for ti in (0..n_frames).step_by(4) {
                    let path = format!("{out_dir}/chunk{:02}_{:03}.ppm", f.chunk, idx);
                    if write_ppm(&path, &f.frames[ti * f.w * f.h * 3..], f.w, f.h).is_ok() {
                        frame_files += 1;
                    }
                    idx += 1;
                }
            }
            PipelineEvent::Done(s) => {
                println!("{:-<96}", "");
                println!(
                    "done: {} chunks, {} frames in {:.0} ms | avg RTF {:.2} | KV constant at {:.1} KiB | params {:.1} M",
                    s.chunks,
                    s.frames,
                    s.total_ms,
                    s.avg_rtf,
                    s.kv_cache_bytes as f64 / 1024.0,
                    s.params as f64 / 1e6
                );
            }
            _ => {}
        },
    );
    println!("wrote {frame_files} PPM frames to {out_dir}/");
}

fn write_ppm(path: &str, rgb: &[u8], w: usize, h: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    Ok(())
}
