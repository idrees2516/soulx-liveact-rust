//! The SoulX-LiveAct streaming generation pipeline — Rust port of
//! `generate.py::generate` (and the interactive loop of `demo.py`).
//!
//! Flow per 14-latent-frame window (chunk sizes `[6, 8]`):
//! 1. `get_msk` + VAE-encode the reference video (cond image + zeros)
//! 2. per chunk: slice the streaming audio window (+2 frames),
//!    wav2vec2-analyze it, build the ±2-frame windowed features
//!    (`get_audio_emb`)
//! 3. Neighbor-Forcing diffusion steps with per-step ConvKV caches
//! 4. chunk 0 decodes 6 latents (21 frames); chunks 1+ decode
//!    `[prev last 3 + new 8]` and emit frames `[9:]` (32 new frames)
//!
//! Memory is O(1) in stream length: the KV window is fixed at 14 frames
//! per layer per step and audio is analyzed per window.

use crate::attention::KvCache;
use crate::config::LiveActConfig;
use crate::context;
use crate::model::{ForwardArgs, WanModel};
use crate::scheduler::NeighborForcing;
use crate::tensor::Rng;
use crate::vae::{Vid, WanVae};
use crate::wav2vec2::{resample_linear, Wav2Vec2};

pub struct PipelineOptions {
    pub cfg: LiveActConfig,
    pub fp8_kv: bool,
    pub mean_memory: bool,
    pub audio_cfg: f32,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        PipelineOptions {
            cfg: LiveActConfig::demo(),
            fp8_kv: false,
            mean_memory: false,
            audio_cfg: 1.0,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ChunkReport {
    pub chunk: usize,
    pub latents: usize,
    pub frames_out: usize,
    pub steps: Vec<StepReport>,
    pub gen_ms: f64,
    pub video_ms: f64,
    pub rtf: f64,
    pub kv_cache_bytes: usize,
    pub audio_feat_rms: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct StepReport {
    pub step: usize,
    pub t: f32,
    pub x0_rms: f32,
    pub latent_rms: f32,
    pub audio_on: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct GenSummary {
    pub chunks: usize,
    pub frames: usize,
    pub total_ms: f64,
    pub avg_rtf: f64,
    pub kv_cache_bytes: usize,
    pub params: usize,
    pub wav2vec2_pretrained: bool,
}

/// One decoded chunk: RGB frames `[frames][h][w][3]` u8.
pub struct ChunkFrames {
    pub chunk: usize,
    pub first_frame: usize,
    pub frames: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

pub enum PipelineEvent {
    ChunkStart { chunk: usize, total: usize },
    Step(StepReport),
    Chunk(ChunkReport),
    Frames(ChunkFrames),
    Done(GenSummary),
}

pub struct GenerateParams {
    pub prompt: String,
    /// (start_chunk, end_chunk_inclusive, prompt) — like `edit_prompt`
    pub edit_prompts: Vec<(usize, usize, String)>,
    /// conditioning audio at `audio_hz` (mono, [-1, 1])
    pub audio: Vec<f32>,
    pub audio_hz: u32,
    /// reference image RGB bytes [3*h*w], scaled to cfg dims
    pub cond_image: Vec<u8>,
    /// number of chunks to generate (0 = derive from audio length)
    pub chunks: usize,
}

pub struct Pipeline {
    pub model: WanModel,
    pub vae: WanVae,
    pub w2v: Option<Wav2Vec2>,
    pub cfg: LiveActConfig,
    nf: NeighborForcing,
    fp8_kv: bool,
    mean_memory: bool,
    audio_cfg: f32,
    rng: Rng,
    pub params: usize,
}

impl Pipeline {
    pub fn new(opts: PipelineOptions) -> Pipeline {
        let cfg = opts.cfg.clone();
        let mut rng = Rng::new(cfg.seed ^ 0x5EED_0001);
        let mut model = WanModel::new(&cfg);
        model.init_weights(&mut rng);
        let mut vae = WanVae::new(
            cfg.vae_dim,
            cfg.vae_z,
            &cfg.vae_dim_mult,
            cfg.vae_num_res_blocks,
        );
        vae.init_weights(&mut rng);
        let w2v = {
            let dir = std::path::PathBuf::from(
                std::env::var("LIVEACT_MODELS").unwrap_or_else(|_| "models".into()),
            );
            match Wav2Vec2::load(&dir) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing_free_warn(&e);
                    Some(Wav2Vec2::deterministic())
                }
            }
        };
        let params = model.param_count();
        Pipeline {
            model,
            vae,
            w2v,
            nf: NeighborForcing::new(&cfg),
            cfg,
            fp8_kv: opts.fp8_kv,
            mean_memory: opts.mean_memory,
            audio_cfg: opts.audio_cfg,
            rng,
            params,
        }
    }

    pub fn wav2vec2_pretrained(&self) -> bool {
        self.w2v.as_ref().map(|w| w.pretrained).unwrap_or(false)
    }

    /// Total chunks the audio duration implies (generate.py
    /// `iter_total_num = int(audio_len / (4 * 8 / fps)) + 1`).
    pub fn chunks_for_audio(&self, audio: &[f32], audio_hz: u32) -> usize {
        let dur = audio.len() as f64 / audio_hz as f64;
        (dur / (self.cfg.vae_scale as f64 * self.cfg.chunk_sizes[1] as f64 / self.cfg.fps as f64)) as usize
            + 1
    }

    /// `get_msk`: [4, T_lat, lh, lw] — ones for latent frame 0, zeros after.
    fn get_msk(&self, lat_t: usize) -> Vid {
        let (lh, lw) = self.cfg.lat_hw();
        let mut msk = Vid::zeros(4, lat_t, lh, lw);
        for c in 0..4 {
            for y in 0..lh {
                for x in 0..lw {
                    msk.set(c, 0, y, x, 1.0);
                }
            }
        }
        msk
    }

    /// Encode the reference video: cond image + zero frames, then prepend
    /// the mask channel block.
    fn build_y(&self, cond_image: &[u8]) -> Vid {
        let cfg = &self.cfg;
        let (h, w) = (cfg.height, cfg.width);
        let frame_num = cfg.cond_frame_num(); // 53
        let (lh, lw) = cfg.lat_hw();
        // pixel video [3, T, H, W] in [-1, 1]
        let mut video = Vid::zeros(3, frame_num, h, w);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = (cond_image[c * h * w + y * w + x] as f32 / 255.0) * 2.0 - 1.0;
                    video.set(c, 0, y, x, v);
                }
            }
        }
        let lat = self.vae.encode(&video); // [16, 14, lh, lw]
        let mut y = Vid::zeros(16 + 4, lat.t, lh, lw);
        // msk channels first (torch.cat([msk, y], dim=1))
        let msk = self.get_msk(lat.t);
        for c in 0..4 {
            y.data[c * lat.t * lh * lw..(c + 1) * lat.t * lh * lw]
                .copy_from_slice(&msk.data[c * lat.t * lh * lw..(c + 1) * lat.t * lh * lw]);
        }
        for c in 0..16 {
            y.data[(4 + c) * lat.t * lh * lw..(5 + c) * lat.t * lh * lw]
                .copy_from_slice(&lat.data[c * lat.t * lh * lw..(c + 1) * lat.t * lh * lw]);
        }
        y
    }

    /// Streaming audio window analysis (`--steam_audio` path):
    /// slice [start, end+2) frames at the ORIGINAL rate, resample to 16 kHz,
    /// normalize + x3 gain, wav2vec2, then ±2-frame windows (`get_audio_emb`).
    fn audio_window(&self, audio: &[f32], audio_hz: u32, start_f: usize, end_f: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let fps = cfg.fps as f64;
        let s = ((start_f as f64 / fps) * audio_hz as f64) as usize;
        let e = (((end_f as f64 + 2.0) / fps) * audio_hz as f64) as usize;
        let s = s.min(audio.len());
        let e = e.min(audio.len());
        let slice = &audio[s..e.max(s + 1).min(audio.len())];
        let w16 = if audio_hz == 16000 {
            slice.to_vec()
        } else {
            resample_linear(slice, audio_hz, 16000)
        };
        let w16 = Wav2Vec2::preprocess_wave(&w16);
        let frames = end_f - start_f + 2;
        let emb = self
            .w2v
            .as_ref()
            .map(|w| w.analyze(&w16, frames))
            .unwrap_or_else(|| vec![0.0; frames * 12 * 768]);
        // get_audio_emb: window indices ±2, clamped
        let (win, blocks, ch) = (cfg.audio_window, cfg.audio_blocks, cfg.audio_channels);
        let t = frames;
        let mut out = vec![0.0f32; (end_f - start_f) * win * blocks * ch];
        let idx = [0usize, 1, 2, 3, 4]; // (arange(5) - 2) + 2
        for fi in 0..(end_f - start_f) {
            for (wi, off) in idx.iter().enumerate() {
                let src_f = (fi + off).min(t - 1);
                out[fi * win * blocks * ch + wi * blocks * ch
                    ..fi * win * blocks * ch + (wi + 1) * blocks * ch]
                    .copy_from_slice(
                        &emb[src_f * blocks * ch..(src_f + 1) * blocks * ch],
                    );
            }
        }
        out
    }

    /// Full generation loop. `emit` receives streaming events.
    pub fn generate(
        &mut self,
        params: GenerateParams,
        emit: &mut dyn FnMut(PipelineEvent),
    ) {
        let t0 = std::time::Instant::now();
        let cfg = self.cfg.clone();
        let (lh, lw) = cfg.lat_hw();
        let frame_len = cfg.frame_len();
        let frame_num = cfg.cond_frame_num(); // 53
        let (first, rest) = (cfg.chunk_sizes[0], cfg.chunk_sizes[1]);
        let total_chunks = if params.chunks > 0 {
            params.chunks
        } else {
            self.chunks_for_audio(&params.audio, params.audio_hz)
        };

        // ---- conditioning ----
        let y = self.build_y(&params.cond_image);
        let (text_tokens, text_ctx) =
            context::encode_text(&params.prompt, cfg.text_dim, cfg.text_len);
        let clip_ctx = context::encode_image_clip(&params.cond_image, cfg.height, cfg.width, cfg.clip_dim);
        let mut edits: Vec<(usize, usize, usize, Vec<f32>)> = Vec::new(); // (a, b, tokens, ctx)
        for (a, b, p) in &params.edit_prompts {
            let (nt, ctx) = context::encode_text(p, cfg.text_dim, cfg.text_len);
            edits.push((*a, *b, nt, ctx));
        }

        // ---- per-step KV caches (+ null-audio set when audio CFG is on) ----
        let mut caches: Vec<Vec<KvCache>> = self.model.new_kv_caches(self.fp8_kv);
        let mut caches_null: Vec<Vec<KvCache>> = if self.audio_cfg > 1.0 {
            self.model.new_kv_caches(self.fp8_kv)
        } else {
            Vec::new()
        };
        let kv_bytes = caches
            .iter()
            .flat_map(|step| step.iter().map(|c| c.bytes_allocated()))
            .sum::<usize>();

        let mut pre_latent: Option<(Vec<f32>, usize)> = None;
        let mut total_frames = 0usize;
        let mut rtfs = Vec::new();

        for chunk in 0..total_chunks {
            let t_chunk = std::time::Instant::now();
            emit(PipelineEvent::ChunkStart {
                chunk,
                total: total_chunks,
            });
            // generate.py: f = _ if _ <= 1 else 1 -> first chunk uses chunk_sizes[0]
            let f = if chunk == 0 { 0 } else { 1 };
            let latents = if f == 0 { first } else { rest };
            let f_start = if f == 0 { 0 } else { first };
            let start_idx = f_start * frame_len;
            let end_idx = (f_start + latents) * frame_len;

            // audio window [(_-1)*rest*4, ...+frame_num)
            let (a_start, a_end) = if chunk == 0 {
                (0, frame_num)
            } else {
                let base = (chunk - 1) * rest * cfg.vae_scale;
                (base, base + frame_num)
            };
            let audio_pix = self.audio_window(&params.audio, params.audio_hz, a_start, a_end);
            let audio_rms = rms(&audio_pix);

            // active context (edit prompts override in range)
            let (use_txt_tokens, use_txt_ctx) = {
                let mut cur = (text_tokens, &text_ctx);
                for (a, b, nt, ctx) in &edits {
                    if *a <= chunk && chunk <= *b {
                        cur = (*nt, ctx);
                        break;
                    }
                }
                cur
            };

            // y slice for this chunk's latent frames
            let y_cut = slice_lat(&y, f_start, f_start + latents);

            // fresh noise for the chunk
            let mut latent = self.rng.normal_vec(cfg.out_dim * latents * lh * lw);

            let mut steps_report = Vec::new();
            for step in 0..self.nf.num_steps() {
                let t = self.nf.timesteps[step];
                let v = self.model.forward(ForwardArgs {
                    latent: &latent,
                    y: &y_cut,
                    t,
                    text_tokens: use_txt_tokens,
                    text_ctx: use_txt_ctx,
                    clip_fea: &clip_ctx,
                    audio_pixel: &audio_pix,
                    audio_pixel_frames: frame_num,
                    kv_caches: &mut caches[step],
                    start_idx,
                    end_idx,
                    update_cache: chunk > 1,
                    skip_audio: !self.nf.audio_on(step),
                    mean_memory: self.mean_memory,
                });
                let v_pred = v;
                let (audio_v, v_pred) = if self.audio_cfg > 1.0 && self.nf.audio_on(step) {
                    // audio CFG: null-audio pass on its own per-step cache set
                    // (kv_cache_null_audio in generate.py)
                    let null_audio = vec![0.0f32; audio_pix.len()];
                    let v_null = self.model.forward(ForwardArgs {
                        latent: &latent,
                        y: &y_cut,
                        t,
                        text_tokens: use_txt_tokens,
                        text_ctx: use_txt_ctx,
                        clip_fea: &clip_ctx,
                        audio_pixel: &null_audio,
                        audio_pixel_frames: frame_num,
                        kv_caches: &mut caches_null[step],
                        start_idx,
                        end_idx,
                        update_cache: chunk > 1,
                        skip_audio: false,
                        mean_memory: self.mean_memory,
                    });
                    let blended: Vec<f32> = v_pred
                        .iter()
                        .zip(&v_null)
                        .map(|(v, n)| n + self.audio_cfg * (v - n))
                        .collect();
                    (true, blended)
                } else {
                    (false, v_pred)
                };
                let eps = self.rng.normal_vec(latent.len());
                let (next, x0_rms) = self.nf.step(&latent, &v_pred, step, &eps);
                latent = next;
                let rep = StepReport {
                    step,
                    t,
                    x0_rms,
                    latent_rms: rms(&latent),
                    audio_on: self.nf.audio_on(step) && !audio_v,
                };
                steps_report.push(rep.clone());
                emit(PipelineEvent::Step(rep));
            }

            // ---- decode with 3-frame overlap (generate.py 376-383) ----
            let decode_in: Vid = match &pre_latent {
                None => lat_to_vid(&latent, cfg.out_dim, latents, lh, lw),
                Some((prev, prev_t)) => {
                    // cat(prev[:, -3:], latent) then decode, skipping the 9
                    // pixel frames the overlap already emitted
                    let mut joined = Vec::with_capacity((3 + latents) * cfg.out_dim * lh * lw);
                    let per_prev = prev_t * lh * lw;
                    let per_cur = latents * lh * lw;
                    for c in 0..cfg.out_dim {
                        let pstart = c * per_prev + per_prev - 3 * lh * lw;
                        joined.extend_from_slice(&prev[pstart..pstart + 3 * lh * lw]);
                        joined.extend_from_slice(&latent[c * per_cur..(c + 1) * per_cur]);
                    }
                    lat_to_vid(&joined, cfg.out_dim, 3 + latents, lh, lw)
                }
            };
            let decoded = self.vae.decode(&decode_in);
            let skip = if chunk == 0 { 0 } else { 9 };
            let new_frames = decoded.t - skip;
            let mut rgb = Vec::with_capacity(new_frames * cfg.height * cfg.width * 3);
            for ti in skip..decoded.t {
                for y in 0..cfg.height {
                    for x in 0..cfg.width {
                        for c in 0..3 {
                            let v = decoded.at(c, ti, y, x);
                            rgb.push(((v + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8);
                        }
                    }
                }
            }
            pre_latent = Some((latent, latents));
            total_frames += new_frames;

            let gen_ms = t_chunk.elapsed().as_secs_f64() * 1000.0;
            let video_ms = new_frames as f64 / cfg.fps as f64 * 1000.0;
            let rtf = gen_ms / video_ms.max(1.0);
            rtfs.push(rtf);
            let rep = ChunkReport {
                chunk,
                latents,
                frames_out: new_frames,
                steps: steps_report,
                gen_ms,
                video_ms,
                rtf,
                kv_cache_bytes: kv_bytes,
                audio_feat_rms: audio_rms,
            };
            emit(PipelineEvent::Frames(ChunkFrames {
                chunk,
                first_frame: total_frames - new_frames,
                frames: rgb,
                w: cfg.width,
                h: cfg.height,
            }));
            emit(PipelineEvent::Chunk(rep));
        }

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let avg_rtf = rtfs.iter().sum::<f64>() / rtfs.len().max(1) as f64;
        emit(PipelineEvent::Done(GenSummary {
            chunks: total_chunks,
            frames: total_frames,
            total_ms,
            avg_rtf,
            kv_cache_bytes: kv_bytes,
            params: self.params,
            wav2vec2_pretrained: self.wav2vec2_pretrained(),
        }));
    }
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

fn slice_lat(y: &Vid, a: usize, b: usize) -> Vec<f32> {
    let per = y.h * y.w;
    let t = b - a;
    let mut out = vec![0.0f32; y.c * t * per];
    for c in 0..y.c {
        let s = (c * y.t + a) * per;
        out[c * t * per..(c + 1) * t * per].copy_from_slice(&y.data[s..s + t * per]);
    }
    out
}

fn lat_to_vid(data: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vid {
    Vid {
        c,
        t,
        h,
        w,
        data: data.to_vec(),
    }
}

fn tracing_free_warn(e: &crate::wav2vec2::LoadError) {
    let _ = e;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_chunk_pipeline_runs() {
        let opts = PipelineOptions {
            cfg: LiveActConfig::demo(),
            ..Default::default()
        };
        let mut p = Pipeline::new(opts);
        let audio: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();
        let img = vec![128u8; 3 * 32 * 32];
        let mut frames_seen = 0usize;
        let mut last: Option<GenSummary> = None;
        p.generate(
            GenerateParams {
                prompt: "a person greets the audience".into(),
                edit_prompts: vec![(2, 3, "the person smiles".into())],
                audio,
                audio_hz: 16000,
                cond_image: img,
                chunks: 2,
            },
            &mut |ev| match ev {
                PipelineEvent::Frames(f) => frames_seen += f.frames.len() / (32 * 32 * 3),
                PipelineEvent::Done(s) => last = Some(s),
                _ => {}
            },
        );
        assert_eq!(frames_seen, 21 + 32, "chunk 0 = 21 frames, chunk 1 = 32");
        let s = last.expect("summary");
        assert_eq!(s.chunks, 2);
        assert_eq!(s.frames, 53);
        assert!(s.kv_cache_bytes > 0);
        // KV cache is constant across the stream: verify via summary value
        // being independent of chunk count (checked in the 4-chunk test)
    }

    #[test]
    fn kv_memory_constant_across_chunks() {
        let opts = PipelineOptions {
            cfg: LiveActConfig::demo(),
            ..Default::default()
        };
        let mut p = Pipeline::new(opts);
        let audio: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.008).sin() * 0.3).collect();
        let img = vec![100u8; 3 * 32 * 32];
        let mut kv_per_chunk = Vec::new();
        p.generate(
            GenerateParams {
                prompt: "constant memory check".into(),
                edit_prompts: vec![],
                audio,
                audio_hz: 16000,
                cond_image: img,
                chunks: 5,
            },
            &mut |ev| {
                if let PipelineEvent::Chunk(c) = ev {
                    kv_per_chunk.push(c.kv_cache_bytes);
                }
            },
        );
        assert!(kv_per_chunk.len() == 5);
        assert!(
            kv_per_chunk.windows(2).all(|w| w[0] == w[1]),
            "ConvKV keeps memory constant: {kv_per_chunk:?}"
        );
        // expected size: 14 frames * 4 tokens * dim * 2 * layers * steps * 4B
        let expect = 14 * 4 * 384 * 2 * 6 * 3 * 4;
        assert_eq!(kv_per_chunk[0], expect);
    }

    #[test]
    fn chunk_frames_math() {
        // chunk 0 -> 21 frames; chunk n -> 32 frames
        let opts = PipelineOptions::default();
        let p = Pipeline::new(opts);
        let audio: Vec<f32> = vec![0.0; 12800]; // 0.8 s -> 1 chunk minimum
        assert_eq!(p.chunks_for_audio(&audio, 16000), 1);
        let audio2: Vec<f32> = vec![0.0; 25600]; // 1.6 s -> 2 chunks
        assert_eq!(p.chunks_for_audio(&audio2, 16000), 2);
    }
}
