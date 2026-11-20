# LiveAct-RS

**A system-level Rust reimplementation of [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct): a real-time, streaming talking-avatar system — server *and* frontend in 100% Rust, using the 82 MB Kokoro-82M TTS model.**

Live deployment: **https://8080-agq7tznykj3vpll2i2zm63cmjm.instabox.tech**

```
┌──────────────────── Browser (Rust → WASM) ────────────────────────────────┐
│  liveact-web (wasm-bindgen, no JS framework)                              │
│   • Live studio: WebSocket client, Web Audio PCM scheduler (24 kHz),      │
│     viseme timeline + coarticulation, procedural avatar renderer          │
│   • Video Model Lab: NDJSON stream reader, frame player/filmstrip,        │
│     Neighbor-Forcing step telemetry, implemented-parts checklist          │
└──────────────▲──────────────────────────────┬─────────────────────────────┘
     JSON + PCM16 /ws          NDJSON events + base64 frames /api/model/*
               │                              │
┌──────────────┴──────────────── Axum server (native) ──────────────────────┐
│  liveact-server                                                            │
│   • /ws  session pipeline: sentence chunks, bounded 14 s look-ahead, O(1)  │
│   • Kokoro-82M TTS: espeak-ng → IPA → vocab → ONNX (ort, CPU)              │
│   • /api/model/* → liveact-model (below), the neural video pipeline        │
│   • Static frontend embedded in the binary (include_dir)                   │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │
┌───────────────────────────────▼ liveact-model (pure Rust) ─────────────────┐
│  wav2vec2-base (real pretrained weights, 377 MB)                           │
│   → ±2-frame windowing → AudioProjModel (5/8-window → 32 tokens/frame)     │
│   → WanModel DiT (patchify, adaLN, 3D RoPE, text+CLIP cross-attn,          │
│     per-frame audio cross-attn, ConvKV memory, fp8 KV option)              │
│   → Neighbor-Forcing scheduler (3 steps, per-step caches, x0 re-noise)     │
│   → causal 3D VAE (streaming encode/decode, 4×/8× strides, frame caches)   │
│   → RGB frames, O(1) stream memory                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Mapping to the original SoulX-LiveAct system

| SoulX-LiveAct (Python, 2×H100) | LiveAct-RS (Rust, CPU-only) |
| --- | --- |
| Flask server + HLS video stream | Axum server + WebSocket stream + NDJSON model events |
| Wan2.1 DiT 18B (`model_memory.py`) | `liveact-model::model/attention/block` — same architecture, compact runnable config (`LiveActConfig::full_14b()` fields all honored; `tiny()/demo()` run on CPU) |
| **ConvKV Memory** (14-frame rolling cache, 5:1 depthwise compression, fp8 KV) | `liveact-model::attention::KvCache` — faithful port incl. slot layout `[2 anchor \| 2 compressed \| 2 recent \| 8 current]`, `--fp8_kv_cache` e4m3, `--mean_memory` |
| **Neighbor Forcing** (3-step schedule, per-step caches, x0 + re-noise) | `liveact-model::scheduler` + `pipeline` — `t=[1000, 937.5, 833.3, 0]`, audio on steps 1–2, audio CFG |
| wav2vec2 audio features (chinese-wav2vec2-base) | `liveact-model::wav2vec2` — real pretrained weights (377 MB), conv stack, linear interp, 12-layer post-LN encoder, stacked hidden states |
| AudioProjModel + `get_audio_emb` windowing | `liveact-model::audio_proj` (5/8-window → 32 tokens/frame) + `pipeline::audio_window` |
| Causal 3D VAE (`wan/modules/vae.py`, streaming) | `liveact-model::vae` — CausalConv3d caches, Resample up/down 3D, `(L-1)*4+1` frame math, 16-ch mean/std scaling |
| WanModel text conditioning (umt5-xxl T5 + CLIP) | `liveact-model::context` — same interfaces, deterministic stand-ins (11 GB GPU-class weights noted in docs) |
| Kokoro TTS (PyTorch) | Kokoro-82M ONNX `q8f16` (**82 MB**) via `ort` + espeak-ng FFI |
| 18B diffusion rendering (GPU-only) | Live studio: procedural avatar (WASM canvas, RTF≈0.42) · Model Lab: the real neural pipeline above |
| Timed "Edit Prompt" segments | Same: per-chunk prompt overrides in the pipeline + timed segments in the studio |
| Task status: chunks / stream-ready / done | Same fields live (plus RTF, x0 RMS, KV bytes, wav2vec2 stats) |

The released 18B checkpoint needs a GPU, so the shipped demo runs the same
architecture at compact scale (`demo`: 44.6 M params, 32×32) with the real
pretrained wav2vec2 — every component is instantiated, exercised and
observable in the Video Model Lab. The live avatar studio keeps the
procedural renderer for real-time interaction (RTF ≈ 0.42).

## Performance (4 vCPU InstaBox machine)

- **RTF ≈ 0.42–0.45** (synthesizes ~2.2× faster than real time)
- First chunk (TTFA) ≈ 0.4–0.6 s; chunk streaming adds ~10–40 ms per chunk
- 20 bundled voices (en-us, en-gb, es, fr, hi, it, pt), 24 kHz mono PCM
- Server binary: ~22 MB, models: 82 MB + ~10 MB voices (+ 377 MB wav2vec2)
- Video Model Lab (4 vCPU): full pipeline ~3–6 s per 1.28 s chunk (RTF 3–5),
  ConvKV memory constant at 3024 KiB regardless of stream length

## Build & run

Requirements: Rust (stable), `espeak-ng` + `libespeak-ng1` (runtime data),
optional `wasm-pack` (only if you change the frontend — a prebuilt `pkg`
is committed under `assets/dist/pkg`).

```bash
./scripts/fetch_models.sh                          # 82 MB Kokoro + voices
python3 scripts/convert_wav2vec2.py models         # wav2vec2-base weights (377 MB)
wasm-pack build crates/web --target web --release --out-dir ../../assets/dist/pkg  # optional
cargo build --release -p liveact-server
LIVEACT_ADDR=0.0.0.0:8080 ./target/release/liveact-server
```

Environment variables:

| Var | Default | Meaning |
| --- | --- | --- |
| `LIVEACT_ADDR` | `0.0.0.0:8080` | Listen address |
| `LIVEACT_MODELS` | `models` | Model directory |
| `LIVEACT_MODEL` | `<models>/kokoro-q8f16.onnx` | Any Kokoro v1.0 ONNX (fp32 `model.onnx` gives ~3× faster RTF at 310 MB) |
| `LIVEACT_ESPEAK_DATA` | auto-detected | Directory containing `espeak-ng-data` |
| `LIVEACT_ORT_OPT` | off | Re-enable ORT graph optimization (off by default: ORT 1.23.2 segfaults on this graph during level-3 transforms) |
| `LIVEACT_MODELS` | `models` | Also holds `wav2vec2-weights.bin` + `wav2vec2-manifest.json` (from `scripts/convert_wav2vec2.py`); without them the audio encoder falls back to deterministic init |

## Wire protocol (`/ws`)

JSON control frames (`type`-tagged, see `crates/proto`), binary frames are
`[u32 chunk_idx][u32 sample_count][i16 PCM …]` right after each
`chunk_start`. Client → server: `start` (script, voice, speed, segments),
`enqueue` (live queueing), `stop`, `ping`. Server → client: `hello`,
`started`, `chunk_start` (viseme spans with absolute timings), `chunk_end`,
`progress`, `done` (RTF), `error`, `pong`.

## Test

```bash
cargo test -p liveact-server          # phonemizer, normalizer, splitter, fixes
cargo test -p liveact-model           # 39 tests: ConvKV layout, RoPE axis split,
                                      # NF step math, VAE frame math (53<->14),
                                      # fp8 KV roundtrip, wav2vec2 pretrained,
                                      # streaming pipeline + constant memory
cargo run -p liveact-model --release --example nf_demo   # full offline pipeline
```

## Deployment (InstaBox)

```bash
tar czf liveact-rs.tar.gz --exclude=liveact-rs/target --exclude=liveact-rs/models liveact-rs
curl -X PUT "https://api.instabox.sh/v1/machines/<id>/files?path=/data/home/liveact-rs.tar.gz&parents=true" \
     -H "X-API-Key: $INSTABOX_API_KEY" --data-binary @liveact-rs.tar.gz
# exec: tar xzf …, scripts/setup-machine.sh, then run the server on :8080
curl -X POST "https://api.instabox.sh/v1/machines/<id>/ports" \
     -H "X-API-Key: $INSTABOX_API_KEY" -H 'Content-Type: application/json' \
     -d '{"port": 8080, "name": "LiveAct-RS web"}'
```

## Notes & limitations

- Chinese/Japanese voices are not bundled (Kokoro zh requires pinyin-based
  phonemization; this pipeline is espeak-based, like the original repo's
  streaming kokoro path).
- Quantized models (`q8f16` 82 MB, `int8` 88 MB) carry unfused QDQ overhead
  on CPU; the fp32 graph (310 MB) reaches RTF ≈ 0.15 on the same machine.
  Switch with `LIVEACT_MODEL`.
- Credits: Kokoro-82M by hexgrad (Apache-2.0), ONNX builds by
  onnx-community, phonemization reference from misaki/kokoro-js/Kokoros.
