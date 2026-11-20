#!/usr/bin/env bash
# Fetch Kokoro-82M ONNX (82MB q8f16, timestamped) + voice packs
set -e
cd "$(dirname "$0")/.."
mkdir -p models/voices
BASE="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped/resolve/main"
VOICES="af_sky af_bella af_nicole af_nova af_sarah af_heart am_adam am_michael am_onyx am_echo bf_emma bf_isabella bm_george ef_dora ff_siwis hf_alpha if_sara im_nicola pf_dora pm_alex"
if [ ! -f models/kokoro-q8f16.onnx ]; then
  echo "Downloading model (82MB)..."
  curl -L --fail -o models/kokoro-q8f16.onnx "$BASE/onnx/model_q8f16.onnx"
fi
for v in $VOICES; do
  if [ ! -f "models/voices/$v.bin" ]; then
    echo "voice $v"
    curl -L --fail -o "models/voices/$v.bin" "$BASE/voices/$v.bin"
  fi
done
echo "done: $(du -sh models | cut -f1)"
