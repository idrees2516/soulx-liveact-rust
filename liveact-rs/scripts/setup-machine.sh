#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu InstaBox machine:
# system deps + Rust + models + release build.
set -e
cd "$(dirname "$0")/.."

echo "== system deps =="
if command -v sudo >/dev/null 2>&1; then SUDO="sudo -n"; else SUDO=""; fi
$SUDO apt-get update -qq || true
$SUDO apt-get install -y -qq espeak-ng libespeak-ng1 curl build-essential pkg-config ca-certificates || \
  apt-get install -y espeak-ng libespeak-ng1 curl build-essential pkg-config ca-certificates

echo "== rust =="
if ! command -v cargo >/dev/null 2>&1; then
  if [ ! -f /tmp/rustup-init ]; then
    curl -fsSL -o /tmp/rustup-init https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
    chmod +x /tmp/rustup-init
  fi
  /tmp/rustup-init -y --default-toolchain stable --profile minimal
  . "$HOME/.cargo/env"
fi
cargo --version
rustc --version

echo "== models =="
./scripts/fetch_models.sh

echo "== build (release) =="
# wasm pkg is shipped prebuilt in assets/dist/pkg
touch crates/server/src/http.rs crates/server/src/main.rs
cargo build --release -p liveact-server

echo "== done =="
ls -la target/release/liveact-server
