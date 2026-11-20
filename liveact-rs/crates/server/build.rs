// Link against the system espeak-ng library (text -> IPA phonemes).
// Some systems only ship the SONAME (libespeak-ng.so.1) without the
// dev symlink, so we materialize a symlink into OUT_DIR ourselves.
use std::path::{Path, PathBuf};

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let candidates: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so",
        "/usr/lib/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so",
    ];
    for cand in candidates {
        let p = Path::new(cand);
        if p.exists() {
            let dst = out_dir.join("libespeak-ng.so");
            let _ = std::fs::remove_file(&dst);
            #[cfg(unix)]
            let ok = std::os::unix::fs::symlink(p, &dst).is_ok();
            #[cfg(not(unix))]
            let ok = std::fs::copy(p, &dst).is_ok();
            if ok {
                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=dylib=espeak-ng");
                println!("cargo:rerun-if-changed={cand}");
                return;
            }
        }
    }
    // No system espeak-ng: the crate still builds; engine init will report
    // a clear error at runtime.
    println!("cargo:warning=libespeak-ng not found; TTS phonemization will be unavailable");
}
