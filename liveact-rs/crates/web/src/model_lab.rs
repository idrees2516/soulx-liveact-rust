//! Video Model Lab: interactive front-end for the complete SoulX-LiveAct
//! neural video pipeline (wav2vec2 -> AudioProj -> DiT + ConvKV +
//! Neighbor Forcing -> causal 3D VAE), streamed as NDJSON from
//! `/api/model/generate`.

use std::rc::Rc;

use wasm_bindgen::{closure::Closure, JsCast, UnwrapThrowExt};
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, ReadableStreamDefaultReader, Request,
    RequestInit, Response,
};

use crate::dom;

struct Lab {
    frames: Vec<(usize, Vec<u8>)>, // (first_frame, rgb)
    n_frames: usize,
    w: usize,
    h: usize,
    playing: bool,
    play_pos: usize,
    cur_chunk: usize,
    total_chunks: usize,
    steps: Vec<(f32, f32, f32)>, // (x0_rms per step) of current chunk
    parts_installed: bool,
}

type LabRef = Rc<std::cell::RefCell<Lab>>;

pub fn boot() -> Result<(), String> {
    console_error_panic_hook::set_once();
    let lab: LabRef = Rc::new(std::cell::RefCell::new(Lab {
        frames: Vec::new(),
        n_frames: 0,
        w: 32,
        h: 32,
        playing: false,
        play_pos: 0,
        cur_chunk: 0,
        total_chunks: 0,
        steps: Vec::new(),
        parts_installed: false,
    }));

    dom::on_click("ml_run", {
        let lab = lab.clone();
        move || {
            let lab2 = lab.clone();
            spawn_local(async move {
                run_generation(lab2).await;
            });
        }
    });


    {
        let lab = lab.clone();
        dom::on_click("ml_play", move || {
            let l = lab.clone();
            toggle_play(l);
        });
    }

    {
        let lab = lab.clone();
        dom::on_input("ml_frame", move || {
            let pos = dom::input("ml_frame").value_as_number() as usize;
            let l = lab.borrow();
            draw_at(&l, pos);
        });
    }

    // playback ticker
    {
        let lab = lab.clone();
        let lab2 = lab.clone();
        let cb = Closure::<dyn FnMut()>::new(move || tick(lab2.clone()));
        dom::window()
            .set_interval_with_callback_and_timeout_and_arguments_0(cb.as_ref().unchecked_ref(), 40)
            .map_err(|e| format!("{e:?}"))?;
        cb.forget();
    }

    spawn_local(load_status(lab.clone()));
    Ok(())
}

// ---------------------------------------------------------------------------
// streaming generation
// ---------------------------------------------------------------------------

async fn run_generation(lab: LabRef) {
    let text = dom::textarea("ml_text").value();
    if text.trim().is_empty() {
        dom::set_text("ml_status", "enter some text first");
        return;
    }
    let prompt = dom::input("ml_prompt").value();
    let voice = dom::select("ml_voice").value();
    let chunks: usize = dom::input("ml_chunks").value_as_number().max(0.0) as usize;

    // reset UI
    {
        let mut l = lab.borrow_mut();
        l.frames.clear();
        l.n_frames = 0;
        l.playing = false;
        l.play_pos = 0;
        l.steps.clear();
    }
    clear_log();
    if let Some(el) = dom::document().get_element_by_id("ml_filmstrip") {
        el.set_text_content(Some(""));
    }
    dom::set_text("ml_status", "starting (TTS → wav2vec2 → DiT → VAE)…");
    dom::get("ml_run")
        .dyn_into::<web_sys::HtmlButtonElement>()
        .unwrap_throw()
        .set_disabled(true);

    let body = serde_json::json!({
        "text": text,
        "prompt": if prompt.trim().is_empty() { serde_json::Value::Null } else { serde_json::Value::String(prompt) },
        "voice": voice,
        "chunks": if chunks == 0 { serde_json::Value::Null } else { serde_json::Value::from(chunks) },
    });

    let mut init = RequestInit::new();
    init.method("POST");
    {
        let h = web_sys::Headers::new().unwrap_throw();
        h.set("content-type", "application/json").unwrap_throw();
        init.headers(&h);
    }
    let body = web_sys::Request::new_with_str_and_init(
        "/api/model/generate",
        &{
            init.body(Some(&serde_json::to_string(&body).unwrap_or_default().into()));
            init.clone()
        },
    );
    let req = match body {
        Ok(r) => r,
        Err(e) => {
            dom::set_text("ml_status", &format!("request build failed: {e:?}"));
            return;
        }
    };

    let resp_val = match JsFuture::from(dom::window().fetch_with_request(&req)).await {
        Ok(v) => v,
        Err(e) => {
            dom::set_text("ml_status", &format!("fetch failed: {e:?}"));
            dom::get("ml_run")
                .dyn_into::<web_sys::HtmlButtonElement>()
                .unwrap_throw()
                .set_disabled(false);
            return;
        }
    };
    let resp: Response = resp_val.dyn_into().unwrap_throw();
    if resp.status() != 200 {
        let txt = JsFuture::from(resp.text().unwrap_throw())
            .await
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        dom::set_text("ml_status", &format!("HTTP {}: {}", resp.status(), txt));
        dom::get("ml_run")
            .dyn_into::<web_sys::HtmlButtonElement>()
            .unwrap_throw()
            .set_disabled(false);
        return;
    }
    let stream = match resp.body() {
        Some(s) => s,
        None => {
            dom::set_text("ml_status", "no response body");
            return;
        }
    };
    let reader: ReadableStreamDefaultReader = stream.get_reader().unchecked_into();

    // incremental NDJSON parse
    let mut buf = String::new();
    loop {
        let read = match JsFuture::from(reader.read()).await {
            Ok(v) => v,
            Err(e) => {
                dom::set_text("ml_status", &format!("stream error: {e:?}"));
                break;
            }
        };
        let done = js_sys::Reflect::get(&read, &"done".into())
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if done {
            break;
        }
        let value = js_sys::Reflect::get(&read, &"value".into()).unwrap_throw();
        let bytes = js_sys::Uint8Array::new(&value).to_vec();
        buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(pos) = buf.find('\n') {
            let line: String = buf.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            handle_event(lab.clone(), line);
        }
    }

    dom::get("ml_run")
        .dyn_into::<web_sys::HtmlButtonElement>()
        .unwrap_throw()
        .set_disabled(false);
    {
        let l = lab.borrow();
        if l.n_frames > 0 {
            dom::set_text("ml_status", "done — press Play");
            if let Some(slider) = dom::document().get_element_by_id("ml_frame") {
                let s = slider.dyn_into::<web_sys::HtmlInputElement>().unwrap_throw();
                s.set_max(&(l.n_frames as f64 - 1.0).to_string());
                s.set_value("0");
            }
        }
    }
}

fn handle_event(lab: LabRef, line: &str) {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return,
    };
    let ty = v["type"].as_str().unwrap_or("");
    match ty {
        "status" => {
            let w2v = v["wav2vec2_pretrained"].as_bool().unwrap_or(false);
            let params = v["params"].as_f64().unwrap_or(0.0);
            dom::set_text(
                "ml_status",
                &format!(
                    "running · {}M params · wav2vec2 {}",
                    (params / 1e6) as u64,
                    if w2v { "pretrained" } else { "fallback" }
                ),
            );
            dom::set_text("ml_model_info", &format!("DiT+VAE {:.1} M params · wav2vec2 {}", params / 1e6, if w2v { "pretrained 377 MB" } else { "deterministic" }));
        }
        "chunk_start" => {
            let mut l = lab.borrow_mut();
            l.cur_chunk = v["chunk"].as_u64().unwrap_or(0) as usize;
            l.total_chunks = v["total"].as_u64().unwrap_or(0) as usize;
            l.steps.clear();
            dom::set_text(
                "ml_status",
                &format!(
                    "chunk {}/{} · Neighbor-Forcing diffusion…",
                    l.cur_chunk + 1,
                    l.total_chunks
                ),
            );
        }
        "step" => {
            let x0 = v["x0_rms"].as_f64().unwrap_or(0.0) as f32;
            let lat = v["latent_rms"].as_f64().unwrap_or(0.0) as f32;
            let step = v["step"].as_u64().unwrap_or(0) as usize;
            let audio = v["audio_on"].as_bool().unwrap_or(false);
            let mut l = lab.borrow_mut();
            while l.steps.len() <= step {
                l.steps.push((0.0, 0.0, 0.0));
            }
            // store (x0, latent, audio_on as flag)
            l.steps[step] = (x0, lat, if audio { 1.0 } else { 0.0 });
            render_steps(&l);
        }
        "chunk" => {
            let chunk = v["chunk"].as_u64().unwrap_or(0);
            let ms = v["gen_ms"].as_f64().unwrap_or(0.0);
            let rtf = v["rtf"].as_f64().unwrap_or(0.0);
            let kv = v["kv_cache_bytes"].as_f64().unwrap_or(0.0) / 1024.0;
            let frames = v["frames_out"].as_u64().unwrap_or(0);
            let latents = v["latents"].as_u64().unwrap_or(0);
            let audio_rms = v["audio_feat_rms"].as_f64().unwrap_or(0.0);
            log_line(&format!(
                "chunk {chunk:>2}: {latents} latents → {frames} frames · {ms:.0} ms · RTF {rtf:.2} · ConvKV {kv:.0} KiB (constant) · wav2vec2 rms {audio_rms:.3}"
            ));
        }
        "frames" => {
            let chunk = v["chunk"].as_u64().unwrap_or(0) as usize;
            let first = v["first_frame"].as_u64().unwrap_or(0) as usize;
            let w = v["w"].as_u64().unwrap_or(32) as usize;
            let h = v["h"].as_u64().unwrap_or(32) as usize;
            let n = v["n_frames"].as_u64().unwrap_or(0) as usize;
            let rgb = match b64_decode(v["rgb"].as_str().unwrap_or("")) {
                Some(d) => d,
                None => return,
            };
            let frame_size = w * h * 3;
            let mut l = lab.borrow_mut();
            l.w = w;
            l.h = h;
            for i in 0..n {
                let start = i * frame_size;
                if start + frame_size > rgb.len() {
                    break;
                }
                l.frames
                    .push((first + i, rgb[start..start + frame_size].to_vec()));
            }
            l.n_frames = l.frames.last().map(|(f, _)| f + 1).unwrap_or(0);
            draw_at(&l, l.n_frames.saturating_sub(1));
            let play = dom::get("ml_play")
                .dyn_into::<web_sys::HtmlButtonElement>()
                .unwrap_throw();
            play.set_disabled(false);
            play.style().set_property("opacity", "1").ok();
            if let Some(slider) = dom::document().get_element_by_id("ml_frame") {
                let s = slider.dyn_into::<web_sys::HtmlInputElement>().unwrap_throw();
                s.set_max(&(l.n_frames as f64 - 1.0).to_string());
                s.set_value(&(l.n_frames as f64 - 1.0).to_string());
            }
            add_thumb(&l, chunk);
            dom::set_text(
                "ml_counter",
                &format!("{} frames · chunk {}", l.n_frames, chunk),
            );
        }
        "done" => {
            let ms = v["total_ms"].as_f64().unwrap_or(0.0);
            let frames = v["frames"].as_u64().unwrap_or(0);
            let kv = v["kv_cache_bytes"].as_f64().unwrap_or(0.0) / 1024.0;
            log_line(&format!(
                "done: {frames} frames in {ms:.0} ms · ConvKV memory constant at {kv:.0} KiB"
            ));
            dom::set_text("ml_status", "generation complete — press Play");
        }
        "error" => {
            dom::set_text("ml_status", &format!("error: {}", v["message"].as_str().unwrap_or("?")));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// status / parts
// ---------------------------------------------------------------------------

async fn load_status(_lab: LabRef) {
    let req = Request::new_with_str("/api/model/status").unwrap_throw();
    let resp_val = match JsFuture::from(dom::window().fetch_with_request(&req)).await {
        Ok(v) => v,
        Err(_) => return,
    };
    let resp: Response = resp_val.dyn_into().unwrap_throw();
    let txt = match JsFuture::from(resp.text().unwrap_throw()).await {
        Ok(v) => v.as_string().unwrap_or_default(),
        Err(_) => return,
    };
    let v: serde_json::Value = match serde_json::from_str(&txt) {
        Ok(v) => v,
        Err(_) => return,
    };
    // voices dropdown reuse (same ids as the live page if present)
    let initialized = v["initialized"].as_bool().unwrap_or(false);
    let params = v["params"].as_f64().unwrap_or(0.0);
    let kv = v["kv_cache_bytes"].as_f64().unwrap_or(0.0) / 1024.0;
    dom::set_text(
        "ml_model_info",
        &format!(
            "video model: {} · ConvKV window {kv:.0} KiB{}",
            if params > 0.0 {
                format!("{:.1} M params", params / 1e6)
            } else {
                "compact demo config".into()
            },
            if initialized { "" } else { " (loads on first run)" }
        ),
    );
    let list = dom::document().get_element_by_id("ml_parts");
    if let Some(list) = list {
        list.set_text_content(Some(""));
        if let Some(parts) = v["parts"].as_array() {
            for p in parts {
                let name = p["part"].as_str().unwrap_or("");
                let rust = p["rust"].as_str().unwrap_or("");
                let orig = p["original"].as_str().unwrap_or("");
                let row = dom::document()
                    .create_element("div")
                    .unwrap_throw()
                    .dyn_into::<web_sys::HtmlElement>()
                    .unwrap_throw();
                row.set_class_name("ml-part");
                let check = dom::document().create_element("span").unwrap_throw();
                check.set_class_name("ml-check");
                check.set_text_content(Some("✓"));
                let body = dom::document().create_element("div").unwrap_throw();
                body.set_class_name("ml-part-body");
                let t1 = dom::document().create_element("div").unwrap_throw();
                t1.set_text_content(Some(name));
                let t2 = dom::document().create_element("div").unwrap_throw();
                t2.set_class_name("ml-part-src");
                t2.set_text_content(Some(&format!("{rust} ⟵ {orig}")));
                body.append_child(&t1).ok();
                body.append_child(&t2).ok();
                row.append_child(&check).ok();
                row.append_child(&body).ok();
                list.append_child(&row).ok();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

fn ctx_2d(canvas: &HtmlCanvasElement) -> CanvasRenderingContext2d {
    canvas
        .get_context("2d")
        .unwrap_throw()
        .unwrap_throw()
        .dyn_into()
        .unwrap_throw()
}

fn draw_rgb(ctx: &CanvasRenderingContext2d, rgb: &[u8], w: usize, h: usize, scale: f64) {
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb[i * 3];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    let img = web_sys::ImageData::new_with_u8_clamped_array_and_sh(
        wasm_bindgen::Clamped(&rgba),
        w as u32,
        h as u32,
    )
    .unwrap_throw();
    // offscreen canvas at 1:1, then scaled blit
    let off = dom::document()
        .create_element("canvas")
        .unwrap_throw()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap_throw();
    off.set_width(w as u32);
    off.set_height(h as u32);
    let octx = ctx_2d(&off);
    octx.put_image_data(&img, 0.0, 0.0).unwrap_throw();
    ctx.set_image_smoothing_enabled(false);
    ctx.clear_rect(0.0, 0.0, 9999.0, 9999.0);
    ctx.draw_image_with_html_canvas_element_and_dw_and_dh(
        &off,
        0.0,
        0.0,
        w as f64 * scale,
        h as f64 * scale,
    )
    .unwrap_throw();
}

fn draw_at(lab: &Lab, pos: usize) {
    if lab.frames.is_empty() {
        return;
    }
    // binary search frame
    let idx = lab
        .frames
        .partition_point(|(f, _)| *f < pos)
        .saturating_sub(1);
    let (_, rgb) = &lab.frames[idx.min(lab.frames.len() - 1)];
    let canvas = dom::canvas("ml_stage");
    canvas.set_width(640);
    canvas.set_height(640);
    let ctx = ctx_2d(&canvas);
    let scale = 640.0 / lab.w.max(1) as f64;
    draw_rgb(&ctx, rgb, lab.w, lab.h, scale);
}

fn add_thumb(lab: &Lab, chunk: usize) {
    let strip = match dom::document().get_element_by_id("ml_filmstrip") {
        Some(s) => s,
        None => return,
    };
    // one thumbnail per chunk: latest frame of the chunk
    let last = lab.frames.len().saturating_sub(1);
    let (_, rgb) = &lab.frames[last];
    let t = dom::document()
        .create_element("canvas")
        .unwrap_throw()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap_throw();
    t.set_class_name("ml-thumb");
    t.set_width(96);
    t.set_height(96);
    let ctx = ctx_2d(&t);
    draw_rgb(&ctx, rgb, lab.w, lab.h, 96.0 / lab.w.max(1) as f64);
    t.set_title(&format!("chunk {chunk}"));
    strip.append_child(&t).ok();
}

fn render_steps(lab: &Lab) {
    // 3 step chips showing x0 rms (denoising progress)
    let mut html = String::new();
    for (i, (x0, lat, audio)) in lab.steps.iter().enumerate() {
        html.push_str(&format!(
            "<div class='ml-step{}'><b>step {}</b> t={:.0}<br/>x₀ {:.3}<br/>x_t {:.3}{}</div>",
            if *audio > 0.5 { " audio" } else { "" },
            i,
            [1000.0f32, 937.5, 833.33][i.min(2)],
            x0,
            lat,
            if *audio > 0.5 { " ·audio" } else { "" }
        ));
    }
    if let Some(el) = dom::document().get_element_by_id("ml_steps") {
        el.set_inner_html(&html);
    }
}

fn toggle_play(lab: LabRef) {
    let mut l = lab.borrow_mut();
    if l.frames.is_empty() {
        return;
    }
    l.playing = !l.playing;
    if l.playing && l.play_pos + 1 >= l.n_frames {
        l.play_pos = 0;
    }
    dom::set_text(
        "ml_play",
        if l.playing { "Pause" } else { "Play" },
    );
}

fn tick(lab: LabRef) {
    let mut l = lab.borrow_mut();
    if !l.playing || l.n_frames == 0 {
        return;
    }
    l.play_pos += 1;
    if l.play_pos >= l.n_frames {
        l.play_pos = 0;
    }
    let pos = l.play_pos;
    draw_at(&l, pos);
    if let Some(slider) = dom::document().get_element_by_id("ml_frame") {
        let s = slider.dyn_into::<web_sys::HtmlInputElement>().unwrap_throw();
        s.set_value(&(pos as f64).to_string());
    }
}

fn log_line(line: &str) {
    if let Some(el) = dom::document().get_element_by_id("ml_log") {
        let mut text = el.text_content().unwrap_or_default();
        text.push_str(line);
        text.push('\n');
        el.set_text_content(Some(&text));
        if let Some(p) = el.dyn_ref::<web_sys::HtmlElement>() {
            p.set_scroll_top(p.scroll_height());
        }
    }
}

fn clear_log() {
    if let Some(el) = dom::document().get_element_by_id("ml_log") {
        el.set_text_content(Some(""));
    }
}

// ---------------------------------------------------------------------------
// base64
// ---------------------------------------------------------------------------

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut n: u32 = 0;
        for (i, c) in chunk.iter().enumerate() {
            n |= (val(*c)? as u32) << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}
