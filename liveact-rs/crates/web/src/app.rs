//! Application wiring: WebSocket + audio scheduling + timeline +
//! avatar + UI controls. One `App` lives behind an `Rc<RefCell<..>>`
//! shared by the event closures and the render loop.

use std::cell::RefCell;
use std::rc::Rc;

use liveact_proto::{ClientMsg, Segment, ServerMsg, VoiceInfo, VisemeSpan};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, UnwrapThrowExt};
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{AudioBuffer, AudioBufferSourceNode, BaseAudioContext, HtmlImageElement};

use crate::audio::{perf_now, AudioOut};
use crate::avatar::Avatar;
use crate::dom;
use crate::expression::Segments;
use crate::net::{Handlers, WsSession};
use crate::viseme::{EnergyTimeline, Timeline};

pub struct App {
    pub ws: Option<WsSession>,
    pub audio: AudioOut,
    timeline: Timeline,
    energy: Option<EnergyTimeline>,
    energy_buffer: Option<AudioBuffer>,
    energy_play: Option<AudioBufferSourceNode>,
    energy_started_at: Option<f64>,
    avatar: Avatar,
    segments: Segments,
    segment_rows: Vec<String>, // text input ids of live rows
    voices: Vec<VoiceInfo>,
    backdrop: Option<HtmlImageElement>,
    pending: Option<(u32, u32, Vec<VisemeSpan>)>, // (idx, gap_ms, visemes)
    voice: String,
    fps: u64,
    running: bool,
    // stats
    total_chunks: usize,
    generated: usize,
    stream_ready: bool,
    done: bool,
    rtf: Option<f32>,
    latency_ms: Option<f64>,
    ping_t: Option<u64>,
    started_at: f64,
    total_audio_ms: f64,
    last_chip: &'static str,
}

pub type AppRef = Rc<RefCell<App>>;

pub fn boot() -> Result<AppRef, String> {
    console_error_panic_hook::set_once();

    let loc = dom::window().location();
    let proto = if loc.protocol().unwrap_or_default() == "https:" {
        "wss"
    } else {
        "ws"
    };
    let host = loc.host().unwrap_or_else(|_| "localhost:8080".into());
    let url = format!("{proto}://{host}/ws");

    let app: AppRef = Rc::new(RefCell::new(App {
        ws: None,
        audio: AudioOut::new()?,
        timeline: Timeline::new(),
        energy: None,
        energy_buffer: None,
        energy_play: None,
        energy_started_at: None,
        avatar: Avatar::new("af_sky"),
        segments: Segments::default(),
        segment_rows: Vec::new(),
        voices: Vec::new(),
        backdrop: None,
        pending: None,
        voice: "af_sky".into(),
        fps: 30,
        running: false,
        total_chunks: 0,
        generated: 0,
        stream_ready: false,
        done: false,
        rtf: None,
        latency_ms: None,
        ping_t: None,
        started_at: 0.0,
        total_audio_ms: 0.0,
        last_chip: "idle",
    }));

    let on_msg = {
        let app = app.clone();
        Rc::new(RefCell::new(Box::new(move |m: ServerMsg| handle_msg(&app, m))
            as Box<dyn FnMut(ServerMsg)>))
    };
    let on_binary = {
        let app = app.clone();
        Rc::new(RefCell::new(Box::new(move |b: Vec<u8>| handle_binary(&app, b))
            as Box<dyn FnMut(Vec<u8>)>))
    };
    let on_state = {
        let app = app.clone();
        Rc::new(RefCell::new(Box::new(move |s: &'static str| {
            let mut a = app.borrow_mut();
            if s == "connected" {
                a.state_connected();
            }
        }) as Box<dyn FnMut(&'static str)>))
    };

    let ws = WsSession::connect(url, Handlers { on_msg, on_binary, on_state })?;
    app.borrow_mut().ws = Some(ws);
    Ok(app)
}

impl App {
    fn state_connected(&mut self) {
        dom::set_text("connection_state", "online");
        dom::set_text("status-bar", "Connected — ready");
    }
}

fn handle_msg(app: &AppRef, m: ServerMsg) {
    let mut a = app.borrow_mut();
    match m {
        ServerMsg::Hello {
            voices, model, ..
        } => {
            a.voices = voices.clone();
            let sel = dom::select("voice");
            sel.set_length(0);
            let mut chosen = None;
            for vinfo in &voices {
                let opt = dom::document().create_element("option").unwrap_throw();
                opt.set_text_content(Some(&format!(
                    "{} · {} ({})",
                    vinfo.name, vinfo.id, vinfo.language
                )));
                opt.set_attribute("value", &vinfo.id).unwrap_throw();
                sel.append_child(&opt).unwrap_throw();
                if vinfo.id == "af_sky" {
                    chosen = Some(vinfo.id.clone());
                }
            }
            let pick = chosen
                .or_else(|| voices.first().map(|v| v.id.clone()))
                .unwrap_or_else(|| "af_sky".into());
            sel.set_value(&pick);
            a.voice = pick.clone();
            a.avatar.set_voice(&pick);
            update_voice_lang(&a);
            dom::set_text("model_name", &model);
        }
        ServerMsg::Started { total_chunks } => {
            a.total_chunks = total_chunks;
            a.generated = 0;
            a.stream_ready = false;
            a.done = false;
            a.running = true;
            a.total_audio_ms = 0.0;
            a.rtf = None;
            a.started_at = now_ms();
            a.timeline.clear();
            a.audio.reset();
            a.energy_play = None;
            a.energy_started_at = None;
            dom::set_text("task_state", "Running");
            dom::set_text("task_done", "No");
            dom::set_text("stream_ready", "No");
            dom::set_text("task_message", "Synthesizing first chunk…");
        }
        ServerMsg::ChunkStart {
            idx,
            visemes,
            gap_ms,
            ..
        } => {
            a.pending = Some((idx as u32, gap_ms, visemes));
            if !a.stream_ready {
                a.stream_ready = true;
                dom::set_text("stream_ready", "Yes");
                dom::set_text("task_message", "Streaming");
            }
        }
        ServerMsg::ChunkEnd { .. } => {
            a.pending = None;
        }
        ServerMsg::Progress {
            state,
            generated,
            total,
            message,
        } => {
            a.generated = generated;
            a.total_chunks = total;
            dom::set_text("task_state", &capitalize(&state));
            dom::set_text("task_message", &message);
            if state == "stopped" {
                a.running = false;
            }
        }
        ServerMsg::Done { audio_ms, rtf, .. } => {
            a.rtf = Some(rtf);
            a.total_audio_ms = audio_ms as f64;
            a.done = true;
            a.running = false;
            dom::set_text("task_done", "Yes");
        }
        ServerMsg::Error { message } => {
            dom::set_text("task_message", &format!("Error: {message}"));
            dom::set_text("status-bar", &message);
        }
        ServerMsg::Pong { t } => {
            if a.ping_t == Some(t) {
                a.ping_t = None;
                a.latency_ms = Some(now_ms() - t as f64);
            }
        }
    }
}

fn handle_binary(app: &AppRef, bytes: Vec<u8>) {
    let mut a = app.borrow_mut();
    if bytes.len() < 8 {
        return;
    }
    let idx = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if bytes.len() < 8 + n * 2 {
        return;
    }
    let mut pcm = Vec::with_capacity(n);
    for i in 0..n {
        let o = 8 + i * 2;
        pcm.push(i16::from_le_bytes([bytes[o], bytes[o + 1]]));
    }

    if let Some((pending_idx, gap_ms, visemes)) = a.pending.take() {
        if pending_idx == idx {
            let start_ms = a.audio.schedule_chunk(&pcm, gap_ms);
            a.timeline.push_chunk(&visemes, start_ms);
            let dur = pcm.len() as f64 * 1000.0 / liveact_proto::SAMPLE_RATE as f64;
            a.total_audio_ms = (start_ms as f64 + dur).max(a.total_audio_ms);
        } else {
            a.pending = Some((pending_idx, gap_ms, visemes));
        }
    }
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

pub fn wire_ui(app: &AppRef) {
    // voice select
    {
        let app = app.clone();
        dom::on_input("voice", move || {
            let sel = dom::select("voice");
            let v = sel.value();
            let mut a = app.borrow_mut();
            a.voice = v.clone();
            a.avatar.set_voice(&v);
            update_voice_lang(&a);
        });
    }
    // speed label
    dom::on_input("speed", move || {
        let v = dom::input("speed").value();
        if let Ok(f) = v.parse::<f32>() {
            dom::set_text("speed_val", &format!("{f:.2}x"));
        }
    });
    // fps
    {
        let app = app.clone();
        dom::on_input("fps", move || {
            let v = dom::input("fps").value();
            if let Ok(f) = v.parse::<u64>() {
                app.borrow_mut().fps = f;
                dom::set_text("fps_val", &v);
            }
        });
    }
    // segments
    {
        let app = app.clone();
        dom::on_click("btn_add_segment", move || {
            let i = app.borrow().segment_rows.len();
            let doc = dom::document();
            let row = doc.create_element("div").unwrap_throw();
            row.set_class_name("prompt-item");
            let from_id = format!("seg_from_{i}");
            let to_id = format!("seg_to_{i}");
            let text_id = format!("seg_text_{i}");
            let from_default = if i == 0 { 0.0 } else { (i as f32) * 2.0 };
            row.set_inner_html(&format!(
                r#"<input type="number" id="{from_id}" value="{from_default}" min="0" step="0.5">
                <span class="time-label">to</span>
                <input type="number" id="{to_id}" value="{}" min="0" step="0.5">
                <span class="time-label">s :</span>
                <input type="text" class="p-content" id="{text_id}" placeholder="smiling and waving">
                <span class="remove-btn" title="remove">×</span>"#,
                from_default + 2.0
            ));
            // remove handler: hides the row and drops it from state
            if let Some(btn) = row.query_selector(".remove-btn").unwrap_throw() {
                let app2 = app.clone();
                let row_id = text_id.clone();
                let btn2 = btn.clone();
                let c = Closure::<dyn FnMut()>::new(move || {
                    let mut a = app2.borrow_mut();
                    a.segment_rows.retain(|id| id != &row_id);
                    drop(a);
                    hide_parent_row(&btn2);
                });
                btn.add_event_listener_with_callback("click", c.as_ref().unchecked_ref())
                    .unwrap_throw();
                c.forget();
            }
            dom::get("prompt-list").append_child(&row).unwrap_throw();
            app.borrow_mut().segment_rows.push(text_id);
        });
    }
    // image upload
    {
        let app = app.clone();
        dom::on_input("img_i", move || {
            let Some(file) = dom::input("img_i").files().and_then(|f| f.item(0)) else {
                return;
            };
            let fr = web_sys::FileReader::new().unwrap_throw();
            let fr2 = fr.clone();
            let app2 = app.clone();
            let c = Closure::<dyn FnMut(web_sys::Event)>::new(move |_: web_sys::Event| {
                if let Ok(r) = fr2.result() {
                    if let Some(url) = r.as_string() {
                        let img = HtmlImageElement::new().unwrap_throw();
                        img.set_src(&url);
                        app2.borrow_mut().backdrop = Some(img);
                        let img_el = dom::get("img_p")
                            .dyn_into::<HtmlImageElement>()
                            .unwrap_throw();
                        img_el.set_src(&url);
                        img_el.set_attribute("style", "display:block").unwrap_throw();
                        dom::get("img_t")
                            .set_attribute("style", "display:none")
                            .unwrap_throw();
                    }
                }
            });
            fr.set_onload(Some(c.as_ref().unchecked_ref()));
            c.forget();
            fr.read_as_data_url(&file).unwrap_throw();
        });
    }
    // audio upload (audio-driven mode)
    {
        let app = app.clone();
        dom::on_input("audio_i", move || {
            let Some(file) = dom::input("audio_i").files().and_then(|f| f.item(0)) else {
                return;
            };
            let name = file.name();
            let fr = web_sys::FileReader::new().unwrap_throw();
            let fr2 = fr.clone();
            let app2 = app.clone();
            let c = Closure::<dyn FnMut(web_sys::Event)>::new(move |_: web_sys::Event| {
                if let Ok(r) = fr2.result() {
                    if let Ok(buf) = r.clone().dyn_into::<js_sys::ArrayBuffer>() {
                        let app3 = app2.clone();
                        let name = name.clone();
                        spawn_local(async move {
                            let (ctx, decode) = {
                                let a = app3.borrow();
                                (a.audio.ctx().clone(), a.audio.ctx().decode_audio_data(&buf))
                            };
                            let decoded = JsFuture::from(decode.unwrap_throw())
                                .await
                                .ok()
                                .and_then(|v| v.dyn_into::<AudioBuffer>().ok());
                            let Some(buffer) = decoded else {
                                dom::set_text("status-bar", "Could not decode audio file");
                                return;
                            };
                            let pcm = buffer.get_channel_data(0).unwrap_throw();
                            let energy =
                                EnergyTimeline::from_pcm(&pcm, buffer.sample_rate() as u32);
                            let mut a = app3.borrow_mut();
                            a.energy = Some(energy);
                            a.energy_buffer = Some(buffer);
                            dom::set_text("audio_t", &format!("Loaded: {name}"));
                            dom::set_text(
                                "status-bar",
                                "Audio-driven mode armed — press Generate & Stream",
                            );
                            dom::button("btn_clear_audio").set_disabled(false);
                        });
                    }
                }
            });
            fr.set_onload(Some(c.as_ref().unchecked_ref()));
            c.forget();
            fr.read_as_array_buffer(&file).unwrap_throw();
        });
    }
    // clear audio
    {
        let app = app.clone();
        dom::on_click("btn_clear_audio", move || {
            let mut a = app.borrow_mut();
            a.energy = None;
            a.energy_buffer = None;
            dom::set_text("audio_t", "Drive from Audio File");
            dom::button("btn_clear_audio").set_disabled(true);
            dom::input("audio_i").set_value("");
        });
    }
    // start
    {
        let app = app.clone();
        dom::on_click("btn_start", move || {
            let mut a = app.borrow_mut();
            a.audio.wake();
            if a.energy.is_some() {
                start_audio_driven(&mut a);
            } else {
                start_tts(&mut a);
            }
        });
    }
    // stop
    {
        let app = app.clone();
        dom::on_click("btn_stop", move || {
            let mut a = app.borrow_mut();
            if a.energy_play.is_some() {
                stop_audio_driven(&mut a);
            } else {
                if let Some(ws) = &a.ws {
                    ws.send(&ClientMsg::Stop {});
                }
                a.running = false;
                a.audio.reset();
                a.timeline.clear();
                a.pending = None;
                dom::set_text("task_state", "Stopped");
                dom::set_text("speaking_state", "idle");
                dom::set_text("btn_stop", "Stop");
            }
        });
    }
    // enqueue (button + Enter key)
    {
        let app = app.clone();
        let send_it = std::rc::Rc::new(move || {
            let t = dom::input("enqueue_text").value();
            if t.trim().is_empty() {
                return;
            }
            let ok = {
                let a = app.borrow();
                a.ws.as_ref().map(|w| w.send(&ClientMsg::Enqueue { text: t })).unwrap_or(false)
            };
            if ok {
                dom::input("enqueue_text").set_value("");
            }
        });
        let s1 = send_it.clone();
        dom::on_click("btn_enqueue", move || s1());
        dom::on_keydown_enter("enqueue_text", move || send_it());
    }
}

fn hide_parent_row(el: &web_sys::EventTarget) {
    if let Some(e) = el.dyn_ref::<web_sys::Element>() {
        if let Some(p) = e.parent_element() {
            let _ = p.set_attribute("style", "display:none");
        }
    }
}

fn update_voice_lang(a: &App) {
    let lang = a
        .voices
        .iter()
        .find(|v| v.id == a.voice)
        .map(|v| v.language.clone())
        .unwrap_or_default();
    dom::set_text("voice_lang", &lang);
}

fn start_tts(a: &mut App) {
    let script = dom::textarea("script").value();
    if script.trim().is_empty() {
        dom::set_text("status-bar", "Write a script first");
        return;
    }
    let voice = {
        let sel = dom::select("voice");
        if sel.value().is_empty() {
            a.voice.clone()
        } else {
            sel.value()
        }
    };
    let speed = dom::input("speed").value().parse::<f32>().unwrap_or(1.0);
    a.fps = dom::input("fps").value().parse::<u64>().unwrap_or(30);

    let mut segs: Vec<Segment> = Vec::new();
    let row_ids = a.segment_rows.clone();
    for (i, text_id) in row_ids.iter().enumerate() {
        let from = dom::input(&format!("seg_from_{i}")).value().parse::<f32>().unwrap_or(0.0);
        let to = dom::input(&format!("seg_to_{i}")).value().parse::<f32>().unwrap_or(0.0);
        let text = dom::input(text_id).value();
        if !text.trim().is_empty() {
            segs.push(Segment { from_s: from, to_s: to, text });
        }
    }
    a.segments.set(
        &segs.iter()
            .map(|s| (s.from_s, s.to_s, s.text.clone()))
            .collect::<Vec<_>>(),
    );
    a.voice = voice.clone();
    a.avatar.set_voice(&voice);

    let sent = a
        .ws
        .as_ref()
        .map(|w| w.send(&ClientMsg::Start { script, voice, speed, segments: segs }))
        .unwrap_or(false);
    if !sent {
        dom::set_text("status-bar", "Not connected — reload the page");
    }
}

fn start_audio_driven(a: &mut App) {
    let Some(buffer) = a.energy_buffer.clone() else { return };
    a.audio.reset();
    a.timeline.clear();
    let ctx = a.audio.ctx().clone();
    let src = ctx.create_buffer_source().unwrap_throw();
    src.set_buffer(Some(&buffer));
    let _ = src.connect_with_audio_node(&ctx.destination());
    let _ = src.start();
    a.energy_play = Some(src);
    a.energy_started_at = Some(perf_now());
    a.running = true;
    a.done = false;
    a.stream_ready = true;
    a.started_at = now_ms();
    dom::set_text("task_state", "Running");
    dom::set_text("stream_ready", "Yes");
    dom::set_text("task_done", "No");
    dom::set_text("task_message", "Audio-driven playback");
}

fn stop_audio_driven(a: &mut App) {
    if let Some(s) = a.energy_play.take() {
        let _ = s.stop_with_when(0.0);
    }
    a.energy_started_at = None;
    a.running = false;
    dom::set_text("task_state", "Stopped");
    dom::set_text("speaking_state", "idle");
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

pub fn start_render_loop(app: AppRef) {
    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let state = Rc::new(RefCell::new((0.0f64, 0.0f64, 0.0f64))); // last_t, last_render, last_stats
    let f2 = f.clone();
    let st = state.clone();
    let app2 = app.clone();

    *f.borrow_mut() = Some(Closure::new(move |t: f64| {
        let (dt, do_render, do_stats) = {
            let mut s = st.borrow_mut();
            let dt = if s.0 == 0.0 { 0.016 } else { (t - s.0) / 1000.0 };
            s.0 = t;
            let fps = app2.borrow().fps;
            let do_render = t - s.1 >= 1000.0 / fps as f64 - 1.0;
            if do_render {
                s.1 = t;
            }
            let do_stats = t - s.2 >= 250.0;
            if do_stats {
                s.2 = t;
            }
            (dt, do_render, do_stats)
        };
        if do_render {
            render_frame(&app2, dt);
        }
        if do_stats {
            update_stats(&app2, t);
        }
        request_next_frame(f2.clone());
    }));
    request_next_frame(f.clone());
}

fn request_next_frame(f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>) {
    let borrow = f.borrow();
    if let Some(c) = borrow.as_ref() {
        let _ = dom::window().request_animation_frame(c.as_ref().unchecked_ref());
    }
}

fn render_frame(app: &AppRef, dt: f64) {
    let mut a = app.borrow_mut();

    // stream position
    let stream_ms = if let Some(t0) = a.energy_started_at {
        let e_dur = a.energy.as_ref().map(|e| e.duration_ms()).unwrap_or(0.0);
        (perf_now() - t0).clamp(0.0, e_dur)
    } else {
        a.audio.stream_pos_ms()
    };

    // mouth
    let (mouth, stressed) = if let Some(e) = &a.energy {
        if a.energy_started_at.is_some() {
            (e.eval(stream_ms), false)
        } else {
            (crate::viseme::Mouth::rest(), false)
        }
    } else {
        a.timeline.eval(stream_ms)
    };

    let (expr, gesture) = a.segments.eval(stream_ms as f32 / 1000.0);
    a.avatar.update(dt, mouth, stressed, expr, gesture);

    // draw
    let canvas = dom::canvas("stage");
    let w = canvas.width() as f64;
    let h = canvas.height() as f64;
    let ctx = canvas
        .get_context("2d")
        .unwrap_throw()
        .and_then(|c| c.dyn_into::<web_sys::CanvasRenderingContext2d>().ok())
        .unwrap_throw();
    let bd = a.backdrop.as_ref();
    a.avatar.render(&ctx, w, h, stream_ms, bd, mouth.open > 0.04);

    a.last_chip = if a.running {
        if mouth.open > 0.05 {
            "speaking"
        } else if a.done {
            "done"
        } else {
            "waiting"
        }
    } else {
        "idle"
    };
}

fn update_stats(app: &AppRef, wall_t: f64) {
    let mut a = app.borrow_mut();

    if a.ping_t.is_none() {
        let t = wall_t as u64;
        a.ping_t = Some(t);
        if let Some(ws) = &a.ws {
            ws.send(&ClientMsg::Ping { t });
        }
    }

    let buffered = a.audio.buffered_ahead_ms();
    let pos = a.audio.stream_pos_ms();
    let rtf = a.rtf.map(|r| format!("{r:.2}")).unwrap_or_else(|| "—".into());
    let lat = a
        .latency_ms
        .map(|l| format!("{l:.0} ms"))
        .unwrap_or_else(|| "—".into());
    let sched = a.audio.scheduled_ms();
    dom::set_text(
        "perf",
        &format!(
            "RTF {rtf} · buffer {buffered:.0} ms · latency {lat} · pos {:.1}s · sched {:.1}s",
            pos / 1000.0,
            sched / 1000.0
        ),
    );
    dom::set_text(
        "chunk_progress",
        &format!("{} / {}", a.generated, a.total_chunks),
    );
    dom::set_text("speaking_state", a.last_chip);
    if a.running {
        let elapsed = (wall_t - a.started_at) / 1000.0;
        dom::set_text(
            "status-bar",
            &format!(
                "Live {elapsed:.0}s · {}/{} chunks · {:.1}s audio scheduled",
                a.generated,
                a.total_chunks,
                a.total_audio_ms / 1000.0
            ),
        );
    }

    // keep timeline memory bounded (hour-scale sessions stay O(1))
    let pos = a.audio.stream_pos_ms();
    let keep_from = (pos - 10_000.0).max(0.0) as u32;
    a.timeline.prune(keep_from);
}

fn now_ms() -> f64 {
    dom::window()
        .performance()
        .map(|p| p.now())
        .unwrap_or_else(|| js_sys::Date::now())
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}
