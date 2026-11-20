//! Procedural avatar renderer: a stylized digital human bust drawn on
//! canvas 2D, driven by blended viseme mouth parameters, expressions
//! and gestures. All animation state is O(1) (bounded timers), which
//! keeps hour-scale sessions cheap.

use crate::expression::{Expr, Gesture};
use crate::viseme::Mouth;
use wasm_bindgen::UnwrapThrowExt;
use web_sys::{CanvasRenderingContext2d, HtmlImageElement};

pub struct Persona {
    pub female: bool,
    pub skin: &'static str,
    pub skin_shade: &'static str,
    pub hair: &'static str,
    pub hair_hi: &'static str,
    pub jacket: &'static str,
    pub jacket_hi: &'static str,
    pub shirt: &'static str,
    pub lip: &'static str,
    pub backdrop_a: &'static str,
    pub backdrop_b: &'static str,
}

pub fn persona_for(voice: &str) -> Persona {
    let female = voice.starts_with("af_")
        || voice.starts_with("bf_")
        || voice.starts_with("ef_")
        || voice.starts_with("ff_")
        || voice.starts_with("hf_")
        || voice.starts_with("if_")
        || voice.starts_with("jf_")
        || voice.starts_with("pf_")
        || voice.starts_with("zf_");
    // deterministic variety from the voice name
    let h = voice.bytes().fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    const SKINS: [(&str, &str); 4] = [
        ("#f2c9a4", "#d9a87e"),
        ("#e8b98c", "#cf9d70"),
        ("#c98d63", "#b0764f"),
        ("#f4d3b4", "#dbb18a"),
    ];
    const HAIRS: [(&str, &str); 4] = [
        ("#3b2d28", "#57433a"),
        ("#23211f", "#3a3733"),
        ("#5a3a22", "#75512f"),
        ("#1d1b22", "#33303d"),
    ];
    const JACKETS: [(&str, &str, &str); 4] = [
        ("#232b45", "#33406622", "#94b8ff"),
        ("#2a2340", "#3d336022", "#c2a8ff"),
        ("#123136", "#1e4c5022", "#7fe3d2"),
        ("#33202a", "#4d2e3d22", "#ff9fb0"),
    ];
    let skin = SKINS[(h % 4) as usize];
    let hair = HAIRS[((h >> 3) % 4) as usize];
    let jacket = JACKETS[((h >> 6) % 4) as usize];
    Persona {
        female,
        skin: skin.0,
        skin_shade: skin.1,
        hair: hair.0,
        hair_hi: hair.1,
        jacket: jacket.0,
        jacket_hi: jacket.1,
        shirt: jacket.2,
        lip: "#b06a63",
        backdrop_a: "#141c30",
        backdrop_b: "#05070c",
    }
}

#[derive(Clone, Copy)]
struct ExprParams {
    brow_inner: f32, // -1 angry (down) .. 1 sad/surprised (up)
    brow_outer: f32,
    corner: f32, // -1 sad .. 1 happy
    squint: f32, // 0..1 (happy eyes)
    blush: f32,
    eye_open: f32, // 0..1 multiplier (sleepy lowers lids)
}

impl ExprParams {
    fn of(e: Expr) -> Self {
        match e {
            Expr::Neutral => ExprParams { brow_inner: 0.0, brow_outer: 0.0, corner: 0.0, squint: 0.0, blush: 0.0, eye_open: 1.0 },
            Expr::Happy => ExprParams { brow_inner: 0.25, brow_outer: 0.4, corner: 0.9, squint: 0.45, blush: 0.55, eye_open: 0.95 },
            Expr::Sad => ExprParams { brow_inner: 0.8, brow_outer: -0.5, corner: -0.8, squint: 0.1, blush: 0.0, eye_open: 0.8 },
            Expr::Angry => ExprParams { brow_inner: -0.9, brow_outer: -0.3, corner: -0.45, squint: 0.35, blush: 0.15, eye_open: 1.0 },
            Expr::Surprised => ExprParams { brow_inner: 0.9, brow_outer: 0.9, corner: 0.15, squint: 0.0, blush: 0.0, eye_open: 1.15 },
            Expr::Sleepy => ExprParams { brow_inner: 0.1, brow_outer: -0.2, corner: -0.1, squint: 0.55, blush: 0.0, eye_open: 0.45 },
            Expr::Wink => ExprParams { brow_inner: 0.3, brow_outer: 0.35, corner: 0.6, squint: 0.2, blush: 0.2, eye_open: 1.0 },
        }
    }
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        ExprParams {
            brow_inner: a.brow_inner + (b.brow_inner - a.brow_inner) * t,
            brow_outer: a.brow_outer + (b.brow_outer - a.brow_outer) * t,
            corner: a.corner + (b.corner - a.corner) * t,
            squint: a.squint + (b.squint - a.squint) * t,
            blush: a.blush + (b.blush - a.blush) * t,
            eye_open: a.eye_open + (b.eye_open - a.eye_open) * t,
        }
    }
}

pub struct Avatar {
    pub persona: Persona,
    mouth: Mouth,
    blink_countdown: f64,
    blink_phase: f64, // 0 = open, runs 0..1 over the blink
    wink_hold: f64,
    saccade_target: (f32, f32),
    saccade: (f32, f32),
    saccade_countdown: f64,
    breath: f64,
    expr: ExprParams,
    brow_pulse: f32,
    gesture: Gesture,
    gesture_t: f64,
}

impl Avatar {
    pub fn new(voice: &str) -> Self {
        Avatar {
            persona: persona_for(voice),
            mouth: Mouth::rest(),
            blink_countdown: 2.2,
            blink_phase: 1.0, // finished
            wink_hold: 0.0,
            saccade_target: (0.0, 0.0),
            saccade: (0.0, 0.0),
            saccade_countdown: 1.4,
            breath: 0.0,
            expr: ExprParams::of(Expr::Neutral),
            brow_pulse: 0.0,
            gesture: Gesture::None,
            gesture_t: 0.0,
        }
    }

    pub fn set_voice(&mut self, voice: &str) {
        self.persona = persona_for(voice);
    }

    /// dt in seconds; wall-clock driven.
    pub fn update(
        &mut self,
        dt: f64,
        mouth_target: Mouth,
        stressed: bool,
        expr: Expr,
        gesture: Gesture,
    ) {
        // mouth smoothing: fast attack, slower release
        let attack = if mouth_target.open > self.mouth.open { 28.0 } else { 62.0 };
        let k = 1.0 - (-dt * attack).exp();
        self.mouth = Mouth::lerp(self.mouth, mouth_target, k as f32);

        // blink
        self.blink_countdown -= dt;
        if self.blink_countdown <= 0.0 {
            self.blink_countdown = 1.8 + (self.blink_countdown.rem_euclid(1.0) + 0.13).abs() * 3.4;
            self.blink_phase = 0.0;
        }
        if self.blink_phase < 1.0 {
            self.blink_phase = (self.blink_phase + dt / 0.16).min(1.0);
        }

        // wink one-shot
        self.wink_hold = (self.wink_hold - dt).max(0.0);
        if matches!(expr, Expr::Wink) && self.wink_hold <= 0.0 && self.blink_phase >= 1.0 {
            self.wink_hold = 0.85;
        }

        // saccades
        self.saccade_countdown -= dt;
        if self.saccade_countdown <= 0.0 {
            self.saccade_countdown = 0.9 + ((self.saccade_countdown * 7.3).rem_euclid(1.0)).abs() * 2.6;
            let mut x = (self.saccade_target.0 * 3.1 + 0.7).sin();
            let y = (self.saccade_target.1 * 5.3 + 2.1).sin() * 0.5;
            // pseudo-random within [-1, 1]
            x = (x * 91.7).rem_euclid(2.0) - 1.0;
            self.saccade_target = (x as f32 * 0.9, y as f32);
        }
        let sk = 1.0 - (-dt * 10.0).exp();
        self.saccade.0 += (self.saccade_target.0 - self.saccade.0) * sk as f32;
        self.saccade.1 += (self.saccade_target.1 - self.saccade.1) * sk as f32;

        self.breath += dt;

        // expression blend
        let target = ExprParams::of(expr);
        let e = 1.0 - (-dt * 5.0).exp();
        let cur = self.expr;
        self.expr = ExprParams::lerp(cur, target, e as f32);

        // stressed brow pulse
        if stressed {
            self.brow_pulse = 1.0;
        }
        self.brow_pulse = (self.brow_pulse - dt as f32 * 4.0).max(0.0);

        // gesture machine
        if gesture != self.gesture {
            self.gesture = gesture;
            self.gesture_t = 0.0;
        }
        if !matches!(self.gesture, Gesture::None) {
            self.gesture_t += dt;
        }
    }

    fn head_offsets(&self, t_ms: f64) -> (f32, f32) {
        // idle sway + breathing bob + speaking bob
        let t = self.breath;
        let sway = ((t * 0.53).sin() * 0.6 + (t * 0.21).sin() * 0.4) as f32;
        let bob = ((t * 1.9).sin() * 0.5 + (t * 0.8).sin() * 0.5) as f32;
        let speak = self.mouth.open * 2.2;
        let rot = sway * 0.035 + speak * (t_ms as f32 / 90.0).sin() * 0.006;
        let nod = match self.gesture {
            Gesture::Nod => {
                let p = self.gesture_t as f32;
                (p * 7.0).sin() * (1.0 - (p / 1.2).min(1.0)).max(0.0) * 0.10
            }
            Gesture::Shake => {
                let p = self.gesture_t as f32;
                (p * 9.0).sin() * (1.0 - (p / 1.1).min(1.0)).max(0.0) * 0.12
            }
            _ => 0.0,
        };
        (rot + nod, bob * 2.0 + speak)
    }

    pub fn render(
        &self,
        ctx: &CanvasRenderingContext2d,
        w: f64,
        h: f64,
        t_ms: f64,
        backdrop: Option<&HtmlImageElement>,
        speaking: bool,
    ) {
        ctx.set_global_alpha(1.0);
        self.draw_backdrop(ctx, w, h, backdrop);

        let cx = w * 0.5;
        let cy = h * 0.44;
        let s = (w / 720.0).min(h / 720.0); // scale relative to design size

        let (head_rot, head_dy) = self.head_offsets(t_ms);
        let breath = (self.breath as f32 * 0.9).sin() * 0.012 + 1.0;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.scale(s, s);

        // ---- torso ----
        self.draw_torso(ctx, breath);

        // ---- gesture hands (over torso) ----
        self.draw_gesture_hands(ctx);

        // ---- head ----
        ctx.save();
        ctx.translate(0.0, head_dy as f64);
        ctx.rotate(head_rot as f64 * 0.5);
        self.draw_back_hair(ctx);
        self.draw_head(ctx);
        ctx.restore();

        ctx.restore();
        let _ = speaking;
    }

    fn draw_backdrop(&self, ctx: &CanvasRenderingContext2d, w: f64, h: f64, img: Option<&HtmlImageElement>) {
        if let Some(img) = img {
            if img.natural_width() > 0 {
                // cover
                let iw = img.natural_width() as f64;
                let ih = img.natural_height() as f64;
                let scale = (w / iw).max(h / ih);
                let dw = iw * scale;
                let dh = ih * scale;
                ctx.set_fill_style_str("#000");
                ctx.fill_rect(0.0, 0.0, w, h);
                ctx.set_global_alpha(0.5);
                ctx.draw_image_with_html_image_element_and_dw_and_dh(
                    img, (w - dw) / 2.0, (h - dh) / 2.0, dw, dh,
                )
                .unwrap_throw();
                ctx.set_global_alpha(1.0);
                // dark veil so the character pops
                ctx.set_fill_style_str("rgba(5,7,12,0.55)");
                ctx.fill_rect(0.0, 0.0, w, h);
            } else {
                self.gradient_backdrop(ctx, w, h);
            }
        } else {
            self.gradient_backdrop(ctx, w, h);
        }
    }

    fn gradient_backdrop(&self, ctx: &CanvasRenderingContext2d, w: f64, h: f64) {
        let g = ctx
            .create_radial_gradient(w * 0.5, h * 0.32, 40.0, w * 0.5, h * 0.5, w * 0.85)
            .unwrap_throw();
        g.add_color_stop(0.0, self.persona.backdrop_a).unwrap_throw();
        g.add_color_stop(1.0, self.persona.backdrop_b).unwrap_throw();
        ctx.set_fill_style_str("#05070c");
        ctx.fill_rect(0.0, 0.0, w, h);
        ctx.set_fill_style_canvas_gradient(&g);
        ctx.fill_rect(0.0, 0.0, w, h);
        // floor glow
        ctx.set_global_alpha(0.5);
        let g2 = ctx
            .create_radial_gradient(w * 0.5, h * 1.06, 20.0, w * 0.5, h * 1.06, w * 0.6)
            .unwrap_throw();
        g2.add_color_stop(0.0, "rgba(96,120,255,0.28)").unwrap_throw();
        g2.add_color_stop(1.0, "rgba(96,120,255,0)").unwrap_throw();
        ctx.set_fill_style_canvas_gradient(&g2);
        ctx.fill_rect(0.0, h * 0.6, w, h * 0.4);
        ctx.set_global_alpha(1.0);
    }

    fn draw_torso(&self, ctx: &CanvasRenderingContext2d, breath: f32) {
        let y0: f64 = 118.0; // shoulders top
        let by = y0 * (breath as f64 * 0.15 + 0.85);
        // neck
        ctx.set_fill_style_str(self.persona.skin_shade);
        rounded_rect(ctx, -34.0, by - 46.0, 68.0, 90.0, 24.0);
        ctx.fill();
        // neck shadow
        ctx.set_global_alpha(0.35);
        ctx.set_fill_style_str("#00000055");
        rounded_rect(ctx, -34.0, by - 46.0, 68.0, 26.0, 18.0);
        ctx.fill();
        ctx.set_global_alpha(1.0);

        // shoulders / jacket
        ctx.begin_path();
        ctx.move_to(-320.0, 340.0);
        ctx.bezier_curve_to(-300.0, by + 36.0, -190.0, by - 12.0, -90.0, by - 26.0);
        ctx.quadratic_curve_to(-44.0, by - 34.0, 0.0, by - 34.0);
        ctx.quadratic_curve_to(44.0, by - 34.0, 90.0, by - 26.0);
        ctx.bezier_curve_to(190.0, by - 12.0, 300.0, by + 36.0, 320.0, 340.0);
        ctx.close_path();
        ctx.set_fill_style_str(self.persona.jacket);
        ctx.fill();

        // shirt v
        ctx.begin_path();
        ctx.move_to(-42.0, by - 28.0);
        ctx.line_to(0.0, by + 66.0);
        ctx.line_to(42.0, by - 28.0);
        ctx.quadratic_curve_to(0.0, by - 40.0, -42.0, by - 28.0);
        ctx.close_path();
        ctx.set_fill_style_str(self.persona.shirt);
        ctx.set_global_alpha(0.9);
        ctx.fill();
        ctx.set_global_alpha(1.0);

        // lapels
        ctx.set_fill_style_str(self.persona.jacket_hi);
        ctx.set_global_alpha(0.9);
        for dir in [-1.0f64, 1.0] {
            ctx.begin_path();
            ctx.move_to(dir * 40.0, by - 30.0);
            ctx.line_to(dir * 96.0, by - 8.0);
            ctx.line_to(dir * 52.0, by + 60.0);
            ctx.line_to(dir * 14.0, by + 18.0);
            ctx.close_path();
            ctx.fill();
        }
        ctx.set_global_alpha(1.0);
    }

    fn draw_gesture_hands(&self, ctx: &CanvasRenderingContext2d) {
        let p = &self.persona;
        // simple sleeve + hand shapes rising from bottom right
        let hand = |x: f64, y: f64, wave: f64| {
            ctx.set_fill_style_str(p.jacket);
            rounded_rect(ctx, x - 26.0, y + 18.0, 52.0, 130.0, 24.0);
            ctx.fill();
            ctx.set_fill_style_str(p.skin);
            ctx.begin_path();
            ctx.ellipse(x, y, 24.0, 20.0 + wave * 2.0, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();
        };
        match self.gesture {
            Gesture::Wave => {
                let t = self.gesture_t;
                let rise = smooth01(t / 0.5).min(1.0) * smooth01((3.2 - t) / 0.6).min(1.0);
                if rise > 0.01 {
                    let base_x = 236.0;
                    let base_y = 330.0 - rise * 128.0;
                    let wob = if t < 2.6 { (t * 10.0).sin() * 16.0 } else { 0.0 };
                    hand(base_x + wob, base_y, wob.abs() as f64);
                }
            }
            Gesture::Point => {
                let t = self.gesture_t;
                let rise = smooth01(t / 0.4).min(1.0) * smooth01((2.6 - t) / 0.5).min(1.0);
                if rise > 0.01 {
                    hand(190.0, 330.0 - rise * 90.0, 0.0);
                }
            }
            Gesture::Think => {
                let t = self.gesture_t;
                let rise = smooth01(t / 0.6).min(1.0) * smooth01((3.4 - t) / 0.7).min(1.0);
                if rise > 0.01 {
                    let x = -108.0 + rise * 52.0;
                    let y = 330.0 - rise * 96.0;
                    hand(x, y, 0.0);
                }
            }
            _ => {}
        }
    }

    fn draw_back_hair(&self, ctx: &CanvasRenderingContext2d) {
        let p = &self.persona;
        ctx.set_fill_style_str(p.hair);
        if p.female {
            // long hair behind head and shoulders
            ctx.begin_path();
            ctx.ellipse(0.0, -8.0, 168.0, 196.0, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();
            ctx.begin_path();
            ctx.move_to(-150.0, 40.0);
            ctx.quadratic_curve_to(-176.0, 210.0, -120.0, 320.0);
            ctx.line_to(-58.0, 320.0);
            ctx.quadratic_curve_to(-104.0, 190.0, -96.0, 60.0);
            ctx.close_path();
            ctx.fill();
            ctx.begin_path();
            ctx.move_to(150.0, 40.0);
            ctx.quadratic_curve_to(176.0, 210.0, 120.0, 320.0);
            ctx.line_to(58.0, 320.0);
            ctx.quadratic_curve_to(104.0, 190.0, 96.0, 60.0);
            ctx.close_path();
            ctx.fill();
        } else {
            ctx.begin_path();
            ctx.ellipse(0.0, -12.0, 152.0, 172.0, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();
        }
    }

    fn draw_head(&self, ctx: &CanvasRenderingContext2d) {
        let p = &self.persona;

        // ears
        ctx.set_fill_style_str(p.skin);
        for dir in [-1.0f64, 1.0] {
            ctx.begin_path();
            ctx.ellipse(dir * 132.0, 26.0, 18.0, 30.0, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();
        }

        // face
        ctx.set_fill_style_str(p.skin);
        ctx.begin_path();
        ctx.move_to(-136.0, -50.0);
        ctx.bezier_curve_to(-140.0, -158.0, -76.0, -196.0, 0.0, -196.0);
        ctx.bezier_curve_to(76.0, -196.0, 140.0, -158.0, 136.0, -50.0);
        ctx.bezier_curve_to(132.0, 26.0, 78.0, 118.0, 0.0, 124.0);
        ctx.bezier_curve_to(-78.0, 118.0, -132.0, 26.0, -136.0, -50.0);
        ctx.close_path();
        ctx.fill();
        // jaw shading
        ctx.set_global_alpha(0.16);
        ctx.set_fill_style_str(p.skin_shade);
        ctx.begin_path();
        ctx.ellipse(0.0, 96.0, 78.0, 42.0, 0.0, 0.0, std::f64::consts::TAU)
            .unwrap_throw();
        ctx.fill();
        ctx.set_global_alpha(1.0);

        // front hair
        ctx.set_fill_style_str(p.hair_hi);
        if p.female {
            ctx.begin_path();
            ctx.move_to(-140.0, -60.0);
            ctx.bezier_curve_to(-148.0, -170.0, -80.0, -206.0, 0.0, -206.0);
            ctx.bezier_curve_to(80.0, -206.0, 148.0, -170.0, 140.0, -60.0);
            ctx.bezier_curve_to(120.0, -120.0, 96.0, -140.0, 62.0, -128.0);
            ctx.bezier_curve_to(20.0, -140.0, -96.0, -150.0, -140.0, -60.0);
            ctx.close_path();
            ctx.fill();
            // side fringe
            ctx.set_fill_style_str(p.hair);
            ctx.begin_path();
            ctx.move_to(-138.0, -70.0);
            ctx.quadratic_curve_to(-120.0, 30.0, -84.0, 58.0);
            ctx.quadratic_curve_to(-118.0, -10.0, -104.0, -110.0);
            ctx.close_path();
            ctx.fill();
        } else {
            ctx.begin_path();
            ctx.move_to(-138.0, -62.0);
            ctx.bezier_curve_to(-146.0, -168.0, -78.0, -202.0, 0.0, -202.0);
            ctx.bezier_curve_to(78.0, -202.0, 146.0, -168.0, 138.0, -62.0);
            ctx.bezier_curve_to(110.0, -128.0, 40.0, -148.0, -30.0, -128.0);
            ctx.bezier_curve_to(-70.0, -118.0, -110.0, -100.0, -138.0, -62.0);
            ctx.close_path();
            ctx.fill();
        }

        // brows
        let bi = self.expr.brow_inner;
        let bo = self.expr.brow_outer;
        let pulse = self.brow_pulse * 6.0;
        let brow = |dir: f64, inner_y: f64, outer_y: f64| {
            ctx.set_fill_style_str(p.hair);
            ctx.begin_path();
            ctx.move_to(dir * 38.0, inner_y);
            ctx.quadratic_curve_to(dir * 70.0, inner_y - 6.0, dir * 98.0, outer_y);
            ctx.quadratic_curve_to(dir * 68.0, inner_y + 7.0, dir * 38.0, inner_y + 5.0);
            ctx.close_path();
            ctx.fill();
        };
        let base = -66.0 - bi as f64 * 14.0 - pulse as f64;
        brow(-1.0, base, base + bo as f64 * -8.0 + 4.0);
        brow(1.0, base, base + bo as f64 * -8.0 + 4.0);

        // eyes
        let blink = if self.blink_phase < 1.0 {
            let x = self.blink_phase;
            // fast close, slower open
            if x < 0.4 { x / 0.4 } else { 1.0 - (x - 0.4) / 0.6 }
        } else {
            0.0
        };
        let wink_right = self.wink_hold > 0.0;
        for (dir, is_right) in [(-1.0, false), (1.0, true)] {
            let ex = dir * 62.0;
            let ey = -24.0;
            let open = (1.0 - blink as f32) * self.expr.eye_open
                * if is_right && wink_right { 0.12 } else { 1.0 };
            let open = open.clamp(0.06, 1.15);
            // white
            ctx.set_fill_style_str("#f7f5f2");
            ctx.begin_path();
            ctx.ellipse(ex, ey, 34.0, 22.0 * open as f64, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();
            // iris + pupil (follow saccade)
            let ix = ex + self.saccade.0 as f64 * 9.0;
            let iy = ey + self.saccade.1 as f64 * 6.0;
            ctx.set_fill_style_str("#4a6076");
            ctx.begin_path();
            ctx.ellipse(ix, iy, 13.0, 13.0 * open as f64, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();
            ctx.set_fill_style_str("#10151c");
            ctx.begin_path();
            ctx.ellipse(ix, iy, 6.4, 6.4 * open as f64, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();
            // glint
            if open > 0.35 {
                ctx.set_fill_style_str("#ffffff");
                ctx.begin_path();
                ctx.ellipse(ix + 4.0, iy - 4.0 * open as f64, 2.4, 2.4, 0.0, 0.0, std::f64::consts::TAU)
                    .unwrap_throw();
                ctx.fill();
            }
            // upper lid (happy squint)
            if self.expr.squint > 0.05 {
                ctx.set_fill_style_str(p.skin);
                ctx.begin_path();
                ctx.ellipse(ex, ey - 22.0 * open as f64, 36.0, 10.0 * self.expr.squint as f64, 0.0, 0.0, std::f64::consts::TAU)
                    .unwrap_throw();
                ctx.fill();
            }
        }

        // nose
        ctx.set_global_alpha(0.5);
        ctx.set_fill_style_str(p.skin_shade);
        ctx.begin_path();
        ctx.move_to(-4.0, 8.0);
        ctx.quadratic_curve_to(2.0, 30.0, 16.0, 38.0);
        ctx.quadratic_curve_to(4.0, 44.0, -6.0, 38.0);
        ctx.close_path();
        ctx.fill();
        ctx.set_global_alpha(1.0);

        // mouth (viseme-driven)
        self.draw_mouth(ctx);

        // blush
        if self.expr.blush > 0.03 {
            ctx.set_global_alpha(self.expr.blush as f64 * 0.35);
            ctx.set_fill_style_str("#e2708a");
            for dir in [-1.0f64, 1.0] {
                ctx.begin_path();
                ctx.ellipse(dir * 88.0, 44.0, 26.0, 13.0, 0.0, 0.0, std::f64::consts::TAU)
                    .unwrap_throw();
                ctx.fill();
            }
            ctx.set_global_alpha(1.0);
        }
    }

    fn draw_mouth(&self, ctx: &CanvasRenderingContext2d) {
        let p = &self.persona;
        let m = &self.mouth;
        let cy = 82.0;

        let width = 42.0 + m.width as f64 * 20.0 - m.round as f64 * 16.0;
        let open = m.open as f64;
        let jaw = open * 34.0;
        let corner = self.expr.corner as f64 * 7.0;

        // outer lips
        ctx.set_fill_style_str(p.lip);
        ctx.begin_path();
        ctx.move_to(-width, cy + corner * 0.4);
        // upper lip
        ctx.quadratic_curve_to(-width * 0.4, cy - 11.0 - open * 4.0, 0.0, cy - 9.0 - open * 3.0);
        ctx.quadratic_curve_to(width * 0.4, cy - 11.0 - open * 4.0, width, cy + corner * 0.4);
        // lower lip
        ctx.quadratic_curve_to(width * 0.55, cy + jaw * 0.8 + 14.0, 0.0, cy + jaw * 0.85 + 15.0);
        ctx.quadratic_curve_to(-width * 0.55, cy + jaw * 0.8 + 14.0, -width, cy + corner * 0.4);
        ctx.close_path();
        ctx.fill();

        // inner cavity
        if open > 0.06 {
            let iw = width * 0.8;
            let ih = (jaw * 0.62).max(3.0);
            ctx.set_fill_style_str("#3a1418");
            ctx.begin_path();
            ctx.ellipse(0.0, cy + jaw * 0.28, iw, ih, 0.0, 0.0, std::f64::consts::TAU)
                .unwrap_throw();
            ctx.fill();

            // upper teeth
            if m.teeth > 0.08 {
                ctx.set_global_alpha(m.teeth as f64);
                ctx.set_fill_style_str("#f8f6f3");
                rounded_rect(ctx, -iw * 0.82, cy + jaw * 0.28 - ih, iw * 1.64, ih * 0.5, 3.0);
                ctx.fill();
                ctx.set_global_alpha(1.0);
            }
            // tongue
            if m.tongue > 0.08 {
                ctx.set_global_alpha(m.tongue as f64 * 0.9);
                ctx.set_fill_style_str("#c76b70");
                ctx.begin_path();
                ctx.ellipse(
                    0.0,
                    cy + jaw * 0.28 + ih * 0.55,
                    iw * 0.6,
                    (ih * 0.45).max(2.0),
                    0.0,
                    0.0,
                    std::f64::consts::TAU,
                )
                .unwrap_throw();
                ctx.fill();
                ctx.set_global_alpha(1.0);
            }
        } else if m.press > 0.4 {
            // pressed lips line
            ctx.set_fill_style_str("#7d4a46");
            rounded_rect(ctx, -width * 0.9, cy - 2.0, width * 1.8, 4.0, 2.0);
            ctx.fill();
        }
    }
}

fn rounded_rect(ctx: &CanvasRenderingContext2d, x: f64, y: f64, w: f64, h: f64, r: f64) {
    let r = r.min(w / 2.0).min(h / 2.0);
    ctx.begin_path();
    ctx.move_to(x + r, y);
    ctx.line_to(x + w - r, y);
    ctx.quadratic_curve_to(x + w, y, x + w, y + r);
    ctx.line_to(x + w, y + h - r);
    ctx.quadratic_curve_to(x + w, y + h, x + w - r, y + h);
    ctx.line_to(x + r, y + h);
    ctx.quadratic_curve_to(x, y + h, x, y + h - r);
    ctx.line_to(x, y + r);
    ctx.quadratic_curve_to(x, y, x + r, y);
    ctx.close_path();
}

fn smooth01(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}
