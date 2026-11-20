//! Timed action/emotion segments: the original demo's "Edit Prompt"
//! feature, parsed into an expression/gesture timeline.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Expr {
    Neutral,
    Happy,
    Sad,
    Angry,
    Surprised,
    Sleepy,
    Wink,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Gesture {
    None,
    Wave,
    Nod,
    Shake,
    Point,
    Think,
}

#[derive(Clone, Copy)]
struct Item {
    from: f32,
    to: f32,
    expr: Option<Expr>,
    gesture: Option<Gesture>,
}

#[derive(Default)]
pub struct Segments {
    items: Vec<Item>,
}

fn parse_expr(s: &str) -> Option<Expr> {
    let s = s.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| s.contains(w));
    if has(&["smile", "smiling", "happy", "joy", "laugh", "cheerful", "excited"]) {
        Some(Expr::Happy)
    } else if has(&["sad", "cry", "tears", "upset", "sorry"]) {
        Some(Expr::Sad)
    } else if has(&["angry", "mad", "furious", "annoyed", "rage"]) {
        Some(Expr::Angry)
    } else if has(&["surprise", "surprised", "shock", "shocked", "gasp", "wow"]) {
        Some(Expr::Surprised)
    } else if has(&["sleep", "sleepy", "tired", "yawn", "bored"]) {
        Some(Expr::Sleepy)
    } else if has(&["wink"]) {
        Some(Expr::Wink)
    } else {
        None
    }
}

fn parse_gesture(s: &str) -> Option<Gesture> {
    let s = s.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| s.contains(w));
    if has(&["wave", "waving", "greet", "hello", "hi ", "goodbye"]) {
        Some(Gesture::Wave)
    } else if has(&["nod", "nodding", "agree", "yes"]) {
        Some(Gesture::Nod)
    } else if has(&["shake", "shaking", "deny", "disagree", "no "]) {
        Some(Gesture::Shake)
    } else if has(&["point", "pointing"]) {
        Some(Gesture::Point)
    } else if has(&["think", "thinking", "ponder", "wonder"]) {
        Some(Gesture::Think)
    } else {
        None
    }
}

impl Segments {
    pub fn set(&mut self, raw: &[(f32, f32, String)]) {
        self.items = raw
            .iter()
            .map(|(from, to, text)| Item {
                from: *from,
                to: (*to).max(*from),
                expr: parse_expr(text),
                gesture: parse_gesture(text),
            })
            .filter(|i| i.expr.is_some() || i.gesture.is_some())
            .collect();
    }

    /// Evaluate the expression and gesture at a stream time (seconds).
    /// Later segments win when overlapping.
    pub fn eval(&self, t_s: f32) -> (Expr, Gesture) {
        let mut expr = Expr::Neutral;
        let mut gesture = Gesture::None;
        for it in &self.items {
            if t_s >= it.from && t_s <= it.to {
                if let Some(e) = it.expr {
                    expr = e;
                }
                if let Some(g) = it.gesture {
                    gesture = g;
                }
            }
        }
        (expr, gesture)
    }
}
