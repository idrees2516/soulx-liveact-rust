//! Viseme timeline evaluation with coarticulation blending (the
//! "neighbor forcing" analogue: mouth shapes blend across span
//! boundaries so chunk and word transitions stay smooth).

use liveact_proto::viseme as v;
use liveact_proto::VisemeSpan;

/// Parametric mouth shape; the renderer blends these directly.
#[derive(Clone, Copy, Debug)]
pub struct Mouth {
    /// 0 closed .. 1 wide open (jaw drop)
    pub open: f32,
    /// -1 rounded (oo) .. +1 spread (ee)
    pub width: f32,
    /// lip protrusion 0..1
    pub round: f32,
    /// upper teeth visible 0..1
    pub teeth: f32,
    /// tongue tip visible 0..1 (th/l/t)
    pub tongue: f32,
    /// lips pressed together (m/b/p)
    pub press: f32,
}

impl Default for Mouth {
    fn default() -> Self {
        Self::rest()
    }
}

impl Mouth {
    pub const fn rest() -> Self {
        Mouth { open: 0.0, width: 0.0, round: 0.0, teeth: 0.0, tongue: 0.0, press: 0.0 }
    }

    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        Mouth {
            open: a.open + (b.open - a.open) * t,
            width: a.width + (b.width - a.width) * t,
            round: a.round + (b.round - a.round) * t,
            teeth: a.teeth + (b.teeth - a.teeth) * t,
            tongue: a.tongue + (b.tongue - a.tongue) * t,
            press: a.press + (b.press - a.press) * t,
        }
    }
}

fn target(code: u8) -> Mouth {
    match code {
        v::MBP => Mouth { press: 1.0, ..Mouth::rest() },
        v::FV => Mouth { open: 0.12, width: 0.15, teeth: 1.0, ..Mouth::rest() },
        v::AA => Mouth { open: 0.95, width: 0.1, ..Mouth::rest() },
        v::AH => Mouth { open: 0.5, width: 0.0, ..Mouth::rest() },
        v::E => Mouth { open: 0.45, width: 0.6, teeth: 0.3, ..Mouth::rest() },
        v::IH => Mouth { open: 0.25, width: 1.0, teeth: 0.55, ..Mouth::rest() },
        v::OH => Mouth { open: 0.55, width: -0.6, round: 0.55, ..Mouth::rest() },
        v::UW => Mouth { open: 0.25, width: -1.0, round: 1.0, ..Mouth::rest() },
        v::SS => Mouth { open: 0.12, width: 0.75, teeth: 1.0, ..Mouth::rest() },
        v::TT => Mouth { open: 0.3, width: 0.2, tongue: 1.0, teeth: 0.35, ..Mouth::rest() },
        v::LL => Mouth { open: 0.4, width: 0.3, tongue: 1.0, ..Mouth::rest() },
        v::RR => Mouth { open: 0.3, width: -0.45, round: 0.5, tongue: 0.5, ..Mouth::rest() },
        v::KK => Mouth { open: 0.42, width: 0.05, ..Mouth::rest() },
        _ => Mouth::rest(),
    }
}

/// The absolute-time viseme timeline for the whole stream.
pub struct Timeline {
    /// spans sorted by t0, absolute ms
    spans: Vec<VisemeSpan>,
}

impl Timeline {
    pub fn new() -> Self {
        Timeline { spans: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// Append a chunk's spans, offset to their absolute stream position.
    pub fn push_chunk(&mut self, spans: &[VisemeSpan], offset_ms: u32) {
        for s in spans {
            self.spans.push(VisemeSpan {
                v: s.v,
                t0_ms: s.t0_ms + offset_ms,
                t1_ms: s.t1_ms + offset_ms,
                stress: s.stress,
            });
        }
        // O(1) amortized: drop everything older than 30s behind playback
        // is done by `prune`.
    }

    pub fn prune(&mut self, before_ms: u32) {
        let cut = self.spans.iter().position(|s| s.t1_ms >= before_ms);
        if let Some(cut) = cut {
            if cut > 0 {
                self.spans.drain(..cut);
            }
        } else {
            self.spans.clear();
        }
    }

    pub fn last_end_ms(&self) -> u32 {
        self.spans.last().map(|s| s.t1_ms).unwrap_or(0)
    }

    /// Evaluate the blended mouth at stream time `t_ms`.
    pub fn eval(&self, t_ms: f64) -> (Mouth, bool) {
        if self.spans.is_empty() {
            return (Mouth::rest(), false);
        }
        let t = t_ms as f32;

        // Binary search for the active span.
        let idx = match self
            .spans
            .binary_search_by(|s| {
                if (s.t0_ms as f32) > t {
                    std::cmp::Ordering::Greater
                } else if (s.t1_ms as f32) < t {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            }) {
            Ok(i) => Some(i),
            Err(0) => None, // before first span
            Err(i) => {
                // between spans: use previous if gap is short
                let prev = &self.spans[i - 1];
                let next = self.spans.get(i);
                if let Some(n) = next {
                    let gap = n.t0_ms as f32 - t;
                    if gap < 70.0 {
                        return (Mouth::rest(), false);
                    }
                }
                let _ = prev;
                None
            }
        };

        match idx {
            Some(i) => {
                let span = &self.spans[i];
                let mut m = target(span.v);
                // coarticulation: blend into the next span over the tail
                if let Some(next) = self.spans.get(i + 1) {
                    let span_len = (span.t1_ms - span.t0_ms) as f32;
                    let blend_win = 45.0f32.min(span_len * 0.45);
                    let remaining = span.t1_ms as f32 - t;
                    if remaining < blend_win && blend_win > 0.0 {
                        let k = 1.0 - (remaining / blend_win).clamp(0.0, 1.0);
                        let k = k * k * (3.0 - 2.0 * k); // smoothstep
                        m = Mouth::lerp(m, target(next.v), k * 0.55);
                    }
                }
                // stressed syllables get a slightly stronger articulation
                if span.stress {
                    m.open = (m.open * 1.15).min(1.0);
                    m.width *= 1.1;
                }
                (m, span.stress)
            }
            None => (Mouth::rest(), false),
        }
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Energy-driven lipsync (audio-file mode): derived mouth from an RMS
// envelope, since we have no phoneme stream for uploaded audio.
// ---------------------------------------------------------------------------

pub struct EnergyTimeline {
    /// one entry per 20 ms: rms in 0..1
    pub rms: Vec<f32>,
    /// one entry per 20 ms: zero-crossing rate 0..1
    pub zcr: Vec<f32>,
}

impl EnergyTimeline {
    pub fn from_pcm(pcm: &[f32], sample_rate: u32) -> Self {
        let win = (sample_rate as usize * 20 / 1000).max(1);
        let mut rms = Vec::new();
        let mut zcr = Vec::new();
        for chunk in pcm.chunks(win) {
            if chunk.is_empty() {
                break;
            }
            let sum: f32 = chunk.iter().map(|s| s * s).sum();
            rms.push((sum / chunk.len() as f32).sqrt().min(1.0));
            let cross = chunk
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            zcr.push((cross as f32 / chunk.len() as f32).min(1.0));
        }
        // light smoothing
        let smooth = |v: &mut Vec<f32>| {
            let orig = v.clone();
            for i in 1..v.len().saturating_sub(1) {
                v[i] = (orig[i - 1] + orig[i] * 2.0 + orig[i + 1]) / 4.0;
            }
        };
        smooth(&mut rms);
        smooth(&mut zcr);
        EnergyTimeline { rms, zcr }
    }

    pub fn eval(&self, t_ms: f64) -> Mouth {
        let idx = (t_ms / 20.0) as usize;
        if idx >= self.rms.len() {
            return Mouth::rest();
        }
        let r = self.rms[idx];
        let z = self.zcr[idx];
        let open = (r * 5.2).clamp(0.0, 0.85);
        // high zero-crossing => sibilant-ish spread; low => voiced/round
        let width = if z > 0.22 { 0.7 } else if z > 0.12 { 0.2 } else { -0.3 };
        Mouth {
            open,
            width: width * (0.4 + open),
            round: (0.25 - z).max(0.0) * open,
            teeth: if z > 0.2 { 0.8 } else { 0.2 } * open.min(1.0),
            tongue: 0.0,
            press: if open < 0.05 { 0.35 } else { 0.0 },
        }
    }

    pub fn duration_ms(&self) -> f64 {
        self.rms.len() as f64 * 20.0
    }
}
