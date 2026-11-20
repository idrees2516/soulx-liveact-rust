//! IPA phoneme -> viseme class mapping for the avatar renderer.
//! The renderer blends neighbouring viseme shapes (coarticulation),
//! so a coarse mapping is sufficient.

use liveact_proto::viseme as v;

/// Returns the viseme code for a phoneme character, or None when the
/// character is not mouth-relevant (punctuation, stress marks...).
pub fn viseme_for(ch: char) -> Option<u8> {
    Some(match ch {
        // bilabial closure: p b m ʘ
        'p' | 'b' | 'm' | '\u{298}' => v::MBP,
        // labiodental: f v ⱱ
        'f' | 'v' | '\u{2b1}' => v::FV,
        // open vowels: ɑ æ ʌ ɐ ɒ a
        '\u{251}' | '\u{e6}' | '\u{28c}' | '\u{250}' | '\u{252}' | 'a' => v::AA,
        // central vowels: ə ɚ ɜ ɝ
        '\u{259}' | '\u{25a}' | '\u{25c}' | '\u{25d}' => v::AH,
        // mid front: e ɛ
        'e' | '\u{25b}' => v::E,
        // close front: i ɪ j ɨ
        'i' | '\u{26a}' | 'j' | '\u{268}' => v::IH,
        // rounded mid: o ɔ ø ɵ
        'o' | '\u{254}' | '\u{f8}' | '\u{275}' => v::OH,
        // close back: u ʊ w ɯ ʉ
        'u' | '\u{28a}' | 'w' | '\u{26f}' | '\u{289}' => v::UW,
        // sibilants: s z ʃ ʒ
        's' | 'z' | '\u{283}' | '\u{292}' => v::SS,
        // dental / alveolar: t d n θ ð ɾ
        't' | 'd' | 'n' | '\u{3b8}' | '\u{f0}' | '\u{27e}' => v::TT,
        // laterals: l ɫ ɭ
        'l' | '\u{26b}' | '\u{26d}' => v::LL,
        // rhotics: ɹ ɻ
        '\u{279}' | '\u{27b}' => v::RR,
        // velar / glottal: k ɡ g ŋ x h ʔ ɦ
        'k' | '\u{261}' | 'g' | '\u{272}' | 'x' | 'h' | '\u{294}' | '\u{266}' => v::KK,
        _ => return None,
    })
}

/// ˈ ˌ
pub fn is_stress_mark(ch: char) -> bool {
    ch == '\u{2c8}' || ch == '\u{2cc}'
}

/// ː ˑ
pub fn is_length_mark(ch: char) -> bool {
    ch == '\u{2d0}' || ch == '\u{2d1}'
}
