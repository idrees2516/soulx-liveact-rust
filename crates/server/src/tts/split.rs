//! Script -> chunk splitting: mirrors the original demo's streaming block
//! pipeline (sentences are the streaming units), with abbreviation,
//! decimal and URL guards so common mid-sentence periods don't split.

const ABBREVIATIONS: [&str; 49] = [
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "sgt", "col", "gen", "rep", "sen", "gov",
    "lt", "maj", "capt", "st", "mt", "etc", "co", "inc", "ltd", "dept", "vs", "p", "pg",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "sun", "mon", "tu", "tue", "tues", "wed", "th", "thu", "thur", "thurs", "fri",
];

fn is_end_punct(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '\u{2026}' | '\u{3002}' | '\u{ff01}' | '\u{ff1f}')
}

/// Lowercased last word (letters only) ending just before index `upto`.
fn last_word(chars: &[char], upto: usize) -> String {
    let mut start = upto;
    while start > 0 && chars[start - 1].is_alphanumeric() {
        start -= 1;
    }
    chars[start..upto].iter().collect::<String>().to_lowercase()
}

fn looks_like_url_or_email(chars: &[char], upto: usize) -> bool {
    let mut start = upto;
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let word: String = chars[start..upto].iter().collect();
    word.contains("://") || word.contains('@')
}

/// Split a script into sentence-ish chunks of at most `max_chars`.
pub fn split_script(script: &str, max_chars: usize) -> Vec<String> {
    let chars: Vec<char> = script.chars().collect();
    let n = chars.len();
    let mut sentences: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;

    while i < n {
        let c = chars[i];
        // Explicit newline boundaries act like sentence ends too.
        if c == '\n' && i > start {
            let s: String = chars[start..i].iter().collect();
            if !s.trim().is_empty() {
                sentences.push(s.trim().to_string());
            }
            start = i + 1;
            i += 1;
            continue;
        }
        if is_end_punct(c) {
            // Consume the whole run of end punctuation (e.g. "...")
            let mut j = i;
            while j + 1 < n && is_end_punct(chars[j + 1]) {
                j += 1;
            }
            // Guard: decimal number (3.5)
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = j + 1 < n && chars[j + 1].is_ascii_digit();
            if prev_digit && next_digit {
                i = j + 1;
                continue;
            }
            // Guard: single letter initial followed by capitalized word
            let word = last_word(&chars, i);
            let prev_is_initial = word.len() == 1 && chars[i - 1].is_alphabetic();
            let next_char_upper = chars
                .get(j + 1)
                .or_else(|| chars.get(j + 2))
                .is_some_and(|c| c.is_uppercase() || *c == '"' || *c == '\u{201c}');
            if prev_is_initial && next_char_upper {
                i = j + 1;
                continue;
            }
            // Guard: known abbreviation ("Mr." / "etc.")
            if ABBREVIATIONS.contains(&word.as_str()) {
                i = j + 1;
                continue;
            }
            // Guard: URL or email
            if looks_like_url_or_email(&chars, i) {
                i = j + 1;
                continue;
            }
            // Boundary must be followed by whitespace or EOF
            let boundary_ok = j + 1 >= n || chars[j + 1].is_whitespace();
            if boundary_ok {
                let s: String = chars[start..=j].iter().collect();
                if !s.trim().is_empty() {
                    sentences.push(s.trim().to_string());
                }
                start = j + 1;
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    if start < n {
        let s: String = chars[start..].iter().collect();
        if !s.trim().is_empty() {
            sentences.push(s.trim().to_string());
        }
    }

    // Merge very short sentences into neighbours.
    let mut merged: Vec<String> = Vec::new();
    for s in sentences {
        let short = s.chars().count() < 24;
        if short {
            if let Some(last) = merged.last_mut() {
                let candidate_len = last.chars().count() + s.chars().count() + 1;
                if candidate_len < max_chars {
                    last.push(' ');
                    last.push_str(&s);
                    continue;
                }
            }
        }
        merged.push(s);
    }

    // Enforce the max length by falling back to comma / space splits.
    let mut out: Vec<String> = Vec::new();
    for s in merged {
        if s.chars().count() <= max_chars {
            out.push(s);
            continue;
        }
        let mut rest = s;
        while rest.chars().count() > max_chars {
            let rc: Vec<char> = rest.chars().collect();
            let mut cut = rc[..max_chars]
                .iter()
                .rposition(|&c| c == ',' || c == ';' || c == ' ')
                .unwrap_or(max_chars - 1);
            // include the punctuation
            if rc[cut] == ',' || rc[cut] == ';' {
                cut += 1;
            }
            let head: String = rc[..cut].iter().collect();
            let tail: String = rc[cut..].iter().collect();
            if head.trim().is_empty() {
                break;
            }
            out.push(head.trim().to_string());
            rest = tail.trim().to_string();
        }
        if !rest.is_empty() {
            out.push(rest);
        }
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Split normalized text into (segment, punctuation-run) pairs, the way
/// the reference phonemizer does: punctuation is hidden from espeak and
/// restored afterwards. Decimal points between digits are protected.
pub fn split_punctuation(text: &str) -> Vec<(String, String)> {
    const MARKS: &[char] = &[
        ';', ':', ',', '.', '!', '?', '\u{a1}', '\u{bf}', '\u{2014}', '\u{2026}', '"', '\u{ab}',
        '\u{bb}', '\u{201c}', '\u{201d}', '(', ')', '{', '}', '[', ']',
    ];
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut seg = String::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if MARKS.contains(&c) {
            // decimal / thousands guard: digit . , digit
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = chars.get(i + 1).is_some_and(|c| c.is_ascii_digit());
            if prev_digit && next_digit && (c == '.' || c == ',') {
                seg.push(c);
                i += 1;
                continue;
            }
            // consume the whole run of marks + surrounding whitespace
            let mut j = i;
            while j < n && (MARKS.contains(&chars[j]) || chars[j] == ' ') {
                j += 1;
            }
            let run: String = chars[i..j].iter().collect();
            let run = run.trim().to_string();
            if !seg.trim().is_empty() {
                out.push((std::mem::take(&mut seg).trim().to_string(), run));
            } else if !run.is_empty() {
                // leading punctuation: attach to previous piece if any
                if let Some(last) = out.last_mut() {
                    last.1.push_str(&run);
                }
            }
            i = j;
            continue;
        }
        seg.push(c);
        i += 1;
    }
    if !seg.trim().is_empty() {
        out.push((seg.trim().to_string(), String::new()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences() {
        let chunks = split_script(
            "Hello world! Mr. Smith paid 3.5 dollars. Nice... right? A very long and winding road",
            320,
        );
        assert!(chunks.len() >= 3);
        assert!(chunks[0].ends_with("Hello world!"));
    }

    #[test]
    fn punctuation_pieces() {
        let pieces = split_punctuation("hello, world! it's 3.5 kg");
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].1, ",");
        assert_eq!(pieces[1].1, "!");
        assert_eq!(pieces[2].0, "it's 3.5 kg");
    }
}
