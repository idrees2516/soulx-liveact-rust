//! Text normalization before phonemization, ported from the reference
//! kokoro implementations (quotes, CJK punctuation, titles, numbers:
//! times / years / money / decimals / ranges, possessives, initials).
//! Implemented with token scanning instead of regex lookarounds.

fn replace_all_few(s: &str, pairs: &[(&str, &str)]) -> String {
    let mut out = s.to_string();
    for (a, b) in pairs {
        if out.contains(a) {
            out = out.replace(a, b);
        }
    }
    out
}

/// Expand a 4-digit year (optionally with a trailing 's') the way the
/// reference does: 1984 -> "19 84", 1900s -> "19 hundred s", 2007 -> "20 07".
fn expand_year(tok: &str) -> String {
    let (digits, suffix) = match tok.strip_suffix('s') {
        Some(d) => (d, "s"),
        None => (tok, ""),
    };
    if digits.len() != 4 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return tok.to_string();
    }
    let a: u32 = digits.parse().unwrap_or(0);
    if a < 1100 || a % 1000 < 10 {
        return tok.to_string();
    }
    let (yy, rr) = (&digits[0..2], &digits[2..4]);
    let r: u32 = rr.parse().unwrap_or(0);
    let a_mod = a % 1000;
    if (100..=999).contains(&a_mod) {
        if r == 0 {
            return format!("{yy} hundred{suffix}");
        }
        if r < 10 {
            return format!("{yy} oh {r}{suffix}");
        }
    }
    format!("{yy} {rr}{suffix}")
}

/// Expand a h:mm time token: 5:30 -> "5 30", 5:00 -> "5 o'clock", 5:05 -> "5 oh 5".
fn expand_time(tok: &str) -> Option<String> {
    let (h, m) = tok.split_once(':')?;
    if h.is_empty() || m.len() != 2 || !h.chars().all(|c| c.is_ascii_digit()) || !m.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let hh: u32 = h.parse().ok()?;
    if !(1..=12).contains(&hh) {
        return None;
    }
    let mm: u32 = m.parse().ok()?;
    if mm == 0 {
        Some(format!("{h} o'clock"))
    } else if mm < 10 {
        Some(format!("{h} oh {mm}"))
    } else {
        Some(format!("{h} {mm}"))
    }
}

/// Expand money: $5 -> "5 dollars", $5.50 -> "5 dollars and 50 cents".
fn expand_money(tok: &str) -> Option<String> {
    let (cur, rest) = tok.split_at(1);
    let (unit, frac_unit, frac_name) = match cur {
        "$" => ("dollar", "dollars", "cent"),
        "\u{a3}" => ("pound", "pounds", "pence"),
        _ => return None,
    };
    if !rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return None;
    }
    match rest.split_once('.') {
        None => {
            let plural = if rest == "1" { unit } else { frac_unit };
            Some(format!("{rest} {plural}"))
        }
        Some((int_part, frac)) => {
            if frac.is_empty() || frac.len() > 2 || !frac.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let cents = if frac.len() == 1 {
                format!("{frac}0")
            } else {
                frac.to_string()
            };
            let cents: u32 = cents.parse().unwrap_or(0);
            let cents = if cents == 1 {
                format!("1 {frac_name}")
            } else {
                format!("{cents} {frac_name}s")
            };
            let plural = if int_part == "1" { unit } else { frac_unit };
            Some(format!("{int_part} {plural} and {cents}"))
        }
    }
}

/// Expand a decimal: 3.14 -> "3 point 1 4".
fn expand_decimal(tok: &str) -> Option<String> {
    let (a, b) = tok.split_once('.')?;
    if a.is_empty() || b.is_empty() {
        return None;
    }
    if !a.chars().all(|c| c.is_ascii_digit()) || !b.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut spaced = String::with_capacity(b.len() * 2);
    for (i, c) in b.chars().enumerate() {
        if i > 0 {
            spaced.push(' ');
        }
        spaced.push(c);
    }
    Some(format!("{a} point {spaced}"))
}

const ABBREV_TITLES: [&str; 6] = ["mr.", "mrs.", "ms.", "dr.", "prof.", "st."];

fn normalize_token(tok: &str, next: Option<&str>) -> String {
    // Titles & fixed words
    let lower = tok.to_lowercase();
    if ABBREV_TITLES.contains(&lower.as_str()) {
        let expanded = match lower.as_str() {
            "mr." => "Mister",
            "mrs." => "Missus",
            "ms." => "Miss",
            "dr." => "Doctor",
            "prof." => "Professor",
            _ => "Saint",
        };
        return expanded.to_string();
    }
    if lower == "etc." {
        // Don't expand when a capitalized word follows (likely a sentence start).
        if !next.is_some_and(|n| n.chars().next().is_some_and(|c| c.is_uppercase())) {
            return "etc".to_string();
        }
        return tok.to_string();
    }
    if lower == "yeah" || lower == "yea" {
        return "ye'a".to_string();
    }

    // Ranges of digits: 3-5 -> 3 to 5
    if let Some((a, b)) = tok.split_once('-') {
        if !a.is_empty()
            && !b.is_empty()
            && a.chars().all(|c| c.is_ascii_digit())
            && b.chars().all(|c| c.is_ascii_digit())
        {
            return format!("{a} to {b}");
        }
    }

    // Times: 5:30
    if tok.contains(':') && tok.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        if let Some(t) = expand_time(tok) {
            return t;
        }
    }

    // Money: $5, £3.50
    if tok.starts_with('$') || tok.starts_with('\u{a3}') {
        if let Some(m) = expand_money(tok) {
            return m;
        }
    }

    // Thousands separators: 1,000,000 -> 1000000
    if tok.chars().all(|c| c.is_ascii_digit() || c == ',') && tok.contains(',') {
        return tok.replace(',', "");
    }

    // Years & decimals (pure digit forms)
    if tok.chars().all(|c| c.is_ascii_digit() || c == 's') {
        let stripped = tok.strip_suffix('s').unwrap_or(tok);
        if stripped.len() == 4 {
            return expand_year(tok);
        }
    }
    if tok.contains('.') {
        if let Some(d) = expand_decimal(tok) {
            return d;
        }
    }

    // Initials and acronyms: U.S.A -> U-S-A, J.R.R -> J-R-R
    let mut is_initials = tok.len() > 3;
    if is_initials {
        let chars: Vec<char> = tok.chars().collect();
        // pattern: letter '.' letter '.' ... optionally trailing letter
        let mut expect_letter = true;
        for &c in &chars {
            if expect_letter {
                if !c.is_ascii_alphabetic() {
                    is_initials = false;
                    break;
                }
            } else if c != '.' {
                is_initials = false;
                break;
            }
            expect_letter = !expect_letter;
        }
    }
    if is_initials {
        return tok.replace('.', "-");
    }

    // Possessives: Bob's -> Bob'S, X'S -> X's (reference quirk)
    if let Some(stem) = tok.strip_suffix("'s") {
        if stem.chars().last().is_some_and(|c| c.is_ascii_uppercase() && !"AEIOU".contains(c)) {
            return format!("{stem}'S");
        }
    }
    if let Some(stem) = tok.strip_suffix("'S") {
        if stem == "X'" || stem.ends_with("X'") {
            return format!("{stem}'s");
        }
    }

    tok.to_string()
}

pub fn normalize(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    // Quotes and CJK punctuation
    let s = replace_all_few(
        text,
        &[
            ("\u{2018}", "'"),
            ("\u{2019}", "'"),
            ("\u{ab}", "\u{201c}"),
            ("\u{bb}", "\u{201d}"),
            ("\u{201c}", "\""),
            ("\u{201d}", "\""),
            ("(", "\u{ab}"),
            (")", "\u{bb}"),
            ("\u{3001}", ", "),
            ("\u{3002}", ". "),
            ("\u{ff01}", "! "),
            ("\u{ff0c}", ", "),
            ("\u{ff1a}", ": "),
            ("\u{ff1b}", "; "),
            ("\u{ff1f}", "? "),
        ],
    );

    // Tokenize on whitespace and normalize each token with lookahead.
    let raw: Vec<&str> = s.split_whitespace().collect();
    let out: Vec<String> = raw
        .iter()
        .enumerate()
        .map(|(i, t)| normalize_token(t, raw.get(i + 1).copied()))
        .collect();
    out.join(" ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        assert_eq!(normalize("Mr. Smith said 3.5 kg"), "Mister Smith said 3 point 5 kg");
        assert_eq!(normalize("in 1984,"), "in 19 84,");
        assert_eq!(normalize("$5.50 only"), "5 dollars and 50 cents only");
        assert_eq!(normalize("hello  world"), "hello world");
        assert_eq!(normalize("U.S.A. is"), "U-S-A- is");
        assert_eq!(normalize("it's Bob's"), "it's Bob'S");
    }
}
