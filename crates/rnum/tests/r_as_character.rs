//! `rformat::r_as_character` against R 4.5.0 in the production image: every line of
//! `data/r_as_character.tsv` is a double's bits and R's `as.character()` of it. Regenerate with
//! `data/r_as_character_inputs.py` (column 1) and `data/r_as_character.R` (column 2).

use rnum::rformat::r_as_character;

#[test]
fn matches_r_as_character() {
    let text = include_str!("data/r_as_character.tsv");
    let mut n = 0;
    let mut bad = Vec::new();
    for line in text.lines() {
        let (bits, want) = line.split_once('\t').unwrap();
        let x = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
        let got = r_as_character(x);
        if got != want {
            bad.push(format!("{x:e}: R {want}, port {got}"));
        }
        n += 1;
    }
    assert!(n >= 5000, "only {n} values");
    assert!(
        bad.is_empty(),
        "{} of {n} differ: {:?}",
        bad.len(),
        &bad[..bad.len().min(10)]
    );
}
