//! `glibm::{exp, ln}` and `glibm_log1p::log1p` against the reference R's glibc, bit for bit
//! (`$MD_COUNT_CORPUS_DIR/reference-glibm/glibm_ref.bin`, written by deseq2-rust
//! `corpus/count-reference/ref_glibm.R`). Skips when the corpus is absent.
use rnum::{glibm, glibm_log1p};

fn corpus_dir() -> std::path::PathBuf {
    match std::env::var("MD_COUNT_CORPUS_DIR") {
        Ok(d) => d.into(),
        Err(_) => std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("wd/md-count-golden-corpus"),
    }
}

#[test]
fn glibm_matches_reference_glibc() {
    let path = corpus_dir().join("reference-glibm/glibm_ref.bin");
    let Ok(b) = std::fs::read(&path) else {
        eprintln!("SKIP: {} not found", path.display());
        return;
    };
    let v: Vec<f64> = b
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (ne, nl, np) = (v[0] as usize, v[1] as usize, v[2] as usize);
    let mut o = 3;
    let mut take = |n: usize| {
        let s = v[o..o + n].to_vec();
        o += n;
        s
    };
    let (xe, ye, xl, yl, xp, yp) = (take(ne), take(ne), take(nl), take(nl), take(np), take(np));
    let bad = |x: &[f64], y: &[f64], f: fn(f64) -> f64| {
        x.iter()
            .zip(y)
            .filter(|(a, b)| f(**a).to_bits() != b.to_bits())
            .count()
    };
    let (be, bl, bp) = (
        bad(&xe, &ye, glibm::exp),
        bad(&xl, &yl, glibm::ln),
        bad(&xp, &yp, glibm_log1p::log1p),
    );
    println!("glibm mismatches: exp {be}/{ne}, log {bl}/{nl}, log1p {bp}/{np}");
    assert_eq!((be, bl, bp), (0, 0, 0));
}
