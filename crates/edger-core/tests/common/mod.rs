//! Shared helpers for the edgeR golden tests: locate the count corpus, list the edgeR runs and
//! read the reference CSVs and `reference.json`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

/// `MD_COUNT_CORPUS_DIR`, defaulting to `~/wd/md-count-golden-corpus`.
pub fn corpus_dir() -> PathBuf {
    match std::env::var("MD_COUNT_CORPUS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join("wd/md-count-golden-corpus")
        }
    }
}

pub fn ref_dir(run: &str) -> PathBuf {
    corpus_dir().join("reference").join(run)
}

/// Every `count_edger_*` and `edge_edger_*` run in the reference directory, sorted.
pub fn edger_runs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(corpus_dir().join("reference"))
        .expect("corpus reference dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("count_edger_") || n.starts_with("edge_edger_"))
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no edgeR runs under {:?}", corpus_dir());
    v
}

pub fn reference_json(run: &str) -> serde_json::Value {
    let s = std::fs::read_to_string(ref_dir(run).join("reference.json")).expect("reference.json");
    serde_json::from_str(&s).expect("json")
}

pub fn manifest(run: &str) -> serde_json::Value {
    let p = corpus_dir().join("runs").join(run).join("manifest.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("manifest")).expect("json")
}

pub fn scalar(run: &str, name: &str) -> f64 {
    let j = reference_json(run);
    let v = &j["scalars"][name];
    v.as_f64()
        .or_else(|| v.as_array().and_then(|a| a[0].as_f64()))
        .unwrap_or_else(|| panic!("{run}: scalar {name} missing: {v}"))
}

/// Whether the run is an expected-error run (`manifest.status == "error"`).
pub fn is_error_run(run: &str) -> bool {
    manifest(run)["status"].as_str() == Some("error")
}

/// A CSV as header + string rows.
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {name} in {:?}", self.header))
    }
    pub fn strings(&self, name: &str) -> Vec<String> {
        let j = self.col(name);
        self.rows.iter().map(|r| r[j].clone()).collect()
    }
    pub fn f64s(&self, name: &str) -> Vec<f64> {
        let j = self.col(name);
        self.rows.iter().map(|r| parse_num(&r[j])).collect()
    }
    pub fn bools(&self, name: &str) -> Vec<bool> {
        let j = self.col(name);
        self.rows.iter().map(|r| r[j] == "TRUE").collect()
    }
    /// All columns but the first, as a row-major numeric matrix.
    pub fn matrix(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .map(|r| r[1..].iter().map(|s| parse_num(s)).collect())
            .collect()
    }
    pub fn index(&self) -> HashMap<String, usize> {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r[0].clone(), i))
            .collect()
    }
}

pub fn parse_num(s: &str) -> f64 {
    match s {
        "NA" | "" => f64::NAN,
        "Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        _ => s.parse().unwrap_or_else(|_| panic!("not a number: {s:?}")),
    }
}

pub fn read_csv(run: &str, name: &str) -> Table {
    let p = ref_dir(run).join(format!("{name}.csv"));
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&p)
        .unwrap_or_else(|e| panic!("{p:?}: {e}"));
    let header = rdr
        .headers()
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows = rdr
        .records()
        .map(|r| r.unwrap().iter().map(|s| s.to_string()).collect())
        .collect();
    Table { header, rows }
}

pub fn has_csv(run: &str, name: &str) -> bool {
    ref_dir(run).join(format!("{name}.csv")).exists()
}

/// Relative difference as `compare_vec` in `ref_common.R` computes it, with an exact NA/Inf
/// pattern check. Returns the max relative difference or panics with the first mismatch.
pub fn max_rel(what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut m: f64 = 0.0;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if a.is_nan() || b.is_nan() {
            assert!(
                a.is_nan() && b.is_nan(),
                "{what}[{i}]: NA pattern, got {a} want {b}"
            );
            continue;
        }
        if a.is_infinite() || b.is_infinite() {
            assert_eq!(a, b, "{what}[{i}]: Inf mismatch");
            continue;
        }
        if a != b {
            m = m.max((a - b).abs() / b.abs().max(1e-300));
        }
    }
    m
}

pub fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    let m = max_rel(what, got, want);
    assert!(m <= tol, "{what}: max rel {m:e} > {tol:e}");
}

/// The engine inputs a stage test needs, rebuilt from the goldens: the design, the kept counts
/// (gene-major) and the offset row from the golden library sizes and norm factors.
pub struct Fixture {
    pub sample_ids: Vec<String>,
    pub all_ids: Vec<String>,
    pub all_counts: Vec<f64>,
    pub design: Vec<f64>,
    pub p: usize,
    pub nlib: usize,
    pub ids: Vec<String>,
    pub counts: Vec<f64>,
    pub lib_eff: Vec<f64>,
    pub offset: Vec<f64>,
}

/// Counts columns follow the design row order (`countMatrix[, rownames(sampleInfo)]`).
pub fn fixture(run: &str) -> Fixture {
    let des = read_csv(run, "edger_design");
    let sample_ids = des.strings("id");
    let nlib = sample_ids.len();
    let p = des.header.len() - 1;
    let dm = des.matrix();
    let mut design = vec![0.0; nlib * p];
    for (i, r) in dm.iter().enumerate() {
        for j in 0..p {
            design[j * nlib + i] = r[j];
        }
    }
    let cnt = read_csv(run, "input_counts");
    let cols: Vec<usize> = sample_ids.iter().map(|s| cnt.col(s)).collect();
    let all_ids = cnt.strings("id");
    let mut all_counts = Vec::with_capacity(all_ids.len() * nlib);
    for r in &cnt.rows {
        for &c in &cols {
            all_counts.push(parse_num(&r[c]));
        }
    }
    let mut ids = Vec::new();
    let mut counts = Vec::new();
    let mut lib_eff = vec![f64::NAN; nlib];
    if has_csv(run, "edger_filter") {
        let keep = read_csv(run, "edger_filter").bools("keep");
        for (g, &k) in keep.iter().enumerate() {
            if k {
                ids.push(all_ids[g].clone());
                counts.extend_from_slice(&all_counts[g * nlib..(g + 1) * nlib]);
            }
        }
    }
    if has_csv(run, "edger_norm") {
        let nt = read_csv(run, "edger_norm");
        let lib = nt.f64s("lib_size");
        let nf = nt.f64s("norm_factor");
        lib_eff = lib.iter().zip(&nf).map(|(a, b)| a * b).collect();
    }
    let offset = lib_eff.iter().map(|&v| rnum::glibm::ln(v)).collect();
    Fixture {
        sample_ids,
        all_ids,
        all_counts,
        design,
        p,
        nlib,
        ids,
        counts,
        lib_eff,
        offset,
    }
}

/// Rows of a gene-major matrix whose mask is true.
pub fn rows_where(m: &[f64], ncol: usize, mask: &[bool]) -> Vec<f64> {
    m.chunks(ncol)
        .zip(mask)
        .filter(|(_, &k)| k)
        .flat_map(|(r, _)| r.to_vec())
        .collect()
}

/// Like `max_rel`, but relative to `max(|want|, floor)`: for quantities whose scale makes tiny
/// absolute values meaningless (deviances of genes fitted to zero counts sit at ~1e-7 and are
/// chaotic in the last bits of the offsets).
pub fn max_rel_floor(what: &str, got: &[f64], want: &[f64], floor: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut m: f64 = 0.0;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if a.is_nan() || b.is_nan() {
            assert!(
                a.is_nan() && b.is_nan(),
                "{what}[{i}]: NA pattern, got {a} want {b}"
            );
            continue;
        }
        if a.is_infinite() || b.is_infinite() {
            assert_eq!(a, b, "{what}[{i}]: Inf mismatch");
            continue;
        }
        if a != b {
            m = m.max((a - b).abs() / b.abs().max(floor));
        }
    }
    m
}

/// The F gate: `|dF| <= 1e-8 * max(F, 1)`. F is a deviance difference, so for genes where the
/// null and full fits nearly agree its noise is absolute (the deviances' own ~1e-12 cancellation
/// noise), and anything computed from F inherits that conditioning.
pub fn f_tol(f: f64) -> f64 {
    1e-8 * f.abs().max(1.0)
}

/// Per-element `[lo, hi]` image of `g(F, df)` over the F gate and a `1e-7` relative band on
/// `df` (the QL prior df, see `stage_ql`), from the four corners (each `g` used here is monotone
/// in both arguments).
pub fn band(f: &[f64], df: &[f64], g: impl Fn(usize, f64, f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![f64::NAN; f.len()];
    let mut hi = vec![f64::NAN; f.len()];
    for i in 0..f.len() {
        let d = f_tol(f[i]);
        let mut vals = Vec::with_capacity(4);
        for fi in [(f[i] - d).max(0.0), f[i] + d] {
            for di in [df[i] * (1.0 - 1e-7), df[i] * (1.0 + 1e-7)] {
                vals.push(g(i, fi, di));
            }
        }
        lo[i] = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        hi[i] = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }
    (lo, hi)
}

/// Pass when `got` is within 1e-8 relative of `want`, or inside `[lo, hi]` widened by 1e-8
/// relative. Exact NA pattern. Returns the max plain relative gap (for reporting) and the number
/// of elements that needed the band.
pub fn assert_in_band(
    what: &str,
    got: &[f64],
    want: &[f64],
    lo: &[f64],
    hi: &[f64],
) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut m: f64 = 0.0;
    let mut nband = 0;
    for i in 0..got.len() {
        let (a, b) = (got[i], want[i]);
        if a.is_nan() || b.is_nan() {
            assert!(
                a.is_nan() && b.is_nan(),
                "{what}[{i}]: NA pattern, got {a} want {b}"
            );
            continue;
        }
        if a == b {
            continue;
        }
        let rel = (a - b).abs() / b.abs().max(1e-300);
        m = m.max(rel);
        if rel <= 1e-8 {
            continue;
        }
        let (l, h) = (lo[i].min(hi[i]), lo[i].max(hi[i]));
        let ok = a >= l - 1e-8 * l.abs() && a <= h + 1e-8 * h.abs();
        assert!(
            ok,
            "{what}[{i}]: got {a} want {b} (rel {rel:e}) outside the F band [{l}, {h}]"
        );
        nband += 1;
    }
    (m, nband)
}
