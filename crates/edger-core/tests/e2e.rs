//! End to end: the engine run from the reference's dumped inputs against `reference_output.csv`
//! (the production table, ANOVA shaped for anova runs), under the corpus `tolerance_policy`, and
//! every expected-error run against its recorded message.
//!
//! P-values and everything derived from F (stat, SE, CI) inherit F's conditioning, so they are
//! checked against the image of the F gate and the 1e-8 df band (`common::band`), as in
//! `stage_qltest`. On the image's arithmetic (glibm, long double sums, Blue's dnrm2) every run
//! passes plain 1e-8 relative, so the test also asserts that no value needed the band.

mod common;

use common::*;
use edger_core::pipeline::{
    max_abs_log2fc, run_edger, Comparison, Control, ControlKind, EdgerInput, EdgerOutput,
};
use rnum::linalg::p_adjust_bh;
use rnum::nmath::{pf, qt};
use std::collections::HashMap;

/// The manifest's `control_cols` (`{Column, Type}`, scalars or arrays) as (name, kind) pairs.
fn control_specs(run: &str) -> Vec<(String, ControlKind)> {
    let cc = &manifest(run)["params"]["control_cols"];
    if cc.is_null() {
        return vec![];
    }
    let as_vec = |v: &serde_json::Value| -> Vec<String> {
        match v {
            serde_json::Value::Array(a) => {
                a.iter().map(|s| s.as_str().unwrap().to_string()).collect()
            }
            s => vec![s.as_str().unwrap().to_string()],
        }
    };
    as_vec(&cc["Column"])
        .into_iter()
        .zip(as_vec(&cc["Type"]))
        .map(|(c, t)| {
            let k = if t == "numerical" {
                ControlKind::Numerical
            } else {
                ControlKind::Categorical
            };
            (c, k)
        })
        .collect()
}

/// Engine input rebuilt from `input_counts`, `input_sample_info`, `input_comparisons` and the
/// manifest, the way the Python router hands it over.
fn edger_input(run: &str) -> EdgerInput {
    let m = manifest(run);
    let si = read_csv(run, "input_sample_info");
    let sample_ids = si.strings("replicate");
    let cc = m["params"]["condition_col"].as_str().unwrap().to_string();
    let condition = si.strings(&cc);
    let specs = control_specs(run);
    let extra: Vec<&String> = si
        .header
        .iter()
        .filter(|h| **h != "replicate" && **h != cc)
        .collect();
    assert_eq!(
        extra.len(),
        specs.len(),
        "{run}: control columns {extra:?} vs manifest {specs:?}"
    );
    let controls = extra
        .iter()
        .zip(&specs)
        .map(|(name, (_, kind))| Control {
            name: name.to_string(),
            kind: *kind,
            values: si.strings(name),
        })
        .collect();
    let cnt = read_csv(run, "input_counts");
    let cols: Vec<usize> = sample_ids.iter().map(|s| cnt.col(s)).collect();
    let mut counts = Vec::with_capacity(cnt.rows.len() * cols.len());
    for r in &cnt.rows {
        for &c in &cols {
            counts.push(parse_num(&r[c]));
        }
    }
    let cmp = read_csv(run, "input_comparisons");
    let (l, r, el, er) = (
        cmp.strings("left"),
        cmp.strings("right"),
        cmp.strings("encoded_left"),
        cmp.strings("encoded_right"),
    );
    let comparisons = (0..l.len())
        .map(|i| Comparison {
            left: l[i].clone(),
            right: r[i].clone(),
            encoded_left: el[i].clone(),
            encoded_right: er[i].clone(),
        })
        .collect();
    EdgerInput {
        gene_ids: cnt.strings("id"),
        sample_ids,
        counts,
        condition_col: cc,
        condition,
        controls,
        comparisons,
        norm_method: m["params"]["edger_norm_method"]
            .as_str()
            .unwrap()
            .to_string(),
        entity_type: m["entity_type"].as_str().unwrap().to_string(),
    }
}

/// Worst plain relative gap and band count per column kind, for the report line.
#[derive(Default)]
struct Worst(HashMap<&'static str, (f64, usize)>);

impl Worst {
    fn add(&mut self, k: &'static str, (m, n): (f64, usize)) {
        let e = self.0.entry(k).or_insert((0.0, 0));
        e.0 = e.0.max(m);
        e.1 += n;
    }
}

/// `got` reordered to the golden's rows.
fn reorder(v: &[f64], order: &[usize]) -> Vec<f64> {
    order.iter().map(|&i| v[i]).collect()
}

/// The golden test table's F, df_total and ids, reordered to the output rows (NA where filtered).
fn test_cols(run: &str, label: &str, out_ids: &[String]) -> (Vec<f64>, Vec<f64>) {
    let t = read_csv(run, &format!("edger_test_{label}"));
    let idx = t.index();
    let (f, d) = (t.f64s("F"), t.f64s("df_total"));
    let pick = |v: &[f64]| {
        out_ids
            .iter()
            .map(|g| idx.get(g).map_or(f64::NAN, |&i| v[i]))
            .collect::<Vec<_>>()
    };
    (pick(&f), pick(&d))
}

/// F: rel 1e-8, or the F gate `|dF| <= 1e-8 * max(|F|, 1)` (`common::f_tol`), which covers the
/// policy's `edger_f_floor` (abs 1e-8 where |F_golden| < 1e-8) and extends it to F < 1. Returns
/// the largest absolute gap among the elements that needed the gate, and the floor rows.
fn check_f(what: &str, got: &[f64], want: &[f64]) -> (f64, Vec<bool>) {
    let mut m: f64 = 0.0;
    let mut floor = vec![false; want.len()];
    for i in 0..want.len() {
        let (a, b) = (got[i], want[i]);
        if a.is_nan() || b.is_nan() {
            assert!(
                a.is_nan() && b.is_nan(),
                "{what}[{i}]: NA pattern, got {a} want {b}"
            );
            continue;
        }
        if b.abs() < 1e-8 {
            floor[i] = true;
        }
        let rel = (a - b).abs() / b.abs();
        if rel > 1e-8 {
            // Below F = 1 the noise is absolute (see `common::f_tol`): report the absolute gap.
            assert!(
                (a - b).abs() <= f_tol(b),
                "{what}[{i}]: got {a} want {b} rel {rel:e}"
            );
            m = m.max((a - b).abs());
        }
    }
    (m, floor)
}

/// The p-value and BH bands from the golden F and df_total.
fn p_bands(f: &[f64], df: &[f64], df1: f64) -> [Vec<f64>; 4] {
    let (plo, phi) = band(f, df, |_, fi, di| pf(fi, df1, di, false, false));
    // BH over the tested genes only (NA rows are the filtered genes).
    let bh = |v: &[f64]| {
        let ok: Vec<usize> = (0..v.len()).filter(|&i| !v[i].is_nan()).collect();
        let adj = p_adjust_bh(&ok.iter().map(|&i| v[i]).collect::<Vec<_>>());
        let mut out = vec![f64::NAN; v.len()];
        for (k, &i) in ok.iter().enumerate() {
            out[i] = adj[k];
        }
        out
    };
    let (alo, ahi) = (bh(&plo), bh(&phi));
    [plo, phi, alo, ahi]
}

fn assert_sig(what: &str, got: &[f64], want: &[f64]) {
    for i in 0..want.len() {
        if !want[i].is_nan() {
            assert_eq!(
                got[i] < 0.05,
                want[i] < 0.05,
                "{what}[{i}]: significance flag, got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}

/// NA positions must match except on F-floor rows; the remaining values go through the band.
fn check_derived(
    what: &str,
    got: &[f64],
    want: &[f64],
    floor: &[bool],
    g: impl Fn(usize, f64, f64) -> f64,
    f: &[f64],
    df: &[f64],
) -> (f64, usize) {
    let keep: Vec<usize> = (0..want.len()).filter(|&i| !floor[i]).collect();
    let (lo, hi) = band(f, df, g);
    let sel = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
    assert_in_band(what, &sel(got), &sel(want), &sel(&lo), &sel(&hi))
}

fn check_run(run: &str, out: &EdgerOutput, w: &mut Worst) {
    let gold = read_csv(run, "reference_output");
    let anova = manifest(run)["mode"].as_str() == Some("anova");
    let ids = gold.strings("GroupId");
    assert_eq!(ids.len(), out.gene_ids.len(), "{run}: row count");
    let pos: HashMap<&str, usize> = out
        .gene_ids
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    let order: Vec<usize> = ids
        .iter()
        .map(|g| {
            *pos.get(g.as_str())
                .unwrap_or_else(|| panic!("{run}: id {g}"))
        })
        .collect();

    let mut cols = vec!["GroupId".to_string()];
    if anova {
        cols.extend(
            [
                "AveExpr",
                "PValue",
                "AdjPValue",
                "F",
                "MaxLog2FCPair",
                "MaxLog2FC",
            ]
            .map(String::from),
        );
    } else {
        cols.extend(["AveExpr", "F", "PValue", "AdjPValue"].map(String::from));
        for p in &out.pairs {
            for k in [
                "Log2FC",
                "stat",
                "SE",
                "CILeft",
                "CIRight",
                "F",
                "PValue",
                "AdjPValue",
            ] {
                cols.push(format!("{k} {}", p.label));
            }
        }
    }
    assert_eq!(gold.header, cols, "{run}: column names");

    w.add(
        "AveExpr",
        (
            max_rel(
                &format!("{run} AveExpr"),
                &reorder(&out.ave_expr, &order),
                &gold.f64s("AveExpr"),
            ),
            0,
        ),
    );
    assert_close(
        &format!("{run} AveExpr"),
        &reorder(&out.ave_expr, &order),
        &gold.f64s("AveExpr"),
        1e-8,
    );

    // Omnibus F, PValue, AdjPValue.
    let ncon = read_csv(run, "edger_test_omnibus")
        .header
        .iter()
        .filter(|h| h.starts_with("logFC"))
        .count() as f64;
    let (tf, td) = test_cols(run, "omnibus", &ids);
    let (m, _) = check_f(
        &format!("{run} F"),
        &reorder(&out.f, &order),
        &gold.f64s("F"),
    );
    w.add("F_abs", (m, 0));
    let [plo, phi, alo, ahi] = p_bands(&tf, &td, ncon);
    let gp = reorder(&out.pvalue, &order);
    let ga = reorder(&out.adj_pvalue, &order);
    w.add(
        "PValue",
        assert_in_band(
            &format!("{run} PValue"),
            &gp,
            &gold.f64s("PValue"),
            &plo,
            &phi,
        ),
    );
    w.add(
        "AdjPValue",
        assert_in_band(
            &format!("{run} AdjPValue"),
            &ga,
            &gold.f64s("AdjPValue"),
            &alo,
            &ahi,
        ),
    );
    assert_sig(&format!("{run} AdjPValue"), &ga, &gold.f64s("AdjPValue"));

    if anova {
        let (bi, bv) = max_abs_log2fc(&out.pairs);
        let names: Vec<String> = order
            .iter()
            .map(|&i| bi[i].map_or(String::new(), |k| out.pairs[k].label.clone()))
            .collect();
        assert_eq!(names, gold.strings("MaxLog2FCPair"), "{run}: MaxLog2FCPair");
        w.add(
            "MaxLog2FC",
            (
                max_rel(
                    &format!("{run} MaxLog2FC"),
                    &reorder(&bv, &order),
                    &gold.f64s("MaxLog2FC"),
                ),
                0,
            ),
        );
        assert_close(
            &format!("{run} MaxLog2FC"),
            &reorder(&bv, &order),
            &gold.f64s("MaxLog2FC"),
            1e-8,
        );
        return;
    }

    let ci = read_csv(run, "edger_ci_df");
    let ci_idx = ci.index();
    let ci_v = ci.f64s("df_ci");
    let df_ci: Vec<f64> = ids
        .iter()
        .map(|g| ci_idx.get(g).map_or(f64::NAN, |&i| ci_v[i]))
        .collect();
    for (k, p) in out.pairs.iter().enumerate() {
        let c = |s: &str| format!("{s} {}", p.label);
        let lfc_w = gold.f64s(&c("Log2FC"));
        w.add(
            "Log2FC",
            (
                max_rel(&c("Log2FC"), &reorder(&p.log2fc, &order), &lfc_w),
                0,
            ),
        );
        assert_close(
            &format!("{run} {}", c("Log2FC")),
            &reorder(&p.log2fc, &order),
            &lfc_w,
            1e-8,
        );
        let (m, floor) = check_f(
            &format!("{run} {}", c("F")),
            &reorder(&p.f, &order),
            &gold.f64s(&c("F")),
        );
        w.add("F_abs", (m, 0));
        w.add("f_floor_rows", (0.0, floor.iter().filter(|&&b| b).count()));
        let (tf, td) = test_cols(run, &format!("pair{}", k + 1), &ids);
        let [plo, phi, alo, ahi] = p_bands(&tf, &td, 1.0);
        let gp = reorder(&p.pvalue, &order);
        let ga = reorder(&p.adj_pvalue, &order);
        w.add(
            "PValue",
            assert_in_band(
                &format!("{run} {}", c("PValue")),
                &gp,
                &gold.f64s(&c("PValue")),
                &plo,
                &phi,
            ),
        );
        w.add(
            "AdjPValue",
            assert_in_band(
                &format!("{run} {}", c("AdjPValue")),
                &ga,
                &gold.f64s(&c("AdjPValue")),
                &alo,
                &ahi,
            ),
        );
        assert_sig(
            &format!("{run} {}", c("AdjPValue")),
            &ga,
            &gold.f64s(&c("AdjPValue")),
        );

        let sgn = |x: f64| if x == 0.0 { 0.0 } else { x.signum() };
        let stat = |i: usize, f: f64, _d: f64| f.sqrt() * sgn(lfc_w[i]);
        let se = |i: usize, f: f64, _d: f64| lfc_w[i].abs() / f.sqrt();
        let cil = |i: usize, f: f64, d: f64| {
            lfc_w[i] - qt(0.975, d, true, false) * lfc_w[i].abs() / f.sqrt()
        };
        let cir = |i: usize, f: f64, d: f64| {
            lfc_w[i] + qt(0.975, d, true, false) * lfc_w[i].abs() / f.sqrt()
        };
        let gf = gold.f64s(&c("F"));
        let tag = |s: &str| format!("{run} {}", c(s));
        w.add(
            "stat",
            check_derived(
                &tag("stat"),
                &reorder(&p.stat, &order),
                &gold.f64s(&c("stat")),
                &floor,
                stat,
                &gf,
                &df_ci,
            ),
        );
        w.add(
            "SE",
            check_derived(
                &tag("SE"),
                &reorder(&p.se, &order),
                &gold.f64s(&c("SE")),
                &floor,
                se,
                &gf,
                &df_ci,
            ),
        );
        w.add(
            "CI",
            check_derived(
                &tag("CILeft"),
                &reorder(&p.ci_left, &order),
                &gold.f64s(&c("CILeft")),
                &floor,
                cil,
                &gf,
                &df_ci,
            ),
        );
        w.add(
            "CI",
            check_derived(
                &tag("CIRight"),
                &reorder(&p.ci_right, &order),
                &gold.f64s(&c("CIRight")),
                &floor,
                cir,
                &gf,
                &df_ci,
            ),
        );
    }
}

#[test]
fn e2e_tables_match_reference_output() {
    let mut n = 0;
    for run in edger_runs() {
        if is_error_run(&run) {
            continue;
        }
        let out = run_edger(&edger_input(&run)).unwrap_or_else(|e| panic!("{run}: {e}"));
        let mut w = Worst::default();
        check_run(&run, &out, &mut w);
        let mut ks: Vec<_> = w.0.iter().collect();
        ks.sort_by_key(|(k, _)| **k);
        let line: Vec<String> = ks
            .iter()
            .map(|(k, (m, b))| format!("{k} {m:.1e}/{b}"))
            .collect();
        eprintln!("{run}: {}", line.join(", "));
        for (k, (m, b)) in &ks {
            assert!(
                **k == "f_floor_rows" || *b == 0,
                "{run}: {k} needed the F band ({b} values, max rel {m:e})"
            );
        }
        n += 1;
    }
    assert_eq!(n, 25, "edgeR ok runs");
}

fn expect_err(what: &str, input: &EdgerInput, expected: &str) {
    match run_edger(input) {
        Ok(_) => panic!("{what}: expected an error containing {expected:?}"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(expected),
                "{what}: {msg:?} does not contain {expected:?}"
            );
        }
    }
}

#[test]
fn e2e_expected_errors() {
    let mut n = 0;
    for run in edger_runs() {
        if !is_error_run(&run) {
            continue;
        }
        let expected = manifest(&run)["expected_error"]
            .as_str()
            .unwrap()
            .to_string();
        // negative and protein_entity fail in the router before inputs are dumped: rebuild them
        // from an ordinary run.
        let input = match run.as_str() {
            "edge_edger_negative" => {
                let mut i = edger_input("count_edger_airway_all_ctlnone");
                i.counts[7] = -1.0;
                i
            }
            "edge_edger_protein_entity" => {
                let mut i = edger_input("count_edger_airway_all_ctlnone");
                i.entity_type = "protein".into();
                i
            }
            _ => edger_input(&run),
        };
        expect_err(&run, &input, &expected);
        n += 1;
    }
    assert_eq!(n, 5, "edgeR error runs");
}
