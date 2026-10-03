//! `edge_rust._core`: thin PyO3 surface over `edger-core`. Takes a numpy count matrix and plain
//! Python lists, returns a dict of numpy arrays and lists. No pandas, no table shaping: the
//! production join, column naming and ANOVA formatting live in `python/edge_rust`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use edger_core::pipeline::{
    max_abs_log2fc, run_edger_diag, Comparison, Control, ControlKind, EdgerInput,
};

/// Run `f` without the GIL. Engine errors become `ValueError`; a panic becomes `RuntimeError`
/// instead of pyo3's `PanicException`, which derives from `BaseException` and so escapes
/// `except Exception`.
fn guarded<T: Send>(py: Python<'_>, f: impl FnOnce() -> rnum::Result<T> + Send) -> PyResult<T> {
    match py.allow_threads(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))) {
        Ok(r) => r.map_err(|e| PyValueError::new_err(e.to_string())),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            Err(PyRuntimeError::new_err(format!(
                "internal error in the edgeR engine: {msg}"
            )))
        }
    }
}

/// Test hook: a panic inside [`guarded`], to check that it reaches Python as `RuntimeError`.
#[pyfunction]
fn _selftest_panic(py: Python<'_>) -> PyResult<()> {
    guarded(py, || -> rnum::Result<()> { panic!("selftest panic") })
}

fn vector<'py>(py: Python<'py>, data: Vec<f64>) -> Bound<'py, PyAny> {
    Array1::from_vec(data).into_pyarray(py).into_any()
}

/// The edgeR QL engine (`filterByExpr` -> `calcNormFactors` -> `estimateDisp` -> `glmQLFit` ->
/// `glmQLFTest`) on `counts`, `ngenes x nsamples` in sample order.
///
/// `controls` is a list of `(name, kind, values)` with `kind` `"categorical"` or `"numerical"`;
/// `comparisons` a list of `(left, right, encoded_left, encoded_right)`. Returns a dict with
/// `gene_ids`, `ave_expr`, `F`, `PValue`, `AdjPValue` (omnibus), `pairs` (a list of dicts with
/// `label`, `Log2FC`, `stat`, `SE`, `CILeft`, `CIRight`, `F`, `PValue`, `AdjPValue`),
/// `max_pair` (index into `pairs` or None, per gene), `max_log2fc`, `design_cols` and `kept`,
/// every vector over all input genes in input order. Engine errors raise `ValueError` with the
/// production message; an internal panic raises `RuntimeError`.
///
/// With `diagnostics`, the dict also has `diag`: `kept_idx` (indices of the genes
/// `filterByExpr` kept) and, over those genes, `ave_log_cpm`, `trended_disp`, `tagwise_disp`,
/// `s2`, `s2_prior`, `s2_post`, `df_residual_adj`, `df_prior`, `coefficients` and
/// `unshrunk_coefficients` (`nkept x p`, natural log) and `fitted` (`nkept x nsamples`); per
/// sample `lib_size` and `norm_factor`; the transposed design (`p x nsamples`); and the scalars
/// `df_residual`, `fit_dispersion`, `ave_ql_dispersion`, `common_disp` and `disp_prior_df`.
#[pyfunction]
#[pyo3(signature = (counts, gene_ids, sample_ids, condition_col, condition, controls, comparisons, norm_method = "TMM", entity_type = "gene", diagnostics = false))]
#[allow(clippy::too_many_arguments)]
fn edger_pipeline<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    gene_ids: Vec<String>,
    sample_ids: Vec<String>,
    condition_col: String,
    condition: Vec<String>,
    controls: Vec<(String, String, Vec<String>)>,
    comparisons: Vec<(String, String, String, String)>,
    norm_method: &str,
    entity_type: &str,
    diagnostics: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let a = counts.as_array();
    let (ng, nlib) = a.dim();
    if ng != gene_ids.len() || nlib != sample_ids.len() {
        return Err(PyValueError::new_err(format!(
            "counts is {ng} x {nlib} but there are {} gene ids and {} sample ids",
            gene_ids.len(),
            sample_ids.len()
        )));
    }
    let controls = controls
        .into_iter()
        .map(|(name, kind, values)| {
            let kind = match kind.as_str() {
                "categorical" => ControlKind::Categorical,
                "numerical" => ControlKind::Numerical,
                k => {
                    return Err(PyValueError::new_err(format!(
                        "control '{name}': unknown kind '{k}'"
                    )))
                }
            };
            Ok(Control { name, kind, values })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let input = EdgerInput {
        gene_ids,
        sample_ids,
        // Row-major iteration of an ngenes x nsamples array is the gene-major layout.
        counts: a.iter().copied().collect(),
        condition_col,
        condition,
        controls,
        comparisons: comparisons
            .into_iter()
            .map(|(left, right, encoded_left, encoded_right)| Comparison {
                left,
                right,
                encoded_left,
                encoded_right,
            })
            .collect(),
        norm_method: norm_method.to_string(),
        entity_type: entity_type.to_string(),
    };
    let (out, diag, (max_pair, max_log2fc)) = guarded(py, || {
        let (out, diag) = run_edger_diag(&input)?;
        let max = max_abs_log2fc(&out.pairs);
        Ok((out, diag, max))
    })?;

    let d = PyDict::new(py);
    d.set_item("gene_ids", out.gene_ids)?;
    d.set_item("ave_expr", vector(py, out.ave_expr))?;
    d.set_item("F", vector(py, out.f))?;
    d.set_item("PValue", vector(py, out.pvalue))?;
    d.set_item("AdjPValue", vector(py, out.adj_pvalue))?;
    let pairs = PyList::empty(py);
    for p in out.pairs {
        let pd = PyDict::new(py);
        pd.set_item("label", p.label)?;
        pd.set_item("Log2FC", vector(py, p.log2fc))?;
        pd.set_item("stat", vector(py, p.stat))?;
        pd.set_item("SE", vector(py, p.se))?;
        pd.set_item("CILeft", vector(py, p.ci_left))?;
        pd.set_item("CIRight", vector(py, p.ci_right))?;
        pd.set_item("F", vector(py, p.f))?;
        pd.set_item("PValue", vector(py, p.pvalue))?;
        pd.set_item("AdjPValue", vector(py, p.adj_pvalue))?;
        pairs.append(pd)?;
    }
    d.set_item("pairs", pairs)?;
    d.set_item("max_pair", max_pair)?;
    d.set_item("max_log2fc", vector(py, max_log2fc))?;
    if diagnostics {
        let p = out.design_cols.len();
        let dd = PyDict::new(py);
        dd.set_item("kept_idx", diag.kept_idx)?;
        dd.set_item("lib_size", vector(py, diag.lib_size))?;
        dd.set_item("norm_factor", vector(py, diag.norm_factors))?;
        // The column-major nsamples x p design read row-major is its p x nsamples transpose.
        dd.set_item("design", matrix(py, out.design.clone(), p, nlib)?)?;
        dd.set_item("ave_log_cpm", vector(py, diag.disp.ave_logcpm))?;
        dd.set_item("trended_disp", vector(py, diag.disp.trended))?;
        dd.set_item("tagwise_disp", vector(py, diag.disp.tagwise))?;
        dd.set_item("common_disp", diag.disp.common)?;
        dd.set_item("disp_prior_df", diag.disp.prior_df)?;
        let q = diag.ql;
        let nk = q.s2.len();
        dd.set_item("fit_dispersion", q.dispersion)?;
        dd.set_item("ave_ql_dispersion", q.ave_ql_dispersion)?;
        dd.set_item("df_residual", q.df_residual)?;
        dd.set_item("s2", vector(py, q.s2))?;
        dd.set_item("s2_prior", vector(py, q.s2_prior))?;
        dd.set_item("s2_post", vector(py, q.s2_post))?;
        dd.set_item("df_residual_adj", vector(py, q.df_residual_adj))?;
        dd.set_item("df_prior", vector(py, q.df_prior))?;
        dd.set_item("coefficients", matrix(py, q.coefficients, nk, p)?)?;
        dd.set_item(
            "unshrunk_coefficients",
            matrix(py, q.unshrunk_coefficients, nk, p)?,
        )?;
        dd.set_item("fitted", matrix(py, q.fitted, nk, nlib)?)?;
        d.set_item("diag", dd)?;
    }
    d.set_item("design_cols", out.design_cols)?;
    d.set_item("kept", out.kept)?;
    Ok(d)
}

/// A row-major `rows x cols` numpy array.
fn matrix<'py>(
    py: Python<'py>,
    data: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let a = numpy::ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py).into_any())
}

/// R's `as.character()` of each double (`rnum::rformat`); NaN gives "NaN".
#[pyfunction]
fn r_as_character(values: Vec<f64>) -> Vec<String> {
    values
        .into_iter()
        .map(rnum::rformat::r_as_character)
        .collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edger_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(r_as_character, m)?)?;
    m.add_function(wrap_pyfunction!(_selftest_panic, m)?)?;
    Ok(())
}
