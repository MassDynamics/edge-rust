//! `edge_rust._core`: thin PyO3 surface over `edger-core`. Takes a numpy count matrix and plain
//! Python lists, returns a dict of numpy arrays and lists. No pandas, no table shaping: the
//! production join, column naming and ANOVA formatting live in `python/edge_rust`.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use edger_core::pipeline::{
    max_abs_log2fc, run_edger, Comparison, Control, ControlKind, EdgerInput,
};

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
/// production message.
#[pyfunction]
#[pyo3(signature = (counts, gene_ids, sample_ids, condition_col, condition, controls, comparisons, norm_method = "TMM", entity_type = "gene"))]
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
    let out = py
        .allow_threads(|| run_edger(&input))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (max_pair, max_log2fc) = max_abs_log2fc(&out.pairs);

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
    d.set_item("design_cols", out.design_cols)?;
    d.set_item("kept", out.kept)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edger_pipeline, m)?)?;
    Ok(())
}
