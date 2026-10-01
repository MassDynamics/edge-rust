//! `limma-core`: a pure-Rust port of the limma statistical core used by the Mass Dynamics
//! pairwise-comparison job. No R anywhere.
//!
//! The golden corpus produced by the R implementation IS the spec. Every module here is
//! checked against it (see `tests/` and the repo `README.md`), under the tolerance policy
//! recorded in each corpus manifest.
//!
//! Module map:
//!
//! - `nmath`     R's distribution functions and gamma family (`pt`, `qt`, `pf`, `qf`,
//!   `pnorm`, `qnorm`, `pbeta`, `qbeta`, `pgamma`, `qgamma`, `pchisq`, `qchisq`, `digamma`,
//!   `trigamma`) plus limma's `trigammaInverse`. Ported from R's `src/nmath`, not from a
//!   generic stats crate: tails must match R within the corpus tolerances.
//! - `lowess`    line-for-line port of R's C `clowess` (used by `eBayes(trend=TRUE)`).
//! - `fit`       `lmFit` / `lm.series`, NA-aware per-gene refits.
//! - `linpack`   R's LINPACK QR (`dqrdc2`, `dqrsl`, `dqrls`) behind `qr()` and `lm.fit()`.
//! - `linalg`    small dense linear algebra and R vector helpers.
//! - `contrasts` `contrasts.fit`, incl. the non-orthogonal cov/Cholesky route.
//! - `ebayes`    `eBayes`, `squeezeVar`, `fitFDist`, `fitFDistUnequalDF1`, `fitFDistRobustly`.
//! - `optim`     R's `optimize()` and `uniroot()`.
//! - `quad`      `statmod::gauss.quad.prob` for `fitFDistRobustly`.
//! - `splines`   `splines::ns()` for `fitFDist` with a covariate.
//! - `toptable`  `topTable` (CIs, BH adjust, moderated F), `decideTests`.

pub mod contrasts;
pub mod ebayes;
pub mod fit;
pub mod glibm;
pub mod glibm_lgamma;
pub mod glibm_log1p;
pub mod glibm_log2;
pub mod glibm_pow;
pub mod hist;
pub mod lbfgsb;
pub mod ldouble;
pub mod linalg;
pub mod linpack;
pub mod locfit;
pub mod loess;
pub mod lowess;
pub mod nmath;
pub mod optim;
pub mod quad;
pub mod rng;
pub mod splines;
pub mod toptable;

pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Errors surfaced to the Python layer. Messages that reach users must match the R
/// package's `md_error` texts where a corpus manifest records an `expected_error`.
#[derive(Debug, thiserror::Error)]
pub enum LimmaError {
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, LimmaError>;
