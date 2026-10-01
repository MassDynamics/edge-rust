//! R's quasi-Newton optimisers and the finite-difference Hessian, ported from R 4.5.0.
//!
//! - [`vmmin`]: `src/appl/optim.c` `vmmin`, the BFGS variable-metric method behind
//!   `optim(method = "BFGS")`, line for line.
//! - [`optim_bfgs`]: the `optim()` wrapper for that method (`library/stats/src/optim.c`
//!   `optim` + `fminfn` / `fmingr` with an analytic gradient).
//! - [`optimhess`]: `library/stats/src/optim.c` `optimhess`, used by `optimHess()` and by
//!   `optim(hessian = TRUE)`.
//!
//! Arithmetic follows the C statement by statement (same operation order, no fused
//! multiply-add), so results are bit-identical where the inputs and the callbacks are.

/// Errors R raises from inside these routines.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimError {
    /// `fminfn` / `fmingr`: "non-finite value supplied by optim".
    NonFinitePar,
    /// `vmmin`: "initial value in 'vmmin' is not finite".
    InitialNotFinite,
    /// L-BFGS-B: "L-BFGS-B needs finite values of 'fn'".
    LbfgsbNonFiniteFn,
    /// Any other argument error, with R's message.
    Invalid(String),
}

impl std::fmt::Display for OptimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimError::NonFinitePar => write!(f, "non-finite value supplied by optim"),
            OptimError::InitialNotFinite => write!(f, "initial value in 'vmmin' is not finite"),
            OptimError::LbfgsbNonFiniteFn => write!(f, "L-BFGS-B needs finite values of 'fn'"),
            OptimError::Invalid(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for OptimError {}

/// Result of [`vmmin`] (its out-parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct VmminOut {
    pub fmin: f64,
    pub fncount: i32,
    pub grcount: i32,
    /// 0 converged, 1 `maxit` reached.
    pub fail: i32,
}

const STEPREDN: f64 = 0.2;
const ACCTOL: f64 = 0.0001;
const RELTEST: f64 = 10.0;

/// `vmmin(n0, b, Fmin, fminfn, fmingr, maxit, trace = 0, mask, abstol, reltol, ...)`.
///
/// `b` is updated in place to the final parameters. `fminfn(b)` returns the objective,
/// `fmingr(b, g)` writes the gradient; either may fail with an [`OptimError`], which is
/// propagated the way R's `error()` would abort the call.
#[allow(clippy::too_many_arguments)]
pub fn vmmin<F, G>(
    b: &mut [f64],
    mut fminfn: F,
    mut fmingr: G,
    maxit: i32,
    mask: &[bool],
    abstol: f64,
    reltol: f64,
) -> Result<VmminOut, OptimError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimError>,
    G: FnMut(&[f64], &mut [f64]) -> Result<(), OptimError>,
{
    let n0 = b.len();
    if maxit <= 0 {
        let fmin = fminfn(b)?;
        return Ok(VmminOut { fmin, fncount: 0, grcount: 0, fail: 0 });
    }
    let l: Vec<usize> = (0..n0).filter(|&i| mask[i]).collect();
    let n = l.len();
    let mut g = vec![0.0; n0];
    let mut t = vec![0.0; n];
    let mut x = vec![0.0; n];
    let mut c = vec![0.0; n];
    // lower triangle, B[i][j] for j <= i
    let mut bm: Vec<Vec<f64>> = (0..n).map(|i| vec![0.0; i + 1]).collect();
    let mut f = fminfn(b)?;
    if !f.is_finite() {
        return Err(OptimError::InitialNotFinite);
    }
    let mut fmin = f;
    let mut funcount = 1;
    let mut gradcount = 1;
    fmingr(b, &mut g)?;
    let mut iter = 1;
    let mut ilast = gradcount;
    let mut count: usize;

    loop {
        if ilast == gradcount {
            for i in 0..n {
                for j in 0..i {
                    bm[i][j] = 0.0;
                }
                bm[i][i] = 1.0;
            }
        }
        for i in 0..n {
            x[i] = b[l[i]];
            c[i] = g[l[i]];
        }
        let mut gradproj = 0.0;
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..=i {
                s -= bm[i][j] * g[l[j]];
            }
            for j in (i + 1)..n {
                s -= bm[j][i] * g[l[j]];
            }
            t[i] = s;
            gradproj += s * g[l[i]];
        }

        if gradproj < 0.0 {
            // search direction is downhill
            let mut steplength = 1.0;
            let mut accpoint = false;
            loop {
                count = 0;
                for i in 0..n {
                    b[l[i]] = x[i] + steplength * t[i];
                    if RELTEST + x[i] == RELTEST + b[l[i]] {
                        // no change
                        count += 1;
                    }
                }
                if count < n {
                    f = fminfn(b)?;
                    funcount += 1;
                    accpoint = f.is_finite() && (f <= fmin + gradproj * steplength * ACCTOL);
                    if !accpoint {
                        steplength *= STEPREDN;
                    }
                }
                if count == n || accpoint {
                    break;
                }
            }
            let enough = (f > abstol) && (f - fmin).abs() > reltol * (fmin.abs() + reltol);
            // stop if value if small or if relative change is low
            if !enough {
                count = n;
                fmin = f;
            }
            if count < n {
                // making progress
                fmin = f;
                fmingr(b, &mut g)?;
                gradcount += 1;
                iter += 1;
                let mut d1 = 0.0;
                for i in 0..n {
                    t[i] *= steplength;
                    c[i] = g[l[i]] - c[i];
                    d1 += t[i] * c[i];
                }
                if d1 > 0.0 {
                    let mut d2 = 0.0;
                    for i in 0..n {
                        let mut s = 0.0;
                        for j in 0..=i {
                            s += bm[i][j] * c[j];
                        }
                        for j in (i + 1)..n {
                            s += bm[j][i] * c[j];
                        }
                        x[i] = s;
                        d2 += s * c[i];
                    }
                    d2 = 1.0 + d2 / d1;
                    for i in 0..n {
                        for j in 0..=i {
                            bm[i][j] += (d2 * t[i] * t[j] - x[i] * t[j] - t[i] * x[j]) / d1;
                        }
                    }
                } else {
                    // D1 < 0
                    ilast = gradcount;
                }
            } else {
                // no progress
                if ilast < gradcount {
                    count = 0;
                    ilast = gradcount;
                }
            }
        } else {
            // uphill search
            count = 0;
            if ilast == gradcount {
                count = n;
            } else {
                ilast = gradcount;
            }
            // Resets unless has just been reset
        }
        if iter >= maxit {
            break;
        }
        if gradcount - ilast > 2 * n as i32 {
            ilast = gradcount; // periodic restart
        }
        if !(count != n || ilast != gradcount) {
            break;
        }
    }
    Ok(VmminOut { fmin, fncount: funcount, grcount: gradcount, fail: if iter < maxit { 0 } else { 1 } })
}

/// Control values `optim()` passes to the C code (the subset BFGS, L-BFGS-B and
/// `optimhess` read). [`OptimControl::default`] is `optim()`'s default list.
#[derive(Debug, Clone)]
pub struct OptimControl {
    pub fnscale: f64,
    /// `None` means `rep(1, npar)`.
    pub parscale: Option<Vec<f64>>,
    /// `None` means `rep(1e-3, npar)`.
    pub ndeps: Option<Vec<f64>>,
    pub maxit: i32,
    pub abstol: f64,
    pub reltol: f64,
    /// L-BFGS-B only.
    pub lmm: i32,
    pub factr: f64,
    pub pgtol: f64,
}

impl Default for OptimControl {
    fn default() -> Self {
        OptimControl {
            fnscale: 1.0,
            parscale: None,
            ndeps: None,
            maxit: 100,
            abstol: f64::NEG_INFINITY,
            reltol: f64::EPSILON.sqrt(),
            lmm: 5,
            factr: 1e7,
            pgtol: 0.0,
        }
    }
}

impl OptimControl {
    fn parscale(&self, n: usize) -> Vec<f64> {
        self.parscale.clone().unwrap_or_else(|| vec![1.0; n])
    }
    fn ndeps(&self, n: usize) -> Vec<f64> {
        self.ndeps.clone().unwrap_or_else(|| vec![1e-3; n])
    }
}

/// What `optim()` returns (`par`, `value`, `counts`, `convergence`, `message`).
#[derive(Debug, Clone, PartialEq)]
pub struct OptimOut {
    pub par: Vec<f64>,
    pub value: f64,
    pub fncount: i32,
    pub grcount: i32,
    pub convergence: i32,
    pub message: Option<String>,
}

/// `fminfn`: objective on the scaled parameters, divided by `fnscale`.
fn fminfn_scaled<F: FnMut(&[f64]) -> f64>(fun: &mut F, p: &[f64], parscale: &[f64], fnscale: f64) -> Result<f64, OptimError> {
    let mut x = vec![0.0; p.len()];
    for i in 0..p.len() {
        if !p[i].is_finite() {
            return Err(OptimError::NonFinitePar);
        }
        x[i] = p[i] * parscale[i];
    }
    Ok(fun(&x) / fnscale)
}

/// `fmingr` with an analytic gradient: `df = gr(p * parscale) * parscale / fnscale`.
fn fmingr_scaled<G: FnMut(&[f64]) -> Vec<f64>>(
    gr: &mut G,
    p: &[f64],
    df: &mut [f64],
    parscale: &[f64],
    fnscale: f64,
) -> Result<(), OptimError> {
    let mut x = vec![0.0; p.len()];
    for i in 0..p.len() {
        if !p[i].is_finite() {
            return Err(OptimError::NonFinitePar);
        }
        x[i] = p[i] * parscale[i];
    }
    let s = gr(&x);
    if s.len() != p.len() {
        return Err(OptimError::Invalid(format!("gradient in optim evaluated to length {} not {}", s.len(), p.len())));
    }
    for i in 0..p.len() {
        df[i] = s[i] * parscale[i] / fnscale;
    }
    Ok(())
}

/// `optim(par, fn, gr, method = "BFGS", control)` with an analytic gradient.
pub fn optim_bfgs<F, G>(par: &[f64], mut fun: F, mut gr: G, control: &OptimControl) -> Result<OptimOut, OptimError>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let n = par.len();
    let ps = control.parscale(n);
    let fs = control.fnscale;
    let mut dpar: Vec<f64> = (0..n).map(|i| par[i] / ps[i]).collect();
    let mask = vec![true; n];
    let out = vmmin(
        &mut dpar,
        |p| fminfn_scaled(&mut fun, p, &ps, fs),
        |p, df| fmingr_scaled(&mut gr, p, df, &ps, fs),
        control.maxit,
        &mask,
        control.abstol,
        control.reltol,
    )?;
    Ok(OptimOut {
        par: (0..n).map(|i| dpar[i] * ps[i]).collect(),
        value: out.fmin * fs,
        fncount: out.fncount,
        grcount: out.grcount,
        convergence: out.fail,
        message: None,
    })
}

/// `optimhess(par, fn, gr, control)` with an analytic gradient: central differences of
/// the gradient with steps `ndeps / parscale`, then symmetrised. Returns the column-major
/// `npar x npar` matrix (`optimHess()` / `optim(hessian = TRUE)`'s `$hessian`).
pub fn optimhess<G>(par: &[f64], mut gr: G, control: &OptimControl) -> Result<Vec<f64>, OptimError>
where
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let npar = par.len();
    let ps = control.parscale(npar);
    let nd = control.ndeps(npar);
    let fs = control.fnscale;
    let mut ans = vec![0.0; npar * npar];
    let mut dpar: Vec<f64> = (0..npar).map(|i| par[i] / ps[i]).collect();
    let mut df1 = vec![0.0; npar];
    let mut df2 = vec![0.0; npar];
    for i in 0..npar {
        let eps = nd[i] / ps[i];
        dpar[i] += eps;
        fmingr_scaled(&mut gr, &dpar, &mut df1, &ps, fs)?;
        dpar[i] -= 2.0 * eps;
        fmingr_scaled(&mut gr, &dpar, &mut df2, &ps, fs)?;
        for j in 0..npar {
            ans[i * npar + j] = fs * (df1[j] - df2[j]) / (2.0 * eps * ps[i] * ps[j]);
        }
        dpar[i] += eps;
    }
    // now symmetrize
    for i in 0..npar {
        for j in 0..i {
            let tmp = 0.5 * (ans[i * npar + j] + ans[j * npar + i]);
            ans[i * npar + j] = tmp;
            ans[j * npar + i] = tmp;
        }
    }
    Ok(ans)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(x: &[f64]) -> f64 {
        100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1.0 - x[0]) * (1.0 - x[0])
    }
    fn grr(x: &[f64]) -> Vec<f64> {
        vec![-400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]), 200.0 * (x[1] - x[0] * x[0])]
    }
    fn p(s: &str) -> f64 {
        s.parse().unwrap()
    }

    #[test]
    fn bfgs_reproduces_r_optim_on_rosenbrock() {
        // R 4.5.0: optim(c(-1.2,1), fr, grr, method = "BFGS", hessian = TRUE)
        let o = optim_bfgs(&[-1.2, 1.0], fr, grr, &OptimControl::default()).unwrap();
        assert_eq!(o.par, vec![p("0.99999999690491403"), p("0.999999993797419")]);
        assert_eq!(o.value, p("9.5949556437698302e-18"));
        assert_eq!((o.fncount, o.grcount, o.convergence), (110, 43, 0));
        let h = optimhess(&o.par, grr, &OptimControl::default()).unwrap();
        let want = [p("802.00039505280938"), p("-399.99999876196159"), p("-399.99999876196159"), p("200.00000000000017")];
        assert_eq!(h, want.to_vec());
    }

    #[test]
    fn bfgs_and_optimhess_apply_parscale_and_fnscale_like_r() {
        // optim(c(-1.2,1), fr, grr, method = "BFGS", control = list(parscale = c(2, 0.5), fnscale = 3))
        let c = OptimControl { parscale: Some(vec![2.0, 0.5]), fnscale: 3.0, ..Default::default() };
        let o = optim_bfgs(&[-1.2, 1.0], fr, grr, &c).unwrap();
        assert_eq!(o.par, vec![p("1.0000000024774967"), p("1.000000004966787")]);
        assert_eq!(o.value, p("6.1518987440991377e-18"));
        assert_eq!((o.fncount, o.grcount), (55, 28));
        // optimHess(c(0.3, 0.7), fr, grr, control = list(parscale = c(2, 0.5)))
        let c = OptimControl { parscale: Some(vec![2.0, 0.5]), ..Default::default() };
        let h = optimhess(&[0.3, 0.7], grr, &c).unwrap();
        let want = [p("-169.99959999999703"), p("-119.99999999999744"), p("-119.99999999999744"), p("200.00000000000284")];
        assert_eq!(h, want.to_vec());
    }

    #[test]
    fn non_finite_parameters_error_like_r() {
        let r = optim_bfgs(&[f64::NAN, 1.0], fr, grr, &OptimControl::default());
        assert_eq!(r.unwrap_err(), OptimError::NonFinitePar);
    }
}
