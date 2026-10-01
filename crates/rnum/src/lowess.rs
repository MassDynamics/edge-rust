//! Line-for-line port of the two lowess smoothers limma's `loessFit` reaches.
//!
//! 1. R's `src/library/stats/src/lowess.c` (`clowess`, `lowest`) plus the R-level
//!    `stats::lowess()` wrapper (sort by `x`, `delta = 0.01 * diff(range(x))`, `iter = 3`).
//! 2. limma's `src/weighted_lowess.c` (`weighted_lowess`, `find_seeds`, `find_limits`,
//!    `lowess_fit`) plus the R-level `limma::weightedLowess()` wrapper (sort by `x`, the
//!    `npts`-driven default `delta`, results returned in the original order).
//! 3. limma's `loessFit()` (`R/loessFit.R`), the entry point the F-distribution fitters use.
//!
//! Which paths the callers reach (limma 3.66.0):
//! - `fitFDistRobustly` (`R/fitFDistRobustly.R`) calls `loessFit(z, covariate, span=0.4)` with
//!   `weights=NULL`, so it lands in the `lowess(x, y, f=span, iter=iterations-1)` branch.
//! - `fitFDist(x, df1, covariate)` (`R/fitFDist.R`) does NOT call `loessFit` at all: its trend
//!   is `splines::ns(covariate, df=splinedf)` + `lm.fit`. Nothing here serves it.
//! - `loessFit`'s `method="locfit"` and `method="loess"` branches are not ported (neither
//!   caller passes `method`, and the default is `"weightedLowess"`).
//!
//! See `tests/scalar_lowess.rs`.

use std::cmp::Ordering;

use crate::nmath::arith::{fmax2, fmin2};

// ---------------------------------------------------------------------------------------------
// R's stats/src/lowess.c
// ---------------------------------------------------------------------------------------------

/// `fsquare` (lowess.c).
fn fsquare(x: f64) -> f64 {
    x * x
}

/// `fcube` (lowess.c).
fn fcube(x: f64) -> f64 {
    x * x * x
}

/// `lowest` (lowess.c): the weighted local linear fit at `xs`.
///
/// `x`, `y`, `w`, `rw` are 0-based slices; `nleft`/`nright` are the C's 1-based window
/// bounds, so every access is `x[j - 1]`. `w` is the C's `res` scratch array (it is
/// overwritten). Returns the C's `*ok`; `*ys` is only written when it is `true`.
#[allow(clippy::too_many_arguments)]
fn lowest(
    x: &[f64],
    y: &[f64],
    n: usize,
    xs: f64,
    ys: &mut f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
    userw: bool,
    rw: &[f64],
) -> bool {
    let range = x[n - 1] - x[0];
    let h = fmax2(xs - x[nleft - 1], x[nright - 1] - xs);
    let h9 = 0.999 * h;
    let h1 = 0.001 * h;

    /* sum of weights */
    let mut a = 0.;
    let mut j = nleft;
    while j <= n {
        /* compute weights */
        /* (pick up all ties on right) */
        w[j - 1] = 0.;
        let r = (x[j - 1] - xs).abs();
        if r <= h9 {
            if r <= h1 {
                w[j - 1] = 1.;
            } else {
                w[j - 1] = fcube(1. - fcube(r / h));
            }
            if userw {
                w[j - 1] *= rw[j - 1];
            }
            a += w[j - 1];
        } else if x[j - 1] > xs {
            break;
        }
        j += 1;
    }

    /* rightmost pt (may be greater */
    /* than nright because of ties) */
    let nrt = j - 1;
    if a <= 0. {
        return false;
    }

    /* weighted least squares */
    /* make sum of w[j] == 1 */
    for j in nleft..=nrt {
        w[j - 1] /= a;
    }
    if h > 0. {
        a = 0.;

        /*  use linear fit */
        /* weighted center of x values */
        for j in nleft..=nrt {
            a += w[j - 1] * x[j - 1];
        }
        let mut b = xs - a;
        let mut c = 0.;
        for j in nleft..=nrt {
            c += w[j - 1] * fsquare(x[j - 1] - a);
        }
        if c.sqrt() > 0.001 * range {
            b /= c;

            /* points are spread out */
            /* enough to compute slope */
            for j in nleft..=nrt {
                w[j - 1] *= b * (x[j - 1] - a) + 1.;
            }
        }
    }
    *ys = 0.;
    for j in nleft..=nrt {
        *ys += w[j - 1] * y[j - 1];
    }
    true
}

/// `clowess` (R `src/library/stats/src/lowess.c`): Cleveland's lowess on `x` already sorted
/// ascending, `y` alongside. `f` is the smoother span, `nsteps` the number of robustness
/// iterations (`iter` in R), `delta` the interpolation distance. Returns the C's `ys`.
///
/// The C partial-sorts `rw` in place with `rPsort` only to read the one or two middle values
/// for `cmad`; a full sort of a copy yields the same values and `rw` is rebuilt from `res`
/// right after, so nothing observable differs.
pub fn clowess(x: &[f64], y: &[f64], f: f64, nsteps: usize, delta: f64) -> Vec<f64> {
    let n = x.len();
    assert_eq!(y.len(), n, "clowess: x and y have different lengths");
    assert!(n > 0, "clowess: invalid input (n == 0)");
    let mut ys = vec![0.; n];

    if n < 2 {
        ys[0] = y[0];
        return ys;
    }
    let mut rw = vec![0.; n];
    let mut res = vec![0.; n];

    /* at least two, at most n points */
    let ns = ((f * n as f64 + 1e-7) as usize).min(n).max(2);

    /* robustness iterations */
    let mut iter = 1;
    while iter <= nsteps + 1 {
        let mut nleft = 1;
        let mut nright = ns;
        let mut last = 0; /* index of prev estimated point */
        let mut i = 1; /* index of current point */

        loop {
            if nright < n {
                /* move nleft,  nright to right */
                /* if radius decreases */
                let d1 = x[i - 1] - x[nleft - 1];
                let d2 = x[nright] - x[i - 1];

                /* if d1 <= d2 with */
                /* x[nright+1] == x[nright], */
                /* lowest fixes */
                if d1 > d2 {
                    /* radius will not */
                    /* decrease by */
                    /* move right */
                    nleft += 1;
                    nright += 1;
                    continue;
                }
            }

            /* fitted value at x[i] */
            let mut ysi = 0.;
            let ok = lowest(
                x,
                y,
                n,
                x[i - 1],
                &mut ysi,
                nleft,
                nright,
                &mut res,
                iter > 1,
                &rw,
            );
            /* all weights zero */
            /* copy over value (all rw==0) */
            ys[i - 1] = if ok { ysi } else { y[i - 1] };

            if last < i - 1 {
                let denom = x[i - 1] - x[last - 1];

                /* skipped points -- interpolate */
                /* non-zero - proof? */
                for j in last + 1..i {
                    let alpha = (x[j - 1] - x[last - 1]) / denom;
                    ys[j - 1] = alpha * ys[i - 1] + (1. - alpha) * ys[last - 1];
                }
            }

            /* last point actually estimated */
            last = i;

            /* x coord of close points */
            let cut = x[last - 1] + delta;
            i = last + 1;
            while i <= n {
                if x[i - 1] > cut {
                    break;
                }
                if x[i - 1] == x[last - 1] {
                    ys[i - 1] = ys[last - 1];
                    last = i;
                }
                i += 1;
            }
            i = (last + 1).max(i - 1);
            if last >= n {
                break;
            }
        }
        /* residuals */
        for (r, (yi, ysi)) in res.iter_mut().zip(y.iter().zip(&ys)) {
            *r = yi - ysi;
        }

        /* overall scale estimate */
        let mut sc = 0.;
        for r in &res {
            sc += r.abs();
        }
        sc /= n as f64;

        /* compute robustness weights */
        /* except last time */
        if iter > nsteps {
            break;
        }
        /* Note: The following code, biweight_{6 MAD|Ri|}
        is also used in stl(), loess and several other places.
        --> should provide API here (MM) */
        for (rwi, r) in rw.iter_mut().zip(&res) {
            *rwi = r.abs();
        }

        /* Compute   cmad := 6 * median(rw[], n)  ---- */
        let m1 = n / 2;
        /* partial sort, for m1 & m2 */
        let mut sorted = rw.clone();
        sorted.sort_by(f64::total_cmp);
        let cmad = if n % 2 == 0 {
            let m2 = n - m1 - 1;
            3. * (sorted[m1] + sorted[m2])
        } else {
            /* n odd */
            6. * sorted[m1]
        };
        if cmad < 1e-7 * sc {
            /* effectively zero */
            break;
        }
        let c9 = 0.999 * cmad;
        let c1 = 0.001 * cmad;
        for (rwi, res_i) in rw.iter_mut().zip(&res) {
            let r = res_i.abs();
            *rwi = if r <= c1 {
                1.
            } else if r <= c9 {
                fsquare(1. - fsquare(r / cmad))
            } else {
                0.
            };
        }
        iter += 1;
    }
    ys
}

/// R's `order(x)` for a double vector: a stable ascending permutation (ties keep input
/// order, `NaN` last), as the default radix method gives. `-0.0` and `0.0` compare equal.
fn order(x: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..x.len()).collect();
    o.sort_by(|&a, &b| match (x[a].is_nan(), x[b].is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => x[a].partial_cmp(&x[b]).unwrap_or(Ordering::Equal),
    });
    o
}

/// R's `diff(range(x))`: `max(x) - min(x)`, NaN if any element is NaN.
fn range_width(x: &[f64]) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in x {
        lo = fmin2(lo, v);
        hi = fmax2(hi, v);
    }
    hi - lo
}

/// `stats::lowess(x, y, f, iter, delta)` (R `src/library/stats/R/lowess.R`): orders by `x`
/// (stable), permutes `y` alongside, defaults `delta` to `0.01 * diff(range(x))`, and runs
/// [`clowess`]. Returns `(x sorted, fitted)` exactly like the R list's `$x`, `$y`.
pub fn lowess(
    x: &[f64],
    y: &[f64],
    f: f64,
    iter: usize,
    delta: Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    assert_eq!(y.len(), n, "lowess: x and y have different lengths");
    assert!(n > 0, "lowess: invalid input (n == 0)");
    assert!(
        f.is_finite() && f > 0.,
        "lowess: 'f' must be finite and > 0"
    );
    let o = order(x);
    let xs: Vec<f64> = o.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = o.iter().map(|&i| y[i]).collect();
    let delta = delta.unwrap_or_else(|| 0.01 * range_width(&xs));
    assert!(
        delta.is_finite() && delta >= 0.,
        "lowess: 'delta' must be finite and > 0"
    );
    let fit = clowess(&xs, &ys, f, iter, delta);
    (xs, fit)
}

// ---------------------------------------------------------------------------------------------
// limma's src/weighted_lowess.c
// ---------------------------------------------------------------------------------------------

/// `THRESHOLD` (weighted_lowess.c).
const THRESHOLD: f64 = 0.0000001;

/// `find_seeds` (weighted_lowess.c): the indices of the points actually fitted, given
/// `delta`. The first and last point are always included.
fn find_seeds(x: &[f64], delta: f64) -> Vec<usize> {
    let npts = x.len();
    let mut idx = vec![0usize];
    let mut last_pt = 0;
    for pt in 1..npts - 1 {
        if x[pt] - x[last_pt] > delta {
            idx.push(pt);
            last_pt = pt;
        }
    }
    idx.push(npts - 1);
    idx
}

/// `find_limits` (weighted_lowess.c): for each seed, the start and end index of its span
/// and the maximum distance within it. Returns `(start, end, dist)`.
///
/// "We don't use the update-based algorithm in Cleveland's paper, as it ceases to be
/// numerically stable once you throw in double-precision weights."
fn find_limits(
    indices: &[usize],
    x: &[f64],
    w: &[f64],
    spanweight: f64,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let npts = x.len();
    let num = indices.len();
    let mut spbegin = Vec::with_capacity(num);
    let mut spend = Vec::with_capacity(num);
    let mut spdist = Vec::with_capacity(num);

    for &curpt in indices {
        let mut left = curpt;
        let mut right = curpt;
        let mut curw = w[curpt];
        let mut ende = curpt == npts - 1;
        let mut ends = curpt == 0;
        let mut mdist = 0.;

        while curw < spanweight && (!ende || !ends) {
            if ende {
                /* Can only extend backwards. */
                left -= 1;
                curw += w[left];
                if left == 0 {
                    ends = true;
                }
                let ldist = x[curpt] - x[left];
                if mdist < ldist {
                    mdist = ldist;
                }
            } else if ends {
                /* Can only extend forwards. */
                right += 1;
                curw += w[right];
                if right == npts - 1 {
                    ende = true;
                }
                let rdist = x[right] - x[curpt];
                if mdist < rdist {
                    mdist = rdist;
                }
            } else {
                /* Can do either; extending by the one that minimizes the curpt mdist. */
                let ldist = x[curpt] - x[left - 1];
                let rdist = x[right + 1] - x[curpt];
                if ldist < rdist {
                    left -= 1;
                    curw += w[left];
                    if left == 0 {
                        ends = true;
                    }
                    if mdist < ldist {
                        mdist = ldist;
                    }
                } else {
                    right += 1;
                    curw += w[right];
                    if right == npts - 1 {
                        ende = true;
                    }
                    if mdist < rdist {
                        mdist = rdist;
                    }
                }
            }
        }

        /* Extending to ties. */
        while left > 0 && x[left] == x[left - 1] {
            left -= 1;
        }
        while right < npts - 1 && x[right] == x[right + 1] {
            right += 1;
        }

        /* Recording */
        spbegin.push(left);
        spend.push(right);
        spdist.push(mdist);
    }

    (spbegin, spend, spdist)
}

/// `lowess_fit` (weighted_lowess.c): the local linear fit at `curpt` over `[left, right]`
/// with tricube x prior x robustness weights. `work` is the C's `rsdptr` "holding cell".
#[allow(clippy::too_many_arguments)]
fn lowess_fit(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    rw: &[f64],
    curpt: usize,
    left: usize,
    right: usize,
    dist: f64,
    work: &mut [f64],
) -> f64 {
    let mut ymean = 0.;
    let mut allweight = 0.;
    if dist < THRESHOLD {
        for pt in left..=right {
            work[pt] = w[pt] * rw[pt];
            ymean += y[pt] * work[pt];
            allweight += work[pt];
        }
        ymean /= allweight;
        return ymean;
    }
    let mut xmean = 0.;
    for pt in left..=right {
        work[pt] = (1. - ((x[curpt] - x[pt]).abs() / dist).powf(3.0)).powf(3.0) * w[pt] * rw[pt];
        xmean += work[pt] * x[pt];
        ymean += work[pt] * y[pt];
        allweight += work[pt];
    }
    xmean /= allweight;
    ymean /= allweight;

    let mut var = 0.;
    let mut covar = 0.;
    for pt in left..=right {
        let temp = x[pt] - xmean;
        var += temp * temp * work[pt];
        covar += temp * (y[pt] - ymean) * work[pt];
    }
    if var < THRESHOLD {
        return ymean;
    }

    let slope = covar / var;
    let intercept = ymean - slope * xmean;
    slope * x[curpt] + intercept
}

/// `rcmp(x, y, nalast=TRUE)` (R `src/main/sort.c`).
fn rcmp(x: f64, y: f64) -> Ordering {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            if x < y {
                Ordering::Less
            } else if x > y {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }
}

/// `rsort_with_index` (R `src/main/sort.c`): Shell sort of `x` ascending carrying `indx`
/// along. Ported rather than replaced by a library sort because it is not stable, and the
/// order of tied residuals decides which index the cumulative weight crosses `halfweight` at.
fn rsort_with_index(x: &mut [f64], indx: &mut [usize]) {
    let n = x.len();
    let mut h = 1;
    while h <= n / 9 {
        h = 3 * h + 1;
    }
    while h > 0 {
        for i in h..n {
            let v = x[i];
            let iv = indx[i];
            let mut j = i;
            while j >= h && rcmp(x[j - h], v) == Ordering::Greater {
                x[j] = x[j - h];
                indx[j] = indx[j - h];
                j -= h;
            }
            x[j] = v;
            indx[j] = iv;
        }
        h /= 3;
    }
}

/// `weighted_lowess` (limma `src/weighted_lowess.c`): Cleveland's lowess with prior weights
/// folded into the span and the regression. `x` must be sorted ascending with `y` and `w`
/// alongside; `iterations` (the C's `iter`) is the total number of fits, robustness weights
/// being recomputed after every one. Returns `(fitted, robustness weights)`, the C's
/// two-element list.
pub fn weighted_lowess(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    span: f64,
    iterations: usize,
    delta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let npts = x.len();
    assert!(
        npts == y.len() && npts == w.len(),
        "weighted_lowess: weight, covariate and response vectors have unequal lengths"
    );
    assert!(npts >= 2, "weighted_lowess: need at least two points");
    assert!(
        iterations > 0,
        "weighted_lowess: number of robustness iterations should be positive"
    );
    let niter = iterations;
    let spv = span;
    let dv = delta;

    /* Computing the span weight that each span must achieve. */
    let mut totalweight = 0.;
    for &wi in w {
        totalweight += wi;
    }
    let spanweight = totalweight * spv;
    let subrange = (x[npts - 1] - x[0]) / npts as f64;

    /* Setting up the indices of points for sampling; the frame start and end for those indices, and the max dist. */
    let seed_index = find_seeds(x, dv);
    let nseeds = seed_index.len();
    let (frame_start, frame_end, max_dist) = find_limits(&seed_index, x, w, spanweight);

    /* Setting up arrays to hold the fitted values, residuals and robustness weights. */
    let mut fit = vec![0.; npts];
    let mut rsd = vec![0.; npts];
    let mut rob = vec![1.; npts];
    let mut ror = vec![0usize; npts];

    /* Robustness iterations. */
    for _it in 0..niter {
        let mut last_pt = 0;

        /* Computing fitted values for seed points, and interpolating to the intervening points. */
        fit[0] = lowess_fit(
            x,
            y,
            w,
            &rob,
            0,
            frame_start[0],
            frame_end[0],
            max_dist[0],
            &mut rsd,
        );
        for cur_seed in 1..nseeds {
            let pt = seed_index[cur_seed];
            fit[pt] = lowess_fit(
                x,
                y,
                w,
                &rob,
                pt,
                frame_start[cur_seed],
                frame_end[cur_seed],
                max_dist[cur_seed],
                &mut rsd,
            ); /* using rsdptr as a holding cell. */

            if pt - last_pt > 1 {
                /* Some protection is provided against infinite slopes. This shouldn't be
                 * a problem for non-zero delta; the only concern is at the final point
                 * where the covariate distance may be zero. Besides, if delta is not
                 * positive, pt-last_pt could never be 1 so we'd never reach this point.
                 */
                let current = x[pt] - x[last_pt];
                if current > THRESHOLD * subrange {
                    let slope = (fit[pt] - fit[last_pt]) / current;
                    let intercept = fit[pt] - slope * x[pt];
                    for subpt in last_pt + 1..pt {
                        fit[subpt] = slope * x[subpt] + intercept;
                    }
                } else {
                    let endave = 0.5 * (fit[pt] + fit[last_pt]);
                    for f in &mut fit[last_pt + 1..pt] {
                        *f = endave;
                    }
                }
            }
            last_pt = pt;
        }

        /* Computing the weighted MAD of the absolute values of the residuals. */
        let mut resid_scale = 0.;
        for pt in 0..npts {
            rsd[pt] = (y[pt] - fit[pt]).abs();
            resid_scale += rsd[pt];
            ror[pt] = pt;
        }
        resid_scale /= npts as f64;
        rsort_with_index(&mut rsd, &mut ror);

        let mut current = 0.;
        let mut cmad = 0.;
        let halfweight = totalweight / 2.;
        for pt in 0..npts {
            current += w[ror[pt]];
            if current == halfweight {
                /* In the unlikely event of an exact match. */
                // `rsd[pt + 1]` is past the end only when the exact match lands on the last
                // point, i.e. totalweight == 0, where the fit is already NaN; the C reads
                // out of bounds there, this reads the last value instead of panicking.
                let next = if pt + 1 < npts { rsd[pt + 1] } else { rsd[pt] };
                cmad = 3. * (rsd[pt] + next);
                break;
            } else if current > halfweight {
                cmad = 6. * rsd[pt];
                break;
            }
        }

        /* If it's too small, then robustness weighting will have no further effect.
         * Any points with large residuals would already be pretty lowly weighted.
         * This is based on a similar step in lowess.c in the core R code.
         */
        if cmad <= THRESHOLD * resid_scale {
            break;
        }

        /* Computing the robustness weights. */
        for pt in 0..npts {
            if rsd[pt] < cmad {
                rob[ror[pt]] = (1. - (rsd[pt] / cmad).powf(2.0)).powf(2.0);
            } else {
                rob[ror[pt]] = 0.;
            }
        }
    }

    (fit, rob)
}

/// `limma::weightedLowess(x, y, weights, delta, npts, span, iterations)` (`R/weightedLowess.R`)
/// with the default `output.style="loess"`: orders by `x` (stable), derives `delta` from
/// `npts` when not given, runs [`weighted_lowess`], and returns `$fitted` in the ORIGINAL
/// `x` order.
///
/// The default `delta` follows the R exactly: `dx <- sort(diff(x)); cumrange <- cumsum(dx);
/// delta <- min(cumrange[length(dx) - k] / (npts - k))` over `k = 0..npts-1`, and `0` when
/// `npts >= length(x)`. R's `cumsum` accumulates in `long double`, which is plain `double`
/// on arm64 macOS (the corpus machine); this accumulates in `f64`.
pub fn weighted_lowess_r(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    span: f64,
    iterations: usize,
    delta: Option<f64>,
    npts: usize,
) -> Vec<f64> {
    let n = x.len();
    assert_eq!(
        y.len(),
        n,
        "weightedLowess: x and y should have same length"
    );
    assert_eq!(
        weights.len(),
        n,
        "weightedLowess: weights should have same length as x and y"
    );
    let o = order(x);
    let xs: Vec<f64> = o.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = o.iter().map(|&i| y[i]).collect();
    let ws: Vec<f64> = o.iter().map(|&i| weights[i]).collect();

    let delta = match delta {
        Some(d) => d,
        None => {
            assert!(
                npts >= 1,
                "weightedLowess: number of points should be a positive integer"
            );
            if npts >= n {
                0.
            } else {
                let mut dx: Vec<f64> = xs.windows(2).map(|p| p[1] - p[0]).collect();
                dx.sort_by(f64::total_cmp);
                let mut cumrange = Vec::with_capacity(dx.len());
                let mut sum = 0.;
                for d in &dx {
                    sum += d;
                    cumrange.push(sum);
                }
                let m = dx.len();
                let mut delta = f64::INFINITY;
                for k in 0..npts {
                    // R's 1-based `cumrange[length(dx) - k]` is 0-based `cumrange[m - k - 1]`.
                    let v = cumrange[m - k - 1] / (npts - k) as f64;
                    delta = fmin2(delta, v);
                }
                delta
            }
        }
    };

    let (fit_sorted, _robust_weights) = weighted_lowess(&xs, &ys, &ws, span, iterations, delta);

    /* Output in the original order, as for loess() or loessFit() */
    let mut fitted = vec![0.; n];
    for (k, &i) in o.iter().enumerate() {
        fitted[i] = fit_sorted[k];
    }
    fitted
}

// ---------------------------------------------------------------------------------------------
// limma's R/loessFit.R
// ---------------------------------------------------------------------------------------------

/// `lm.wfit(cbind(1, x), y, w)$fitted` for the two-column intercept + slope design: the
/// weighted least-squares line, or the weighted mean when `x` has no spread. Algebraically
/// what R's Householder QR returns, not a port of `dqrls`, so it agrees with R to rounding only.
/// `loessFit` takes this branch when fewer than `4 + 1/span` weighted observations remain, e.g.
/// `fitFDistUnequalDF1` with a trend covariate and only three or four informative features.
fn lm_wfit_line(x: &[f64], y: &[f64], w: &[f64]) -> Vec<f64> {
    let mut sw = 0.;
    let mut swx = 0.;
    let mut swy = 0.;
    for i in 0..x.len() {
        sw += w[i];
        swx += w[i] * x[i];
        swy += w[i] * y[i];
    }
    let xbar = swx / sw;
    let ybar = swy / sw;
    let mut sxx = 0.;
    let mut sxy = 0.;
    for i in 0..x.len() {
        let dx = x[i] - xbar;
        sxx += w[i] * dx * dx;
        sxy += w[i] * dx * (y[i] - ybar);
    }
    if sxx > 0. {
        let slope = sxy / sxx;
        x.iter().map(|&xi| ybar + slope * (xi - xbar)).collect()
    } else {
        vec![ybar; x.len()]
    }
}

/// `limma::loessFit(y, x, weights, span, iterations)` (`R/loessFit.R`) with the defaults
/// `min.weight=1e-5`, `max.weight=1e5`, `equal.weights.as.null=TRUE`,
/// `method="weightedLowess"`. Returns `$fitted` in the original order; positions where `y`
/// or `x` is not finite come back `NaN` (R's `NA`).
///
/// Branches, in the R's order:
/// - no finite observations: all `NaN`;
/// - `span < 1/nobs`: fitted = `y`;
/// - `weights` given: `NA` -> 0, clamped to `[min.weight, max.weight]`, then treated as
///   `NULL` when their range is `< 1e-15`;
/// - `weights == NULL`: `lowess(xobs, yobs, f=span, iter=iterations-1)` with R's default
///   `delta`, put back through `order(xobs)`;
/// - `weights` given and fewer than `4 + 1/span` observations: one observation copies `y`,
///   otherwise `lm.wfit(cbind(1, x), y, w)`;
/// - `weights` given: `weightedLowess(xobs, yobs, wobs, span, iterations, npts=200)`.
///
/// `method="locfit"` and `method="loess"` are not ported (never selected by limma's callers).
pub fn loess_fit(
    y: &[f64],
    x: &[f64],
    weights: Option<&[f64]>,
    span: f64,
    iterations: usize,
) -> Vec<f64> {
    loess_fit_bounded(y, x, weights, span, iterations, 1e-5, 1e5)
}

/// [`loess_fit`] with explicit `min.weight` / `max.weight`, which `fitFDistUnequalDF1`
/// sets to `1e-8` / `1e2`.
pub fn loess_fit_bounded(
    y: &[f64],
    x: &[f64],
    weights: Option<&[f64]>,
    span: f64,
    iterations: usize,
    min_weight: f64,
    max_weight: f64,
) -> Vec<f64> {
    let n = y.len();
    assert_eq!(x.len(), n, "loessFit: y and x have different lengths");
    let mut fitted = vec![f64::NAN; n];

    let obs: Vec<usize> = (0..n)
        .filter(|&i| y[i].is_finite() && x[i].is_finite())
        .collect();
    let xobs: Vec<f64> = obs.iter().map(|&i| x[i]).collect();
    let yobs: Vec<f64> = obs.iter().map(|&i| y[i]).collect();
    let nobs = yobs.len();

    /* If no good obs, exit straight away */
    if nobs == 0 {
        return fitted;
    }

    /* Check span */
    if span < 1. / nobs as f64 {
        for &i in &obs {
            fitted[i] = y[i];
        }
        return fitted;
    }

    /* Check weights */
    let mut wobs: Option<Vec<f64>> = None;
    if let Some(weights) = weights {
        assert_eq!(
            weights.len(),
            n,
            "loessFit: y and weights have different lengths"
        );
        let w: Vec<f64> = obs
            .iter()
            .map(|&i| {
                let mut v = weights[i];
                if v.is_nan() {
                    v = 0.;
                }
                if v < min_weight {
                    v = min_weight;
                }
                if v > max_weight {
                    v = max_weight;
                }
                v
            })
            .collect();
        /* If weights all equal, treat as NULL */
        if range_width(&w) < 1e-15 {
            wobs = None;
        } else {
            wobs = Some(w);
        }
    }

    /* If no weights, so use classic lowess algorithm */
    let Some(wobs) = wobs else {
        assert!(iterations >= 1, "lowess: 'iter' must be finite and >= 0");
        let o = order(&xobs);
        let (_x_sorted, lo_y) = lowess(&xobs, &yobs, span, iterations - 1, None);
        /* out$fitted[obs][o] <- lo$y */
        for (k, &oi) in o.iter().enumerate() {
            fitted[obs[oi]] = lo_y[k];
        }
        return fitted;
    };

    /* Count number of observations with positive weights (must always be positive) */
    // min.weight > 0, so nwobs <- nobs.
    let nwobs = nobs;

    /* Check whether too few obs to estimate lowess curve */
    if (nwobs as f64) < 4. + 1. / span {
        if nwobs == 1 {
            fitted[obs[0]] = yobs[0];
        } else {
            let fit = lm_wfit_line(&xobs, &yobs, &wobs);
            for (k, &i) in obs.iter().enumerate() {
                fitted[i] = fit[k];
            }
        }
        return fitted;
    }

    /* Need to compute lowess with unequal weights: method = "weightedLowess" */
    let fit = weighted_lowess_r(&xobs, &yobs, &wobs, span, iterations, None, 200);
    for (k, &i) in obs.iter().enumerate() {
        fitted[i] = fit[k];
    }
    fitted
}
