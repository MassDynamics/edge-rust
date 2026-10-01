//! `graphics::hist.default(x, breaks = <vector>, plot = FALSE)`, ported from R 4.5.0.
//!
//! Sources: `src/library/graphics/R/hist.R` (`hist.default`) and
//! `src/library/graphics/src/stem.c` (`C_bincount`).
//!
//! Only the explicit-breaks path is ported. DESeq2's `estimateDispersionsPriorVar` always
//! passes `breaks = -20:20/2`, so the `pretty()` / Sturges path that builds breaks from a
//! class count is never reached and is not here.

/// The parts of R's `histogram` object a caller reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    /// Sorted breaks, as given (unfuzzed).
    pub breaks: Vec<f64>,
    /// Bin counts (`length(breaks) - 1`).
    pub counts: Vec<i64>,
    /// `counts / (n * diff(breaks))`, with `n` the number of finite `x`.
    pub density: Vec<f64>,
    /// Bin midpoints.
    pub mids: Vec<f64>,
}

/// R's `median` on a non-empty slice without NAs.
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// `C_bincount` (stem.c).
fn bincount(x: &[f64], breaks: &[f64], right: bool, include_border: bool) -> Vec<i64> {
    let nb1 = breaks.len() - 1;
    let mut count = vec![0i64; nb1];
    for &xi in x {
        if !xi.is_finite() {
            continue;
        }
        let mut lo = 0usize;
        let mut hi = nb1;
        if breaks[lo] <= xi && (xi < breaks[hi] || (xi == breaks[hi] && include_border)) {
            while hi - lo >= 2 {
                let new = (hi + lo) / 2;
                if xi > breaks[new] || (!right && xi == breaks[new]) {
                    lo = new;
                } else {
                    hi = new;
                }
            }
            count[lo] += 1;
        }
    }
    count
}

/// `hist(x, breaks = breaks, plot = FALSE)` with the defaults `include.lowest = TRUE`,
/// `right = TRUE`, `fuzz = 1e-7`.
///
/// Panics if `breaks` has fewer than two values. Returns `Err` with R's message if some
/// finite `x` falls outside the breaks.
pub fn hist(x: &[f64], breaks: &[f64]) -> Result<Histogram, String> {
    hist_full(x, breaks, true, true, 1e-7)
}

/// `hist.default` with `include.lowest`, `right` and `fuzz` given explicitly.
pub fn hist_full(
    x: &[f64],
    breaks: &[f64],
    include_lowest: bool,
    right: bool,
    fuzz: f64,
) -> Result<Histogram, String> {
    assert!(
        breaks.len() > 1,
        "hist: explicit breaks need at least two values"
    );
    let x: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = x.len();
    let mut breaks = breaks.to_vec();
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let nb = breaks.len();
    let h: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();

    let diddle = fuzz
        * if nb > 5 {
            median(&h)
        } else if nb <= 3 {
            let (mn, mx) = x
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            mx - mn
        } else {
            h.iter()
                .copied()
                .filter(|&v| v > 0.0)
                .fold(f64::INFINITY, f64::min)
        };
    let fuzzv: Vec<f64> = if right {
        let mut f = vec![diddle; nb];
        f[0] = if include_lowest { -diddle } else { diddle };
        f
    } else {
        let mut f = vec![-diddle; nb];
        f[nb - 1] = if include_lowest { diddle } else { -diddle };
        f
    };
    let fuzzybreaks: Vec<f64> = breaks.iter().zip(&fuzzv).map(|(b, f)| b + f).collect();
    let counts = bincount(&x, &fuzzybreaks, right, include_lowest);
    let total: i64 = counts.iter().sum();
    if (total as usize) < n {
        return Err("some 'x' not counted; maybe 'breaks' do not span range of 'x'".into());
    }
    let density = counts
        .iter()
        .zip(&h)
        .map(|(&c, &hh)| c as f64 / (n as f64 * hh))
        .collect();
    let mids = (0..nb - 1)
        .map(|i| 0.5 * (breaks[i + 1] + breaks[i]))
        .collect();
    Ok(Histogram {
        breaks,
        counts,
        density,
        mids,
    })
}
