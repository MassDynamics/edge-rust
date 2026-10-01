//! Gaussian quadrature as `fitFDistRobustly` uses it: `statmod::gauss.quad.prob(128,
//! dist = "uniform")` (`fitFDistRobustly.R:121`), whose nodes and weights come from the
//! Fortran `gausq2` in `statmod/src/gaussq2.f` (a modified EISPACK `imtql2`: eigenvalues and
//! first eigenvector components of a symmetric tridiagonal matrix by the implicit QL
//! method). Both are ported here; the R wrapper is `statmod/R/gaussquad.R`.

/// `gausq2(n, d, e, z, ierr)`: eigenvalues (into `d`, ascending) and the first component of
/// each orthonormal eigenvector (into `z`) of the symmetric tridiagonal matrix with diagonal
/// `d` and sub-diagonal `e[0..n-1]`. `z` must hold the first row of the identity on entry.
/// Returns `ierr`: 0 on success, or `j` if the `j`-th eigenvalue did not converge in 30
/// iterations.
#[allow(clippy::many_single_char_names)]
pub fn gausq2(n: usize, d: &mut [f64], e: &mut [f64], z: &mut [f64]) -> usize {
    let machep = 2.0f64.powi(-52);
    if n == 1 {
        return 0;
    }
    e[n - 1] = 0.0;
    // Fortran is 1-based; `l`, `m`, `i` below are 1-based and indexed with `- 1`.
    for l in 1..=n {
        let mut j = 0;
        loop {
            // look for small sub-diagonal element
            let mut m = l;
            while m < n {
                if e[m - 1].abs() <= machep * (d[m - 1].abs() + d[m].abs()) {
                    break;
                }
                m += 1;
            }
            let mut p = d[l - 1];
            if m == l {
                break;
            }
            if j == 30 {
                // set error -- no convergence to an eigenvalue after 30 iterations
                return l;
            }
            j += 1;
            // form shift
            let mut g = (d[l] - p) / (2.0 * e[l - 1]);
            let mut r = (g * g + 1.0).sqrt();
            g = d[m - 1] - p + e[l - 1] / (g + r.copysign(g));
            let mut s = 1.0;
            let mut c = 1.0;
            p = 0.0;
            let mml = m - l;
            // for i=m-1 step -1 until l do --
            for ii in 1..=mml {
                let i = m - ii;
                let f = s * e[i - 1];
                let b = c * e[i - 1];
                if f.abs() < g.abs() {
                    s = f / g;
                    r = (s * s + 1.0).sqrt();
                    e[i] = g * r;
                    c = 1.0 / r;
                    s *= c;
                } else {
                    c = g / f;
                    r = (c * c + 1.0).sqrt();
                    e[i] = f * r;
                    s = 1.0 / r;
                    c *= s;
                }
                g = d[i] - p;
                r = (d[i - 1] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i] = g + p;
                g = c * r - b;
                // form first component of vector
                let f = z[i];
                z[i] = s * z[i - 1] + c * f;
                z[i - 1] = c * z[i - 1] - s * f;
            }
            d[l - 1] -= p;
            e[l - 1] = g;
            e[m - 1] = 0.0;
        }
    }
    // order eigenvalues and eigenvectors
    for ii in 2..=n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d[i - 1];
        for j in ii..=n {
            if d[j - 1] >= p {
                continue;
            }
            k = j;
            p = d[j - 1];
        }
        if k == i {
            continue;
        }
        d[k - 1] = d[i - 1];
        d[i - 1] = p;
        z.swap(i - 1, k - 1);
    }
    0
}

/// Nodes and weights of `gauss.quad.prob(n, dist = "uniform", l = 0, u = 1)`: the Gauss-
/// Legendre rule on `[0, 1]` with weights summing to one.
pub fn gauss_quad_prob_uniform(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![0.5], vec![1.0]);
    }
    let mut a = vec![0.0; n];
    let mut b: Vec<f64> = (1..n)
        .map(|i| {
            let i = i as f64;
            i / (4.0 * i * i - 1.0).sqrt()
        })
        .collect();
    b.push(0.0);
    let mut z = vec![0.0; n];
    z[0] = 1.0;
    let ierr = gausq2(n, &mut a, &mut b, &mut z);
    debug_assert_eq!(ierr, 0, "gausq2 did not converge");
    let weights: Vec<f64> = z.iter().map(|w| w * w).collect();
    let nodes: Vec<f64> = a.iter().map(|x| (x + 1.0) / 2.0).collect();
    (nodes, weights)
}

/// `chooseLowessSpan(n, small.n, min.span, power)` (`limma/R/chooseLowessSpan.R`):
/// `pmin(min.span + (1 - min.span) * (small.n / n)^power, 1)`.
pub fn choose_lowess_span(n: usize, small_n: f64, min_span: f64, power: f64) -> f64 {
    let span = min_span + (1.0 - min_span) * (small_n / n as f64).powf(power);
    span.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn two_and_three_point_rules_match_the_textbook_legendre_values() {
        // R: gauss.quad.prob(2) -> nodes 0.2113249 0.7886751, weights 0.5 0.5
        let (x, w) = gauss_quad_prob_uniform(2);
        let s = 1.0 / 12f64.sqrt();
        assert!(
            close(x[0], 0.5 - s, 1e-14) && close(x[1], 0.5 + s, 1e-14),
            "{x:?}"
        );
        assert!(close(w[0], 0.5, 1e-14) && close(w[1], 0.5, 1e-14), "{w:?}");
        // R: gauss.quad.prob(3) -> nodes 0.1127017 0.5 0.8872983, weights 0.2777778 0.4444444 0.2777778
        let (x, w) = gauss_quad_prob_uniform(3);
        let s = 0.6f64.sqrt() / 2.0;
        assert!(
            close(x[0], 0.5 - s, 1e-14) && close(x[1], 0.5, 1e-14) && close(x[2], 0.5 + s, 1e-14)
        );
        assert!(
            close(w[0], 5.0 / 18.0, 1e-14)
                && close(w[1], 4.0 / 9.0, 1e-14)
                && close(w[2], 5.0 / 18.0, 1e-14)
        );
    }

    #[test]
    fn the_128_point_rule_is_symmetric_normalised_and_exact_on_polynomials() {
        let (x, w) = gauss_quad_prob_uniform(128);
        assert_eq!(x.len(), 128);
        assert!(x.windows(2).all(|p| p[0] < p[1]), "nodes ascending");
        assert!(x[0] > 0.0 && x[127] < 1.0);
        for i in 0..64 {
            assert!(close(x[i] + x[127 - i], 1.0, 1e-13));
            assert!(close(w[i], w[127 - i], 1e-12));
        }
        assert!(close(w.iter().sum::<f64>(), 1.0, 1e-14));
        // int_0^1 x^k dx = 1/(k+1), exact for k <= 255.
        for k in [1, 2, 7, 50, 200] {
            let got: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(k)).sum();
            assert!(close(got, 1.0 / (k as f64 + 1.0), 1e-12), "k={k}: {got}");
        }
    }

    #[test]
    fn choose_lowess_span_matches_the_limma_formula() {
        // R: chooseLowessSpan(1000, small.n=500) -> 0.3 + 0.7*(0.5)^(1/3) = 0.8555...
        let s = choose_lowess_span(1000, 500.0, 0.3, 1.0 / 3.0);
        assert!(close(s, 0.3 + 0.7 * 0.5f64.powf(1.0 / 3.0), 1e-15));
        assert_eq!(choose_lowess_span(10, 500.0, 0.3, 1.0 / 3.0), 1.0);
    }
}
