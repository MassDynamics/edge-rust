# edge-rust

edgeR quasi-likelihood differential expression for the Mass Dynamics platform, ported from
MDFlexiComparisons' `runEdgeRPairwiseStats` and `runANOVA`: a Rust core with a Python entry point
(`edge_rust.run`) and no R at run time.

## Build and test

```sh
uv sync
uv run pytest
cargo test --release --workspace
cargo test --workspace   # debug: overflow checks and debug_assert! only run here
```

The golden tests read the count corpus at `~/wd/md-count-golden-corpus` (or `MD_COUNT_CORPUS_DIR`).
The Python golden tests fail without it (`EDGE_RUST_ALLOW_NO_CORPUS=1` skips them instead); the
Rust `glibm_golden` test skips.

## Consumers

deseq2-rust depends on the `rnum` and `edger-core` crates here through a git dependency pinned to
a full commit SHA, with a `file:///Users/...` URL. That URL resolves on the development machine
only; it is deliberate while both repos are local, and must become a hosted git URL (same `rev`)
before either repo is built anywhere else. After a change here that deseq2-rust needs, commit,
then bump the `rev` in deseq2-rust's `Cargo.toml`.

## Known differences from production

- **Omnibus F with control variables (review deseq2 r5, R5-m1).** On rare genes edgeR's
  `glmLRT` null fit stores a deviance below the full model's, which is impossible at the null's
  optimum, so production reports a negative F and PValue 1. The port's F is the correct one (a
  direct `optim` of the null model agrees), and that gene's BH rank then shifts the omnibus
  AdjPValue of the other genes slightly (median 7e-4 relative on the probe). R's value also changes
  with the dcast row order, so it is not reproduced.
- **One-group fits that do not converge (review r3, R3-2).** When edgeR's one-group
  Newton-Raphson runs out of iterations, `fit_one_group_mat` in `src/glm.c` returns an uninitialised
  `double`, in practice the previous gene's coefficient (or stack garbage for the first gene). The
  port returns the last iterate. Under norm "none" with a library of 1 to 5 counts this changes one
  gene's adjusted profile likelihood at one or two grid points of `estimateDisp`, which moves the
  common dispersion by about 1e-4 relative and AveExpr by up to 3.5e-7. Accepted: R's value is
  undefined behaviour and cannot be matched in general.
