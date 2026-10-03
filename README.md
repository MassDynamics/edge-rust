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

The R references (the corpus, `crates/rnum/tests/data/` and the review probes) come from the
production image `md-flexi-r45-local:latest` (R 4.5.0) run as `linux/amd64` under emulation on an
Apple silicon Mac. Results that depend on x87 long double, such as the near-ties in
`r_as_character.tsv`, are assumed to match native x86-64 production; that has not been checked on
native hardware.

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
- **A one-group fit that does not converge on the first gene of a call (review r3 R3-2, overnight
  r1 M1).** When edgeR's one-group Newton-Raphson runs out of iterations, `fit_one_group_mat`
  (`src/glm.c`) and `average_log_cpm` (`src/compute_cpm.c`) return an output variable they never
  wrote. In practice it still holds the value written for the previous gene in the same call, and
  the port reproduces that (`glm::OneGroup::resolve`, one slot per group in `mglmOneWay`, one in
  `aveLogCPM`). On the first gene of a call R returns stack garbage (6.95e-310 in the oracle),
  and the port returns the last iterate instead. The trigger is any low-depth sample next to
  normal ones, under any normalisation: a library of 1 to 5 counts under "none", or one sample
  thinned to 100 or 1,000 counts under TMM (`tests/edger_cases/k_lib*`, `thin*`, all within
  1e-8 of production; before the emulation they drifted up to 3.5e-7).
