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
Rust `glibm_golden` test skips. CI runs the same tests against `tests/corpus-small`, ten whole
count_synth runs copied from the full corpus by `scripts/build_small_corpus.py`.

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
- **A one-group fit that does not converge before any gene of its call has been written (review
  r3 R3-2, overnight r1 M1 and r2).** When edgeR's one-group Newton-Raphson runs out of
  iterations, `fit_one_group_mat` (`src/glm.c`) and `average_log_cpm` (`src/compute_cpm.c`)
  return an output variable they never wrote. In practice it still holds the last value written
  in the same call, that of the last converged or all-zero gene, which need not be the previous
  gene, and the port reproduces that (`glm::OneGroup::resolve`, one slot per group in
  `mglmOneWay`, one in `aveLogCPM`). For every non-converged gene before the first written gene
  of a call R returns stack garbage (6.95e-310 in the oracle), and the port returns the last
  iterate instead; on the probe where this happens end to end, AveExpr differs by up to 2.4e-7.
  A call is one group of one zero-pattern subgroup (`.comboGroups`) at one grid point of
  `estimateDisp`, one group of one one-way `glmFit`, or one `aveLogCPM`. The two `glmFit` kinds
  (the `estimateDisp` grid fits go through `glmFit` too, `adjustedProfileLik.R:29`) exist only
  for a one-way design (`mglmOneWay`); any other design goes to `mglmLevenberg`, which has no
  such slot. So the result depends on the GroupId order, which the port takes from production
  (numeric for integer ids). The order
  matters without a stall too: it moves df.prior by up to 2e-9 relative, F by up to 2e-10
  relative and by about 1e-12 absolute near 0, and since SE = |Log2FC| / sqrt(F), a gene with F
  near 0 can see SE and the CIs change by any amount (SE C - B on `twozero_ctl` gene 16 fell
  14.1-fold, from 106,798 to 7,560) or stat flip between a number and NA. Either kind of stall
  moves the result by however far the last written gene is from the stalled one, so neither has
  a fixed size. A stall inside `estimateDisp` moves the common dispersion, and through it every
  gene's AveExpr, df.prior and F: by about 3e-8 on `k_lib1_reid_none` (review loop r1, R Minor
  1), but on `k_alc_none` with gene 88 renamed 99999 the common dispersion moves 1.4% (0.20156 to
  0.19872), df.prior 0.6%, and for the genes other than 89 AveExpr by up to 0.023 (gene 211) and
  F by up to 14% where F >= 0.1, jointly with the `aveLogCPM` stall below. The port matches R on
  both orders (AveExpr within 2e-15, F within 7e-13 relative; review loop r3, R Minor 1). A stall
  in `aveLogCPM` gives the gene the AveExpr of the last written gene outright, so a reorder can
  move it by whole log2-CPM units: on `k_alc_none` gene 89 takes gene
  88's 8.558757, and with gene 88 renamed 99999 R gives gene 89 10.186701565417202, bit-identical
  to gene 87 (oracle run 2026-10-04, review loop r2, R Minor 1). The trigger is any low-depth sample
  next to normal ones, under any normalisation: a library of 1 to 5 counts under "none", or one
  sample thinned to 100 or 1,000 counts under TMM (`tests/edger_cases/k_lib*`, `thin*`, all
  within 1e-8 of production; before the emulation they drifted up to 3.5e-7). None of the 25
  edgeR corpus runs that reach the fit has a non-converged gene (counted on the port, whose
  stalls match R's on every logged probe).
