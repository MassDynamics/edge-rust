# Inputs for the two golden-gap runs (review r1, stats e): interior df.prior, a few % of genes
# with df.residual.adj = 0 (gap_zero, no control) or 0 < df.residual.adj < 0.01 (gap_tiny,
# numeric control). Run from this directory: Rscript make_gap_inputs.R
set.seed(11)
ng <- 700; cond <- c(rep("A", 4), rep("B", 4), "C"); n <- length(cond)
mu <- 2^runif(ng, 4, 11); fc <- matrix(1, ng, n); fc[sample(ng, ng %/% 10), cond == "B"] <- 3
phi <- 0.08 * 10 / rchisq(ng, 10)
base <- matrix(rnbinom(ng * n, mu = mu * fc, size = 1 / phi), ng, n)
samples <- sprintf("s%02d", seq_len(n))
write_case <- function(dir, cm, control = NULL) {
  dir.create(dir, showWarnings = FALSE)
  write.csv(data.frame(id = seq_len(ng), cm, check.names = FALSE), file.path(dir, "input_counts.csv"),
            row.names = FALSE, quote = FALSE)
  si <- data.frame(replicate = samples, condition = cond)
  if (!is.null(control)) si$x <- control
  write.csv(si, file.path(dir, "input_sample_info.csv"), row.names = FALSE, quote = FALSE)
  write.csv(data.frame(left = c("B", "C", "C"), right = c("A", "A", "B")),
            file.path(dir, "input_comparisons.csv"), row.names = FALSE, quote = FALSE)
  ctl <- if (is.null(control)) "null" else '{"Column": "x", "Type": "numerical"}'
  writeLines(sprintf('{"condition_col": "condition", "control_cols": %s}', ctl), file.path(dir, "params.json"))
}
# Counts only in the singleton C: df.residual.adj = 0 exactly without a control.
only_c <- function(k) { r <- rep(0, n); r[n] <- k; r }
cm <- base; for (i in 1:21) cm[i, ] <- only_c(50 + 10 * i)
colnames(cm) <- samples; write_case("gap_zero", cm)
# With a numeric control the same genes, and genes seen in one sample per group, fall to
# 0 < df.residual.adj < 0.01.
cm <- base
pats <- list(c(0, 0, 0, 300, 0, 0, 0, 300, 100), c(0, 0, 0, 300, 0, 0, 0, 0, 100), c(0, 0, 0, 0, 0, 0, 0, 300, 0))
for (i in 1:15) cm[i, ] <- pats[[(i - 1) %% 3 + 1]] * (1 + i / 10)
for (i in 16:24) cm[i, ] <- only_c(50 + 10 * i)
colnames(cm) <- samples; write_case("gap_tiny", cm, control = c(1, 2, 3, 4, 1, 2, 3, 4, 2.5))
