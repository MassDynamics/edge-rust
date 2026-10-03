# Rebuilds tests/edger_cases/thin100_tmm and thin1000_tmm inputs: one sample of the corpus case
# count_edger_count_synth_all_ctlnone binomially thinned (review overnight r1, M1). The loop order
# and seed are those of the probe that found the cases, so the RNG stream matches (review overnight
# r2, SE2-m1). input_comparisons.csv and params.json are written here, not copied: the
# comparisons name the corpus's encoded levels and the corpus run has no params.json.
#   docker run --rm --platform linux/amd64 -v "$PWD":/w \
#     -v ~/wd/md-count-golden-corpus/reference:/corp:ro -w /w md-flexi-r45-local:latest \
#     Rscript make_thin_inputs.R
src <- "/corp/count_edger_count_synth_all_ctlnone"
set.seed(7)
counts <- read.csv(file.path(src, "input_counts.csv"), check.names = FALSE)
keep <- c(th_100_TMM_2 = "thin100_tmm", th_1000_TMM_5 = "thin1000_tmm")
for (target in c(20, 100, 1000, 10000)) for (norm in c("none", "TMM")) for (smp in c(2, 5)) {
  cc <- counts; x <- cc[[smp + 1]]
  cc[[smp + 1]] <- rbinom(length(x), x, target / sum(x))
  id <- sprintf("th_%d_%s_%d", target, norm, smp)
  if (!id %in% names(keep)) next
  dir.create(keep[[id]], showWarnings = FALSE)
  write.csv(cc, file.path(keep[[id]], "input_counts.csv"), row.names = FALSE)
  file.copy(file.path(src, "input_sample_info.csv"), keep[[id]], overwrite = TRUE)
  writeLines(c("left,right", "VvbFo,YJWrq", "VvbFo,bdcYa", "YJWrq,bdcYa"),
             file.path(keep[[id]], "input_comparisons.csv"))
  writeLines('{"condition_col": "condition", "control_cols": null, "edger_norm_method": "TMM"}',
             file.path(keep[[id]], "params.json"))
}
