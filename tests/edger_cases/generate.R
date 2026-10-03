# Reference outputs for every case here, from production MDFlexiComparisons (runEdgeRPairwiseStats,
# sourced from R/*.R) in the md-flexi-r45-local image (edgeR 4.8.2). Writes reference_output.csv
# (the production table, %.17g) and r_diag.csv (df.prior and df.residual.adj per kept gene).
# GroupId is integer when every id is one, as in production, so genes are fitted in numeric order;
# a stalled gene takes the last written gene's value, so the order matters (review overnight r2,
# Minor 1).
#   docker run --rm --platform linux/amd64 -v "$PWD":/w -v <MDFlexiComparisons>:/flexi:ro -w /w \
#     md-flexi-r45-local:latest Rscript generate.R
suppressPackageStartupMessages({
  library(data.table); library(edgeR); library(limma); library(log4r); library(glue); library(stringr)
})
for (f in list.files("/flexi/R", pattern = "[.]R$", full.names = TRUE)) source(f)
fmt <- function(v) if (is.numeric(v)) ifelse(is.na(v), "", sprintf("%.17g", v)) else v
for (case in list.dirs(".", recursive = FALSE)) {
  rd <- function(f) read.csv(file.path(case, f), colClasses = "character", check.names = FALSE)
  counts <- rd("input_counts.csv"); si <- rd("input_sample_info.csv"); cmp <- rd("input_comparisons.csv")
  params <- jsonlite::fromJSON(file.path(case, "params.json"))
  ctl <- params$control_cols
  gid <- if (all(grepl("^-?[0-9]+$", counts$id))) as.integer(counts$id) else counts$id
  long <- data.table::melt(data.table(GroupId = gid, counts[-1]), id.vars = "GroupId",
                           variable.name = "replicate", value.name = "intensity", variable.factor = FALSE)
  long$intensity <- as.numeric(long$intensity)
  long <- merge(long, data.table(si), by = "replicate")
  ctlCols <- NULL
  if (!is.null(ctl)) {
    ctlCols <- ctl$Column
    if (ctl$Type == "numerical") long[[ctlCols]] <- as.numeric(long[[ctlCols]])
  }
  norm <- if (is.null(params$edger_norm_method)) "TMM" else params$edger_norm_method
  levs <- sort(unique(si$condition))
  dict <- data.table(original = levs, safe = levs)
  res <- tryCatch({
    out <- runEdgeRPairwiseStats(long, "condition", ctlCols, data.frame(left = cmp$left, right = cmp$right),
                                 "GroupId", " - ", dict, normMethod = norm)
    fwrite(out[, lapply(.SD, fmt)], file.path(case, "reference_output.csv"))
    # The same fit again for the per-gene df (deterministic).
    cm <- .buildCountMatrixFromLongDT(long, "GroupId")
    sinfo <- .buildSampleInfoDF(long, "condition", ctlCols)[colnames(cm), , drop = FALSE]
    sinfo$condition <- factor(sinfo$condition)
    fit <- .fitEdgeRModel(cm, sinfo, "condition", ctlCols, normMethod = norm)$fit
    fwrite(data.table(id = rownames(fit$counts), df_prior = fmt(rep_len(fit$df.prior, nrow(fit$counts))),
                      df_residual_adj = fmt(fit$df.residual.adj)), file.path(case, "r_diag.csv"))
    sprintf("%s: %d genes, df.prior %.4g, df_adj == 0: %d, 0 < df_adj < 0.01: %d", case, nrow(fit$counts),
            fit$df.prior[1], sum(fit$df.residual.adj == 0), sum(fit$df.residual.adj > 0 & fit$df.residual.adj < 0.01))
  }, error = function(e) paste(case, "ERROR", conditionMessage(e)))
  message(res)
}
