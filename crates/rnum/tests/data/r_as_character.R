# Rewrites column 2 of r_as_character.tsv with R's as.character() of the double whose bits are in
# column 1 (written by r_as_character_inputs.py). Run in the production image from this directory:
#   docker run --rm --platform linux/amd64 -v "$PWD":/w -w /w md-flexi-r45-local:latest \
#     Rscript r_as_character.R
d <- read.delim("r_as_character.tsv", header = FALSE, colClasses = "character", quote = "",
                na.strings = character())
hex <- d[[1]]
bytes <- substring(rep(hex, each = 8), seq(15, 1, -2), seq(16, 2, -2))
x <- readBin(as.raw(strtoi(bytes, 16L)), "double", n = length(hex), size = 8, endian = "little")
writeLines(paste0(hex, "\t", as.character(x)), "r_as_character.tsv")
