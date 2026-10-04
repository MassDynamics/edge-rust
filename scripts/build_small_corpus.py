"""Copy the small count-corpus tier into tests/corpus-small/.

The full count corpus (~780 MB) lives outside git and stays the local gate. CI runs the golden
tests against this committed subset instead: whole count_synth runs, never trimmed, because
edgeR's dispersion and QL prior pooling run across every gene, so a trimmed input would no longer
match the R results. Only what the tests read is copied: reference/<run>/ and
runs/<run>/manifest.json. The index files are filtered to the chosen runs and the root index.json
carries "tier": "small", which the Rust run-count checks key on.

Not copied: shared/ (the tests read the inputs dumped under reference/), runs/*/results.* and
.rds files, and reference-glibm/ (9.6 MB; crates/rnum/tests/glibm_golden.rs skips without it).

Usage: uv run python scripts/build_small_corpus.py ~/wd/md-count-golden-corpus
"""

import json
import shutil
import sys
from pathlib import Path

FULL = Path(sys.argv[1]).expanduser()
SMALL = Path(__file__).resolve().parent.parent / "tests" / "corpus-small"

RUNS = [
    # TMM, all pairs, factor + numeric covariates; also the diagnostics run and the base the
    # negative and protein-entity error runs are rebuilt from
    "count_edger_count_synth_all_ctlfactor_numeric",
    # ANOVA (omnibus contrast) with a factor covariate
    "count_edger_count_synth_anova_ctlfactor",
    # the other norm methods, no covariate
    "count_edger_norm_RLE",
    "count_edger_norm_upperquartile",
    "count_edger_norm_none",
    # expected-error runs (all small)
    "edge_edger_filter_drops_all",
    "edge_edger_negative",
    "edge_edger_one_rep_per_condition",
    "edge_edger_protein_entity",
    "edge_edger_rank_deficient",
]


def copy_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".rds":
            shutil.copy2(f, dst / f.name)


if SMALL.exists():
    shutil.rmtree(SMALL)
for run in RUNS:
    copy_dir(FULL / "reference" / run, SMALL / "reference" / run)
    (SMALL / "runs" / run).mkdir(parents=True)
    shutil.copy2(FULL / "runs" / run / "manifest.json", SMALL / "runs" / run / "manifest.json")

index = json.loads((FULL / "index.json").read_text())
index["runs"] = {k: v for k, v in index["runs"].items() if k in RUNS}
index["n_runs"] = len(index["runs"])
index["tier"] = "small"
(SMALL / "index.json").write_text(json.dumps(index, indent=2) + "\n")

ref_index = json.loads((FULL / "reference" / "index.json").read_text())
ref_index["runs"] = [r for r in ref_index["runs"] if r["run_id"] in RUNS]
ref_index["n_runs"] = len(ref_index["runs"])
(SMALL / "reference" / "index.json").write_text(json.dumps(ref_index, indent=2) + "\n")

assert index["n_runs"] == ref_index["n_runs"] == len(RUNS), "a chosen run is missing an index entry"
print(f"{len(RUNS)} runs into {SMALL}")
