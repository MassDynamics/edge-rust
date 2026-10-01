"""edgeR quasi-likelihood differential expression, ported to Rust.

``run`` takes what MDFlexiComparisons hands the edgeR engine (count matrix, sample info,
comparisons, params) and returns the production output table.
"""

from edge_rust.edger import run

__all__ = ["run"]
