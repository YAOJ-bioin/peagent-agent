"""PEAgent Tool: sequence-to-single-cell chromatin accessibility inference for plant genomes."""

from importlib.metadata import PackageNotFoundError, version

from .assets import describe_assets, fetch_species_assets, resolve_metadata_path, resolve_model_path
from .attribution import (
    AttributionResult,
    ContributionScorer,
    compute_and_plot_attribution,
    compute_attribution,
    plot_attribution,
    plot_attribution_panels,
)
from .celltype_prediction import (
    SPECIES_DEFAULTS,
    CellTypePredictor,
    list_valid_celltypes,
    load_predictor,
    predict_celltypes,
)
from .evaluation import (
    binary_classification_metrics,
    safe_auroc,
    safe_average_precision,
    safe_mae,
    safe_pearson,
    safe_spearman,
)
from .preprocessing import encode_sequences, normalize_sequences
from .variant_effect import VariantEffectScorer, VariantRecord, score_ref_alt_sequences, score_variants

__author__ = """Jie Yao"""
__email__ = "JieYao@uga.edu"
try:
    __version__ = version("peagent-agent")  # peagent_tool ships inside the peagent-agent distribution
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "SPECIES_DEFAULTS",
    "AttributionResult",
    "CellTypePredictor",
    "ContributionScorer",
    "VariantEffectScorer",
    "VariantRecord",
    "__version__",
    "binary_classification_metrics",
    "compute_and_plot_attribution",
    "compute_attribution",
    "describe_assets",
    "encode_sequences",
    "fetch_species_assets",
    "list_valid_celltypes",
    "load_predictor",
    "normalize_sequences",
    "plot_attribution",
    "plot_attribution_panels",
    "predict_celltypes",
    "resolve_metadata_path",
    "resolve_model_path",
    "safe_average_precision",
    "safe_auroc",
    "safe_mae",
    "safe_pearson",
    "safe_spearman",
    "score_ref_alt_sequences",
    "score_variants",
]
