"""Robust exports for model.separator."""

__all__ = []


def _safe_import(module_name: str, candidates: list[str]) -> None:
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=candidates)
    except Exception:
        return

    for name in candidates:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)
            __all__.append(name)


_safe_import("frontend", ["FrontendConfig", "STFTFrontend", "SpectrogramFrontend"])
if "STFTFrontend" in globals() and "SpectrogramFrontend" not in globals():
    SpectrogramFrontend = STFTFrontend
    __all__.append("SpectrogramFrontend")

_safe_import("feature_head", ["FeatureHeadConfig", "SeparatorFeatureHead", "FeatureAggregationHead"])
if "SeparatorFeatureHead" in globals() and "FeatureAggregationHead" not in globals():
    FeatureAggregationHead = SeparatorFeatureHead
    __all__.append("FeatureAggregationHead")

_safe_import(
    "resunet_separator",
    ["ResUNetSeparator", "ResUNetSeparatorConfig", "load_pretrained_separator_state_dict"],
)

_safe_import(
    "conditioning",
    ["TimeFiLM2d", "LearnedHiddenStateFusion", "LatentFeatureInjection2d", "gather_class_probability_map"],
)

_safe_import(
    "stage2_sed",
    ["Stage2SEDConfig", "Stage2SEDGuideEncoder", "build_stage2_sed_guide"],
)

_safe_import(
    "dprnn",
    ["DPRNNConfig", "DPRNNBlock2d", "DPRNN2d", "build_dprnn_2d"],
)

_safe_import(
    "iterative_refinement",
    ["IterativeRefinementConfig", "IterativeRefinementInputAdapter", "IterativeRefinementWrapper", "build_iterative_refinement_wrapper"],
)

_safe_import("tfgridnet_separator", ["TFGridNetSeparator", "TFGridNetSeparatorConfig"])