try:
    from .dataset_ff import (
        FF_Dataset,
        build_ff_datasets,
        build_ff_dataloaders,
        MANIP_TYPES,
        LABEL_REAL,
        LABEL_FAKE,
    )
    from .dataset_cached import CachedFeatureDataset, build_cached_dataloaders
    __all__ = [
        "FF_Dataset", "build_ff_datasets", "build_ff_dataloaders",
        "MANIP_TYPES", "LABEL_REAL", "LABEL_FAKE",
        "CachedFeatureDataset", "build_cached_dataloaders",
    ]
except Exception:
    # v2 pipeline imports dataset_ff_v2 and dataset_v2 directly
    # Old imports skipped when not needed
    __all__ = []