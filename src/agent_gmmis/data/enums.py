"""Domain enumerations for medical image segmentation benchmarking."""

from enum import Enum


class ImagingModality(str, Enum):
    """Supported 3D radiological modalities."""
    CT = "CT"
    MRI = "MRI"
    PET = "PET"
    MULTIMODAL = "Multi-modal"
    OTHER = "Other"


class AnatomicalRegion(str, Enum):
    """Standardized anatomical target regions."""
    HEAD = "Head"
    HEAD_NECK = "Head and Neck"
    THORAX = "Thorax"
    ABDOMEN = "Abdomen"
    PELVIS = "Pelvis"
    SPINE = "Spine"
    WHOLE_BODY = "Whole Body"
    OTHER = "Other"


class FusionLevel(str, Enum):
    """Taxonomy of feature fusion levels in generalist models."""
    F0 = "F0 (Early/Input Fusion)"
    F1 = "F1 (Encoder Fusion)"
    F2 = "F2 (Bottleneck Fusion)"
    F3 = "F3 (Decoder Fusion)"
    F4 = "F4 (Late/Decision Fusion)"
    NO_FUSION = "No Fusion"
    UNKNOWN = "Unknown"


class ScoreCategory(str, Enum):
    """Evaluation score provenance."""
    PRIMARY = "primary"
    BEST = "best"


class ModelParadigm(str, Enum):
    """Generalist vs. Task-Specific classification."""
    GENERALIST = "generalist"
    TASK_SPECIFIC = "task_specific"
