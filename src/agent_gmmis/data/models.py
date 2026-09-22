"""Pydantic v2 schemas for medical image segmentation models and benchmarks."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gmmis.data.enums import (
    AnatomicalRegion,
    FusionLevel,
    ImagingModality,
    ModelParadigm,
    ScoreCategory,
)


class ModelMetadata(BaseModel):
    """Metadata for a segmentation model (generalist or task-specific)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = Field(..., alias="ID", description="Unique identifier for the model (e.g. MedSAM, nnU-Net)")
    title: str = Field(..., alias="Title", description="Full title of the model publication")
    major_affiliations: Optional[str] = Field(None, alias="Major Affiliations", description="Publishing institutions/countries")
    first_publication_date: Optional[str] = Field(None, alias="First Publication Date", description="Initial release or preprint date (YYYY-MM)")
    first_publisher: Optional[str] = Field(None, alias="First Publisher", description="Venue of first release (e.g. arXiv, CVPR)")
    first_publication_link: Optional[str] = Field(None, alias="First Publication Link", description="URL to the primary paper")
    first_publication_bibkey: Optional[str] = Field(None, alias="First Publication BibKey", description="BibTeX citation key")
    last_publication_date: Optional[str] = Field(None, alias="Last Publication Date", description="Most recent journal or conference publication date")
    last_publisher: Optional[str] = Field(None, alias="Last Publisher", description="Most recent venue")
    last_publication_link: Optional[str] = Field(None, alias="Last Publication Link", description="URL to latest paper")
    last_publication_bibkey: Optional[str] = Field(None, alias="Last Publication BibKey")
    code: Optional[str] = Field(None, alias="Code", description="Official GitHub/code repository URL")
    framework: Optional[str] = Field(None, alias="Framework", description="Underlying framework (PyTorch, TensorFlow, etc.)")
    architecture: Optional[str] = Field(None, alias="Architecture", description="Structural type (e.g. SAM, Transformer with Convolutions, CNN)")
    visual_backbone: Optional[str] = Field(None, alias="Visual Backbone", description="Vision encoder (e.g. ViT-B, Swin-B, ResNet-50)")
    millions_of_parameters: Optional[float] = Field(None, alias="Millions of Parameters", description="Total parameter count in Millions")
    number_of_gflops: Optional[float] = Field(None, alias="Number of GFlops", description="Computational complexity in GFLOPs")
    resources: Optional[str] = Field(None, alias="Resources", description="Hardware used for training (e.g. 8x NVIDIA A100 80GB)")
    resources_total_vram: Optional[int] = Field(None, alias="Resources Total V-RAM", description="Total GPU VRAM in GB")


class DatasetMetadata(BaseModel):
    """Metadata for a public medical image segmentation dataset."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str = Field(..., alias="Name", description="Canonical short name (e.g. BraTS, AMOS, BTCV)")
    full_name: Optional[str] = Field(None, alias="Full Name", description="Full descriptive title of the challenge/dataset")
    description: Optional[str] = Field(None, alias="Description")
    sub_dataset: Optional[str] = Field(None, alias="Sub-dataset")
    imaging_modality: Optional[str] = Field(None, alias="Imaging Modality", description="3D CT, 3D MRI, PET, etc.")
    objects: Optional[str] = Field(None, alias="Objects", description="Target organs, tumors, or structures segmented")
    main_anatomical_structure: Optional[str] = Field(None, alias="Main Anatomical Structure")
    region: Optional[str] = Field(None, alias="Region", description="Body region (e.g. Abdomen, Thorax, Head)")
    links_url: Optional[str] = Field(None, alias="Links URL", description="Official dataset download or challenge URL")
    links_text: Optional[str] = Field(None, alias="Links Text")
    cit_keys: Optional[str] = Field(None, alias="Cit Keys", description="Dataset citation reference")
    n_tot: Optional[int] = Field(None, alias="N Tot", description="Total number of volumetric scans")
    n_with_labels: Optional[int] = Field(None, alias="N with Labels", description="Number of annotated volumes")


class BenchmarkScore(BaseModel):
    """A validated segmentation score on a given dataset."""

    model_id: str
    dataset_name: str
    score_type: ScoreCategory = ScoreCategory.PRIMARY
    dice_score: float = Field(..., description="Dice Similarity Coefficient, normalized to [0.0, 1.0]")
    source_paper: Optional[str] = None
    comments: Optional[str] = None

    @field_validator("dice_score", mode="before")
    @classmethod
    def validate_dice_range(cls, v: float) -> float:
        """Enforces [0.0, 1.0] range and converts percentage notation automatically."""
        if v is None:
            raise ValueError("Dice score cannot be None")
        val = float(v)
        # If expressed as percentage (e.g. 89.5%), normalize to [0.0, 1.0]
        if val > 1.0 and val <= 100.0:
            val = val / 100.0
        if val < 0.0 or val > 1.0:
            raise ValueError(f"Dice score must be in range [0.0, 1.0], got {val}")
        return round(val, 4)


class InclusionCriteriaCheck(BaseModel):
    """Scientific screening check based on Moglia et al. inclusion/exclusion criteria."""

    is_3d: bool = Field(..., description="Is the imaging modality 3D (volumetric)?")
    is_radiological: bool = Field(..., description="Is it CT, MRI, or PET? (Excludes 2D retina/dermoscopy)")
    is_public_dataset: bool = Field(..., description="Evaluated on publicly available dataset(s)?")
    has_dice_metric: bool = Field(..., description="Reports Dice Similarity Coefficient (DSC)?")
    notes: Optional[str] = None

    @property
    def is_included(self) -> bool:
        """Determines if the candidate paper meets all inclusion criteria."""
        return self.is_3d and self.is_radiological and self.is_public_dataset and self.has_dice_metric

    @property
    def rejection_reasons(self) -> List[str]:
        """Provides human-readable reasons if rejected."""
        reasons = []
        if not self.is_3d:
            reasons.append("Modality is 2D instead of 3D volumetric.")
        if not self.is_radiological:
            reasons.append("Modality is non-radiological (not CT/MRI/PET).")
        if not self.is_public_dataset:
            reasons.append("Evaluated solely on private/proprietary datasets.")
        if not self.has_dice_metric:
            reasons.append("Does not report Dice Similarity Coefficient (DSC).")
        return reasons
