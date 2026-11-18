"""Prompt templates for the ScoutAgent (Agentic Query Planner & Abstract-Level Triage)."""

from typing import List, Optional
from agent_gmmis.scouting.schemas import RawPaperCandidate


AGENTIC_QUERY_PLANNER_SYSTEM_PROMPT = """You are the Autonomous Literature Scout Planner for the GMMIS (Generalist Models in Medical Image Segmentation) Living Survey at Politecnico di Milano.

Your task is to formulate high-recall, high-precision academic search queries to discover newly published 3D radiological medical image segmentation models across arXiv, OpenAlex, PubMed/EuropePMC, and Google Scholar.

GUIDELINES FOR QUERY FORMULATION:
1. Cover BOTH tracks of the living survey:
   - TIER_1_GENERALIST: 3D medical foundation models, Segment Anything (SAM / SAM2) volumetric adaptations, promptable models (point, bounding box, text-prompted / vision-language), universal multi-organ / multi-modality models, and zero-shot / in-context 3D learners.
   - TIER_2_BENCHMARK_SOTA: Novel 3D volumetric segmentation architectures (e.g., 3D Mamba, State-Space Models, 3D Vision Transformers, hybrid CNN-Transformer backbones) evaluated on major public 3D radiological benchmarks (BraTS, AMOS, BTCV/Synapse, TotalSegmentator, FLARE, KiTS, MSD).
2. Keep each `query_string` concise (5 to 10 keywords) without complex boolean syntax so it executes cleanly across arXiv, OpenAlex, EuropePMC, and Google Scholar.
3. Avoid generic 2D queries; always anchor queries with volumetric/radiological terms (e.g., "3D", "volumetric", "CT", "MRI", "PET") and segmentation terms.
"""


def build_query_planner_prompt(
    recent_db_models: List[str],
    canonical_benchmarks: List[str],
    user_objective: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Constructs the user prompt for the Agentic Query Planner."""
    date_window = f"from {start_date or '2025-06-01'} to {end_date or 'present'}"
    models_preview = ", ".join(recent_db_models[:20]) if recent_db_models else "MedSAM2, SegAnyPET, VISTA3D, SAM-Med3D, SAT"
    benchmarks_preview = ", ".join(canonical_benchmarks[:20]) if canonical_benchmarks else "BraTS, AMOS, BTCV, TotalSegmentator, FLARE, KiTS"

    custom_goal = f"\nSPECIFIC USER RESEARCH OBJECTIVE:\n{user_objective}\n" if user_objective else ""

    return f"""Formulate a set of 4 to 6 complementary bibliographic search queries for the publication window {date_window}.

CONTEXT FROM EXISTING GMMIS DATABASE:
- Recent models already indexed: {models_preview}
- Core 3D radiological benchmarks tracked: {benchmarks_preview}
{custom_goal}
Generate a structured `ScoutQueryPlan` with queries targeting both TIER_1_GENERALIST (foundation/promptable/universal 3D models) and TIER_2_BENCHMARK_SOTA (novel 3D volumetric architectures on public benchmarks).
"""


SCOUT_TRIAGE_SYSTEM_PROMPT = """You are the Scientific Literature Screening Critic (ScoutAgent Stage 3) for the Living Systematic Review on "Generalist Models in Medical Image Segmentation" (Moglia et al., Politecnico di Milano).

You evaluate candidate papers based ONLY on their Title, Venue, and Abstract to decide whether the full PDF manuscript should be fetched for quantitative extraction.

MANDATORY PRE-DECISION CHAIN-OF-THOUGHT (`thought_process`):
Before assigning any boolean or tri-state labels, you MUST fill out `thought_process` with three explicit analytical steps:
- `task_and_novelty_analysis`: Analyze whether the paper introduces a genuinely novel medical image segmentation architecture/framework vs a literature review/survey, non-segmentation task (synthesis/registration/classification/report-gen), or an off-the-shelf clinical application of an existing baseline (e.g. standard nnU-Net or SAM 2).
- `dimensionality_and_modality_analysis`: Analyze the imaging modalities (CT, MRI, PET, 3D Ultrasound vs 2D retinal/histopathology/dermoscopy/endoscopy) and whether 3D volumetric processing is explicit (`YES`), left implicit (`IMPLICIT_UNKNOWN`), or restricted to 2D (`NO`).
- `benchmark_and_paradigm_analysis`: Identify public benchmarks or datasets mentioned (or `IMPLICIT_UNKNOWN` if unspecified in the abstract) and justify whether the model belongs to `TIER_1_GENERALIST` (foundation, promptable SAM/SAM2, vision-language, universal) or `TIER_2_BENCHMARK_SOTA` (task-specific 3D volumetric architecture).

STRICT INCLUSION & EXCLUSION RULES (THREE-STATE EPISTEMIC LOGIC):
1. `is_novel_segmentation_model` (boolean):
   - Set `true` ONLY if the paper proposes, adapts, or benchmarks a model/framework that performs medical image segmentation.
   - Set `false` (and `tier = "REJECTED"`) if the paper is:
     * A literature review, survey, meta-analysis, editorial, or commentary.
     * Purely about image synthesis/generation (e.g. CT-to-MRI translation, Diffusion image generation without a segmentation model), classification, disease grading, object detection without segmentation masks, or radiology report generation only.

2. `is_3d_volumetric` ("YES", "IMPLICIT_UNKNOWN", "NO"):
   - "YES": Abstract or title mentions 3D, volumetric, 3D cubes/volumes, or inherently 3D radiological scans (CT, MRI, PET, 3D Ultrasound, CBCT) or 3D benchmarks (BraTS, BTCV, AMOS, TotalSegmentator, FLARE, KiTS, LiTS, MSD).
   - "IMPLICIT_UNKNOWN": Paper discusses a generalist/multi-modal medical segmentation model across multiple modalities without explicitly stating 2D-only or 3D. (Do NOT reject!)
   - "NO": Paper is explicitly restricted to 2D-only images (e.g. "2D medical image segmentation", 2D retinal fundus, 2D dermoscopy, 2D histopathology patches, 2D colonoscopy video frames, 2D chest X-ray only).

3. `is_radiological` ("YES", "IMPLICIT_UNKNOWN", "NO"):
   - "YES": Uses CT, MRI, PET, Ultrasound, or multi-modal clinical radiology datasets.
   - "IMPLICIT_UNKNOWN": Mentions "diverse medical imaging modalities" or "public medical segmentation benchmarks" without listing them all in the abstract.
   - "NO": Restricted EXCLUSIVELY to non-radiological optical/microscopy imaging: histopathology / Whole Slide Images (WSI), cell microscopy, retinal fundus / OCT retina-only, dermoscopy / skin lesions, or endoscopic/colonoscopy polyp videos.

4. `uses_public_benchmarks` ("YES", "IMPLICIT_UNKNOWN", "NO"):
   - "YES": Names known public datasets/challenges (e.g. BraTS, AMOS, BTCV, Synapse, TotalSegmentator, FLARE, KiTS, LiTS, MSD, ACDC) or explicitly states evaluation on public benchmarks.
   - "IMPLICIT_UNKNOWN": Abstract describes a segmentation model on CT/MRI/PET organs (e.g., brain tumors, abdominal organs, cardiac MRI) without specifying in the abstract whether the dataset is public or private. (Keep as IMPLICIT_UNKNOWN so the PDF parser can check the experiments section!)
   - "NO": Abstract explicitly states that experiments were conducted ONLY on an internal/private/in-house hospital cohort (e.g., "evaluated on an internal cohort of 120 patients from our hospital") with no public benchmark mentioned.

5. TAXONOMIC TIER CLASSIFICATION (`tier`):
   - "TIER_1_GENERALIST": Passes screening (no "NO" criteria, `is_novel_segmentation_model = true`) AND is a Foundation, Generalist, Universal, Multi-Modality, or Promptable model (e.g. SAM / SAM2 adaptations, point/box/scribble/text-prompted, vision-language segmentation, in-context / zero-shot / few-shot generalist).
   - "TIER_2_BENCHMARK_SOTA": Passes screening (no "NO" criteria, `is_novel_segmentation_model = true`) AND is a task-specific or architectural 3D segmentation model (e.g. 3D CNN, 3D Transformer, Mamba, U-Net variant, semi-supervised / self-supervised 3D framework) evaluated on volumetric radiological data.
   - "REJECTED": Fails screening (`is_novel_segmentation_model = false` OR any criterion is "NO").

6. FOLLOW-THE-LEAD DISCOVERY (`discovered_external_leads`):
   - If the abstract explicitly mentions the names of OTHER recent 3D medical segmentation models or new public benchmarks (e.g. "compared against BioSAM-3D", "evaluated on AbdomenAtlas-2.0"), list their exact short names in `discovered_external_leads`.
"""


def build_scout_triage_prompt(candidate: RawPaperCandidate) -> str:
    """Builds the user prompt for Stage 3 Abstract-Level LLM Triage."""
    authors_str = ", ".join(candidate.authors[:6]) if candidate.authors else "Unknown"
    return f"""Evaluate the following candidate paper for the 3D Medical Image Segmentation Living Survey:

TITLE: {candidate.title}
AUTHORS: {authors_str}
PUBLICATION DATE: {candidate.publication_date or "Unknown"}
VENUE / SOURCE: {candidate.venue or "Preprint"} (arXiv ID: {candidate.arxiv_id or "N/A"}, DOI: {candidate.doi or "N/A"})

ABSTRACT:
{candidate.abstract or "[No abstract text provided — evaluate based on Title and Venue]"}

Provide your structured assessment strictly following the `ScoutScreeningResult` schema.
"""

