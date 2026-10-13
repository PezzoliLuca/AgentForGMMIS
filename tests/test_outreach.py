"""Unit tests for OutreachAgent ('Delega Autori') - Phase 3.2 Part 2."""

import json
from pathlib import Path
import pytest

from agent_gmmis.auditing.outreach import OutreachAgent
from agent_gmmis.auditing.schemas import (
    AuditReport,
    AuditStatus,
    AuthorContactInfo,
    MissingDataDiagnosis,
    OutreachDraft,
    OutreachStatus,
)
from agent_gmmis.data.models import InclusionCriteriaCheck
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.extraction.schemas import ExtractedPaperData, RawExtractedScore


@pytest.fixture
def outreach_agent(tmp_path):
    return OutreachAgent(output_dir=tmp_path / "outreach_drafts")


@pytest.fixture
def mock_medsam_manuscript():
    text_page_1 = """
Segment Anything in Medical Images
Jun Ma1,2,3, Yuting He4, Feifei Li1, Lin Han5, Chenyu You6,
Bo Wang1,2,3,7,8*
1Peter Munk Cardiac Centre, University Health Network, Toronto, Canada.
2Department of Laboratory Medicine and Pathobiology, University of Toronto, Canada.
3Vector Institute, Toronto, Canada.
*Corresponding author(s). E-mail(s): bowang@vectorinstitute.ai;
Contributing authors: junma.ma@mail.utoronto.ca; yhe668@uwo.ca;
Abstract
Medical image segmentation is a critical component in clinical practice...
"""
    return ParsedManuscript(
        file_path="mock_medsam.pdf",
        total_pages=14,
        title="Segment Anything in Medical Images",
        authors="Jun Ma, Yuting He, Bo Wang",
        pages_text={1: text_page_1},
    )


@pytest.fixture
def mock_universeg_manuscript():
    text_page_1 = """
UniverSeg: Universal Medical Image Segmentation
Victor Ion Butoi*
MIT CSAIL
vbutoi@mit.edu
Jose Javier Gonzalez Ortiz*
MIT CSAIL
josejg@mit.edu
Tianyu Ma
Cornell University
Abstract
While deep learning models have become the predominant method...
"""
    return ParsedManuscript(
        file_path="mock_universeg.pdf",
        total_pages=16,
        title="UniverSeg: Universal Medical Image Segmentation",
        authors="Victor Ion Butoi, Jose Javier Gonzalez Ortiz",
        pages_text={1: text_page_1},
    )


class TestOutreachAgent:
    """Verifies contact extraction, target figure diagnosis, email composition, and draft persistence."""

    def test_extract_contact_info_footnote_format(self, outreach_agent, mock_medsam_manuscript):
        """Standard Nature/Springer format with explicit corresponding author footnote."""
        contact = outreach_agent.extract_contact_info(mock_medsam_manuscript)

        assert contact.name == "Bo Wang"
        assert contact.email == "bowang@vectorinstitute.ai"
        assert contact.is_corresponding is True
        assert contact.confidence == 1.0
        assert "Vector Institute" in (contact.affiliation or "") or "Toronto" in (contact.affiliation or "")

    def test_extract_contact_info_inline_format(self, outreach_agent, mock_universeg_manuscript):
        """Conference/arXiv format with email directly below author name and asterisk."""
        contact = outreach_agent.extract_contact_info(mock_universeg_manuscript)

        assert contact.name == "Victor Ion Butoi"
        assert contact.email == "vbutoi@mit.edu"
        assert contact.is_corresponding is True
        assert "MIT CSAIL" in (contact.affiliation or "")

    def test_extract_contact_info_fallback_no_email(self, outreach_agent):
        """Manuscript without email on front page falls back gracefully with reduced confidence."""
        manuscript = ParsedManuscript(
            file_path="mock_no_email.pdf",
            total_pages=8,
            title="Novel 3D Brain Segmentation",
            authors="Alice Smith, Bob Jones",
            pages_text={
                1: "Novel 3D Brain Segmentation\nAlice Smith, Bob Jones\nDepartment of Radiology, Harvard University\nAbstract\nWe present...",
            },
        )
        contact = outreach_agent.extract_contact_info(manuscript)

        assert contact.name == "Alice Smith"
        assert contact.email is None
        assert contact.is_corresponding is False
        assert contact.confidence < 1.0
        assert "Harvard University" in (contact.affiliation or "")

    def test_identify_missing_targets_from_scores_and_captions(self, outreach_agent):
        """Identifies target datasets lacking scalar values and figures with visual evaluations."""
        manuscript = ParsedManuscript(
            file_path="mock_paper.pdf",
            total_pages=10,
            pages_text={
                1: "Paper Header...",
                5: "Figure 3: Comparative evaluation boxplot showing Dice score (DSC) on BraTS and AMOS across models.",
                6: "Fig. 4: Radar chart comparing mean DSC over multi-organ abdomen targets.",
            },
        )
        extracted = ExtractedPaperData(
            paper_title="Merlin: Generalist Medical Segmentation",
            proposed_model_id="Merlin",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BraTS", organ_or_task="Brain Tumor", dice_score=None, table_or_section_ref="Figure 3"),
                RawExtractedScore(dataset_name="AMOS", organ_or_task="Multi-organ", dice_score=None, table_or_section_ref="Figure 3"),
            ],
        )

        datasets, figures = outreach_agent.identify_missing_targets(extracted, manuscript)

        assert "BraTS" in datasets
        assert "AMOS" in datasets
        assert any("Figure 3" in f for f in figures)
        assert any("Fig. 4" in f for f in figures)

    def test_compose_email_formatting(self, outreach_agent):
        """Email text must be polite, formal, in English, and cite DEIB Politecnico di Milano."""
        extracted = ExtractedPaperData(
            paper_title="FLAP-SAM: Fast Medical Segmentation",
            proposed_model_id="FLAP-SAM",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
        )
        contact = AuthorContactInfo(
            name="John Doe",
            email="johndoe@stanford.edu",
            affiliation="Stanford University",
            is_corresponding=True,
        )

        subject, body = outreach_agent.compose_email(
            extracted=extracted,
            contact=contact,
            target_datasets=["BTCV", "KiTS"],
            target_figures=["Figure 4"],
        )

        assert "Quantitative evaluation metrics" in subject
        assert "FLAP-SAM" in subject
        assert "Dear Prof./Dr. John Doe," in body
        assert "Politecnico di Milano" in body
        assert "FLAP-SAM" in body
        assert "BTCV" in body and "KiTS" in body
        assert "Figure 4" in body
        assert "Mean Dice Similarity Coefficient" in body
        assert "luca.pezzoli@polimi.it" in body

    def test_generate_and_save_draft(self, outreach_agent, mock_medsam_manuscript, tmp_path):
        """Generates draft and writes both valid JSON and Markdown files to disk."""
        extracted = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=None, table_or_section_ref="Figure 2"),
            ],
        )

        draft = outreach_agent.generate_draft(extracted, mock_medsam_manuscript)
        assert draft.status == OutreachStatus.DRAFT_PENDING_REVIEW
        assert draft.recipient.email == "bowang@vectorinstitute.ai"

        json_path, md_path = outreach_agent.save_draft(draft, output_dir=tmp_path / "outreach_tests")

        assert json_path.exists()
        assert md_path.exists()

        # Check JSON validity by deserializing back to OutreachDraft
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = OutreachDraft.model_validate(data)
        assert loaded.model_id == "MedSAM"
        assert loaded.recipient.name == "Bo Wang"

        # Check Markdown content
        md_text = md_path.read_text(encoding="utf-8")
        assert "# Academic Outreach Inquiry: MedSAM" in md_text
        assert "bowang@vectorinstitute.ai" in md_text
        assert "### Email Body" in md_text

    def test_handle_audit_report_needs_outreach(self, outreach_agent, mock_medsam_manuscript):
        """Auditor report with NEEDS_OUTREACH triggers draft generation automatically."""
        extracted = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
            extracted_scores=[
                RawExtractedScore(dataset_name="BTCV", organ_or_task="Abdomen", dice_score=None),
            ],
        )
        report = AuditReport(
            model_id="MedSAM",
            status=AuditStatus.NEEDS_OUTREACH,
            passed_sanity_checks=True,
            passed_inclusion_screening=True,
            data_diagnosis=MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND,
        )

        draft = outreach_agent.handle_audit_report(extracted, mock_medsam_manuscript, report, auto_save=True)

        assert draft is not None
        assert draft.model_id == "MedSAM"
        assert draft.draft_filepath is not None
        assert Path(draft.draft_filepath).exists()

    def test_handle_audit_report_verified_skipped(self, outreach_agent, mock_medsam_manuscript):
        """Auditor report with VERIFIED status does not generate an outreach draft."""
        extracted = ExtractedPaperData(
            paper_title="Segment Anything in Medical Images",
            proposed_model_id="MedSAM",
            screening=InclusionCriteriaCheck(is_3d=True, is_radiological=True, is_public_dataset=True, has_dice_metric=True),
        )
        report = AuditReport(
            model_id="MedSAM",
            status=AuditStatus.VERIFIED,
            passed_sanity_checks=True,
            passed_inclusion_screening=True,
            data_diagnosis=MissingDataDiagnosis.NONE,
        )

        draft = outreach_agent.handle_audit_report(extracted, mock_medsam_manuscript, report, auto_save=True)
        assert draft is None
