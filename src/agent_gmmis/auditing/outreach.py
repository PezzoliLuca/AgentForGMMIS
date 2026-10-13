"""OutreachAgent for the Delegation Layer of AgentForGMMIS.

Operationalizes the scholarly inquiry methodology established by Moglia et al.:
"In cases where the Dice score was only available in graphical form without
explicit numerical values, we contacted the corresponding authors."

Extracts author contact info from manuscript front matter, identifies missing
scalar targets (radar plots, boxplots, bar charts), composes formal academic
inquiry emails in English, and persists drafts for Human-in-the-Loop review.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from agent_gmmis.auditing.schemas import (
    AuditReport,
    AuditStatus,
    AuthorContactInfo,
    ConsensusAuditReport,
    MissingDataDiagnosis,
    OutreachDraft,
    OutreachStatus,
)
from agent_gmmis.config import PROJECT_ROOT
from agent_gmmis.extraction.pdf_parser import ParsedManuscript
from agent_gmmis.extraction.schemas import ExtractedPaperData

logger = logging.getLogger(__name__)

# Standard institutional keywords to recognize academic affiliations
AFFILIATION_KEYWORDS: Set[str] = {
    "university",
    "institute",
    "department",
    "hospital",
    "centre",
    "center",
    "college",
    "school",
    "laboratory",
    "lab",
    "clinic",
    "faculty",
    "foundation",
    "academy",
    "health network",
}

# Standard email regex pattern conforming to RFC 5322 subset
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


class OutreachAgent:
    """Automates scholarly author outreach drafting for papers with graphical-only metrics."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        """Initialize the OutreachAgent.

        Args:
            output_dir: Directory where JSON and Markdown drafts are stored.
                        Defaults to data/outreach_drafts/.
            llm_client: Optional LLMClient for structured LLM-assisted metadata fallback.
        """
        if output_dir is None:
            self.output_dir = PROJECT_ROOT / "data" / "outreach_drafts"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client

    # -------------------------------------------------------------------------
    # 1. Contact Information Extraction
    # -------------------------------------------------------------------------
    def extract_contact_info(
        self,
        manuscript: ParsedManuscript,
        extracted: Optional[ExtractedPaperData] = None,
    ) -> AuthorContactInfo:
        """Extracts corresponding author name, email, and affiliation from manuscript.

        Employs hierarchical heuristics:
        1. Parse Page 1 text for corresponding author footnotes and email markers (*, ✉, 'e-mail').
        2. Match email username with listed author names.
        3. Extract institution from affiliation lines.
        4. Fallback to first author / paper authors with reduced confidence if no email found.
        """
        page_1_text = manuscript.pages_text.get(1, "") if manuscript.pages_text else ""
        if not page_1_text:
            # Fallback if pages_text is empty
            page_1_text = manuscript.get_full_text(max_pages=2)

        # 1. Extract all emails on front page
        emails = EMAIL_REGEX.findall(page_1_text)
        # Clean trailing punctuation from emails
        cleaned_emails = [re.sub(r"[;,.:]+$", "", e).strip() for e in emails]

        # 2. Identify corresponding author marker block
        # Examples: "*Corresponding author", "Correspondence to:", "E-mail:", "✉"
        corr_patterns = [
            r"(?i)\*?\s*corresponding\s+author[s]?\s*[:*†‡§\(]?[^\n\r]*?(?:e-?mail[s]?\s*[:*†‡§]?)?\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
            r"(?i)correspondence\s+(?:to|at)\s*[:*†‡§]?\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
            r"(?i)e-?mail[s]?\s*[:*†‡§]?\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
        ]

        primary_email = None
        for pat in corr_patterns:
            m = re.search(pat, page_1_text)
            if m:
                primary_email = re.sub(r"[;,.:]+$", "", m.group(1)).strip()
                break

        if not primary_email and cleaned_emails:
            primary_email = cleaned_emails[0]

        # 3. Identify Author Name
        author_name = self._identify_author_name(page_1_text, primary_email, manuscript.authors)

        # 4. Identify Affiliation
        affiliation = self._identify_affiliation(page_1_text, author_name, extracted)

        is_corr = bool(
            re.search(r"(?i)corresponding\s+author|correspondence\s+to|\*|✉", page_1_text)
            and primary_email
        )

        confidence = 1.0 if (primary_email and author_name != "Corresponding Author") else (0.7 if primary_email else 0.4)

        return AuthorContactInfo(
            name=author_name,
            email=primary_email,
            affiliation=affiliation,
            is_corresponding=is_corr,
            confidence=confidence,
        )

    def _identify_author_name(
        self,
        page_1_text: str,
        primary_email: Optional[str],
        known_authors: str,
    ) -> str:
        """Identifies the target author name using email usernames, footnote asterisks, and author blocks."""
        # 1. If explicit footnote specifies the author: e.g. "Bo Wang1,2,3,7,8*" or "*Corresponding author: Bo Wang"
        explicit_named_corr = re.search(
            r"(?i)\*?\s*corresponding\s+author[s]?\s*[:]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
            page_1_text,
        )
        if explicit_named_corr:
            name = explicit_named_corr.group(1).strip()
            if not any(kw in name.lower() for kw in ("e-mail", "email", "university", "department")):
                return name

        # 2. Check for author with asterisk: "Bo Wang1,2,3*" or "Victor Ion Butoi*"
        asterisk_name = re.search(
            r"(?:^|\n)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})[\d,\s]*\*(?!\*)",
            page_1_text,
        )
        if asterisk_name:
            name = asterisk_name.group(1).strip()
            if not any(kw in name.lower() for kw in ("abstract", "introduction", "university", "department")):
                return name

        # 3. If primary_email is known, try to match username to names on front page
        if primary_email:
            username = primary_email.split("@")[0].lower()
            # Clean non-alpha
            clean_user = re.sub(r"[^a-z]", "", username)

            # Search in title / author header before Abstract
            pre_abstract = page_1_text.split("Abstract")[0] if "Abstract" in page_1_text else page_1_text[:1200]
            candidate_names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", pre_abstract)
            for c_name in candidate_names:
                parts = [p.lower() for p in c_name.split()]
                # Check if parts appear in username (e.g. bowang -> bo + wang, vbutoi -> butoi)
                if any(p in clean_user for p in parts if len(p) >= 3):
                    return c_name

        # 4. Fallback to first author in known_authors or first candidate name
        if known_authors and known_authors.strip():
            first = known_authors.split(",")[0].split(";")[0].strip()
            if first:
                return first

        pre_abstract = page_1_text.split("Abstract")[0] if "Abstract" in page_1_text else page_1_text[:800]
        candidate_names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", pre_abstract)
        for c in candidate_names:
            if not any(kw in c.lower() for kw in ("abstract", "arxiv", "paper", "deep", "learning", "segmentation", "medical")):
                return c

        return "Corresponding Author"

    def _identify_affiliation(
        self,
        page_1_text: str,
        author_name: str,
        extracted: Optional[ExtractedPaperData],
    ) -> Optional[str]:
        """Extracts the primary institution or department for the author."""
        if extracted and extracted.authors_affiliations:
            return extracted.authors_affiliations.strip()

        lines = [line.strip() for line in page_1_text.split("\n") if line.strip()]

        # 1. Look for lines directly following author_name before the next author or email
        if author_name and author_name != "Corresponding Author":
            clean_target = re.sub(r"[^a-zA-Z\s]", "", author_name).lower().strip()
            for idx, line in enumerate(lines[:30]):
                clean_line = re.sub(r"[^a-zA-Z\s]", "", line).lower().strip()
                if clean_target in clean_line or clean_line in clean_target:
                    # Found author line; check next line(s)
                    if idx + 1 < len(lines):
                        next_line = lines[idx + 1]
                        # If next line is not an email and not an author name or abstract, it's the affiliation
                        if "@" not in next_line and not next_line.lower().startswith("abstract") and len(next_line) >= 3:
                            cleaned = re.sub(r"^[\d,\s*†‡§]+", "", next_line).strip()
                            return cleaned

        # 2. Look for lines containing academic keywords in page 1
        affil_lines = []
        for line in lines[:40]:
            l_lower = line.lower()
            if any(kw in l_lower for kw in AFFILIATION_KEYWORDS):
                cleaned = re.sub(r"^[\d,\s*†‡§]+", "", line).strip()
                if len(cleaned) >= 5 and not cleaned.lower().startswith("abstract"):
                    affil_lines.append(cleaned)
                    if len(affil_lines) >= 2:
                        break

        if affil_lines:
            return ", ".join(affil_lines)
        return None

    # -------------------------------------------------------------------------
    # 2. Missing Targets & Figures Identification
    # -------------------------------------------------------------------------
    def identify_missing_targets(
        self,
        extracted: ExtractedPaperData,
        manuscript: Optional[ParsedManuscript] = None,
    ) -> Tuple[List[str], List[str]]:
        """Identifies target benchmarks and specific figures where scalar metrics are missing.

        Returns:
            Tuple of (target_datasets, target_figures)
        """
        target_datasets: List[str] = []
        target_figures: List[str] = []

        # 1. Datasets extracted with dice_score is None
        for score in extracted.extracted_scores:
            if score.dice_score is None and score.dataset_name:
                if score.dataset_name not in target_datasets:
                    target_datasets.append(score.dataset_name)
            if score.table_or_section_ref and any(fig_w in score.table_or_section_ref.lower() for fig_w in ("fig", "figure", "chart", "plot")):
                if score.table_or_section_ref not in target_figures:
                    target_figures.append(score.table_or_section_ref)

        # 2. If no target datasets found from null scores, use all unique datasets extracted
        if not target_datasets:
            unique_sets = {s.dataset_name for s in extracted.extracted_scores if s.dataset_name}
            target_datasets = sorted(list(unique_sets))

        # 3. Search manuscript for figure captions presenting quantitative metrics
        if manuscript:
            fig_pattern = re.compile(
                r"(?i)\b(Fig(?:ure|\.)\s*\d+[a-z]?)\s*[:.]\s*([^\n\r]+)",
            )
            for page_num, text in manuscript.pages_text.items():
                matches = fig_pattern.findall(text)
                for fig_id, caption in matches:
                    c_low = caption.lower()
                    # Check if caption relates to performance evaluation
                    if any(kw in c_low for kw in ("dice", "dsc", "comparison", "quantitative", "evaluation", "radar", "boxplot", "bar chart", "performance", "results")):
                        clean_fig = fig_id.strip()
                        if clean_fig not in target_figures:
                            target_figures.append(clean_fig)

        # Fallback if no specific figures detected
        if not target_figures:
            target_figures = ["evaluation figures"]

        if not target_datasets:
            target_datasets = ["3D medical segmentation benchmarks"]

        return target_datasets, target_figures

    # -------------------------------------------------------------------------
    # 3. Email Drafting Engine
    # -------------------------------------------------------------------------
    def compose_email(
        self,
        extracted: ExtractedPaperData,
        contact: AuthorContactInfo,
        target_datasets: List[str],
        target_figures: List[str],
    ) -> Tuple[str, str]:
        """Composes a professional, formal academic inquiry email in English.

        Returns:
            Tuple of (email_subject, email_body)
        """
        model_id = extracted.proposed_model_id or "Your Model"
        paper_title = extracted.paper_title or "your manuscript"

        subject = f"[Academic Inquiry] Quantitative evaluation metrics for \"{paper_title}\" ({model_id})"

        # Salutation
        if contact.name and contact.name != "Corresponding Author":
            salutation = f"Dear Prof./Dr. {contact.name},"
        else:
            salutation = f"Dear Authors of \"{paper_title}\","

        # Datasets clause
        if len(target_datasets) == 1:
            dataset_clause = f"the {target_datasets[0]} benchmark"
        elif len(target_datasets) <= 3:
            dataset_clause = f"the {', '.join(target_datasets[:-1])} and {target_datasets[-1]} benchmarks"
        else:
            dataset_clause = f"several benchmark datasets (including {', '.join(target_datasets[:3])}, among others)"

        # Figures clause
        if len(target_figures) == 1 and target_figures[0] == "evaluation figures":
            figures_clause = "in visual graphical form (charts/plots)"
        elif len(target_figures) == 1:
            figures_clause = f"in graphical form in {target_figures[0]}"
        else:
            figures_clause = f"in graphical format ({', '.join(target_figures)})"

        body = f"""{salutation}

I hope this email finds you well.

I am reaching out on behalf of the Generalist Medical 3D Segmentation (GMMIS) benchmark initiative at the Department of Electronics, Information and Bioengineering (DEIB), Politecnico di Milano.

We have read with great interest your paper:
\"{paper_title}\",
which presents {model_id}. As part of our comprehensive open-access study cataloging 3D medical foundation models, we are systematically evaluating published architectures across standardized radiological benchmarks.

In your manuscript, the quantitative results for {dataset_clause} appear to be reported primarily {figures_clause} without explicit scalar numerical values (e.g. Mean Dice Similarity Coefficient / standard deviation in tabular format).

To ensure that {model_id} is accurately and fairly represented in the benchmark and comparative meta-analysis, without relying on approximate coordinate readings from charts, would your team be willing to share the underlying numerical Mean Dice scores for these evaluations?

Even a brief tabular summary, raw CSV export, or evaluation log snippet would be immensely valuable for the community. Your work will be fully cited and credited in the GMMIS public database and corresponding survey publication.

Thank you very much for your time, consideration, and dedication to open science.

Sincerely,

Luca Pezzoli
Department of Electronics, Information and Bioengineering (DEIB)
Politecnico di Milano, Milan, Italy
Email: luca.pezzoli@polimi.it
GitHub: https://github.com/PezzoliLuca/AgentForGMMIS
"""
        return subject, body

    # -------------------------------------------------------------------------
    # 4. Draft Generation & Persistence
    # -------------------------------------------------------------------------
    def generate_draft(
        self,
        extracted: ExtractedPaperData,
        manuscript: ParsedManuscript,
        target_datasets: Optional[List[str]] = None,
        target_figures: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> OutreachDraft:
        """Generates a complete, structured academic outreach draft."""
        contact = self.extract_contact_info(manuscript, extracted=extracted)

        if target_datasets is None or target_figures is None:
            auto_datasets, auto_figures = self.identify_missing_targets(extracted, manuscript)
            target_datasets = target_datasets or auto_datasets
            target_figures = target_figures or auto_figures

        subject, body = self.compose_email(
            extracted=extracted,
            contact=contact,
            target_datasets=target_datasets,
            target_figures=target_figures,
        )

        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        return OutreachDraft(
            model_id=extracted.proposed_model_id,
            paper_title=extracted.paper_title,
            recipient=contact,
            target_datasets=target_datasets,
            target_figures=target_figures,
            email_subject=subject,
            email_body=body,
            created_at=now_iso,
            status=OutreachStatus.DRAFT_PENDING_REVIEW,
            notes=notes or "Automatically generated by OutreachAgent for human-in-the-loop review.",
        )

    def save_draft(
        self,
        draft: OutreachDraft,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """Saves the outreach draft in both JSON (machine-readable) and Markdown (human-reviewable) formats.

        Returns:
            Tuple of (json_path, md_path)
        """
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", draft.model_id).lower()
        json_path = target_dir / f"{clean_id}_outreach.json"
        md_path = target_dir / f"{clean_id}_outreach.md"

        # Update draft filepath
        draft.draft_filepath = str(json_path)

        # 1. Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(draft.model_dump_json(indent=2))

        # 2. Write Markdown
        md_content = f"""# Academic Outreach Inquiry: {draft.model_id}

> **Status**: `{draft.status.value.upper()}`  
> **Created**: `{draft.created_at}`  
> **Model ID**: `{draft.model_id}`  
> **Paper Title**: {draft.paper_title}  
> **Recipient**: {draft.recipient.name} {f'<{draft.recipient.email}>' if draft.recipient.email else '(No email extracted)'}  
> **Affiliation**: {draft.recipient.affiliation or 'Not specified'}  
> **Confidence**: {draft.recipient.confidence * 100:.1f}%  
> **Target Benchmarks**: {', '.join(draft.target_datasets) if draft.target_datasets else 'General 3D'}  
> **Target Figures**: {', '.join(draft.target_figures) if draft.target_figures else 'None specified'}  

---

### Email Subject
```text
{draft.email_subject}
```

### Email Body (Ready to Review & Send)
```text
{draft.email_body}
```

---
*Note: This draft was automatically generated by AgentForGMMIS OutreachAgent following the methodology of Moglia et al. Please review before sending.*
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Saved outreach drafts to {json_path} and {md_path}")
        return json_path, md_path

    # -------------------------------------------------------------------------
    # 5. Auditor Integration Workflow
    # -------------------------------------------------------------------------
    def handle_audit_report(
        self,
        extracted: ExtractedPaperData,
        manuscript: ParsedManuscript,
        report: Union[AuditReport, ConsensusAuditReport],
        auto_save: bool = True,
    ) -> Optional[OutreachDraft]:
        """Triage helper: if an audit report requires outreach, generates and saves the draft.

        Args:
            extracted: Structured extraction output.
            manuscript: Parsed manuscript instance.
            report: AuditReport or ConsensusAuditReport.
            auto_save: If True, immediately writes JSON and Markdown to disk.

        Returns:
            OutreachDraft if outreach was necessary, None otherwise.
        """
        needs_outreach = (
            report.status == AuditStatus.NEEDS_OUTREACH
            or report.data_diagnosis == MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND
        )

        if not needs_outreach:
            return None

        notes = f"Triggered by AuditReport: status={report.status.value}, diagnosis={report.data_diagnosis.value}"
        draft = self.generate_draft(extracted, manuscript, notes=notes)

        if auto_save:
            self.save_draft(draft)

        return draft
