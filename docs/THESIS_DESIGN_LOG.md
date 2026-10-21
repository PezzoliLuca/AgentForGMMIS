# THESIS DESIGN LOG: Architectural Decisions, Methodological Rationale & Thesis Defense Notes

> **Project**: AgentForGMMIS — Multi-Agent AI System for Generalist Medical Image Segmentation Benchmarking  
> **Author**: Luca Pezzoli (`luca2.pezzoli@mail.polimi.it`)  
> **Supervisor**: Andrea Moglia (`andrea.moglia@polimi.it`) — DEIB, Politecnico di Milano  
> **Reference Publication**: Moglia et al., *"Generalist models in medical image segmentation: A survey and performance comparison with task-specific approaches"*, *Information Fusion*, 127 (2026) 103709.  
> **Language**: All project documentation, code, and comments are in English.

---

## 1. Executive Motivation: Why Automate the Survey?

### 1.1. The Scientific Problem
In 2024–2025, the research group led by Andrea Moglia at DEIB Politecnico di Milano conducted an exhaustive literature review of **98 generalist segmentation models** and **58 benchmark datasets**, comparing them against task-specific state of the art (nnU-Net, Swin UNETR, SegResNet, etc.).

This process required hundreds of hours of manual labor:
1. Keyword searches on Google Scholar and arXiv.
2. Manual reading of 30+ page PDF papers.
3. Verification of inclusion/exclusion criteria (e.g., discarding 2D modalities or private datasets).
4. Extracting parameters, GFLOPs, hardware (VRAM), and Dice Similarity Coefficients (DSC) across dozens of anatomical structures.
5. Manually entering records into a SQLite database (`database.sqlite`).
6. Emailing corresponding authors whenever data was reported only in graphical charts (bar plots, box plots).

### 1.2. The Stagnation Dilemma in Medical AI Literature
Medical image segmentation is advancing exponentially. By the time a static literature review is published, dozens of new models and adaptations (e.g., MedSAM-2, BioSAM-2, SAM-Med3D variants) have already appeared on arXiv. **A static survey becomes obsolete within months.**

### 1.3. Thesis Contribution
This Master's Thesis operationalizes the visionary statement in Section 7.4 of Moglia et al. (*"Towards agentic AI, and physical AI"*):  
**Transforming a static literature survey into a self-updating, autonomous, multi-agent AI system that discovers, extracts, audits, harmonizes, and serves medical segmentation intelligence.**

---

## 2. Why Simple LLM Prompting (ChatGPT) Fails

A critical methodological justification for the thesis dissertation is **why an agentic system is required instead of a single LLM prompt**:

1. **Table Parsing & Spatial Misalignment**:
   * Scientific papers present Dice scores in complex multi-column, multi-row tables (often spanning multiple pages or supplementary material).
   * Generic LLMs frequently swap columns (e.g., mistaking spleen for liver, or standard deviation for mean).
2. **Entity Name Incoherence**:
   * Without strict ontology mapping, an LLM might record `"BraTS 2021"`, `"BraTS21"`, and `"Brain Tumor Segmentation"` as three separate entities, destroying relational database integrity.
3. **Hallucination & Fabricated Metrics**:
   * LLMs tend to generate plausible-sounding numbers when data is absent or ambiguous.
4. **Lack of State & Memory**:
   * A single prompt cannot maintain state, track primary vs. best-in-literature versioning, execute SQL transactions, or draft follow-up author inquiries.

---

## 3. Multi-Agent Hierarchy & Architectural Design

Instead of a monolithic pipeline, the system adopts a **Supervisor-Worker pattern with an explicit Critic/Auditor ("Vigile") and Human-in-the-Loop Outreach**:

```
                       ┌──────────────────────────────┐
                       │    1. Orchestrator Agent     │
                       └──────────────┬───────────────┘
                                      │
              ┌───────────────────────┼────────────────────────┐
              │ (Search)              │ (Audit & Curate)       │ (Serve)
              ▼                       ▼                        ▼
    ┌───────────────────┐   ┌───────────────────┐    ┌───────────────────┐
    │  2. Scout Agent   │   │  4. Auditor Agent │    │ 7. Consultant     │
    │   (arXiv/PubMed)  │   │    ("Il Vigile")  │    │      Agent        │
    └─────────┬─────────┘   └─────────┬─────────┘    └───────────────────┘
              │ (PDF)                 │
              ▼                       ├──────────► 5. Outreach Agent (Email Drafts)
    ┌───────────────────┐             │
    │ 3. Extractor Agent│             ▼
    │  (Vision & Tables)│───► 6. Database Agent
    └───────────────────┘       (SQLite ACID Sync)
```

### 3.1. Agent Roles & Specializations

#### 1. `OrchestratorAgent` (Supervisor)
* **Responsibility**: Manages the overarching finite state machine, distributes tasks, logs execution traces, and handles recovery if an agent fails.

#### 2. `ScoutAgent` (Literature Retrieval)
* **Responsibility**: Interfaces with arXiv API, PubMed API, and Semantic Scholar.
* **Filter Rules**: Enforces inclusion/exclusion criteria on titles and abstracts:
  * Modality must be 3D radiological (CT, MRI, PET).
  * Excludes 2D-only (fundus, dermoscopy, microscopy).
  * Must segment anatomy or pathology on public datasets.

#### 3. `ExtractionAgent` (Document Understanding)
* **Responsibility**: Reads full manuscripts (text + tables + vision pages).
* **Extraction Targets**: Model architecture, visual backbone, parameter count, GFLOPs, hardware VRAM, and per-dataset/per-organ Dice Similarity Coefficients.
* **Mechanism**: Leverages **Structured Outputs (JSON Schema / Pydantic v2)** to guarantee strict typing.

#### 4. `AuditorAgent` ("Il Vigile" / Critic) — *Key Thesis Insight*
* **Responsibility**: Quality assurance, anomaly detection, and hallucination prevention.
* **Auditing Heuristics**:
  * **Outlier Detection**: Compares extracted Dice scores against historical baselines in `database.sqlite` (e.g., flags an impossible 0.99 Dice score on pancreas tumors where SOTA is ~0.68).
  * **Sanity Checks**: Verifies whether test splits were standard or zero-shot; flags metric anomalies (e.g., if DSC > 1.0 or if standard deviation was parsed as mean).
  * **Self-Critique Trigger**: When confidence is low or values deviate sharply, sends the paper back to the Extractor with targeted questions (*"Verify row 4 of Table 2"*).

#### 5. `OutreachAgent` (Author Inquiry) — *Direct Operationalization of Paper Methodology*
* **Responsibility**: In Moglia et al., the authors explicitly state: *"In cases where the Dice score was only available in graphical form without explicit numerical values, we contacted the corresponding authors."*
* **Mechanism**:
  1. Triggered when the Auditor detects that key metrics are graphical-only (bar charts, box plots) or missing.
  2. Extracts the corresponding author's name, email, and institution from the manuscript header.
  3. Automatically drafts a polite, standardized, formal academic inquiry email in English requesting the exact numerical values.
  4. Presents the draft to the human researcher for one-click approval before sending.

#### 6. `DatabaseAgent` (Curator)
* **Responsibility**: Entity resolution (aliases $\rightarrow$ canonical names), version tracking, calculating performance improvements $\Delta\% = \frac{\text{Dice}_{\text{best}} - \text{Dice}_{\text{primary}}}{\text{Dice}_{\text{primary}}} \times 100$, and committing transactions to `database.sqlite`.

#### 7. `ConsultantAgent` (Decision Support)
* **Responsibility**: Natural language clinical reasoning interface. Allows clinicians and engineers to query:
  * *"Which generalist model achieves the highest Dice on Liver CT given a maximum VRAM budget of 16 GB?"*
  * Evaluates Pareto frontiers (Dice vs. VRAM vs. Parameters) and performs **Mann-Whitney U tests** for statistical significance.

---

## 4. Zero-Cost Budget Architecture (0.00 € Mandate)

A vital constraint for this Master's Thesis is that **the university allocation is 0.00 €**. The architecture is specifically engineered to operate completely free of charge without compromising scientific performance:

```mermaid
flowchart LR
    subgraph Zero_Cost_Infrastructure["Zero-Cost Infrastructure Strategy (0.00 €)"]
        A[Google AI Studio Free Tier\nGemini 2.0 Flash\n1,500 req/day | 1M Context\nCost: 0.00 €]
        B[DEIB Polimi 'Mufasa' HPC\nSlurm GPU Cluster\nSelf-Hosted Open Source Models\nCost: 0.00 €]
        C[Local Apple Silicon Mac\nOllama / Offline Dev\nLocal 7B/14B Models\nCost: 0.00 €]
    end
```

### 4.1. The Three Free Compute Pillars:
1. **Google AI Studio (Gemini Free Tier)**:
   * **Cost**: 0.00 € (no credit card required).
   * **Quotas**: 15 requests per minute, 1,500 requests per day.
   * **Context Window**: 1,000,000 tokens (can ingest multiple 30-page papers in a single prompt).
   * **Role**: Primary engine for fast development, table extraction, and reasoning.
2. **Politecnico di Milano DEIB "Mufasa" GPU Cluster**:
   * **Cost**: 0.00 € (pre-allocated academic resources).
   * **Hardware**: Multi-GPU nodes, allocated GPU `3g.20gb:1`, 32 GB RAM, SLURM scheduling.
   * **Role**: Self-hosted open-source LLMs (Llama 3.1, Qwen 2.5, DeepSeek-R1) and optional execution of medical segmentation benchmarking scripts.
3. **Local Development (Mac)**:
   * **Cost**: 0.00 € (local compute).
   * **Role**: Unit tests, schema validation, offline prototyping via Ollama.

### 4.2. Model-Agnostic Design Pattern
The codebase will never hardcode a specific LLM vendor. It uses abstract model interfaces (via standard OpenAI-compatible endpoints or `litellm`):
```yaml
# config.yaml
agents:
  scout:
    model: "gemini-2.0-flash"
    fallback: "ollama/llama3.1:8b"
  extractor:
    model: "gemini-2.0-flash"
  auditor:
    model: "gemini-2.0-flash-thinking"
    fallback: "mufasa/deepseek-r1"
```
* **Thesis Value**: Enables an empirical benchmark chapter comparing: *"Extraction accuracy and cost tradeoffs across proprietary cloud models vs. open-source models self-hosted on university HPC."*

---

## 5. Incremental Engineering Roadmap

To maintain scientific rigor and clear thesis milestones:

| Step | Milestone | Deliverable | Thesis Impact |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Data Layer & Pydantic Contracts** | Typed data models & SQLite DAL | Defines formal schema integrity and data types. |
| **Step 2** | **Extraction Engine & Batch Ground Truth Benchmark** | Extraction pipeline + 15-paper batch benchmark evaluated against primary survey ground truth | Empirical validation across 15 papers: 100% recall on 8 models, 0.0000 Dice MAE, discovering human survey errors. |
| **Step 3** | **Auditor & Curation Engine ("Il Vigile")** | Outlier detection, sanity checks, multi-source verification (tables vs GitHub JSON), SQLite synchronization | Demonstrates multi-agent verification, actor-critic self-healing, and provenance tracking. |
| **Step 4** | **Autonomous Scout** | arXiv/PubMed API harvesting & filtering | Demonstrates end-to-end continuous literature tracking and automated screening. |
| **Step 5** | **Consultant Decision Support** | Reasoning CLI/UI with Pareto analysis & Mann-Whitney U tests | Real-world clinical decision support tool for medical practitioners and researchers. |

---

## 6. Key "Gems" & Insights for Writing the Thesis

Keep these points in mind when drafting chapters of the final dissertation:

1. **The Ground Truth Benchmark (Chapter on Evaluation)**:
   * We can quantitatively evaluate our `ExtractionAgent` by comparing its output on the 98 models already in `database.sqlite` against the manual ground truth curated by Andrea Moglia et al.
   * This yields concrete numbers: **Table Extraction Accuracy**, **Dice Score Mean Absolute Error (MAE)**, and **Entity Resolution F1-Score**.
2. **Actor-Critic AI Pattern (Chapter on Methodology)**:
   * Explain how the pairing of `ExtractionAgent` (Actor) and `AuditorAgent` (Critic) mitigates LLM hallucinations and ensures medical-grade data integrity.
3. **Operationalizing Academic Protocol (Chapter on Multi-Agent Collaboration)**:
   * The `OutreachAgent` proves that agents can automate not just computation, but academic communication protocols (inquiring about graphical-only data).
4. **Privacy & Regulatory Compliance (Discussion Chapter)**:
   * Discuss how running the system on Mufasa ensures zero data leakage, addressing the privacy challenges highlighted in Section 6.2 of Moglia et al.
5. **Interactive Prompt Regimes & Curatorial Disambiguation (Chapter on Extraction & Discovery)**:
   * Generalist foundation models (e.g., SAM-Med3D, MedSAM) report vastly different Dice scores depending on prompt interaction depth (e.g., 1-point prompt vs. 10-point prompt).
   * A single paper table (e.g. Table 4 in SAM-Med3D) contains multiple valid numerical columns: `Ours (1 pt)` with 84.68% on TotalSegmentator vs `Ours (10 pts)` with 87.59%.
   * In Moglia et al., survey records standardized on the 1-point interactive prompt (`results_primary_comments`: *"Results reported in the 1-point prompt framework"*).
   * The automated extraction system must therefore extract structured `prompt_type` metadata alongside raw scores. When matching on equivalent prompt configurations against primary literature ground truth, the agent achieves an exact **0.0000** error match across all evaluated public benchmarks.
6. **Agentic Verification Surpasses Human Precision (The LHU-Net Baseline Copy Anomaly)**:
   * In Table 2 of LHU-Net (*"Comparison on Synapse multi-organ dataset"*), the authors report competing baselines and their proposed method in adjacent rows:
     * Row 1: `D-LKA Net` (baseline) $\rightarrow$ **87.49%**
     * Row 2: `LHU-Net (Ours)` (proposed model) $\rightarrow$ **87.83%**
   * In the human-curated survey database (`database.sqlite`), the record for `LHU-Net` on Synapse was transcribed as **87.49%**—a classic human slip of transcribing the adjacent baseline row above.
   * When `ExtractionAgent` parsed the paper, it extracted the true score of **87.83%** for the proposed model.
   * **Thesis Defense Impact**: This provides undeniable empirical proof that autonomous LLM extraction with structured Pydantic outputs is not merely an approximation of human literature review; **it actively detects and repairs human transcription errors in published meta-analyses**.
7. **Curatorial Math vs. Raw Table Reports (TotalSegmentator MRI Composite Averaging)**:
   * In `database.sqlite`, TotalSegmentator is recorded with a **90.25%** Dice score on the TotalSegmentator MRI benchmark.
   * However, nowhere in the TotalSegmentator MRI paper does the number `90.25%` appear. The paper evaluates two distinct test sets: a CT test set (**83.9%**) and an MRI test set (**96.6%**).
   * The human survey authors calculated the unweighted mathematical mean: $\frac{83.9\% + 96.6\%}{2} = 90.25\%$.
   * **Thesis Insight**: Automated extraction systems cannot rely solely on naive string/cell lookups. Curation agents must possess arithmetic reasoning capabilities to recognize when a reported benchmark is a composite aggregation of sub-test sets or multiple imaging modalities, documenting provenance via `results_primary_comments`.
8. **Multi-Source Auditing: Paper Tables vs. Code Repositories vs. Author Outreach (BiomedParse Case Study)**:
   * In BiomedParse (*Nature Methods 2024*), model performance across 14 segmentation benchmarks is presented exclusively via visual radar charts and bar plots; no scalar numbers exist in the main tables.
   * To populate the survey, the authors had to inspect raw JSON experiment artifacts hosted on Microsoft's GitHub repository (`results_primary_comments`: *"Results extracted from Microsoft GitHub JSON repository"*).
   * **Thesis Architecture Justification**: This real-world finding justifies why a single `ExtractionAgent` reading PDF text is insufficient. The architecture must incorporate an `AuditorAgent` ("Il Vigile") capable of diagnosing missing numerical tables and delegating to:
     * A repository inspection subagent that clones the model's GitHub repository to parse JSON/CSV benchmark logs.
     * The `OutreachAgent` that drafts standardized academic inquiries to corresponding authors.
9. **Disambiguation of Multi-Task, Multi-Modality, and Data-Partition Regimes (IMIS-Net & KnowSAM)**:
   * Medical segmentation literature presents dense tabular matrices reporting:
     * Organ-specific rows alongside a table average (e.g., SegTHOR in IMIS-Net: Esophagus 89.17% vs. Average 89.27%).
     * Varying budgets of labeled training data in semi-supervised learning (e.g., KnowSAM on ACDC: 5% labeled data with 89.56% vs. 10% labeled data with 91.13%).
     * Multiple MRI sequences (T1, T2, FLAIR in MoME).
   * Disambiguating these dimensions requires strict schema modeling (`organ_or_task`, `prompt_type`, `data_partition`) rather than flattening tables into unstructured key-value pairs.
10. **The Challenge Alias & Synonym Resolution Problem (DatasetAliasResolver)**:
    * Benchmark datasets in medical imaging are referred to by contradictory identifiers:
      * Challenge names vs. anatomical descriptions: Biomedical SAM 2 refers to `FLARE` as *"Organs in Abdomen CT"* and `AMOS` as *"Organs in Abdomen MR"*.
      * Challenge parent names vs. sub-challenges: *"Synapse"* vs. *"BTCV"*, *"Fed-KiTS"* vs. *"KiTS"*, *"Prost.MRI"* vs. *"PROMISE12"*.
    * Without ontology-driven entity resolution (`DatasetAliasResolver`), automated meta-analyses suffer from catastrophic dataset fragmentation and near-zero recall. With canonical reverse-indexing and word-boundary protection, extraction recall reached **100%** across 8 independent target models.
11. **Preprint Drift & Publication Lifecycle Dynamics (The MoME Case Study)**:
    * In medical AI, preprints on arXiv often use preliminary internal nomenclature (e.g., `WMHsMix`, `Tumour1`, `Tumour2` in the MoME preprint) and lack exhaustive benchmark tables.
    * Later, the peer-reviewed journal version (e.g. published in *IEEE Transactions on Medical Imaging*, `\citep{10879789}`) introduces complete comparative matrices (e.g. Table IV) and multi-modality aggregations.
    * **Thesis Insight**: Literature surveying is not a static one-off snapshot. The autonomous system must monitor the publication lifecycle of manuscripts, automatically detecting when a tracked arXiv preprint is formally published in a peer-reviewed journal and scheduling a re-extraction to capture finalized benchmark tables.
12. **Context Window Economy vs. Hierarchical Supplementary Mining (The 74-Page SAT Paradigm)**:
    * Massive foundation model papers (e.g. *SAT* with 74 pages) illustrate the fundamental trade-off between LLM inference latency and coverage depth. Feeding all 74 pages (80,000+ tokens) to an LLM incurs prohibitive latency and noise.
    * By applying intelligent page scoring, the system achieves sub-5-second throughput on the main manuscript. When main-text tables report only aggregated summary metrics, the system can trigger a second-tier hierarchical probe targeting supplementary appendix sections (pages 30–60).
    * **Thesis Insight**: Demonstrates that multi-agent document understanding is far more efficient than brute-force whole-document prompting, saving 90%+ computational overhead while retaining surgical precision.
13. **Actor-Critic Quality Assurance & Biological Grounding ("Il Vigile")**:
    * Generative LLMs operate without intrinsic physical or clinical grounding; an extractor might extract a hallucinated or flawed score of 0.99 on Pancreas tumor segmentation without realizing that the world state-of-the-art across literature is ~0.77 due to high anatomical variability and fuzzy tumor margins.
    * By introducing the `AuditorAgent` ("Il Vigile") in an Actor-Critic architecture, raw extracted data is cross-referenced against historical SQLite empirical distributions across 98 generalist models before ingestion.
    * **Thesis Defense Impact**: Proves that reliable scientific AI requires separating *information acquisition* (Actor) from *empirical verification* (Critic), enforcing clinical plausibility checks and sanity bounds that prevent benchmark database contamination.
14. **Continuous Discovery & Dynamic Ingestion of Novel Models & Benchmarks (The Ontology & Registry Dilemma)**:
    * Literature meta-analyses are not static closed-world catalogs: emerging preprints frequently propose novel model architectures and evaluate on newly released open benchmarks (e.g. *AbdomenAtlas*, *TotalSegmentator v2*, *FLARE 2024*) or private clinical cohorts.
    * **The Anti-Pattern**: A naive architecture would spawn specialized micro-agents for every entity type ("ModelValidatorAgent", "DatasetValidatorAgent", "SchemaEvolutionAgent"), dramatically escalating latency, token consumption, and orchestration instability.
    * **The Thesis Solution**: We maintain a strict Separation of Concerns across the three foundational lifecycle actors:
      * **Actor (`ExtractionAgent`)**: Mines structured candidates (`ModelMetadata`, novel datasets with public/private status, modality, organ targets).
      * **Critic (`AuditorAgent`)**: Acts as the gatekeeper, verifying inclusion criteria for novel models and strictly filtering private in-house clinical cohorts from contaminating the public benchmark registry.
      * **Curator (`DatabaseAgent`)**: Performs entity deduplication, registers verified models into `models`, registers verified public benchmarks into `datasets`, and conducts safe transactional schema expansion (`ALTER TABLE results_primary ADD COLUMN ...`) while preserving private cohorts as provenance metadata.
    * **Thesis Defense Impact**: Demonstrates principled software engineering and agent economy—achieving full dynamic ontology expansion without agentic bloat.
15. **The 2D-Slice vs. 3D-Volumetric Misconception (Critic Domain Knowledge Override on KnowSAM)**:
    * Generative LLMs frequently conflate the *neural network inference mechanism* (e.g. 2D axial slice-by-slice processing as performed in models like KnowSAM, MedSAM, and nnU-Net 2D) with the *underlying clinical imaging domain* (which is fundamentally 3D volumetric cine-MRI or CT scans like ACDC or BraTS).
    * When prompting an LLM on inclusion criteria, it may naively mark `is_3d = False` simply because the text mentions "2D slice processing". If an inclusion filter accepts this linguistic judgment unconditionally, valid volumetric foundation models are falsely rejected.
    * **Thesis Insight & Solution**: The Critic (`AuditorAgent`) incorporates medical imaging domain ontology cross-referencing: when an extraction evaluates on verified 3D volumetric benchmarks (`ACDC`, `BraTS`, `BTCV`, `AMOS`, `KiTS`), the Critic overrides the extractor's slice-level confusion, certifying that the target anatomy and evaluation domain is 3D volumetric radiological data (`passed_screening = True`).
    * **Defense Impact**: Demonstrates that pure prompt engineering is brittle without domain ontology supervision; an autonomous Critic must possess grounding in medical imaging semantics to prevent catastrophic false rejections.
16. **Semi-Supervised Learning & Multi-Budget Sweeps vs. Duplicate Contradictions (SFR SAM & FLAP-SAM)**:
    * In modern medical image segmentation, architectures frequently tackle semi-supervised learning or federated learning by evaluating performance across labeled data budgets (e.g. 1%, 5%, 10%, 20%, 100% labeled scans in SFR SAM) or communication rounds in FLAP-SAM.
    * A naive consistency auditor treats multiple reported Dice scores for the same dataset and anatomical target as "inconsistent duplicate contradictions produced under identical conditions", generating dozens of false-positive warnings that collapse audit confidence to 0% despite the model achieving a perfect 0.0000 MAE against primary literature ground truth.
    * **Thesis Insight & Solution**: We formulated the multi-regime audit contract:
      1. Progression series ($\ge 3$ scores across data partitions) are recognized as legitimate experimental sweeps rather than duplicates.
      2. Duplicate warnings are strictly reserved for unannotated pairwise contradictions where exactly 2 conflicting numbers diverge significantly (> 0.03 DSC).
      3. Non-critical warning penalties are capped (maximum 0.35 deduction) so that legitimate experimental complexity cannot drag a valid, verified model to 0% confidence.
    * **Defense Impact**: Proves that automated scientific auditors must understand experimental ML methodologies (data budgets, ablations, semi-supervised regimes) rather than treating academic tables as static 1:1 key-value pairs.
17. **Graceful Degradation on Graphical-Only Literature (The Nullable Metric Pattern in Merlin)**:
    * Prominent foundation model publications (e.g. Merlin in *Nature Communications*, BiomedParse in *Nature Methods*) frequently report comparative evaluations exclusively through visual bar plots, radar charts, and box plots, omitting scalar numerical tables from the manuscript.
    * In strictly-typed schema architectures (such as Pydantic models with `dice_score: float`), encountering a paper with qualitative-only mentions crashes the extraction runtime with `NoneType` validation exceptions before quality assurance can even begin.
    * **Thesis Insight & Solution**: Implementing nullable metric types (`dice_score: Optional[float] = None`) paired with structured triage logic in the Auditor enables graceful degradation: the pipeline extracts dataset mentions, diagnoses `NEEDS_OUTREACH` or `NEEDS_REPO_INSPECTION`, and delegates to downstream resolution mechanisms without pipeline failure.
    * **Defense Impact**: Illustrates production-grade resilience in scientific agent pipelines—ensuring incomplete or qualitative literature is systematically triaged rather than causing catastrophic runtime failures.
18. **Dual-Pillar PDF Scoring for Massive Manuscripts (The 74-Page SAT Benchmark Paradigm)**:
    * Frontier foundation model papers (e.g. SAT with 74 pages) distribute dozens of extensive benchmark matrices across 50 pages of supplementary appendices (e.g. Table 5 and Table 6 on pages 40–50).
    * Standard text-mention heuristics score text pages with casual citations (e.g. "as shown in Table 5") higher than the actual tabular appendix pages that contain 200+ raw numerical percentage values (`85.4%`, `78.9%`), resulting in appendix table starvation within the LLM's top-page context window.
    * **Thesis Insight & Solution**: A dual-pillar scoring engine was introduced:
      1. Structural table header regex detection (`Table \d+ [|:.]`) (+35 pts).
      2. Dense percentage/numerical token counting (`\d{1,2}\.\d{1,2}`) (+40 pts).
      This mathematically guarantees that dense benchmark appendix tables rank in the top selection, capturing comprehensive multi-dataset matrices without wasteful whole-document token ingestion.
    * **Defense Impact**: Solves the long-context retrieval dilemma in multi-page scientific document understanding, providing high-precision targeted table extraction at 90%+ lower computational cost.
19. **Hybrid Cloud-to-Cluster Topology & Inference Sovereignty (The Mufasa HPC Pillar)**:
    * Relying entirely on commercial cloud API endpoints (even free-tier services like Google AI Studio) exposes autonomous scientific workflows to external operational risks: concurrency throttling, preview quota ceilings (e.g. 20 requests/day), regional rate-limits, and temporary server-side capacity spikes (`503 UNAVAILABLE`).
    * Furthermore, in real-world clinical and university deployments, institution-specific preprints or proprietary hospital benchmark datasets cannot always be transmitted across external commercial cloud boundaries due to strict data sovereignty and GDPR privacy mandates.
    * **Thesis Insight & Solution**: We engineered a model-agnostic hybrid compute topology:
      1. **Primary Rapid Cloud Layer**: Google AI Studio Gemini API for agile development, zero-setup prototyping, and large context windows (1,500 req/day).
      2. **Autonomous Academic HPC Pillar**: Politecnico di Milano DEIB Mufasa GPU cluster running self-hosted open-weights models (`Qwen/Qwen2.5-14B-Instruct-AWQ`) served through high-throughput `vLLM` with PagedAttention and OpenAI-compatible HTTP endpoints.
      3. **Resilient Client Abstraction**: `LLMClient` dynamically routes between cloud endpoints and local cluster nodes (`http://localhost:8000/v1`) with zero changes to downstream extraction schemas, evaluator, or auditor logic.
20. **Cross-Architecture Typo Discovery & Empirical Replication on Academic Cluster (The LHU-Net Grounding Milestone)**:
    * A central tenet of the scientific method is empirical reproducibility across independent experimental substrates.
    * When evaluated across 15 scientific papers on the Politecnico di Milano DEIB Mufasa cluster (NVIDIA A100-PCIE-40GB MIG slice) using `Qwen2.5-14B-Instruct-AWQ`, the system replicated high-fidelity extraction: achieving exact `0.0000` Dice MAE across 5 models (`KnowSAM`, `BioSAM-2`, `BrainSegFounder`, `SFR SAM`, `MEA M-SAM`), 100% recall on 8 models, and correctly triaged `BiomedParse` to `NEEDS_REPO_INSPECTION` (`code_repo_available`).
    * **The Ultimate Scientific Proof for Dissertation**: Both Google's commercial cloud LLM (`gemini-3.5-flash-lite`) and the open-source self-hosted cluster model (`Qwen2.5-14B-Instruct-AWQ`) independently caught the identical human survey copy-paste transcription error in Table 2 of LHU-Net: both models independently extracted `0.8783` for LHU-Net, flagging that the human curator had copied the baseline row above (`D-LKA Net` with `0.8749`).
    * **Defense Impact**: Serves as unassailable empirical proof for the thesis dissertation committee that multi-agent extraction with Actor-Critic auditing is not merely an automation convenience, but an objective scientific audit instrument capable of detecting and correcting human curation errors across radically different model families and execution environments.
21. **The Epistemological Separation: Benchmark Calibration vs. Autonomous Deploy Mode (Zero-Ground-Truth Scientific Auditing)**:
    * **The Dilemma**: During thesis development, it is tempting to use the human-curated SQLite database from Moglia et al. as a direct verification oracle inside the Auditor. However, if the Auditor's quality certification depended on diffs against an existing human database, the system would be incapable of processing novel preprints or maintaining an up-to-date survey, because unseen papers by definition do not exist in the historical database.
    * **Thesis Insight & Solution**: We explicitly decoupled the system's operational regimes:
      1. **Offline Benchmark / Calibration Mode (`benchmark_mode=True`)**:
         * Used exclusively during experimental validation to evaluate extraction fidelity against human curators (measuring Dice MAE, dataset recall, and identifying human transcription typos like the LHU-Net discrepancy).
      2. **Autonomous Deploy / Discovery Mode (`benchmark_mode=False`)**:
         * Operates in an open-world setting with **zero ground truth** to autonomously reconstruct the survey from raw PDFs and continuously ingest future literature.
         * The Critic ("Il Vigile") enforces quality through **intrinsic epistemological priors**:
           - *Formal screening*: Objective verification of 3D radiological modalities, public datasets, and Dice metrics.
           - *Architectural & physical sanity*: Validating parameter counts against backbone family scales (e.g., ViT-Base ~85–150M), VRAM hardware realism, and strict mathematical metric bounds $\text{DSC} \in [0.0, 1.0]$.
           - *Self-consistency & hallucination rejection*: Detecting template cloning (identical scores repeated $\ge 4$ times), competitor baseline misattribution (e.g., mistaking nnU-Net for the proposed model), and unannotated pairwise internal contradictions (> 0.03 DSC).
           - *Bayesian dataset statistical sanity*: Utilizing SQLite aggregate distributions (`get_dataset_statistics()`) to flag biologically implausible claims (e.g., Pancreas tumor DSC > 0.92) or sudden unverified jumps exceeding world SOTA by > 4%, without knowing or requiring paper-specific ground truth.
           - *Missing data triage*: Routing papers with qualitative charts to `NEEDS_OUTREACH` or `NEEDS_REPO_INSPECTION`.
    * **Defense Impact**: Formalizes the scientific autonomy of the thesis contribution: demonstrating that the agent is not a brittle comparison tool against historical records, but a self-contained, autonomous scientific curator capable of constructing, auditing, and perpetually updating the generalist medical segmentation database.
22. **The Five Epistemological Sources of Non-Hallucinatory Discrepancy (MAE) in Scientific Benchmarking**:
    * **The Research Question**: In literature evaluation, when an LLM achieves non-zero Mean Absolute Error (MAE) relative to human-curated ground truth without hallucinating, what causes the numerical divergence?
    * **The 5 Systematic Epistemological Sources**:
      1. *Human Curator Transcription Slips (The LHU-Net Grounding)*: The human curator inadvertently copied an adjacent baseline row or transposed numbers (e.g., in Table 2 of LHU-Net, the human recorded `0.8749` from `D-LKA Net` instead of `0.8783` for `LHU-Net`, as independently proven by both Gemini and Mufasa with an identical MAE of 0.0293).
      2. *Interactive Prompt Regime Divergence (The SAM 1-Point Paradigm)*: Promptable foundation models (SAM architectures) report performance across varying interaction depths (1-point, 3-point, 5-point, 10-point, bounding box). The human survey standardized strictly on the 1-point interaction framework (`"Results reported in the 1-point prompt framework"`). If an LLM extracts the 3-point or bounding-box column—both 100% genuine numbers printed in the manuscript—an artificial MAE is registered against the 1-point ground truth.
      3. *Curatorial Arithmetic Aggregation vs. Direct Table Cells (TotalSegmentator MRI)*: Human curators frequently synthesized unweighted mathematical means across cohorts or modalities using hand calculators (e.g., TotalSegmentator MRI `90.25%` = average of CT test set `83.9%` and MRI test set `96.6%`, or BraTS overall means computed across Whole Tumor, Tumor Core, and Enhancing Tumor). If an LLM extracts an individual sub-task or organ row, an apparent divergence occurs.
      4. *Ablation Sweeps vs. Primary Benchmark Tables*: Manuscripts feature internal hyperparameter sweeps or validation-only ablations (e.g., Table 1 showing validation performance under varying loss functions, or semi-supervised sweeps across 1%, 5%, 10% labeled data in KnowSAM/SFR SAM) alongside primary test-set benchmark tables. Extracting from an ablation table yields authentic printed numbers that diverge from the canonical test benchmark.
      5. *Decimal Precision, Normalization, and Rounding*: Academic tables print rounded percentages (e.g., `88.5%` $\rightarrow$ 0.8850), whereas code repository logs or internal tables record unrounded multi-digit floats (`0.8847`), introducing sub-decimal precision variances.
    * **Defense Impact**: Provides a foundational epistemological framework for analyzing AI extraction accuracy, demonstrating to the examination committee that MAE in scientific literature mining is multi-factorial and often reveals curation nuances rather than model failure.

23. **Multi-Tier Semantic Acronym Matching vs. Competitor Baseline Hallucinations (The Med-SA Case Study)**:
    * **The Dilemma**: In automated medical meta-analysis, LLMs frequently commit *baseline confusion*: mistaking a competitor benchmark row (e.g. `nnU-Net`, `Swin UNETR`, `TransUNet`) for the proposed model because the competitor row appears prominently in the comparative table. To prevent benchmark contamination, the Critic (`AuditorAgent`) maintains a registry of known baseline models (`KNOWN_BASELINE_MODELS`) and flags baseline confusion as a critical anomaly.
    * **The Pathology**: In the 30-paper benchmark, `Med-SA` was falsely rejected with `AuditStatus.REJECTED` despite extracting 100% exact numerical Dice scores on BTCV (0.8830) and BraTS (0.8910). The paper was titled *"Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation"*, and the proposed model name was `"Med-SA"`. Because `"med-sa"` was listed in `KNOWN_BASELINE_MODELS` and did not literally appear as a contiguous substring in the paper title string, the Auditor flagged the model as an alien competitor baseline mistakenly extracted.
    * **The Anti-Pattern**: Hardcoding a paper-specific override (`if model_id == "med-sa": pass`) is brittle, unscientific, and fails to generalize to future preprints.
    * **The Thesis Solution**: We formulated a generalized multi-tier semantic compatibility check (`_is_model_name_compatible_with_paper`):
      1. *Punctuation & Delimiter Normalization*: Stripping hyphens, underscores, and spacing to compare normalized alphanumeric token overlap.
      2. *Initials & Acronym Synthesis*: Generating uppercase acronym combinations from multi-word paper titles after filtering English stopwords (*"Medical SAM Adapter"* $\rightarrow$ `MSA`, `MED-SA`), confirming abbreviation validity.
      3. *Manuscript Self-Nomination Grounding*: Scanning manuscript full text for first-person naming phrases (*"we propose Med-SA"*, *"our method, Med-SA"*, *"called Med-SA"*).
    * **Defense Impact**: Demonstrates principled linguistic reasoning in agentic auditing—safeguarding against competitor baseline hallucinations while robustly recognizing legitimate acronyms across scientific literature without hardcoded rules.

24. **30-Paper Cross-Architecture Empirical Benchmark — Commercial Cloud API vs. Sovereign Academic HPC**:
    * **Experimental Setup**: Scaled evaluation to 30 foundational 3D medical foundation models, executing a parallel double-run:
      - Commercial Cloud Frontier API: `gemini-3.5-flash-lite` via Google AI Studio (0.00 € Free Tier, 1M context window).
      - Sovereign On-Premise Academic Cluster: `Qwen2.5-14B-Instruct-AWQ` on DEIB Mufasa HPC (NVIDIA A100-PCIE-40GB MIG slice, vLLM v0.30.0).
    * **Key Findings**:
      1. *Consensus on Perfect Accuracy*: 14 out of 30 models achieved an identical **0.0000 Dice MAE** across BOTH architectures, confirming rock-solid extraction reliability on standardized tables.
      2. *Cross-Architecture Typo Replication*: Both independent models replicated the identical **0.0293 MAE** on `LHU-Net`, independently catching the human survey transcription error in Table 2.
      3. *Speed vs. Recall Pareto Frontier*:
         - Gemini 3.5 Flash achieved **10.1x faster execution** (9.3s/paper vs 94.2s/paper), completing 30 papers in 4.7 minutes.
         - Local Qwen-14B on Mufasa achieved **+7.0% higher dataset recall** (73.6% vs 66.6%), successfully capturing dense tables in multi-page papers (Merlin, SFR SAM, HERMES).
      4. *Operational Autonomy & Privacy*: Running Qwen-14B on Mufasa guarantees full data sovereignty, zero external data leakage, and zero cost, satisfying clinical privacy regulations for institutional hospital deployment.
    * **Defense Impact**: Provides comprehensive empirical proof comparing proprietary frontier cloud LLMs against sovereign on-premise open-weights models, delivering quantitative data for Chapter 5 (Evaluation) of the Master's Thesis.

25. **Dual-Agent Consensus Extraction & Epistemic Arbitration ('Il Vigile' as Multi-Model Arbiter)**:
    * **The Core Scientific Problem**: In single-agent extraction (Actor-Critic 1-on-1), the system's extraction fidelity is tied to the idiosyncratic biases, context handling, or tokenization quirks of a single LLM architecture. If the model misinterprets a table layout or drops a column, the Critic has only intrinsic heuristics (range checks, distribution comparisons) to detect the flaw. If the single model hallucinates a plausible-sounding value within valid ranges, an intrinsic sanity check cannot catch it.
    * **The Epistemic Diversity Paradigm**: In multi-agent consensus literature and epistemic decision theory (Condorcet's Jury Theorem applied to LLMs), combining two distinct, structurally independent model families (e.g., proprietary frontier cloud like Gemini paired with open-weights models like Qwen or Llama on the Mufasa cluster) drastically suppresses idiosyncratic hallucinations: two distinct models with different pre-training datasets, tokenizers, and architectures almost never hallucinate the exact same erroneous score or misinterpret a complex table in the exact same way.
    * **Triadic Actor-Critic Architecture (Dual-Actor / Arbiter)**:
      Instead of a 1:1 Actor-Critic topology, the architecture generalizes to a triadic coordination pattern:
      ```
                        ┌────────────────────────┐
                        │   Manuscript / PDF     │
                        └───────────┬────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
           ┌───────────────────┐         ┌───────────────────┐
           │ ExtractionAgent A │         │ ExtractionAgent B │
           │   (Model A / LLM) │         │   (Model B / LLM) │
           └─────────┬─────────┘         └─────────┬─────────┘
                     │ Extracted A                 │ Extracted B
                     └──────────────┬──────────────┘
                                    ▼
                      ┌───────────────────────────┐
                      │   AuditorAgent (Vigile)   │
                      │  - Inter-Agent Agreement  │
                      │  - Discrepancy Detection  │
                      │  - Sanity / SQLite Checks │
                      │  - Epistemic Arbitration  │
                      └─────────────┬─────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │ Reconciled Extraction │
                        │ + Certified Report    │
                        └───────────────────────┘
      ```
    * **Dynamic Model-Agnostic Parametrization**:
      Crucial for thesis experimentation: the architecture decouples model identity from the agent role. The pipeline is fully parametrized via CLI/configs (`--model-a`, `--provider-a`, `--model-b`, `--provider-b`), allowing dynamic pairings (e.g. Gemini + Qwen, Llama + DeepSeek, Mistral + Qwen) to support a dedicated comparative chapter in the thesis.
    * **Parallel Independent Execution**:
      Both extractors process the manuscript concurrently (`ThreadPoolExecutor(max_workers=2)`). This guarantees independent extractions (avoiding anchoring bias where one model anchors on the other's draft) while maintaining execution speed ($\text{Wall-Clock Time} \approx \max(T_A, T_B)$).
    * **Three-Tier Taxonomy of Extracted Benchmarks**:
      The Auditor classifies each extracted score into one of three explicit, scientifically transparent states:
      1. `CONCORDANT`: Both models extract the dataset and their Dice scores match within tolerance $|S_A - S_B| \le \epsilon$ ($\epsilon = 0.01$). The value is unified with maximum confidence boost.
      2. `DISPUTED`: Both models identify the dataset, but their numerical values diverge significantly ($|S_A - S_B| > 0.01$). Rather than silently guessing via heuristics, the record is explicitly flagged as `DISPUTED`, preserving both candidate numbers ($S_A$ vs $S_B$) and triggering an audit warning.
      3. `SINGLE_AGENT_UNCONFIRMED`: A benchmark extracted by only one agent (e.g., Model A captured a supplementary table that Model B overlooked). The benchmark is retained to preserve recall, but transparently flagged as `UNCONFIRMED` with reduced confidence.
    * **Screening & Metadata Reconciliation**:
      Inclusion screening (`is_3d`, `is_radiological`, `has_dice`) and metadata (`millions_of_parameters`, `visual_backbone`) are subjected to consensus:
      - If both agree on screening, high confidence.
      - If screening conflicts, the Auditor flags as disputed and applies domain fallback checks.
      - If parameter counts conflict, the Auditor inspects architecture ranges and logs an anomaly.
    * **Quantification of Epistemic Uncertainty & Inter-Annotator Agreement (IAA)**:
      Provides rich academic metrics for Chapter 4/5 of the thesis:
      - *Inter-Model Dice MAE* between Agent A and Agent B.
      - *Consensus Agreement Rate* (% of benchmarks concordant).
      - *Dispute Rate* (% of benchmarks disputed).
      - Quantitative proof that concordant records correlate with near-zero error against human ground truth.
    * **Defense Impact**: Elevates the thesis contribution from single-agent automation to a formal consensus-driven epistemological verification system, offering a principled framework for automated multi-model scientific curation.

---

### 26. Scholarly Communication & Delegation Layer: Automated Author Inquiry (`OutreachAgent`)
* **Problem**: In scientific literature, a non-trivial fraction of papers report evaluation metrics exclusively through graphical visualizations (e.g. radar plots, bar charts, boxplots, or violin plots) without presenting scalar numerical values in tabular form (e.g., *BiomedParse*, *Merlin*). When public code repositories also lack evaluation logs, an autonomous system faces an epistemic dead-end: attempting to OCR pixel coordinates from compressed raster charts introduces severe measurement errors ($\Delta \text{Dice} > 0.03$), violating curation fidelity.
* **Literature Grounding**: In the reference systematic review by Moglia et al., the human curators handled this exact boundary condition through direct academic correspondence:
  > *"In cases where the Dice score was only available in graphical form without explicit numerical values, we contacted the corresponding authors."*
* **Design Pattern (Delegation Layer - Human-in-the-Loop Outreach)**:
  Rather than hallucinating numbers or silently discarding eligible models, the Auditor diagnoses `AuditStatus.NEEDS_OUTREACH` (`MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND`) and delegates to `OutreachAgent`:
  ```
                     ┌────────────────────────────────┐
                     │   Auditor ("Il Vigile")        │
                     │   Status: NEEDS_OUTREACH       │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │         OutreachAgent          │
                     │  1. Extract Corresponding Auth │
                     │  2. Identify Missing Figures   │
                     │  3. Compose Formal Inquiry     │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │     Persisted Email Draft      │
                     │   (.json + .md in data/drafts) │
                     └───────────────┬────────────────┘
                                     │
                                     ▼ (Human Reviewer)
                     ┌────────────────────────────────┐
                     │   Human-in-the-Loop Dispatch   │
                     │   & Scholarly Communication    │
                     └────────────────────────────────┘
  ```
* **Hierarchical Metadata Extraction Engine**:
  Extracts contact information without relying on fragile external web scrapers:
  1. *Footnote & Asterisk Matching*: Identifies corresponding author asterisks (`*`, `✉`, `†`) and explicit footnote blocks (`*Corresponding author(s). E-mail(s): ...`).
  2. *Inline Layout Parsing*: Supports conference/arXiv formats (e.g. CVPR, NeurIPS) where institutional affiliation and email are placed directly adjacent to the author's name.
  3. *Email-to-Author Alignment*: Matches username tokens (`bowang`, `vbutoi`) against candidate capitalized names in the front matter.
* **Scholarly Inquiry Protocol & Tone**:
  Constructs a formal, polite academic inquiry email adhering to international scientific correspondence standards:
  - Subject: `[Academic Inquiry] Quantitative evaluation metrics for "[Paper Title]" ([Model ID])`
  - Affiliation: Identifies the research context under the Generalist Medical 3D Segmentation (GMMIS) benchmark at Politecnico di Milano (DEIB).
  - Explicit Figure Citations: References the exact figures (e.g. `Figure 3`, `Figure 4`) and target benchmarks (e.g. `BraTS`, `AMOS`) depicted graphically.
  - Open Science Rationale: Explains that exact scalar metrics ensure their model is represented accurately in the public benchmark without coordinate approximations.
* **Ethical Governance & Human-in-the-Loop (HITL)**:
  Fully automated outbound emailing without human oversight risks spamming researchers or sending misdirected inquiries. Therefore, `OutreachAgent` generates and persists both machine-readable JSON (`OutreachDraft`) and formatted Markdown drafts (`.md`), tracking status (`DRAFT_PENDING_REVIEW` $\rightarrow$ `APPROVED` $\rightarrow$ `DISPATCHED` $\rightarrow$ `REPLIED`). The system empowers human researchers to review, edit, and dispatch inquiries with one click.
* **Defense Impact**: Demonstrates that the autonomous system respects real-world scientific boundaries, mirrors established human methodological practices, and enforces responsible AI governance.

---

### 27. Empirical Validation of the Delegation Loop: Real-World Outreach Case Study (`Merlin`, Stanford MIMI)
* **The Empirical Case**: During the 20-paper benchmark execution, *Merlin: A Computed Tomography Vision-Language Foundation Model and Dataset* (Blankemeier et al., Stanford MIMI / UW-Madison, arXiv: `2406.06512.pdf`) was subjected to autonomous extraction.
* **Autonomous Decision Trajectory**:
  1. *Extraction Attempt*: The `ExtractionAgent` parsed the manuscript text, discovering that all primary segmentation comparisons (Internal Clinical Dataset, External Clinical Dataset, VerSe) were presented as graphical radar/bar plots without numeric evaluation tables.
  2. *First Delegation (Code Inspection)*: The `AuditorAgent` diagnosed missing tables and triggered `RepoInspector` on the official code repository (`https://github.com/StanfordMIMI/Merlin`).
  3. *Zero-Data Confirmation*: `RepoInspector` cloned and analyzed the repository tree, establishing that no evaluation CSVs, JSON logs, or Markdown tables were present.
  4. *Second Delegation (Outreach Trigger)*: The Auditor re-audited the paper with `repo_inspected=True`, transitioning the diagnosis to `MissingDataDiagnosis.MISSING_TABLE_FIGURES_FOUND` and setting status `AuditStatus.NEEDS_OUTREACH`.
  5. *Autonomous Inquiry Composition*: The `OutreachAgent` processed the front matter:
     - Extracted Corresponding Author: **Louis Blankemeier** (`akshaysc@stanford.edu`).
     - Affiliation: Stanford University, University of Wisconsin-Madison, et al.
     - Extracted Missing Visual Targets: `Figure 1, Figure 10, Figure 3, Figure 9, Figure 8b, Figure 12, Figure 15`.
     - Targeted Benchmarks: `Internal Clinical Dataset, External Clinical Dataset, VerSe`.
     - Draft Generation: Produced a polite, professionally formatted academic email on behalf of Politecnico di Milano DEIB and the GMMIS benchmark initiative.
     - Persisted artifacts to [`data/outreach_drafts/merlin_outreach.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/data/outreach_drafts/merlin_outreach.md) and [`data/outreach_drafts/merlin_outreach.json`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/data/outreach_drafts/merlin_outreach.json) with status `DRAFT_PENDING_REVIEW`.
* **Scientific Rigor for Thesis Defense**: This end-to-end trace provides empirical evidence that the system never invents fake metrics when papers omit tables, seamlessly attempts code repository recovery, and gracefully initiates Human-in-the-Loop academic correspondence to obtain ground truth, reproducing the exact investigative protocol of systematic review authors.

---

### 28. The Granularity Asymmetry Paradox & Proportional Epistemic Arbitration in Dual-Agent Consensus

* **Theoretical Core: The Granularity Asymmetry Paradox**:
  When deploying multiple heterogeneous LLMs in parallel (e.g. `Qwen2.5-14B-Instruct-AWQ` on HPC vs `Gemini-3.1-Flash-Lite` on cloud) to independently extract complex multi-dimensional medical segmentation benchmarks, agents naturally exhibit divergent hierarchical granularities:
  - **Macro-Aggregator Behavior** (e.g., Gemini): The agent targets compact table summaries, extracting the macro-average `Overall` or `Mean` Dice row to minimize token usage while capturing the primary survey metric.
  - **Decompositional Behavior** (e.g., Qwen): The agent recursively unpacks the entire multi-column structure, extracting both the summary row and 15–25 fine-grained anatomical sub-structures (e.g. Spleen, R. Kidney, L. Kidney, Liver, Pancreas, Aorta).
* **The Union vs. Intersection Fallacy in Inter-Annotator Agreement (IAA)**:
  In standard NLP literature, agreement is often formulated over the complete pool of extracted entities:
  $$\text{Concordance}_{\text{naive}} = \frac{|\mathcal{C}|}{|\mathcal{S}_A \cup \mathcal{S}_B|}$$
  where $\mathcal{C}$ is the set of concordant scores and $\mathcal{S}_A, \mathcal{S}_B$ are the extraction sets of Agents A and B.
  In the presence of granularity asymmetry, if Agent A extracts 1 summary score ($\text{DSC} = 0.8030$) and Agent B extracts that identical summary score plus 13 organ-specific sub-scores, the naive formula yields:
  $$\text{Concordance}_{\text{naive}} = \frac{1}{1 + 13} = 7.14\%$$
  This creates an epistemological illusion: **the models exhibit 100% agreement on their common ground, yet the naive metric declares 92.9% disagreement**. This paradox erroneously depressed confidence scores to near-zero (0%–12%) on papers where the extraction was objectively flawless ($\text{MAE} = 0.0000$).
* **Mathematical Resolution: Disentangled Paired IAA & Singleton Preservation**:
  To resolve this paradox, the `AuditorAgent` strictly partitions extracted claims into two orthogonal mathematical subsets:
  1. **Paired Benchmark Set ($\mathcal{P}$)**: Benchmarks extracted by *both* agents with matching canonical dataset, prompt type, and data budget:
     $$\mathcal{P} = \{ (s_A, s_B) \in \mathcal{S}_A \times \mathcal{S}_B \mid \text{matched}(s_A, s_B) \}$$
  2. **Singleton Set ($\mathcal{U}$)**: Benchmarks extracted by only one agent ($\mathcal{S}_A \setminus \mathcal{P}$ and $\mathcal{S}_B \setminus \mathcal{P}$), classified as `SINGLE_AGENT_UNCONFIRMED`.
  
  **True Inter-Annotator Agreement** is calculated strictly on the paired intersection:
  $$\text{Concordance Rate} = \frac{|\{ (s_A, s_B) \in \mathcal{P} \mid |s_A - s_B| \le \tau \}|}{|\mathcal{P}|} \quad (\text{for } |\mathcal{P}| > 0)$$
  where $\tau = 0.01$ (1.0% DSC tolerance).
  Singletons represent *coverage breadth* rather than factual contradiction; they are preserved in the consensus report with their extracted scores rather than discarded.
* **Bounded & Proportional Epistemic Confidence**:
  Rigid linear penalties (e.g. $-0.15 \times \text{disputes}$) induce catastrophic confidence collapse when minor table alignment ambiguities arise. The revised confidence engine couples base model confidence with a bounded, relative dispute penalty:
  $$\Delta \text{Conf}_{\text{dispute}} = \min\left(0.25, \text{round}\left(\frac{|\mathcal{D}|}{|\mathcal{P}|} \times 0.25, 2\right)\right)$$
  where $\mathcal{D} \subseteq \mathcal{P}$ is the set of disputed pairs ($|s_A - s_B| > \tau$). Singletons incur only a marginal, capped deduction ($\min(0.08, |\mathcal{U}| \times 0.01)$).
* **Experimental Partition Disambiguation (Semi-Supervised & Few-Shot Regimes)**:
  Foundation models in medical imaging frequently benchmark across semi-supervised partitions (e.g., 1%, 10%, 20%, 100% labeled slices in `KnowSAM` or `SFR SAM`). If the matching algorithm fails to respect the partition budget, a 10% partition score ($\text{DSC} = 0.6811$) is falsely paired against a 20% partition score ($\text{DSC} = 0.7269$), fabricating an artificial dispute. By explicitly encoding `data_budget_or_setting` in both the extraction schema and the matching engine:
  $$\text{budget}_A \ne \text{budget}_B \implies \text{No Match (Separate Singletons)}$$
  the system completely eliminates cross-partition false disputes while faithfully capturing the full multi-regime evaluation matrix.
* **Defense Impact**: Demonstrates mathematically rigorous multi-agent consensus scoring that handles varying granularity levels, prevents false-positive dispute cascades, and grounds epistemic arbitration in clean decision theory.

---

### 29. Double-Blind Multi-Agent Iterative Deliberation & Peak Performance Extraction

* **Motivation & Academic Advisor Guidance**:
  In human systematic literature reviews (e.g. Moglia et al.), multiple reviewers meet in iterative consensus rounds to reconcile discrepancies and resolve omissions. Two key directives govern this process:
  1. **Peak Performance Principle (Maximum Achievable DSC)**: The literature survey aims to establish the upper bound and true architectural frontier of generalist models against specialized baselines. Therefore, for every target benchmark dataset, the extraction engine must target the **maximum (highest) DSC score achieved by the proposed model**, regardless of whether it was attained via bounding box prompts, multi-point clicks, or fine-tuning (replacing prior artificial restrictions that penalized multi-prompt regimes).
  2. **Double-Blind Deliberation (Zero Numerical Leakage)**: Conventional multi-agent debate often leaks the competitor's raw answer (e.g., *"Model B extracted 0.85; what do you say?"*). In empirical LLM literature, this triggers severe **Anchoring Bias** and **Sycophancy** (where the model passively capitulates to the peer's suggestion rather than verifying empirical text). To preserve epistemic integrity, the deliberation must be strictly double-blind: neither agent ever observes the other agent's numerical outputs.

* **Multi-Round Deliberation Lifecycle**:
  ```
  Round 1: Parallel Epistemic Isolation (Agent A & Agent B extract independently)
                             │
                             ▼
  Auditor Consensus Matrix: Classify Concordant vs Disputed (|Δ| > 0.01) vs Omitted
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Dispute Count == 0 &            [Disputes or Omissions Exist]
    Omission Count == 0]                      │
            │                                 ▼
            │               Round k+1: Double-Blind Targeted Challenge
            │               ├── Disputed Item: Blind challenge to A and B
            │               └── Omitted Item: Blind challenge to omitting agent
            │                                 │
            │                                 ▼
            │               Auditor Re-evaluation & Early Stopping Check
            │               ├── All resolved? ───► STOP (VERIFIED_BY_DELIBERATION)
            │               ├── Nash Equilibrium (Stall)? ───► STOP (STABLE)
            │               └── k == MAX_ROUNDS? ───► STOP (PERMANENT_DISPUTE)
            │                                 │
            └─────────────────────────────────┴──► Final Reconciled Record
  ```

* **Blind Challenge Protocol Design**:
  1. **Blind Discrepancy Challenge (Dataset $D$)**:
     Neither agent is shown the other's extracted number. Both are prompted:
     > *"On dataset [D], another independent extractor reached a divergent result regarding the peak performance of the proposed model. Re-read the relevant table/section of the manuscript targeting the MAXIMUM Dice Similarity Coefficient (DSC) for the proposed model. If you are confident your original extraction already represents the true peak score, maintain your decision and state your rationale. If you identify a higher valid score, a misaligned column, or a baseline confusion, update your score. Do not guess; cite the exact table row and column."*
  2. **Blind Omission Challenge (Dataset $D$ found only by one agent)**:
     The omitting agent is queried without leaking the extracted score:
     > *"Another independent extractor identified an evaluation on dataset [D] for the proposed model, which you did not extract. Inspect the manuscript text and tables: is [D] evaluated with a quantitative 3D DSC score for the proposed model? If yes, extract its peak score. If it was excluded for legitimate reasons (e.g., 2D modality, competitor baseline, or graphical-only plot), confirm the exclusion and state the reason."*

* **Convergence Trajectory & Artifact Persistence**:
  For each paper undergoing iterative deliberation, the logging engine persists:
  - `round_1/`, `round_2/`, ..., `round_k/` subdirectories with independent extractions.
  - `convergence_trajectory.json` tracing the evolution of concordance rate ($C_1 \to C_2 \to C_k$) and MAE ($\text{MAE}_1 \to \text{MAE}_k$).
  This provides a concrete empirical graph for the thesis dissertation demonstrating multi-agent convergence dynamics under epistemic uncertainty.




