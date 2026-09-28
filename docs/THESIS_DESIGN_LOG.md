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
    * **Defense Impact**: Elevates the thesis from an API-calling script to a production-grade, enterprise-ready multi-agent system. Enables a dedicated experimental chapter: *"Empirical comparison of table extraction fidelity, inference latency, and operational autonomy between commercial cloud LLMs and open-source models self-hosted on academic HPC."*




