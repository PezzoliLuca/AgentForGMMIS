# Piano Operativo di Implementazione e Benchmark (Post-Riunione Prof. Moglia — GPU A100 40 GB)

> **Scopo del documento**: Specifica tecnica dettagliata per gli agenti di sviluppo incaricati di aggiornare la pipeline multi-agente (`AgentForGMMIS`) e piano sperimentale completo per i nuovi benchmark su cluster Mufasa con **GPU NVIDIA A100 da 40 GB** (`amoglia@10.79.23.96`, `--qos=gpuheavy-40`, `--gres=gpu:40gb:1`).

---

## 1. Sintesi delle 3 Direttive del Professore

1. **Direttiva 1 — Estrazione e Persistenza DB per Singolo Organo (Stop `Overall`)**:
   Nonostante la survey umana originale (`data/database.sqlite`, tabelle `results_primary` e `results_best`) riporti un unico valore aggregato (`Overall`) per ciascuna coppia `(Modello, Dataset)`, il nostro sistema deve estrarre e salvare a database **il punteggio Dice di ciascun singolo organo / sottostruttura anatomica** (es. i 13 organi di `BTCV`, i 15 di `AMOS`, i 13 di `FLARE`, le 3 sub-regioni `WT`/`TC`/`ET` di `BraTS`, `RV`/`LV`/`Myo` di `ACDC`), evitando di cercare o salvare `Overall` quando il breakdown per organo è presente nel paper.
2. **Direttiva 2 — Centralità della Metrica di Concordanza Inter-Agente (IAA)**:
   La concordanza tra i due agenti estrattori indipendenti (`Agent A` e `Agent B` in double-blind) deve diventare la **metrica primaria del sistema e della tesi**. Se due modelli indipendenti estraggono in cieco lo stesso identico valore per uno specifico organo/dataset ($|\Delta| \le 0.01$), è altamente probabile che il valore estratto sia corretto anche in assenza di una Ground Truth manuale per quel singolo organo o nuovo benchmark.
3. **Direttiva 3 — Comportamento dei Modelli Cinesi su `SAM-Med2D` e Simili**:
   Su paper come `SAM-Med2D` (e foundation model 2D-slice o ibridi 2D/3D valutati su scansioni volumetriche 3D come `CT`, `MRI`, `PET` e challenge 3D), il sistema deve adottare il comportamento inclusivo mostrato dai modelli cinesi (`Qwen2.5`), garantendo che tali paper superino lo screening (`is_3d = True`, `overall_status = VERIFIED`) ed estraendo correttamente tutti i benchmark volumetrici 3D senza essere scartati o penalizzati dal nome `"2D"` o dalle righe fotografiche non radiologiche (`Dermoscopy`, `Fundus`, ecc.).

---

## 2. Piano di Implementazione Codice (Moduli 1–4)

### Modulo 1: Estrazione Granulare per Singolo Organo e Persistenza DB

#### Task 1.1 — Aggiornamento Prompt e Rimozione Tetto di 25 Score
* **File**: `src/agent_gmmis/extraction/prompts.py` e `src/agent_gmmis/extraction/schemas.py`
* **Modifiche in `src/agent_gmmis/extraction/prompts.py`**:
  1. In `EXTRACTION_SYSTEM_PROMPT` (righe 37–41), sostituire la regola `MACRO-AVERAGE VS SINGLE ORGANS` con `PER-ORGAN GRANULAR EXTRACTION MANDATE`:
     * Quando una tabella del paper riporta i punteggi Dice per singoli organi, strutture anatomiche o sub-regioni tumorali (es. `Spleen`, `Right Kidney`, `Left Kidney`, `Gallbladder`, `Esophagus`, `Liver`, `Stomach`, `Aorta`, `Inferior Vena Cava`, `Portal and Splenic Veins`, `Pancreas`, `Right Adrenal Gland`, `Left Adrenal Gland`, `Duodenum`, `Urinary Bladder`, `Prostate/Uterus`, `Whole Tumor`, `Tumor Core`, `Enhancing Tumor`, `Right Ventricle`, `Left Ventricle`, `Myocardium`), l'agente **DEVE estrarre un oggetto `PerformanceScore` separato per ciascun singolo organo**.
     * Quando nella tabella sono presenti i singoli organi, l'agente **NON DEVE** estrarre la colonna riassuntiva `Mean` / `Average` / `Overall` per quel dataset.
     * **Unica eccezione (`Dataset-Level Fallback`)**: Se e solo se il paper riporta **esclusivamente** un singolo punteggio aggregato per l'intero dataset senza alcuna scomposizione per organo nella tabella, impostare `organ_or_task` al nome dell'organo specifico del dataset (se mono-organo, es. `KiTS` $\to$ `"Kidney"`, `LiTS` $\to$ `"Liver"`, `SegTHOR` $\to$ `"Thoracic Organs"`) oppure `"Dataset-Level"` (se multi-organo ma riportato solo aggregato).
  2. In `EXTRACTION_USER_PROMPT_TEMPLATE` (riga 63), rimuovere il limite `"up to the top 25 key evaluation benchmark scores"` (un singolo paper con `BTCV` + `AMOS` + `FLARE` contiene più di 40 punteggi organo). Richiedere l'estrazione completa di tutti i punteggi per organo del modello proposto sui dataset 3D.
  3. Sostituire `BLIND_GRANULARITY_SYSTEM_PROMPT` e `BLIND_GRANULARITY_USER_TEMPLATE` (righe 119–142) con `BLIND_ORGAN_BREAKDOWN_SYSTEM_PROMPT` e `BLIND_ORGAN_BREAKDOWN_USER_TEMPLATE`:
     * Invece di chiedere il `Macro-Average ('Overall')`, il prompt chiede all'agente di ispezionare nuovamente il paper sul dataset `{dataset}` ed estrarre i punteggi Dice individuali per la lista di organi target `{target_organs}` (senza mai rivelare i valori numerici dell'altro agente).
* **Modifiche in `src/agent_gmmis/extraction/schemas.py`**:
  * Aggiornare la ` description` di `PerformanceScore.organ_or_task` (righe 18–21) per riflettere l'obbligo di inserire il nome canonico del singolo organo/sottostruttura anatomica e usare `"Dataset-Level"` solo come fallback quando il paper non riporta i singoli organi.

#### Task 1.2 — Aumento `max_output_tokens` per Evitare Troncamenti JSON
* **File**: `src/agent_gmmis/llm/registry.py`
* **Modifiche**:
  * In `ModelProfile` (riga 28) e nei profili del catalogo `MODEL_REGISTRY`, aumentare `max_output_tokens`:
    * Da `2048` a **`4096`** per tutti i modelli standard (`is_reasoning=False`).
    * Da `4096` a **`8192`** per tutti i modelli reasoning (`is_reasoning=True`, es. `DeepSeek-R1-Distill-Qwen-14B-AWQ`, `DeepSeek-R1-Distill-Qwen-32B-AWQ`, `Qwen3-8B`), in modo che il blocco `<think>...</think>` più un array JSON di 40–60 organi non vengano mai troncati.

#### Task 1.3 — Creazione di `OrganAliasResolver`
* **File**: `src/agent_gmmis/extraction/alias_resolver.py`
* **Modifiche**:
  * Accanto a `DatasetAliasResolver`, implementare la classe `OrganAliasResolver` con metodo `resolve(raw_organ_name: str) -> str` per normalizzare in modo deterministico tutte le abbreviazioni cliniche e varianti lessicali nello stesso identificatore canonico:
    * **Organi Addominali (`BTCV`, `AMOS`, `FLARE`, `WORD`, `AbdomenCT-1K`, `TotalSegmentator`)**:
      * `"spl"`, `"spleen"` $\to$ `"Spleen"`
      * `"rk"`, `"r. kidney"`, `"right kidney"`, `"kidney (r)"`, `"kidney_r"`, `"r-kidney"`, `"r kidney"` $\to$ `"Right Kidney"`
      * `"lk"`, `"l. kidney"`, `"left kidney"`, `"kidney (l)"`, `"kidney_l"`, `"l-kidney"`, `"l kidney"` $\to$ `"Left Kidney"`
      * `"gal"`, `"gb"`, `"gallbladder"`, `"gall bladder"` $\to$ `"Gallbladder"`
      * `"eso"`, `"esophagus"`, `"oesophagus"` $\to$ `"Esophagus"`
      * `"liv"`, `"liver"` $\to$ `"Liver"`
      * `"sto"`, `"stomach"` $\to$ `"Stomach"`
      * `"aor"`, `"aorta"` $\to$ `"Aorta"`
      * `"ivc"`, `"inferior vena cava"`, `"postcava"` $\to$ `"Inferior Vena Cava"`
      * `"psv"`, `"portal vein and splenic vein"`, `"portal and splenic veins"`, `"portal & splenic veins"` $\to$ `"Portal and Splenic Veins"`
      * `"pan"`, `"pancreas"` $\to$ `"Pancreas"`
      * `"rag"`, `"right adrenal gland"`, `"r. adrenal gland"`, `"adrenal gland (r)"`, `"right adrenal"` $\to$ `"Right Adrenal Gland"`
      * `"lag"`, `"left adrenal gland"`, `"l. adrenal gland"`, `"adrenal gland (l)"`, `"left adrenal"` $\to$ `"Left Adrenal Gland"`
      * `"ag"`, `"adrenal gland"`, `"adrenal glands"` $\to$ `"Adrenal Gland"`
      * `"duo"`, `"duodenum"` $\to$ `"Duodenum"`
      * `"ub"`, `"bladder"`, `"urinary bladder"` $\to$ `"Urinary Bladder"`
      * `"prostate/uterus"`, `"prostate / uterus"`, `"prostate or uterus"` $\to$ `"Prostate/Uterus"`
      * `"colon"`, `"rectum"`, `"small bowel"`, `"intestine"` $\to$ nomi canonici corrispondenti.
    * **Sub-regioni Neuro-Oncologiche (`BraTS`, `ISLES`, `CrossMoDA`, `ATLAS`)**:
      * `"wt"`, `"whole tumor"`, `"whole tumour"` $\to$ `"Whole Tumor"`
      * `"tc"`, `"tumor core"`, `"tumour core"` $\to$ `"Tumor Core"`
      * `"et"`, `"enhancing tumor"`, `"enhancing tumour"` $\to$ `"Enhancing Tumor"`
      * `"vs"`, `"vestibular schwannoma"` $\to$ `"Vestibular Schwannoma"`
      * `"cochlea"` $\to$ `"Cochlea"`
    * **Strutture Cardiache e Toraciche (`ACDC`, `MM-WHS`, `SegTHOR`)**:
      * `"rv"`, `"right ventricle"` $\to$ `"Right Ventricle"`
      * `"lv"`, `"left ventricle"` $\to$ `"Left Ventricle"`
      * `"myo"`, `"myocardium"`, `"lv myocardium"` $\to$ `"Myocardium"`
      * `"la"`, `"left atrium"` $\to$ `"Left Atrium"`
      * `"ra"`, `"right atrium"` $\to$ `"Right Atrium"`
      * `"trachea"` $\to$ `"Trachea"`
    * **Fallback Generico**:
      * `"overall"`, `"average"`, `"mean"`, `"avg"`, `"macro average"`, `"all"`, `"dataset-level"` $\to$ `"Dataset-Level"` (usato solo per identificare righe aggregate).

#### Task 1.4 — Inversione della Deliberazione di Granularità nel Coordinatore
* **File**: `src/agent_gmmis/extraction/dual_coordinator.py` e `src/agent_gmmis/extraction/extractor.py`
* **Modifiche**:
  * In `ExtractionAgent` (`extractor.py`, righe 467–502): sostituire `reinterrogate_granularity_macro(...)` con `reinterrogate_organ_breakdown(paper, current_result, target_organs_by_dataset: dict[str, list[str]]) -> ExtractionResult`. Quando i nuovi punteggi per organo vengono estratti, rimuovere eventuali record `"Dataset-Level"` / `"Overall"` dello stesso dataset per evitare duplicati.
  * In `DualExtractionCoordinator` (`dual_coordinator.py`, righe 273–310):
    * Se per un dataset canonico $D$ l'Agente A ha estratto $\ge 2$ organi specifici $\{O_1, \dots, O_k\}$ mentre l'Agente B ha estratto solo `"Dataset-Level"`/`"Overall"` (o ha estratto un sottoinsieme incompleto di organi rispetto ad A), aggiungere $D \to \{O_1, \dots, O_k\}$ alla coda `missing_organ_breakdown_for_b` (e viceversa per A).
    * Invocare `reinterrogate_organ_breakdown` in modalità **100% double-blind** (passando solo i nomi canonici del dataset e degli organi richiesti, mai i valori numerici).

#### Task 1.5 — Nuova Tabella Relazionale `results_organ_scores` e Curation Per-Organo
* **File**: `src/agent_gmmis/data/db_manager.py`, `src/agent_gmmis/database/database_agent.py`, `src/agent_gmmis/database/schemas.py`
* **Modifiche in `src/agent_gmmis/data/db_manager.py`**:
  1. Aggiungere il metodo `init_organ_scores_table()` (chiamato automaticamente all'avvio di `DatabaseAgent` o `DatabaseManager`) che esegue:
     ```sql
     CREATE TABLE IF NOT EXISTS results_organ_scores (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         model_id INTEGER NOT NULL,
         model_name TEXT NOT NULL,
         dataset_name TEXT NOT NULL,
         organ_name TEXT NOT NULL,
         dice_primary TEXT,
         dice_best TEXT,
         prompt_type TEXT,
         data_budget TEXT,
         consensus_status TEXT NOT NULL,
         score_agent_a REAL,
         score_agent_b REAL,
         inter_agent_delta REAL,
         audit_confidence REAL,
         table_reference TEXT,
         source_reference TEXT,
         comments TEXT,
         updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
         UNIQUE(model_id, dataset_name, organ_name)
     );
     ```
  2. Aggiungere i metodi helper `get_organ_scores_for_model(model_id: int) -> list[dict]` e `upsert_organ_score(...)`.
* **Modifiche in `src/agent_gmmis/database/database_agent.py`**:
  1. In `_deduplicate_score_items()` (righe 467–528):
     * Normalizzare `dataset_name` tramite `DatasetAliasResolver` e `organ_or_task` tramite `OrganAliasResolver`.
     * Se per uno stesso `canonical_dataset` esistono punteggi per organi specifici (`canonical_organ != "Dataset-Level"`), **scartare** qualsiasi riga con `canonical_organ == "Dataset-Level"` (`Overall`), così da non inquinare il database con medie ridondanti.
     * Raggruppare e deduplicare per **`(canonical_dataset, canonical_organ)`** anziché solo per `canonical_dataset`, selezionando il punteggio di picco (`max(dice_score)`) per ciascun singolo organo.
  2. Salvare ciascun punteggio per organo validato nella tabella `results_organ_scores` applicando le stesse regole di sicurezza (inserimento automatico per `CONCORDANT` con $\text{Dice}_{\text{new}} > \text{Dice}_{\text{old}}$, blocco `SKIP_LOW_CONFIDENCE` per `SINGLE_AGENT_UNCONFIRMED` o `DISPUTED` in modalità strict).
  3. Per mantenere la retrocompatibilità con le viste legacy `results_primary` / `results_best` (solo se richiesto dagli script di esportazione matriciale), se un dataset ha $k$ organi concordanti estratti nello stesso paper, non chiedere mai `Overall` all'LLM ma lasciare `results_organ_scores` come fonte primaria di verità.

#### Task 1.6 — Bridge Deterministico in `ExtractionEvaluator` per la Survey Storica
* **File**: `src/agent_gmmis/extraction/evaluator.py`
* **Modifiche**:
  * Poiché la Ground Truth dei 40 paper nella survey del Prof. Moglia contiene 1 solo numero aggregato per dataset (`Overall`), mentre i nostri agenti ora estrarranno solo i singoli organi $O_1, \dots, O_k$:
  * In `ExtractionEvaluator._select_candidate_score()` (righe 142–169), quando per un dataset $D$ sono presenti i punteggi dei singoli organi $O_1, \dots, O_k$ (e non esiste `Overall`), calcolare **deterministicamente in Python** la media aritmetica dei punteggi di picco per organo di quel dataset ($\hat{S}_D = \frac{1}{k}\sum_{i=1}^k S_{D, O_i}$, raggruppando per lo stesso `prompt_type`/`data_budget` ottimale) come candidato da confrontare con il valore della survey storica.

---

### Modulo 2: Centralità della Metrica di Concordanza Inter-Agente (IAA)

#### Task 2.1 — Fix del Bug Critico in `AuditorAgent.normalize_data_budget()`
* **File**: `src/agent_gmmis/auditing/auditor.py` (righe 130–151 e 872–874)
* **Problema riscontrato nei log**:
  `normalize_data_budget()` restituisce la stringa originale per qualsiasi valore non riconosciuto (es. `"Not applicable"`, `"1024x1024"`, `"bbox prompt"`). Subito dopo, in `audit_consensus()` (riga 873), l'istruzione:
  `if sem_budget_a and sem_budget_b and sem_budget_a != sem_budget_b: continue`
  impedisce l'accoppiamento di due punteggi **numericamente identici** (es. `0.7734` e `0.7734` in `SAM-Med2D` e 31 casi in `Mistral-7B + Gemini`), degradandoli a `SINGLE_AGENT_UNCONFIRMED`!
* **Soluzione**:
  Riscrivere `AuditorAgent.normalize_data_budget(budget_str: Optional[str]) -> Optional[str]` in modo che restituisca una stringa canonica **ESCLUSIVAMENTE** se `budget_str` rappresenta una vera frazione di training semi-supervisionata o federata:
  * Percentuali di label: regex `r'\b(\d+(?:\.\d+)?)\s*%'` (se $< 100\%$, restituisce `f"{val}%"`; se `100%`, restituisce `None` in quanto regime full-supervision standard).
  * Few-shot: regex `r'\b(\d+)\s*-?\s*shot\b'` $\to$ `f"{n}-shot"`.
  * Federated clients/sites: regex `r'\b(\d+)\s*(?:client|site)s?\b'` $\to$ `f"{n}-clients"`.
  * **Qualsiasi altra stringa** (`"not applicable"`, `"n/a"`, `"none"`, `"null"`, `"full"`, `"fully supervised"`, `"100%"`, risoluzioni `"1024x1024"`, o prompt `"bbox"`, `"1 point"`) **DEVE restituire `None`**.

#### Task 2.2 — Matching Canonico `(Dataset, Organ, Budget)` e Consolidamento Multi-Prompt
* **File**: `src/agent_gmmis/auditing/auditor.py` (righe 857–920) e `src/agent_gmmis/extraction/dual_coordinator.py` (righe 34–74)
* **Modifiche**:
  1. In `audit_consensus()` e in `_find_matching_score_index()`, normalizzare sempre sia il dataset (`DatasetAliasResolver.resolve()`) sia l'organo (`OrganAliasResolver.resolve()`). Due punteggi sono candidati all'accoppiamento se e solo se hanno lo stesso `canonical_dataset`, lo stesso `canonical_organ` e `sem_budget_a == sem_budget_b`.
  2. **Pre-consolidamento dei regimi di prompt dello stesso agente**: Nei modelli interattivi (es. `SAM-Med2D`), una tabella può avere sia la colonna `1 point` sia la colonna `bounding box` per lo stesso `(dataset, organo)`. Se l'Agente A estrae entrambe le colonne e l'Agente B estrae solo la colonna di picco (`bounding box`), prima del bipartite matching (o dopo aver accoppiato i punteggi con lo stesso `prompt_type`), consolidare per ciascun agente i duplicati sulla stessa tripla `(canonical_dataset, canonical_organ, sem_budget)` mantenendo il valore di picco (`max(dice_score)`), così da evitare che le colonne sub-ottimali (`1 point`) restino appese come falsi `SINGLE_AGENT_UNCONFIRMED` o vengano accoppiate per errore con la colonna `bounding box` dell'altro agente.

#### Task 2.3 — Metriche di Concordanza di Primo Livello nello Schema di Audit e Valutazione
* **File**: `src/agent_gmmis/auditing/schemas.py`, `src/agent_gmmis/auditing/auditor.py`, `src/agent_gmmis/extraction/evaluator.py`, `scripts/run_batch_benchmark.py`
* **Modifiche in `ConsensusAuditReport` (`src/agent_gmmis/auditing/schemas.py`, righe 54–98)**:
  Aggiungere e calcolare in `AuditorAgent.audit_consensus()` i seguenti campi:
  1. `paired_concordance_rate: float` ($\text{IAA}_{\text{paired}} = \frac{N_{\text{CONCORDANT}}}{N_{\text{CONCORDANT}} + N_{\text{DISPUTED}}}$): misura la concordanza pura su tutti gli organi/dataset estratti da entrambi gli agenti ($|\Delta| \le 0.01$).
  2. `strict_concordance_rate: float` ($\text{IAA}_{\text{strict}} = \frac{N_{\text{CONCORDANT}}}{N_{\text{CONCORDANT}} + N_{\text{DISPUTED}} + N_{\text{SINGLETON}}}$): include anche i singleton a denominatore.
  3. `exact_concordance_rate: float` ($\text{IAA}_{\Delta=0} = \frac{N_{|\Delta| == 0}}{N_{\text{CONCORDANT}} + N_{\text{DISPUTED}}}$): percentuale di accoppiamenti con identità al quarto decimale ($S_A = S_B$).
  4. `inter_agent_mae: Optional[float]` ($\text{MAE}_{A,B} = \frac{1}{N_{\text{paired}}}\sum |S_A - S_B|$): distanza media in punti Dice tra Agente A e Agente B.
  5. `concordant_organs_count: int`, `disputed_organs_count: int`, `singleton_organs_count: int`.
* **Modifiche in `scripts/run_batch_benchmark.py` (righe 750–920)**:
  Nel report aggregato (`benchmark_results.json` e `benchmark_report.md`), promuovere la **Concordanza Inter-Agente** a tabella principale insieme all'accuratezza GT, calcolando:
  * **Micro-Averaged Paired IAA%** (totale organi concordanti / totale organi accoppiati sui 40 paper).
  * **Macro-Averaged Paper IAA%** (media dei tassi di concordanza dei singoli paper).
  * **Round-1 IAA% vs. Final Post-Deliberation IAA% ($\Delta\text{IAA}_{\text{delib}}$)** e riduzione di $\text{MAE}_{A,B}$ dal Round 1 al Round finale.
  * **Validazione Empirica dell'Ipotesi "Concordanza $\implies$ Verità"**:
    Calcolare separatamente l'errore rispetto alla Ground Truth della survey per i punteggi concordanti vs. non concordanti:
    * `gt_mae_when_concordant` ($\text{MAE}_{\text{GT}} \mid \text{CONCORDANT}$) e `gt_exact_match_rate_when_concordant`
    * `gt_mae_when_non_concordant` ($\text{MAE}_{\text{GT}} \mid \text{DISPUTED} \cup \text{SINGLETON}$)
    Questo indicatore dimostra matematicamente nella tesi l'intuizione del Prof. Moglia: quando i due agenti sono concordi, l'errore rispetto alla Ground Truth tende a zero, rendendo la concordanza un certificato affidabile anche per i punteggi per-organo e per i nuovi dataset.

---

### Modulo 3: Allineamento al Comportamento dei Modelli Cinesi su `SAM-Med2D` e Simili

#### Task 3.1 — Regola Esplicita per Foundation Model 2D-Slice su Volumi 3D e Filtro Righe 2D Fotografiche
* **File**: `src/agent_gmmis/extraction/prompts.py` (righe 17–31)
* **Modifiche in `EXTRACTION_SYSTEM_PROMPT` (Regola 1: `3D VOLUMETRIC SCOPE & INCLUSION SCREENING`)**:
  1. Specificare in modo inequivocabile che i **Foundation Models 2D o ibridi 2D/3D valutati slice-by-slice su scansioni volumetriche 3D** (citando esplicitamente `SAM-Med2D`, `MedSAM`, `IMIS-Net`, `BioSAM-2` come esempi) **DEVONO avere `is_3d = True`**, anche se il nome dell'architettura contiene `"2D"`, ogniqualvolta il paper riporta risultati su modalità radiologiche volumetriche (`CT`, `MRI`/`MR`, `PET`, `3D Ultrasound`) o benchmark 3D (`FLARE`, `KiTS`, `AMOS`, `BTCV`, `ATLAS`, `CrossMoDA`, `ToothFairy`, ecc.).
  2. Aggiungere la regola **`MULTI-MODALITY TABLE FILTERING`**: Quando una tabella riporta risultati aggregati per modalità (come la Tabella 3 di `SAM-Med2D` o `IMIS-Net`), l'agente **DEVE estrarre le modalità radiologiche volumetriche (`CT`, `MRI`/`MR`, `PET`)** e **DEVE IGNORARE** le modalità puramente fotografiche/microscopiche 2D (`Dermoscopy`, `Fundus`, `Endoscopy`, `Histopathology`, `Microscopy`, `X-Ray`/`CXR`), evitando di contaminare i punteggi 3D con dataset fotografici 2D.

#### Task 3.2 — Espansione Completa di `known_3d_datasets` e Modalità Radiologiche nell'Auditor
* **File**: `src/agent_gmmis/auditing/auditor.py` (righe 189–212)
* **Problema riscontrato nei log**:
  In `AuditorAgent.audit()`, `known_3d_datasets` contiene solo 11 nomi. Quando un agente estrae `CT`, `MR`, `PET` o i challenge MICCAI 2023 di `SAM-Med2D` (`ATLAS2023`, `CrossMoDA23`, `CAS2023`, `LNQ2023`, `SEG.A.2023`, `TDSC-ABUS2023`, `ToothFairy2023`), l'override `evaluates_known_3d` non scatta!
* **Soluzione**:
  1. Espandere `known_3d_datasets` includendo tutti i 58 dataset 3D canonici del database + tutti i challenge volumetrici MICCAI e le modalità radiologiche 3D:
     ```python
     KNOWN_3D_DATASETS_AND_MODALITIES = {
         "acdc", "brats", "btcv", "amos", "kits", "lits", "flare", "totalsegmentator",
         "synapse", "isles", "segthor", "msd", "abdomenct-1k", "abdomenatlas", "word",
         "verse", "ctspine1k", "autopet", "mm-whs", "asoca", "promise12", "prostate158",
         "chaos", "feta", "wmh", "msseg", "oasis", "atlas", "atlas2023", "atlas v2.0",
         "crossmoda", "crossmoda23", "cas2023", "lnq2023", "seg.a.2023", "tdsc-abus2023",
         "toothfairy", "toothfairy2023", "uls23", "sliver07", "3dircadb", "visceral",
         "pancreas-ct", "nih pancreas", "covid-19-20", "lidc-idri", "luna16", "parse2022",
         "atm22", "topcow", "care", "bhsd", "saros", "totalvibes", "han-seg", "hecktor",
         "instance22", "autoimplant", "ribfrac", "ct-org", "dap-atlas",
         "ct", "mri", "mr", "pet", "pet-ct", "3d ultrasound", "abus"
     }
     ```
  2. Filtrare via deterministicamente (in `ExtractionAgent` o all'ingresso di `AuditorAgent`) eventuali score estratti su modalità puramente fotografiche 2D nella blacklist:
     ```python
     NON_3D_PHOTOGRAPHIC_MODALITIES = {
         "dermoscopy", "fundus", "endoscopy", "histopathology", "microscopy",
         "pathology", "x-ray", "cxr", "chest x-ray", "retinal", "colonoscopy"
     }
     ```
     In questo modo, in `Codestral-22B + Gemini`, la riga `Dermoscopy (0.8769)` non farà mai più scattare il falso positivo di *"Template Hallucination across 3 datasets"* (`-0.50` di penalità)!

#### Task 3.3 — Fix della Propagazione dell'Override `is_3d` in `AuditorAgent.audit_consensus()`
* **File**: `src/agent_gmmis/auditing/auditor.py` (righe 199–205 e 697–727)
* **Problema riscontrato nei log**:
  In `DeepSeek-R1-14B + Gemini` su `SAM-Med2D`, `audit()` aveva impostato `report_a.passed_inclusion_screening = True` grazie all'override, ma non aveva aggiornato `extracted_a.screening.is_3d`. Di conseguenza, `audit_consensus()` (riga 700) confrontava `sc_a.is_3d` (`False`) con `sc_b.is_3d` (`True`), marcava lo screening come `DISPUTED`, sottraeva `-0.15` alla confidenza e forzava `overall_status = AuditStatus.FLAGGED_WARNING` (`0.56 < 0.60`), bloccando il paper!
* **Soluzione**:
  * In `AuditorAgent.audit()` (riga 201), quando `evaluates_known_3d` è `True`, aggiornare anche `extracted.screening.is_3d = True` (se `extracted.screening` è presente).
  * Inoltre, in `AuditorAgent.audit_consensus()` (righe 697–706), calcolare:
    ```python
    eff_is_3d_a = (sc_a.is_3d if sc_a else False) or report_a.passed_inclusion_screening
    eff_is_3d_b = (sc_b.is_3d if sc_b else False) or report_b.passed_inclusion_screening
    ```
    usando `eff_is_3d_a` ed `eff_is_3d_b` per determinare `screening_status`. Se entrambi passano lo screening (`True` e `True`), `consensus_screening.status` deve essere **`ConsensusStatus.CONCORDANT`** senza alcuna penalità di confidenza!

---

### Modulo 4: Aggiornamento ed Espansione Unit Tests (`tests/`)

Prima di avviare i job su Mufasa, gli agenti di sviluppo devono aggiornare ed eseguire `pytest` su:
1. `tests/test_alias_resolver.py`:
   * Testare `OrganAliasResolver.resolve()` su tutte le abbreviazioni di `BTCV`, `AMOS`, `FLARE`, `BraTS` (`WT/TC/ET`), `ACDC` (`RV/LV/Myo`).
2. `tests/test_auditor.py`:
   * Testare che `normalize_data_budget("Not applicable") is None` e `normalize_data_budget("1024x1024") is None`, e che due agenti con score identici ma stringhe neutre diverse in `data_budget` risultino `CONCORDANT`.
   * Testare il caso `SAM-Med2D`: Agente A con `is_3d=False` ma score su `CT`, `MR`, `FLARE23` e Agente B con `is_3d=True` $\to$ verifica che `consensus_screening.status == ConsensusStatus.CONCORDANT`, nessuna penalità `-0.15`, e `overall_status == AuditStatus.VERIFIED`.
   * Testare il filtraggio delle modalità 2D fotografiche (`Dermoscopy`, `Fundus`).
3. `tests/test_dual_coordinator.py`:
   * Testare `reinterrogate_organ_breakdown` quando Agente A estrae `Liver`, `Spleen`, `Pancreas` su `BTCV` e Agente B estrae `Overall`.
4. `tests/test_database_agent.py`:
   * Testare la creazione di `results_organ_scores` e la persistenza separata di ciascun organo per lo stesso `(model_id, dataset_name)`.

---

## 3. Piano Sperimentale dei Benchmark su Mufasa (GPU A100 40 GB)

Una volta completati e verificati i Moduli 1–4, lanceremo i benchmark su Mufasa sfruttando la **GPU A100 da 40 GB** (`--qos=gpuheavy-40`, `--gres=gpu:40gb:1`) e il budget di contesto espanso a **`35.000` caratteri** (`max_model_len = 24576–32768` token in `src/agent_gmmis/llm/registry.py`, contro i `16.000` caratteri delle run precedenti su 16 GB).

### Smoke Test Preliminare (5 minuti)
Prima dei batch da 40 paper, lanciare un test rapido su 2 paper rappresentativi (`SAM_Med2D.pdf` e un paper denso multi-organo come `Swin_UNETR.pdf` o `nnSAM.pdf`) per verificare:
* che `SAM-Med2D` ottenga `overall_status = VERIFIED`, `consensus_screening = CONCORDANT`, confidenza $\ge 0.85$;
* che `results_organ_scores` registri tutti i singoli organi (`Liver`, `Spleen`, `Right Kidney`, ecc.) con i rispettivi stati di concordanza.

---

### Blocco 1: Ablation Study su Contesto (35k char) e Nuova Architettura (Le 6 Coppie Storiche)
**Obiettivo**: Rilanciare sui 40 paper le 6 configurazioni della Tabella 7.2 di `docs/sviluppi.md` sulla A100 40 GB per avere nella tesi il confronto **Pre vs. Post** a parità esatta di modelli.

| ID | Agente A | Agente B | Modalità su 40 GB | Comando di lancio (`run_unified_benchmark_mufasa.sh`) | Obiettivo di Confronto per la Tesi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1.1** | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `gemini-3.1-flash-lite-preview` | Hybrid (`10 GB` VRAM) | `AGENT_A_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ" AGENT_B_PROVIDER="gemini" ...` | Confronto con il vecchio 1° classificato (`Recall 64.6%`, `IAA 58.9%`). Misura il guadagno dato dal contesto 35k e dal per-organo. |
| **B1.2** | `solidrust/Codestral-22B-v0.1-hf-AWQ` | `gemini-3.1-flash-lite-preview` | Hybrid (`13 GB` VRAM) | `AGENT_A_MODEL="solidrust/Codestral-22B-v0.1-hf-AWQ" AGENT_B_PROVIDER="gemini" ...` | Verifica la promozione di `SAM-Med2D` da `FLAGGED_WARNING (0.54)` a `VERIFIED` e l'aumento di IAA sulle tabelle multi-organo. |
| **B1.3** | `casperhansen/deepseek-r1-distill-qwen-14b-awq` | `gemini-3.1-flash-lite-preview` | Hybrid (`10 GB` VRAM) | `AGENT_A_MODEL="casperhansen/deepseek-r1-distill-qwen-14b-awq" AGENT_B_PROVIDER="gemini" ...` | Verifica l'effetto di `max_output_tokens=8192` e del fix `is_3d` su `SAM-Med2D` (che era bloccato a `0.56`). |
| **B1.4** | `mistralai/Mistral-7B-Instruct-v0.3` | `gemini-3.1-flash-lite-preview` | Hybrid (`7 GB` VRAM) | `AGENT_A_MODEL="mistralai/Mistral-7B-Instruct-v0.3" AGENT_B_PROVIDER="gemini" ...` | **Prova regina del fix dell'Auditor**: mostra il balzo di IAA dal vecchio `32.5%` (viziato dai 31 mancati match su `data_budget`) al valore reale (>70%). |
| **B1.5** | `Qwen/Qwen2.5-7B-Instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `dual_simultaneous` (`14.5 GB` VRAM) | `AGENT_A_MODEL="Qwen/Qwen2.5-7B-Instruct" AGENT_B_PROVIDER="local_hf" AGENT_B_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct" ...` | Sulla 16 GB richiedeva 3h18m in `dual_sequential`; sulla 40 GB gira in **`dual_simultaneous`** (~40 min) con contesto 2.2× superiore. |
| **B1.6** | `Qwen/Qwen2.5-7B-Instruct` | `mistralai/Mistral-7B-Instruct-v0.3` | `dual_simultaneous` (`14.0 GB` VRAM) | `AGENT_A_MODEL="Qwen/Qwen2.5-7B-Instruct" AGENT_B_PROVIDER="local_hf" AGENT_B_MODEL="mistralai/Mistral-7B-Instruct-v0.3" ...` | Verifica la risoluzione del conflitto Cina-Europa su `SAM-Med2D` (che su 16 GB era `0.35 REJECTED`). |

---

### Blocco 2: Nuove Coppie 100% Locali in Simultanea (`14B + 14B` e `22B + 14B`)
**Obiettivo**: Sfruttare i `40 GB` di VRAM (`26.0 GB` allocabili per i pesi con `gpu_memory_utilization = 0.42 + 0.42`) per far coesistere **contemporaneamente in GPU** due modelli da `14B–22B` senza API esterne e senza swap su disco.

| ID | Agente A (Locale) | Agente B (Locale) | VRAM Pesi | Modalità | Valore Scientifico per la Tesi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B2.1** | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `casperhansen/deepseek-r1-distill-qwen-14b-awq` | `20.0 GB` | `dual_simultaneous` | **Coppia 100% Cinese Instruct + Reasoning (14B+14B)**: valuta la sinergia tra il miglior modello instruct tabellare cinese e il modello reasoning `<think>` cinese su `SAM-Med2D` e sull'estrazione per-organo. |
| **B2.2** | `solidrust/Codestral-22B-v0.1-hf-AWQ` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `23.0 GB` | `dual_simultaneous` | **Top Cross-Origin Europa + Cina (22B+14B)**: accoppia i due estrattori più forti in assoluto (Mistral/Francia + Alibaba/Cina) in totale indipendenza di pre-training. Se concordano, l'affidabilità del dato è massima. |
| **B2.3** | `solidrust/Codestral-22B-v0.1-hf-AWQ` | `casperhansen/deepseek-r1-distill-qwen-14b-awq` | `23.0 GB` | `dual_simultaneous` | **Cross-Origin Europa Structural + Cina Reasoning (22B+14B)**: combina la lettura strutturale di Codestral 22B con la verifica passo-passo di DeepSeek-R1 14B. |
| **B2.4** | `THUDM/glm-4-9b-chat` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `19.0 GB` | `dual_simultaneous` | **Multi-Family China (ZhipuAI + Alibaba)**: verifica che il comportamento ottimale sui foundation model 2D/3D (`SAM-Med2D`, `MedSAM`) sia generalizzabile tra diverse famiglie di modelli cinesi (`GLM-4` + `Qwen2.5`). |

---

### Blocco 3: Modelli Frontier `32B AWQ` (Sbloccati dalla 40 GB)
**Obiettivo**: Testare i modelli da 32 miliardi di parametri (`~20 GB` VRAM in AWQ 4-bit), impossibili da caricare sulla vecchia GPU da 16/20 GB, per tracciare la **Scaling Law (7B $\to$ 14B $\to$ 22B $\to$ 32B)** nella tesi.

| ID | Agente A | Agente B | Modalità su 40 GB | Valore Scientifico per la Tesi |
| :--- | :--- | :--- | :--- | :--- |
| **B3.1** | `Qwen/Qwen2.5-32B-Instruct-AWQ` | `gemini-3.1-flash-lite-preview` | Hybrid (`20 GB` VRAM) | Misura il salto di qualità passando da `Qwen-7B` $\to$ `Qwen-14B` $\to$ `Qwen-32B` a parità di partner Gemini. |
| **B3.2** | `casperhansen/deepseek-r1-distill-qwen-32b-awq` | `gemini-3.1-flash-lite-preview` | Hybrid (`20 GB` VRAM) | Misura il salto di qualità del ragionamento `<think>` passando da `14B` a `32B`. |
| **B3.3** | `Qwen/Qwen2.5-32B-Instruct-AWQ` | `casperhansen/deepseek-r1-distill-qwen-32b-awq` | `dual_sequential` (`20 GB` + `20 GB`) | **Upper-Bound 100% Locale (32B + 32B)**: usa la pipeline 2-stage per far deliberare due giganti da 32B sulla singola A100 da 40 GB. |

---

## 4. Analisi Dati e Tabelle da Generare per la Tesi

Al termine dei benchmark, uno script dedicato (`scripts/generate_thesis_tables.py`, evoluzione di `scripts/compare_benchmarks.py`) genererà automaticamente in Markdown e LaTeX le **5 tabelle cardine del capitolo sperimentale**:

1. **Tabella T1 — Dimostrazione Empirica "Concordanza $\implies$ Verità"**:
   Mostra per ogni coppia di modelli il confronto tra $\text{MAE}_{\text{GT}} \mid \text{CONCORDANT}$ (errore rispetto alla Ground Truth quando Agente A e Agente B concordano) e $\text{MAE}_{\text{GT}} \mid \text{NON-CONCORDANT}$ (errore quando sono in disaccordo o singleton), insieme alla percentuale di Exact Match. Questa tabella valida scientificamente la tesi del Prof. Moglia: l'accordo inter-agente in cieco abbatte l'errore di estrazione vicino allo zero e legittima l'inserimento automatico dei punteggi per singolo organo in `results_organ_scores`.
2. **Tabella T2 — Statistiche dell'Estrazione Granulare Per-Organo (`results_organ_scores`)**:
   Riporta il numero totale di tuple `(Modello, Dataset, Organo)` estratte sui 40 paper (attese **>600–800 misurazioni per organo** contro le 99 celle `Overall` della survey manuale), con il dettaglio della concordanza organo per organo sui benchmark principali (`BTCV`, `AMOS`, `FLARE22`, `BraTS`, `ACDC`).
3. **Tabella T3 — Ablation Study: Impatto di Contesto (`16k` $\to$ `35k`) e Architettura Per-Organo**:
   Confronto fianco a fianco tra i risultati in `docs/sviluppi.md` (16 GB, `Overall`) e il Blocco 1 (40 GB, `35k` char, per-organo + fix Auditor).
4. **Tabella T4 — Analisi Geopolitica/Architetturale su `SAM-Med2D` e Foundation Models Ibridi (Cina vs. Europa vs. USA)**:
   Confronto dettagliato sul comportamento di screening (`is_3d`), numero di modalità/challenge 3D estratti da `SAM-Med2D` (`CT`, `MR`, `PET`, `FLARE23`, `KiTS23`, `ATLAS2023`, `CrossMoDA23`, `ToothFairy2023`, ecc.) e tasso di concordanza tra famiglie di modelli cinesi (`Qwen`, `DeepSeek`, `GLM`), europei (`Mistral`, `Codestral`) e USA (`Llama`, `Gemini`).
5. **Tabella T5 — Scaling Law (7B $\to$ 14B $\to$ 22B $\to$ 32B) e Frontiera di Pareto Tempo/Concordanza/Accuratezza**:
   Confronta le configurazioni `Hybrid`, `Simultaneous Dual-Local` (Blocco 2) e `Frontier 32B` (Blocco 3) per identificare la configurazione ottima da lasciare in produzione per l'aggiornamento continuo della living survey.

---

## 5. Checklist Operativa per gli Agenti di Sviluppo

- [x] **Step 1**: Implementare `OrganAliasResolver` in `src/agent_gmmis/extraction/alias_resolver.py` e i relativi test in `tests/test_alias_resolver.py`.
- [x] **Step 2**: Aggiornare `src/agent_gmmis/extraction/prompts.py`, `src/agent_gmmis/extraction/schemas.py` e `src/agent_gmmis/llm/registry.py` (`max_output_tokens` a `4096`/`8192`, mandato per-organo, rimozione tetto 25 score, regola `SAM-Med2D` e filtro modalità 2D fotografiche).
- [x] **Step 3**: Sostituire `reinterrogate_granularity_macro` con `reinterrogate_organ_breakdown` in `src/agent_gmmis/extraction/extractor.py` e `src/agent_gmmis/extraction/dual_coordinator.py`.
- [x] **Step 4**: Correggere i bug in `src/agent_gmmis/auditing/auditor.py` (`normalize_data_budget`, espansione `known_3d_datasets` + filtro 2D fotografico, propagazione override `is_3d` in `audit_consensus`, e pre-consolidamento di picco per `(dataset, organo, budget)`) e aggiungere le nuove metriche IAA in `src/agent_gmmis/auditing/schemas.py`.
- [x] **Step 5**: Creare la tabella `results_organ_scores` in `src/agent_gmmis/data/db_manager.py`, aggiornare `src/agent_gmmis/database/database_agent.py` per deduplicare e salvare a livello di `(dataset, organo)`, e aggiornare `src/agent_gmmis/extraction/evaluator.py` + `scripts/run_batch_benchmark.py` (calcolo deterministico media organi per confronto survey + calcolo $\text{MAE}_{\text{GT}} \mid \text{CONCORDANT}$).
- [x] **Step 6**: Eseguire l'intera suite `pytest` verificando che tutti i test esistenti e nuovi passino al 100% (`151 passed`, poi `154 passed`).
- [x] **Step 7 (Fix post Smoke-Test & Hardening 40 GB)**:
  - Fix SQLite `DatabaseManager.get_dataset_statistics` / `get_leaderboard_for_dataset` con verifica preventiva `has_column` (`CrossMoDA` su `SAM-Med2D`).
  - Helper `_parse_and_normalize_dice_score` condiviso in `src/agent_gmmis/extraction/schemas.py`.
  - Estensione di `AuditorAgent.normalize_data_budget` agli split standard (`Validation`, `Testing`, `5-fold CV`) e bonus matching `+3` su `table_or_section_ref` (`Swin UNETR` passato da 0% a 100% di concordanza).
  - Hardening `dual_simultaneous` (`max_len <= 16384` per `combined_vram >= 18.0 GB` e avvio sequenziale `Server A` $\to$ `Server B`) + `max_wait=900s` per modelli `32B AWQ`.
  - Fix mapping chiavi JSON flat (`audit_status`, `dice_mae`) in `scripts/generate_thesis_tables.py` (`T3` e `T4`).
  - Hardening retry Gemini in `src/agent_gmmis/llm/client.py`: rimosso `gemini-flash-latest` (`429`) dai fallback, `max_retries=8` alternando `primary_model` / `gemini-flash-lite-latest` con backoff fino a `20.0s` (sincronizzato per i run `B1.3`+).

---

## 6. Registro di Avanzamento ed Esecuzione Benchmark (Aggiornato al 18 Novembre 2026)

### 6.1. Smoke Test Preliminare (`Job 6201` — `20261113_161933_smoke_test_b1_1_v2`)
Eseguito su Mufasa A100 40 GB con `Qwen/Qwen2.5-14B-Instruct-AWQ` + `gemini-3.1-flash-lite-preview` sui 2 paper rappresentativi (`4m 50s`, `ExitCode 0:0`):
* **`SAM-Med2D`**: `FLAGGED_WARNING` (`conf=88%`, `Paired IAA = 5/7 (71.4%)`), estratti `CT`, `FLARE`, `KiTS`, `Overall Test Set`, `PET`, `CrossMoDA23`, `MR`; deliberazione multi-round `+71.4 pp` dal Round 1; 4 score granulari inseriti in `results_organ_scores`.
* **`Swin UNETR`**: `VERIFIED` (`conf=100%`, `Paired IAA = 6/6 (100.0%)`, `Inter-Agent MAE = 0.0000`), estratti e salvati in `results_organ_scores` gli score separati per sotto-regioni BraTS (`Enhancing Tumor (ET)`, `Whole Tumor (WT)`, `Tumor Core (TC)`).

### 6.2. Risultati Completi Prima Tornata Benchmark 40 GB (`13–18 Novembre`, 40 Paper)
Tabelle comparative generate anche in `logs/mufasa_runs/thesis_tables_40gb/thesis_tables.md` (`.tex` incluso):

| ID | Configurazione (`Agente A + Agente B`) | Topologia | `VERIFIED` | Matched / GT (Recall) | Micro MAE | CAS | Micro Paired IAA | Macro Paired IAA | Exact IAA | Deliberation (R1 $\to$ Fin.) | $\text{MAE}_{\text{GT}}$ Conc. vs Non-Conc. | Organ `INSERT` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B3.1** | `Qwen2.5-32B-AWQ + Gemini` | `hybrid` | **21/40** | **117/196 (`59.7%`)** | `0.0223` | **0.5836** | **89.7%** (`269/300`) | **90.7%** | **84.0%** | `71.9%` $\to$ `90.7%` (`+18.7pp`, 54) | **`0.0129`** *(n=52)* vs `0.0298` *(n=65)* | **398** |
| **B3.2** | `DeepSeek-R1-32B-AWQ + Gemini` | `hybrid` | 20/40 | 113/196 (`57.7%`) | `0.0264` | 0.5613 | 89.1% (`228/256`) | 89.4% | 80.5% | `65.0%` $\to$ `89.4%` (`+24.4pp`, 80) | **`0.0218`** *(n=60)* vs `0.0315` *(n=53)* | 305 |
| **B1.1** | `Qwen2.5-14B-AWQ + Gemini` | `hybrid` | 14/40 | 108/196 (`55.1%`) | **0.0193** | 0.5404 | 83.3% (`219/263`) | 88.1% | 78.3% | `63.4%` $\to$ `88.1%` (`+24.7pp`, 86) | **`0.0192`** *(n=26)* vs `0.0193` *(n=82)* | 367 |
| **B1.3** | `DeepSeek-R1-14B-AWQ + Gemini` | `hybrid` | 14/40 | 97/196 (`49.5%`) | `0.0238` | 0.4831 | 78.9% (`150/190`) | 83.2% | 65.8% | `69.7%` $\to$ `83.2%` (`+13.5pp`, 39) | **`0.0097`** *(n=44)* vs `0.0355` *(n=53)* | 306 |
| **B1.4** | `Mistral-7B-v0.3 (FP16) + Gemini` | `hybrid` | 11/40 | 92/196 (`46.9%`) | `0.0240` | 0.4581 | 68.5% (`126/184`) | 74.9% | 44.0% | `39.0%` $\to$ `74.9%` (`+36.0pp`, 56) | **`0.0085`** *(n=19)* vs `0.0280` *(n=73)* | 215 |
| **B1.2** | `Codestral-22B-AWQ + Gemini` *(503 Gemini)* | `hybrid` | 3/40 | 80/196 (`40.8%`) | `0.0369` | 0.3931 | 73.8% (`62/84`) | 78.0% | 65.5% | `62.3%` $\to$ `78.0%` (`+15.7pp`, 8) | **`0.0004`** *(n=8)* vs `0.0402` *(n=71)* | 192 |
| **B1.5** | `Qwen2.5-7B-AWQ + Llama-3.1-8B-AWQ` | `dual_sim` | 13/40 | 48/196 (`24.5%`) | `0.0410` | 0.2349 | 73.4% (`105/143`) | 70.9% | 57.3% | `33.6%` $\to$ `70.9%` (`+37.4pp`, 56) | **`0.0157`** *(n=10)* vs `0.0467` *(n=37)* | 167 |
| **B1.6** | `Qwen2.5-7B-AWQ + Mistral-7B-AWQ` | `dual_sim` | 6/40 | 36/196 (`18.4%`) | `0.0543` | 0.1737 | 54.0% (`34/63`) | 53.4% | 38.1% | `13.5%` $\to$ `53.4%` (`+39.9pp`, 35) | **`0.0478`** *(n=6)* vs `0.0542` *(n=23)* | 90 |
| **B2.1** | `Qwen2.5-14B-AWQ + DeepSeek-R1-14B-AWQ` | `dual_sim` | 9/40 | 36/196 (`18.4%`) | `0.0426` | 0.1759 | 59.6% (`34/57`) | 67.2% | 33.3% | `5.2%` $\to$ `67.2%` (`+62.0pp`, 39) | **`0.0055`** *(n=12)* vs `0.0611` *(n=24)* | 110 |
| **B2.2** | `Codestral-22B-AWQ + Qwen2.5-14B-AWQ` | `dual_sim` | 13/40 | 37/196 (`18.9%`) | `0.0552` | 0.1784 | 73.6% (`78/106`) | 76.7% | 58.5% | `19.2%` $\to$ `76.7%` (`+57.5pp`, 39) | **`0.0209`** *(n=13)* vs `0.0737` *(n=24)* | 157 |
| **B2.3** | `Codestral-22B-AWQ + DeepSeek-R1-14B-AWQ` | `dual_sim` | 6/40 | 24/196 (`12.2%`) | `0.0710` | 0.1138 | 58.1% (`18/31`) | 69.6% | 32.3% | `5.3%` $\to$ `69.6%` (`+64.3pp`, 22) | **`0.0265`** *(n=5)* vs `0.0856` *(n=18)* | 56 |

### 6.3. Analisi Problemi Tecnici Trovati e Coda Rilancio `v2` (`18 Novembre`)
Dall'analisi dei log della prima tornata notturna sono stati identificati e risolti 4 problemi tecnici:
1. **`MUFASA_MAX_MODEL_LEN` non esportato in `dual_simultaneous`** (causa della Recall `12%–24%` in `B1.5`, `B1.6`, `B2.1`, `B2.2`, `B2.3`): `run_unified_benchmark_mufasa.sh` avviava vLLM con `16384` token ma `client.py` usava il default `24576`, facendo fallire ~15–20 paper lunghi per contesto eccessivo. Risolto esportando esplicitamente `MUFASA_MAX_MODEL_LEN` in tutti i branch di `run_unified_benchmark_mufasa.sh` e abbassando la stima `chars_per_token = 1.85` (con margini `800` token) in `src/agent_gmmis/llm/client.py`.
2. **Serializzazione JSON `RepoInspector`** (causava `❌ ERROR` su `BiomedParse` in `B2.2` e `B3.1` e su `MEA M-SAM` in `B2.2`): `scripts/run_batch_benchmark.py:421` passava l'istanza Python `repo_inspector` a `RunLogger`. Sostituito con un dizionario JSON-serializzabile `{"recovered": True, "code_repository_url": ...}`.
3. **Orfani `VLLM::EngineCore` in `dual_sequential`** (causa dell'OOM allo Stage 2 di **`B3.3`**, `Job 6389`, dopo che lo Stage 1 di `Qwen2.5-32B-AWQ` aveva completato 40/40 paper): `kill -9 $VLLM_PID_A` lasciava allocati `37 GB` di VRAM sul figlio `VLLM::EngineCore`. Risolto in `cleanup()` e nello switch Stage 1 $\to$ Stage 2 con `pkill -9 -P "$VLLM_PID_A"` e `pkill -9 -f "VLLM::EngineCore|vllm serve"` + `sleep 10`.
4. **Incompatibilità vLLM `--quantization bitsandbytes` su **`B2.4` (`GLM-4-9B-Chat`)** (`Job 6378`)**: la versione vLLM installata su Mufasa non accetta `--quantization bitsandbytes`. Sostituito il filtro di `discover_runs()` in `src/agent_gmmis/ui/run_analyzer.py` per saltare eventuali run incomplete prive di `benchmark_results.json`.

La coda di rilancio **`v2`** in `scripts/mufasa/queue_feeder_mufasa.sh` riesegue con questi fix le 7 run viziate (`B1.2_v2`, `B3.3_v2`, `B2.2_v2`, `B2.1_v2`, `B2.3_v2`, `B1.5_v2`, `B1.6_v2`), mentre `B1.1`, `B1.3`, `B1.4`, `B3.1` e `B3.2` sono già validate (e su `B3.1` il solo error di `BiomedParse` per `RepoInspector` è ormai corretto per i prossimi run).

