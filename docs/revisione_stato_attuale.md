# Revisione Tecnica, Architetturale e Metodologica dello Stato Attuale (`AgentForGMMIS`)

**Data di revisione:** 29 Ottobre 2026  
**Oggetto:** Audit critico end-to-end del codice sorgente (`src/agent_gmmis/`), degli script di orchestrazione (`scripts/`), della documentazione progettuale (`docs/`) e delle ultime 5 run sperimentali su 40 paper (`logs/mufasa_runs/`).  
**Modalità:** Sola lettura (nessun file sorgente o di configurazione è stato modificato).

---

## Sintesi Esecutiva

Il progetto **`AgentForGMMIS`** presenta un'architettura multi-agente concettualmente matura e ben allineata al problema scientifico di partenza (automatizzare la Living Review sulla *Generalist Medical Image Segmentation* di Moglia et al., 2026). L'idea di separare l'estrazione parallela (*Dual Independent Extraction*), la verifica deterministica (*Auditor / "Il Vigile"*), la deliberazione cieca iterativa (*Double-Blind Deliberation*), l'ispezione di fallback (*RepoInspector* e *OutreachAgent*) e la curatela transazionale (*DatabaseAgent*) costituisce un solido contributo ingegneristico e metodologico per una tesi magistrale.

Tuttavia, un'ispezione rigorosa del codice e dei log delle ultime 5 run su 40 paper (`20261026_181522` -> `20261027_212758`) porta alla luce **bug silenziosi ma critici**, **distorsioni nelle metriche di valutazione** (tra cui *data leakage* dell'oracolo nel calcolo del MAE, Recall superiori al 100% e *survivorship bias* sui dataset non estratti), **troncamenti ciechi del testo PDF a 12.000 caratteri** che spiegano la bassa recall complessiva (~31%–41%), e **contraddizioni tra quanto dichiarato nel `THESIS_DESIGN_LOG.md` (§33) e ciò che è realmente accaduto nei log di esecuzione**.

Il documento è organizzato nelle tre sezioni richieste:
1. **[1. Pregi](#1-pregi-scelte-architetturali-e-implementazioni-solide)** — Scelte architetturali e implementazioni solide da valorizzare nella tesi.
2. **[2. Criticità e Difetti](#2-criticità-e-difetti-bug-fragilità-e-analisi-delle-ultime-5-run)** — Bug logici, fragilità strutturali, anti-pattern e disamina chirurgica delle ultime 5 run.
3. **[3. Cambiamenti Suggeriti](#3-cambiamenti-suggeriti-piano-dazione-prioritizzato)** — Proposte concrete, ordinate per priorità, pronte per essere delegate a un agente esecutore.

---

## 1. Pregi: Scelte Architetturali e Implementazioni Solide

### 1.1. Separazione Epistemica tra Estrazione Stocastica e Auditing Deterministico
L'architettura evita l'anti-pattern più diffuso nei sistemi LLM-based: usare un secondo prompt LLM generico per "verificare" il primo.
- Il modulo [`AuditorAgent`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L194-L1083) implementa controlli deterministici e riproducibili ancorati al dominio medico-radiologico (10 criteri di validazione: `RANGE_PLAUSIBILITY`, `SCALE_NORMALIZATION`, `BASELINE_CONTAMINATION`, `PROMPT_REGIME_MIXING`, `AGGREGATION_CONSISTENCY`, `MODALITY_CONGRUENCE`, `DATASET_COVERAGE`, `METRIC_COMPLETENESS`, `EXTRACTION_CONFIDENCE`, `CROSS_BENCHMARK_CONSISTENCY`).
- La normalizzazione automatica della scala `0-100 -> 0.0-1.0` in [`_check_scale_and_range`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L317-L405) viene tracciata senza mutare distruttivamente l'oggetto originale prima dell'audit.

### 1.2. Protocollo di Deliberazione *Double-Blind* contro la *Sycophancy*
Il coordinatore [`DualExtractionCoordinator`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L130-L561) implementa una deliberazione multi-round (fino a $K$ round) ben progettata:
- **Re-interrogazione cieca sulle discrepanze** ([`reinterrogate_discrepancy`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L198-L273)): quando i due estrattori divergono oltre la soglia $\tau = 0.01$, ciascun modello viene invitato a verificare il proprio valore sul testo senza mai conoscere il valore numerico estratto dall'altro agente. Questo previene l'*anchoring bias* e la *sycophancy* tipica dei modelli instruction-tuned.
- **Risoluzione dell'asimmetria di granularità** ([`reinterrogate_granularity_macro`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L350-L426)): quando un agente estrae la media macro (`Overall`) e l'altro i singoli organi (es. i 13 organi di BTCV), il coordinatore rileva il pattern e richiede miratamente la riga di sintesi.
- **Early-stopping per stallo / equilibrio di Nash** ([`dual_coordinator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L488-L537)): il confronto tra la firma delle dispute del round precedente e quello corrente evita cicli inutili di chiamate LLM quando entrambi gli agenti confermano la propria posizione.

### 1.3. *Schema-Embedded Chain-of-Thought* (`ExtractionThoughtProcess`)
In [`ExtractedPaperData`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py#L84-L120), il campo `thought_process` ([`ExtractionThoughtProcess`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py#L8-L31)) è posizionato come **primo attributo assoluto** dello schema JSON.
- Poiché la decodifica autoregressiva genera i token da sinistra verso destra, costringere il modello a compilare `proposed_model_identification`, `tables_identified`, `baseline_models_excluded` e `granularity_strategy` *prima* della lista `extracted_scores` funge da "scratchpad" condizionale senza rompere il vincolo dello structured output.

### 1.4. Disaccoppiamento dell'Accordo Inter-Annotatore (`paired_agreement_rate`) e Bipartite Matching
In [`AuditorAgent.audit_consensus`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L839-L1014):
- L'accoppiamento tra i punteggi dell'Agente A e dell'Agente B non si ferma a una chiave rigida, ma utilizza un *greedy bipartite matching* a 3 passaggi (match esatto su `(dataset, organ_norm, budget_norm)` -> fallback su organo quando uno dei due è `overall` -> fallback su data budget quando uno dei due è `100%_or_default`), ordinando le coppie candidate per distanza assoluta di Dice (`|dice_a - dice_b|`).
- La separazione tra `paired_agreement_rate` (calcolato solo sui benchmark estratti da entrambi) e i punteggi `SINGLE_AGENT_UNCONFIRMED` (penalizzati con un fattore scalare ma non trattati come conflitti numerici) è matematicamente sensata e risolve il paradosso per cui un estrattore più completo veniva punito dal mancato richiamo del modello più debole.

### 1.5. Tracciabilità, Persistenza Granulare (`RunLogger`) e Modalità *Dry-Run*
- [`RunLogger`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/run_logger.py#L28-L624) salva sistematicamente ogni artefatto intermedio (`papers/<model_id>.json`, `curation/<model_id>_curation.json`, `curation/<model_id>_outreach.eml`, `run_summary.json` e `benchmark_report.md`), permettendo un'ispezione post-mortem completa senza dover rieseguire le pipeline.
- [`DatabaseManager`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/data/db_manager.py#L14-L321) utilizza backup automatici pre-modifica e transazioni atomiche (`with self.connection()`), mentre [`DatabaseAgent`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py#L46-L273) separa la pianificazione (`plan_curation`) dall'esecuzione (`execute_curation(dry_run=...)`).

---

## 2. Criticità e Difetti: Bug, Fragilità e Analisi delle Ultime 5 Run

Questa sezione elenca i problemi riscontrati, classificati per gravità e supportati dai riferimenti puntuali al codice e ai log delle ultime 5 run su Mufasa.

### 2.1. Bug Critici nel Valutatore (`ExtractionEvaluator`) che Falsano le Metriche della Tesi

#### A. Bug del Recall > 100% (`150.0%` nei report ufficiali)
Nelle ultime run compaiono valori di `Benchmark Recall` matematicamente impossibili (**`150.0%`**):
- In `20261027_200809_mufasa_mistral_7b_v0_3_gemini_40`: **`SAM3D`** riporta `6/4` dataset matchati (**`150.0%`** Recall).
- In `20261027_172306_mufasa_codestral_22b_gemini_40`: **`Cheap Lunch SAM`** riporta `3/2` dataset matchati (**`150.0%`** Recall).

**Causa radice nel codice ([`evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L79-L189)):**
1. Tra le righe [`81-101`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L81-L101), i punteggi estratti vengono raggruppati nel dizionario `extracted_by_dataset[canonical]`. Quando `self.alias_resolver.resolve(s.dataset_name)` restituisce `None` (perché il nome estratto contiene suffissi o varianti non presenti nel dizionario statico), la chiave diventa `s.dataset_name.strip()` (es. `"CHAOS CT"`, `"CHAOS MRI"`, `"AMOS22 (CT)"`, `"AMOS22 (MRI)"`).
2. Nel ciclo successivo ([`righe 111-179`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L111-L179)), il valutatore itera su `extracted_by_dataset.items()` e, se non trova un match esatto in `gt_map`, usa una regex substring (`re.search(pattern_gt, d_clean)` alle [`righe 120-124`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L120-L124)) per associare `d_name` a `gt_k`.
3. Di conseguenza, **più chiavi distinte di `extracted_by_dataset` fanno match con lo stesso identico dataset di Ground Truth `match_gt_key`**, e alla [`riga 172`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L172) `matched.append(match_gt_key)` inserisce duplicati nella lista `matched`!
4. Alla [`riga 183`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L183), `n_matched = len(matched)` conta i duplicati anziché i dataset unici (`len(set(matched))`), gonfiando sia il numeratore della `dataset_recall` di ogni singolo paper sia la `Overall Benchmark Recall` dell'intera run, e pesando più volte lo stesso dataset nel calcolo di `dice_mae`!

#### B. *Oracle Data Leakage* (Cherry-Picking guidato dalla Ground Truth) nel calcolo del MAE
In [`evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L140-L163), quando per uno stesso dataset vengono estratti più punteggi (`len(scores_list) > 1`), il valutatore calcola prima la media (`avg_dice`), ma poi esegue questo blocco ([`righe 152-156`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L152-L156)):
```python
best_single = min(
    candidate_pool,
    key=lambda s: abs(s.dice_score - true_dice) if s.dice_score is not None else 999.0
)
if best_single.dice_score is not None and abs(best_single.dice_score - true_dice) < 0.015:
    ext_dice = best_single.dice_score
```
**Perché è metodologicamente inaccettabile:**
Il valutatore **sbiricia il valore vero `true_dice` della Ground Truth** per scegliere, tra i vari numeri estratti dall'agente per quel dataset (es. 1-click, 5-click, Bounding Box, split 10%, split 100%, o persino un organo singolo il cui Dice cade casualmente entro $\pm 0.015$ dalla media macro!), quello più vicino alla Ground Truth!
- Se un estrattore riversa nell'output 5 configurazioni diverse senza sapere quale sia quella giusta, l'evaluatore seleziona a posteriori quella che minimizza l'errore rispetto a `true_dice`, abbassando artificialmente il `dice_mae`. In sede di revisione di tesi o peer-review, questo costituisce *test-set oracle leakage*.

#### C. *Survivorship Bias* tra Recall e MAE (Compromesso Precisione/Copertura nascosto)
In [`run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py#L728-L735) e in [`run_logger.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/run_logger.py#L510-L511), il `Mean Dice MAE` globale è calcolato come media non pesata dei soli paper che hanno almeno un dataset matchato (`mae_list = [p["dice_mae"] for p in paper_summaries if p.get("dice_mae") is not None]`).
- Se un modello su un paper difficile estrae `0/27` dataset (`SAT`) o `0/14` (`BiomedParse`), il suo MAE su quel paper è `None` (ignorato).
- Se invece su un altro paper estrae solo `1/13` dataset (`STU-Net`, il più facile) con errore `0.0000`, quel paper pesa nel `Mean Dice MAE` globale esattamente quanto un paper su cui sono stati estratti `13/13` dataset!
- Questo spiega il paradosso per cui una run che fallisce il richiamo sulla maggior parte dei benchmark complessi può esibire un `Mean Dice MAE` apparentemente eccellente.

---

### 2.2. Il Collo di Bottiglia Principale del Recall (~31%–41%): Troncamento Cieco a 12.000 Caratteri e Parsing PDF Piatto

Perché tutte le 5 configurazioni di modelli hanno una `Benchmark Recall` inchiodata tra il **31.7%** e il **41.5%** (circa 90–118 benchmark trovati su 284)? Non è colpa dei modelli LLM, ma di **due colli di bottiglia nel codice di pre-processing**:

#### A. Troncamento Hardcoded a `max_chars = 12000` in `ExtractionAgent._build_context_content`
Si osservi [`extractor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L36-L85):
```python
def _build_context_content(
    self, parsed_paper: ParsedPaper, max_chars: int = 12000
) -> list[Any]:
```
1. `max_chars` è fissato di default a **`12000` caratteri** (~2.800–3.000 token), e nelle chiamate in [`extract()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L140), [`reinterrogate_discrepancy()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L228), [`reinterrogate_omission()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L304) e [`reinterrogate_granularity_macro()`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L379) questo parametro non viene mai sovrascritto!
2. Come costruisce il testo `_build_context_content`?
   - Prima inserisce un **Abstract Preview** dei primi `3.000` caratteri ([`riga 55`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L55)).
   - Rimangono appena **`9.000` caratteri** (~2–3 pagine di un paper scientifico a doppia colonna) per **tutte** le `candidate_table_pages`!
   - Non appena `current_len + len(page_text) > max_chars` ([`riga 66`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L66)), il ciclo fa `break` e **scarta silenziosamente tutte le pagine successive del paper** (inclusi gli esperimenti principali a fine paper e l'Appendice)!
3. **La prova fumante nei log:** Nel report `20261027_172306_mufasa_codestral_22b_gemini_40` sul paper **`KnowSAM`** ([`benchmark_report.md#L150`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/logs/mufasa_runs/20261027_172306_mufasa_codestral_22b_gemini_40/benchmark_report.md#L150)), Gemini Flash scrive esplicitamente nel suo Chain-of-Thought:
   > *"Due to the preview text truncation before full numerical tables appear in the prompt, exact numerical values cannot be fabricated."*
4. Anche qualora si alzasse `max_chars` in `extractor.py`, in [`LLMClient._call_openai_compatible`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L229-L233) è presente un secondo taglio hardcoded a `max_Prompt_chars = 28000` (~7.000 token), nonostante gli script Mufasa avvino vLLM con `--max-model-len 16384` o `32768` e Gemini 2.5 Flash supporti `1.000.000` di token! Inoltre, in `_call_openai_compatible` il troncamento `full_user_text[:28000]` taglia la coda del messaggio utente, ovvero **proprio il blocco `TASK:` con le istruzioni finali** (che in [`extractor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L148-L162) viene appeso *dopo* `context_parts`)!

#### B. Assenza di Visione Multimodale e di Estrazione Strutturata delle Tabelle in `PDFParser`
- Nonostante `PROJECT_CONTEXT.md` parli di *"Vision-LLM / Table Parser"*, [`PDFParser`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/pdf_parser.py#L46-L154) usa esclusivamente `page.get_text("text")` di PyMuPDF ([`riga 71`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/pdf_parser.py#L71)) senza attivare né `page.find_tables()` (disponibile nativamente in PyMuPDF $\ge 1.23$ per esportare le tabelle in Markdown), né il rendering in immagini per i modelli Vision.
- Su tabelle medico-scientifiche complesse a doppia colonna con intestazioni gerarchiche (es. `SAT` con 27 dataset, `STU-Net` con 13 dataset, `TotalSegmentator` o `SAM-Med3D`), `page.get_text("text")` linearizza le celle fuori ordine, separando i nomi delle righe dai valori numerici.
- In aggiunta, l'euristica [`_is_candidate_table_page`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/pdf_parser.py#L112-L153) preserva in ordine cronologico le pagine che contengono la parola `"table"` o `"dice"` con almeno 5 decimali: le prime pagine del paper (Introduzione / Related Work / Ablation preliminare) consumano tutti i 9.000 caratteri disponibili in `_build_context_content`, impedendo alle vere tabelle comparative (spesso a pagina 6–15) di entrare nel prompt!

---

### 2.3. Bug Architetturali e Logici nei Singoli Moduli del Codice

#### 1. `DatabaseAgent` in modalità Single-Agent è un `pass` vuoto (`database_agent.py`)
In [`DatabaseAgent._extract_score_items`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py#L334-L363):
```python
if isinstance(audit_report, ConsensusAuditReport):
    ...
else:
    # Single-agent AuditReport: we must reconstruct score items from auditor context
    # Note: Single-agent AuditReport doesn't carry the raw scores directly unless attached,
    # so we inspect if consensus_scores exist or fallback to flags.
    pass
```
- Il ramo `else:` (Single-Agent `AuditReport`) contiene letteralmente **`pass`** e restituisce una lista vuota `[]`!
- Di conseguenza, contrariamente a quanto dichiarato in [`docs/DATABASE_MANAGER_DESIGN.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/docs/DATABASE_MANAGER_DESIGN.md#L64-L66), se si esegue la pipeline in modalità Single-Agent, `DatabaseAgent` **non inserisce mai alcuno score nel database**!

#### 2. `scripts/run_extraction.py` ha la condizione `if not is_dual:` invertita e crasha in Single-Agent
Si guardi [`scripts/run_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L213-L270):
- Alla [`riga 213`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L213) è scritto `if not is_dual:`: ma dentro quel blocco ([`righe 222-264`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L222-L264)) il codice accede a `audit_report.consensus_scores`, `model_b` ed `extracted_data_a`, variabili che esistono **solo quando `is_dual` è `True`**!
- Viceversa, nel ramo `else:` (eseguito quando `is_dual` è `True`, [`righe 265-270`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L265-L270)), il codice **sovrascrive** il `ConsensusAuditReport` appena calcolato invocando `audit_report = auditor.audit(extracted_data)` (l'audit single-agent solo sull'Agente A!), perdendo completamente il consenso dual-agent nella stampa e nella successiva chiamata a `DatabaseAgent`!
- Se un utente lancia `run_extraction.py --single-agent`, lo script crasha immediatamente alla riga 222 con `AttributeError: 'AuditReport' object has no attribute 'consensus_scores'`.

#### 3. `RepoInspector` e `OutreachAgent` sono Disinnescati in Modalità Dual-Agent (`auditor.py`)
Nelle ultime 5 run da 40 paper ciascuna (200 esecuzioni totali), **`RepoInspector` è stato attivato 0 volte (`0/200`)** e **`OutreachAgent` solo 1 volta (`1/200`, su `Merlin` nella run Codestral)**, anche quando su decine di paper (es. `BiomedParse`, `SAT`, `Vista3D`, `MedSAM-2`) entrambi gli agenti avevano estratto 0 benchmark validi o avevano segnalato l'assenza di tabelle!
**Perché?** Si veda [`AuditorAgent.audit_consensus`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L1043-L1050):
```python
has_repo_inspection = (
    report_a.status == AuditStatus.NEEDS_REPO_INSPECTION
    or report_b.status == AuditStatus.NEEDS_REPO_INSPECTION
) and total_unique_benchmarks == 0
```
1. La condizione `and total_unique_benchmarks == 0` blocca qualsiasi delega a `RepoInspector` o `OutreachAgent` non appena anche **uno solo** dei due agenti estrae un singolo oggetto `ExtractedBenchmarkScore` (anche se `SINGLE_AGENT_UNCONFIRMED`, anche se con `dice_score = None`, o anche se contiene solo una nota testuale)!
2. Inoltre, in [`_determine_status`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L714-L743), `NEEDS_REPO_INSPECTION` scatta solo se `extracted_scores` è vuoto **e** `github_url` è presente. Ma molti paper (come `BiomedParse` nelle run Qwen+Mistral o Mistral+Gemini) contengono nel testo una tabella parziale di un solo dataset secondario o un link HuggingFace/Project Page che il modello non mette in `github_url`, oppure uno dei due modelli allucina un punteggio non confermato (`total_unique_benchmarks = 1`): in tutti questi casi `has_repo_inspection` diventa `False` e il paper finisce in `FLAGGED_WARNING` o `REJECTED` senza mai ispezionare il repository GitHub né generare la mail di Outreach!

#### 4. Inquinamento Massivo dello Schema SQLite (`ALTER TABLE ADD COLUMN`) e Frammentazione delle Entità in `DatabaseAgent`
I dati di curatela delle ultime 5 run mostrano un difetto strutturale grave in [`DatabaseAgent`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py):
- **Esplosione di colonne spazzatura (da +59 a +143 nuove colonne SQLite per run!):**
  In [`_is_private_cohort`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py#L275-L291), qualunque stringa `dataset_name` che non faccia match esatto con [`DatasetAliasResolver`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/alias_resolver.py) e non contenga una delle 9 parole chiave (`internal`, `in-house`, `private`, `proprietary`, `clinical cohort`, `institution`, `hospital`, `local dataset`, `collected at`) viene promossa a **"Nuovo Benchmark Pubblico"** con emissione di `ALTER TABLE ... ADD COLUMN` su tutte le 5 tabelle del database!
  Nei log reali delle 5 run, `DatabaseAgent` ha proposto di aggiungere colonne come:
  - `'Dataset 1'`, `'Dataset 2'`, `'Dataset 3'` (estratti da Codestral su `KnowSAM` in `20261027_172306`!)
  - Singoli organi o strutture anatomiche trattati come dataset: `'Liver'`, `'Spleen'`, `'Right Kidney'`, `'Left Kidney'`, `'Aorta'`, `'Pancreas'`, `'Gallbladder'`, `'Stomach'`, `'Whole Heart'`
  - Sequenze o modalità generiche: `'T1'`, `'T2'`, `'FLAIR'`, `'CT'`, `'MRI'`, `'CT Test Set'`, `'Abdominal CT scans'`, `' Chest X-ray'`
  - Varianti sintattiche di dataset già esistenti a DB: `'BTCV (Synapse)'`, `'MSD (Medical Segmentation Decathlon)'`, `'BraTS 2021'` vs `'BraTS21'` vs `'BraTS2023'`, `'AMOS22 CT'` vs `'AMOS-CT'`.
- **Fallimento della Risoluzione d'Identità dei Modelli (`proposed_model_id`):**
  In [`_resolve_existing_model_id`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py#L293-L332), se l'LLM restituisce come `proposed_model_id` una variante del titolo o un segnaposto, `DatabaseAgent` non conosce l'`expected_model_id` del paper che sta processando e crea una **nuova riga duplicata** nella tabella `models`! Esempi reali dalle ultime 5 run:
  - Nella run `20261027_172306` (`Codestral + Gemini`), per il paper `Cheap Lunch SAM` il modello ha estratto `proposed_model_id = "Ours"`, e `DatabaseAgent` ha generato **`INSERT OR IGNORE INTO "models" ("model_id", ...) VALUES ('Ours', ...)`**!
  - Lo stesso modello già presente nel DB di Moglia et al. viene re-inserito come "nuovo modello" perché estratto con ligature Unicode (`'Diﬀ-UNet'` con il carattere `U+FB00` `ﬀ` invece di `'Diff-UNet'`), o col titolo del paper (`'Large-Vocabulary Segmentation for Medical Images with Text Prompts'` invece di `'SAT'`, `'TotalSegmentator MRI: Robust Sequence-'` invece di `'TotalSegmentator MRI'`, `'One-Prompt Segmentation'` invece di `'One-Prompt'`, `'STU-Net-B'` / `'STU-Net-H'` invece di `'STU-Net'`).
- **Sovrascrittura multipla della stessa cella e vulnerabilità SQL (`database_agent.py`):**
  - Quando un paper presenta più `ConsensusScore` per lo stesso dataset (es. i singoli organi di un dataset non filtrati, oppure diversi regimi di prompt/data budget), il ciclo in [`plan_curation`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py#L121-L214) emette istruzioni `UPDATE` multiple sulla stessa cella `(model_id, canonical_name)` senza prima selezionare il valore macro (`Overall`) o il picco assoluto, sovrascrivendo i commenti in `results_best_comments`.
  - La costruzione delle query SQL avviene per interpolazione manuale di stringhe ([`_escape_sql`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py#L666-L669)), che effettua l'escape solo degli apici singoli (`'`) ma **non** dei doppi apici (`"`) usati per gli identificatori di colonna `ALTER TABLE "{table}" ADD COLUMN "{canonical_name}"`.

#### 5. Fragilità nella Deliberazione Multi-Round (`dual_coordinator.py`) e nella Decodifica JSON vLLM (`client.py`)
1. **Bug di collisione in `_find_matching_score_index` ([`dual_coordinator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L22-L56)):**
   Quando un agente aggiorna il proprio punteggio durante la deliberazione ([`riga 343`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L343)), `_find_matching_score_index` cerca l'elemento da sovrascrivere confrontando solo `(dataset_name, organ_or_task)` e **ignorando `data_budget_or_setting` e `prompt_type`**! Se il modello ha estratto per lo stesso dataset sia lo split `10%` sia lo split `100%` (come in `GVSL` o `C2FNAS`), la deliberazione sovrascrive la riga sbagliata!
2. **Assenza di *Guided JSON Schema Decoding* su vLLM ([`client.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L204-L220)):**
   In `_call_openai_compatible`, viene passato `response_format={"type": "json_object"}` anziché `response_format={"type": "json_schema", "json_schema": ...}` (o `extra_body={"guided_json": response_schema.model_json_schema()}` supportato nativamente da vLLM). Con `json_object`, vLLM garantisce solo che l'output sia un JSON sintatticamente valido, ma **non** che rispetti i tipi e i campi obbligatori di Pydantic!
3. **Crash catastrofico dell'intera coppia Dual-Agent per un singolo elemento invalido (`dual_coordinator.py`):**
   Nelle run con `Mistral-7B-Instruct-v0.3-AWQ` (`20261027_200809` e `20261027_212758`), i 4 crash registrati (`MultiTalent`, `UNETR++`, `HERMES`, `GVSL`) sono tutti causati dal fatto che Mistral ha emesso `is_baseline_comparison: null` (o `'Not applicable (proposed model)'`) oppure ha lasciato incompleto l'ultimo elemento dell'array `extracted_scores`.
   - Poiché [`ExtractedBenchmarkScore.is_baseline_comparison`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py#L64-L67) non ha un validator Pydantic che converta `None`/stringhe in `False`, l'intera validazione di `ExtractedPaperData` fallisce.
   - E poiché [`DualExtractionCoordinator.extract_and_audit`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L459-L471) lancia un `RuntimeError` se anche uno solo dei due agenti fallisce, **viene buttato via anche l'output perfettamente valido dell'altro agente** (es. `Qwen-2.5-7B` o `Gemini 2.5 Flash`)!

#### 6. Packaging, Sicurezza e Configurazione (`pyproject.toml`, `repo_inspector.py`, `outreach.py`)
- **Dipendenze mancanti in [`pyproject.toml`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/pyproject.toml#L9-L16):** `google-genai` (usato in [`client.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L69)) e `certifi` (usato in [`paper_fetcher.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/paper_fetcher.py#L14)) non sono dichiarati nelle `dependencies` del pacchetto.
- **Disabilitazione della verifica SSL in [`repo_inspector.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/repo_inspector.py#L86):** `self._ssl_ctx = ssl._create_unverified_context()` disabilita la verifica dei certificati TLS verso le API GitHub (mentre `paper_fetcher.py` usa correttamente `certifi.where()`).
- **Email mittente incoerente in [`outreach.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/outreach.py#L360):** Nel template di Outreach è cablato `luca.pezzoli@polimi.it`, mentre in [`paper_fetcher.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/paper_fetcher.py#L33) l'indirizzo istituzionale corretto è `luca2.pezzoli@mail.polimi.it`.

---

### 2.4. Disamina Critica delle Ultime 5 Run (`logs/mufasa_runs/`) e Contraddizioni con la Documentazione

Sono state analizzate nel dettaglio le ultime 5 run complete sui 40 paper del benchmark su cluster HPC Mufasa:
1. **`20261026_181522_mufasa_qwen14b_gemini_40`** (`Qwen2.5-14B-Instruct-AWQ` + `Gemini 2.5 Flash`)
2. **`20261026_200829_mufasa_dual_local_40`** (`Qwen2.5-7B-Instruct-AWQ` + `Llama-3.1-8B-Instruct-AWQ`)
3. **`20261027_172306_mufasa_codestral_22b_gemini_40`** (`Codestral-22B-v0.1-AWQ` + `Gemini 2.5 Flash`)
4. **`20261027_200809_mufasa_mistral_7b_v0_3_gemini_40`** (`Mistral-7B-Instruct-v0.3-AWQ` + `Gemini 2.5 Flash`)
5. **`20261027_212758_mufasa_dual_local_Qwen257B_Mistral7B_40`** (`Qwen2.5-7B-Instruct-AWQ` + `Mistral-7B-Instruct-v0.3-AWQ`)

*(Nota: la directory `20261027_171949_mufasa_codestral_22b_gemini_40` è un tentativo abortito dopo 3 paper in cui Codestral era fallito al 100% per errore di avvio vLLM, subito sostituito dalla run `172306`).*

#### Tabella Comparativa Reale (Estratta dai `benchmark_report.md` delle 5 Run)

| Run ID | Architettura (Agent A + Agent B) | Verified | Flagged | Rejected | Hard Errors (`❌`) | Outreach / RepoInsp | Recall (`Matched/GT`) | Mean Dice MAE* | Nuove Colonne SQLite Proposte |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`20261026_181522`** | `Qwen-14B-AWQ` + `Gemini Flash` | 26 | 14 | 0 | **0** | 0 / 0 | 38.4% (`109/284`) | 0.0365 | +108 colonne |
| **`20261026_200829`** | `Qwen-7B-AWQ` + `Llama-3.1-8B` | 27 | 13 | 0 | **0** | 0 / 0 | 33.8% (`96/284`) | 0.0226 | +59 colonne |
| **`20261027_172306`** | `Codestral-22B` + `Gemini Flash` | 30 | 9 | 0 | **0** | 1 / 0 | **41.5%** (`118/284`)* | 0.0376 | +143 colonne |
| **`20261027_200809`** | `Mistral-7B` + `Gemini Flash` | 29 | 9 | 0 | **2** (`HERMES`, `GVSL`) | 0 / 0 | 39.2% (`107/273`)* | 0.0439 | +133 colonne |
| **`20261027_212758`** | `Qwen-7B-AWQ` + `Mistral-7B` | 24 | 12 | 2 | **2** (`MultiTalent`, `UNETR++`) | 0 / 0 | 31.7% (`84/265`) | **0.0170** | +73 colonne |

*\*Nota: i valori di Recall e MAE nelle colonne sopra includono gli effetti distorsivi dei bug descritti nella Sezione 2.1 (es. i falsi `150%` di Recall su `Cheap Lunch SAM` e `SAM3D` e il denominatore ridotto da `284` a `265`/`273` nelle ultime due run perché i paper andati in `❌ ERROR` sottraggono i propri dataset di Ground Truth dal denominatore `total_gt`!).*

#### Rettifiche Critiche rispetto a quanto scritto in `THESIS_DESIGN_LOG.md` (§33)

Nella Sezione 33 di [`docs/THESIS_DESIGN_LOG.md`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/docs/THESIS_DESIGN_LOG.md#L706-L763), l'analisi comparativa contiene affermazioni che **non corrispondono ai log reali** e che esporrebbero la tesi a contestazioni immediate:

1. **Falsa assenza di crash nella Run 5 (`Qwen-7B + Mistral-7B`, Job 5518):**
   - *Cosa dice il Design Log:* che la run `Qwen-7B + Mistral-7B` ha dimostrato stabilità superiore completando in ~35 minuti senza le penalità delle altre run.
   - *Cosa dicono i log reali:* le prime 3 run (`Qwen-14B + Gemini`, `Qwen-7B + Llama-8B`, `Codestral-22B + Gemini`) hanno avuto **0 errori runtime (`40/40` paper completati)**, mentre le uniche due run con crash catastrofici (`2/40` paper falliti con `❌ ERROR`) sono proprio le due che utilizzano `Mistral-7B-Instruct-v0.3-AWQ` (`20261027_200809` e `20261027_212758`). Inoltre, i 2 crash nella run `Qwen-7B + Mistral-7B` hanno rimosso `MultiTalent` (13 benchmark GT) e `UNETR++` (6 benchmark GT) dal denominatore del Recall (`265` invece di `284`).
2. **Perché il MAE `0.0170` di `Qwen-7B + Mistral-7B` è un artefatto di *Survivorship Bias* e Asimmetria "Primary vs Best":**
   - Nella run `Qwen-7B + Mistral-7B`, ben **11 paper su 40** hanno `0` dataset matchati (`Recall = 0.0%`, `MAE = N/A`: `SAT 0/27`, `BiomedParse 0/14`, `MedSAM-2 0/15`, `Vista3D 0/4`, `SegVol 0/3`, `SAM-Med3D 0/1`, `KnowSAM 0/1`, `Merlin 0/1`, `GVSL 0/3`, più i 2 in crash).
   - Perché le 3 run con `Gemini 2.5 Flash` hanno un MAE leggermente più alto (`0.0365` – `0.0439`) ma trovano fino a **34 benchmark in più** (`118` vs `84`)?
     Perché [`prompts.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/prompts.py#L42) ordina esplicitamente di estrarre il **Peak Prompt Regime (Bounding Box)**, mentre [`run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py#L222-L227) valuta di default contro `--compare-source primary` (`results_primary` di Moglia et al., che registra la baseline conservativa 1-Click)!
     Ad esempio, su **`MedSAM`** nella run `Codestral + Gemini`, gli agenti estraggono correttamente il picco Bounding Box su `AMOS22` (`0.8830`), ma il valutatore lo confronta con il valore `primary` (`0.6560`, 1-click), generando un "errore" artificiale di `0.2270` che alza il MAE medio della run! Quando invece un modello locale omette `MedSAM` o estrae casualmente la prima riga (1-click), il suo MAE contro `primary` sembra migliore.
3. **Il `REJECTED` su `Merlin` e `BiomedParse` nella Run 5 NON è un successo di "Epistemic Gatekeeping", ma un mancato innesco dei moduli di Fallback:**
   - *Cosa dice il Design Log (§33.3):* elogia la coppia `Qwen-7B + Mistral-7B` per aver classificato `Merlin` e `BiomedParse` come `❌ REJECTED`.
   - *Perché è una contraddizione architetturale:* nella filosofia stessa della tesi (e come documentato nelle sezioni precedenti del `THESIS_DESIGN_LOG.md`), **`BiomedParse`** (che nel PDF ha solo metriche aggregate in un grafico radar ma pubblica i CSV completi dei 14 dataset su GitHub) è il caso d'uso principe per attivare **`NEEDS_REPO_INSPECTION -> RepoInspector`**! Analogamente, **`Merlin`** (che riporta su `TotalSegmentator` solo F1-score/AUC e non il Dice) è il caso d'uso principe per attivare **`NEEDS_OUTREACH -> OutreachAgent`**!
   - Classificarli come `REJECTED` significa abortire la pipeline anziché delegare al `RepoInspector` o all'`OutreachAgent`. E infatti nella run `Codestral + Gemini` (`20261027_172306`), `Merlin` è stato correttamente indirizzato a `📧 NEEDS_OUTREACH` con generazione del file `.eml`!

---

### 2.5. Lacune nella Suite di Test (`tests/`)

La cartella [`tests/`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests) contiene 8 file di test ben strutturati per i casi nominali, ma presenta tre punti ciechi che hanno permesso ai bug sopra descritti di arrivare in produzione:
1. **Assenza totale di unit test per [`ExtractionEvaluator`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py):** non esiste un `test_evaluator.py`. Per questo motivo il bug del Recall al `150%` (duplicati in `matched`) e l'*oracle cherry-picking* non sono mai stati intercettati.
2. **[`test_database_agent.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_database_agent.py) testa solo `ConsensusAuditReport`:** non esiste alcun test che passi un `AuditReport` single-agent a `DatabaseAgent.plan_curation()`, lasciando inosservato il blocco `else: pass` alla riga 358 di `database_agent.py`.
3. **Nessun test sul delegating di `RepoInspector` / `OutreachAgent` in modalità Dual-Agent quando uno dei due estrattori produce rumore parziale (`total_unique_benchmarks > 0`):** i test in [`test_consensus_auditing.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_consensus_auditing.py) verificano il fallback solo quando entrambi gli agenti restituiscono liste `extracted_scores` completamente vuote (`[]`).

---

## 3. Cambiamenti Suggeriti: Piano d'Azione Prioritizzato

Di seguito vengono elencati gli interventi tecnici e metodologici raccomandati, suddivisi per priorità, pensati per essere eseguiti in modo modulare da un agente di sviluppo.

### Priorità P0 — Correzioni Bloccanti (Metriche di Valutazione e Bug Silenziosi) `[COMPLETATO E VERIFICATO]`

1. **Eliminare i duplicati nel calcolo del Recall e rimuovere l'*Oracle Data Leakage* in [`evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L79-L189):** `[RISOLTO - §4.1.1]`
   - Invertire il ciclo di matching in `ExtractionEvaluator.evaluate()`: iterare sui dataset di Ground Truth (`for gt_key, true_dice in gt_map.items():`) anziché sulle chiavi grezze di `extracted_by_dataset`, oppure deduplicare i match raggruppando tutte le chiavi estratte che puntano allo stesso `match_gt_key` **prima** di calcolare `matched` e `diffs`. Questo garantisce matematicamente $\text{Recall} \le 100\%$.
   - **Rimuovere immediatamente le righe [`152-156`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L152-L156)** (`best_single = min(..., abs(s.dice_score - true_dice))`). Quando per uno stesso dataset sono presenti più punteggi estratti, la selezione del punteggio da confrontare con la Ground Truth deve essere **cieca rispetto a `true_dice`** e basarsi esclusivamente su regole intrinseche:
     1. Preferire i punteggi con stato `AGREEMENT` rispetto a `SINGLE_AGENT_UNCONFIRMED`.
     2. Se esiste una riga con `organ_or_task` equivalente a `"Overall"` / `"Average"` / `"Mean"`, selezionare quella (o il suo massimo se ci sono più prompt regime e si valuta contro `best`, oppure il regime 1-click/automatico se si valuta contro `primary`).
     3. Altrimenti, calcolare la media macro sui diversi organi estratti per quel dataset.
2. **Doppia Valutazione Esplicita (`Primary` vs `Best`) e Metriche Globali Pesate nei Report:** `[RISOLTO - §4.1.2]`
   - Modificare [`ExtractionEvaluator`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py) e [`run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py) affinché ogni run calcoli in automatico **sia** il MAE contro `results_primary` **sia** il MAE contro `results_best` (oltre a un `Minimum Regime-Agnostic MAE` dichiarato trasparentemente come metrica ausiliaria quando la survey ha mescolato i regimi).
   - Nel `run_summary.json` e in `benchmark_report.md`, affiancare al `Mean Paper MAE` il **`Micro-Averaged Benchmark MAE`** (pesato sul numero di benchmark matchati) e il **`Benchmark F1 / Coverage-Adjusted Score`**, mantenendo fisso il denominatore del Recall a tutti i `284` benchmark della Ground Truth anche in presenza di paper andati in `ERROR`.
3. **Correggere l'inversione booleana `if not is_dual:` in [`scripts/run_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L213-L270) e il ramo vuoto in [`database_agent.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py#L358-L361):** `[RISOLTO - §4.1.3]`
   - In `scripts/run_extraction.py`, cambiare `if not is_dual:` alla riga 213 in `if is_dual:` e rimuovere la chiamata ridondante `audit_report = auditor.audit(extracted_data)` che sovrascrive il `ConsensusAuditReport`.
   - In `AuditReport` ([`auditing/schemas.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/schemas.py#L51-L97)), allegare la lista dei punteggi validati (`validated_scores: list[ExtractedBenchmarkScore]`) in modo che `DatabaseAgent._extract_score_items()` possa processare anche i report Single-Agent.

---

### Priorità P1 — Sblocco del Recall (Contesto PDF, Tabelle Strutturate e Fallback) `[COMPLETATO E VERIFICATO]`

4. **Rimuovere il collo di bottiglia `max_chars = 12000` in [`extractor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L36) e il taglio della coda in [`client.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L229-L233):** `[RISOLTO - §4.2.4]`
   - Rendere `max_chars` configurabile dinamicamente in base al backend: almeno **`60.000` – `80.000` caratteri** per Gemini 2.5 Flash e **`35.000` – `45.000` caratteri** per i modelli vLLM serviti con `--max-model-len 16384` / `32768`.
   - In `_build_context_content`, ordinare le `candidate_table_pages` per **densità di numeri decimali e presenza di nomi di benchmark noti** (usando `DatasetAliasResolver`) anziché per semplice ordine di pagina, così che le tabelle principali dei risultati (spesso a metà/fine paper) abbiano priorità assoluta rispetto all'Introduzione.
   - In `LLMClient._call_openai_compatible`, se è necessario troncare `full_user_text`, **troncare solo la sezione centrale del testo del paper preservando sempre intatto il blocco finale `TASK:`**.
5. **Arricchire [`PDFParser`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/pdf_parser.py) con l'estrazione tabellare Markdown di PyMuPDF (`page.find_tables()`):** `[RISOLTO - §4.2.5]`
   - Nelle pagine candidate, invocare `tabs = page.find_tables()` di PyMuPDF e appendere la rappresentazione `.to_markdown()` di ciascuna tabella insieme al testo della pagina. Questo preserva l'allineamento riga-colonna nelle tabelle multi-dataset (es. `SAT`, `STU-Net`, `UniSeg`) a costo computazionale quasi nullo.
6. **Ripristinare l'innesco di `RepoInspector` e `OutreachAgent` in modalità Dual-Agent ([`auditor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L1043-L1055)):** `[RISOLTO - §4.2.6]`
   - Sostituire la condizione troppo restrittiva `and total_unique_benchmarks == 0` con `and confirmed_agreements == 0`: se due agenti non raggiungono **nessun accordo confermato** su metriche Dice valide e almeno uno dei due segnala `NEEDS_REPO_INSPECTION` (o il paper ha un `github_url` noto anche tramite i metadati del DB di Moglia et al.), il coordinatore deve attivare `RepoInspector`!
   - Passare a `RepoInspector` anche il `code_link` già noto nel database (`models.code_link`) come fallback se gli estrattori non trovano l'URL GitHub nelle prime pagine del PDF.

---

### Priorità P2 — Igiene del Database (`DatabaseAgent`) e Risoluzione delle Entità `[COMPLETATO E VERIFICATO]`

7. **Fermare l'inquinamento dello schema SQLite (`ALTER TABLE ADD COLUMN`) in [`DatabaseAgent`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py):** `[RISOLTO - §4.3.7]`
   - **Regola di promozione dei nuovi dataset:** Non eseguire mai `ALTER TABLE ... ADD COLUMN` per punteggi con stato `SINGLE_AGENT_UNCONFIRMED`, né per stringhe che corrispondono a nomi di organi anatomici (es. `Liver`, `Spleen`, `Aorta`, `Pancreas`), modalità generiche (`CT`, `MRI`, `T1`, `T2`, `FLAIR`), o placeholder (`Dataset 1`, `Test Set`). Un nuovo dataset può generare una migrazione di schema (`ADD_DATASET_COLUMN`) **solo se** è confermato in `AGREEMENT` da entrambi gli agenti e supera una whitelist/euristica di validazione per i nomi di benchmark pubblici.
   - **Normalizzazione Unicode e Passaggio di `expected_model_id`:**
     - Applicare `unicodedata.normalize("NFKC", text)` su tutti i `proposed_model_id` e `dataset_name` (risolve istantaneamente il caso `'Diﬀ-UNet'` -> `'Diff-UNet'`).
     - Bloccare identificatori generici come `'Ours'`, `'Proposed'`, `'Baseline'` in `_resolve_existing_model_id`.
     - Quando si aggiorna o si valuta un paper già censito o tracciato dalla pipeline (`run_batch_benchmark.py`), passare il `model_id` noto o risolvere per match sul titolo del paper (`models.paper`) prima di emettere un `INSERT INTO models`.
   - **Deduplicazione pre-UPDATE per `(model_id, canonical_name)`:**
     - Se un `ConsensusAuditReport` contiene più voci che puntano allo stesso `canonical_name` (es. più organi o più split), aggregarle o selezionare la riga `Overall` *prima* di generare gli statement SQL in `plan_curation()`, emettendo al massimo un singolo `UPDATE` per coppia `(model_id, dataset)`.

---

### Priorità P3 — Robustezza Dual-Agent, Guided Decoding, Sicurezza e Test `[COMPLETATO E VERIFICATO]`

8. **Attivare il *Guided JSON Schema Decoding* su vLLM e la tolleranza ai guasti parziali:** `[RISOLTO - §4.4.8]`
   - In [`LLMClient._call_openai_compatible`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L209), utilizzare `extra_body={"guided_json": response_schema.model_json_schema()}` (o `response_format={"type": "json_schema", ...}`) quando si interroga un server vLLM, eliminando alla radice gli errori di validazione Pydantic di `Mistral-7B`.
   - In [`ExtractedBenchmarkScore`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py#L64-L67), aggiungere un `@field_validator("is_baseline_comparison", mode="before")` che converta `None` o stringhe non booleane in `False`, e un `@field_validator("extracted_scores", mode="before")` in `ExtractedPaperData` che scarti eventuali dizionari troncati privi di `dataset_name` anziché far fallire l'intero paper.
   - In [`DualExtractionCoordinator.extract_and_audit`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L459-L471), se dopo i retry uno dei due agenti fallisce ma l'altro ha restituito un `ExtractedPaperData` valido, **non sollevare `RuntimeError`**: degradare con grazia producendo un `ConsensusAuditReport` con i punteggi dell'agente superstite marcati come `SINGLE_AGENT_UNCONFIRMED`.
   - In [`_find_matching_score_index`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L22-L56), includere `normalize_data_budget(s.data_budget_or_setting)` nel criterio di matching per evitare collisioni durante i round di deliberazione.
9. **Allineare Dipendenze, SSL e Suite di Test:** `[RISOLTO - §4.4.9]`
   - Aggiungere `google-genai>=1.0.0` e `certifi>=2024.0.0` in [`pyproject.toml`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/pyproject.toml).
   - Sostituire `ssl._create_unverified_context()` con `ssl.create_default_context(cafile=certifi.where())` in [`repo_inspector.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/repo_inspector.py#L86).
   - Uniformare l'indirizzo email in [`outreach.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/outreach.py#L360) a `luca2.pezzoli@mail.polimi.it`.
   - Creare `tests/test_evaluator.py` per testare l'invariante $\text{Recall} \le 1.0$ e l'assenza di data leakage, ed estendere `tests/test_database_agent.py` al caso Single-Agent.

---

## 4. Registro delle Risoluzioni e Modifiche Applicate (Changelog di Revisione)

Questo registro documenta in modo esaustivo le correzioni architetturali, algoritmiche e di refactoring implementate in risposta ai rilievi dell'agente revisore. Tutte le modifiche sono state coperte da test automatizzati (suite pytest completa: **109 test passati con successo**).

### 4.1. Priorità P0 — Correzioni Bloccanti su Valutazione ed Esecuzione

#### P0.1: Eliminazione duplicati nel Recall e rimozione dell'*Oracle Data Leakage* nel MAE
- **File modificati:** [`src/agent_gmmis/extraction/evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py), [`tests/test_evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_evaluator.py).
- **Risoluzione algoritmica:**
  1. *Inversione del Bipartite Matching:* La funzione `evaluate()` ora raggruppa tutti i punteggi estratti associati a ciascun benchmark canonico di Ground Truth mediante l'helper `_match_dataset_to_gt()`. Ogni chiave di GT viene matchata al massimo una volta (`gt_matched_scores`), azzerando le collisioni multi-alias che causavano Recall fino al 150%. È ora garantito che $\text{Recall} \le 1.0$ e $\text{Precision} \le 1.0$.
  2. *Rimozione dell'Oracle Leakage:* Eliminato il blocco di cherry-picking guidato dalla Ground Truth (`min(..., abs(s.dice_score - true_dice) < 0.015)`). La selezione del candidato (`_select_candidate_score()`) è ora rigorosamente **cieca rispetto a `true_dice`**:
     - Preferisce punteggi in `CONCORDANT` / `AGREEMENT`.
     - Ricerca la riga di sintesi macro (`"Overall"`, `"Average"`, `"Mean"`), selezionando il regime di picco se si valuta contro `best` o il regime standard/1-click se si valuta contro `primary`.
     - Se esistono solo punteggi disaggregati per organo (senza riga Overall), calcola deterministicamente la **media aritmetica macro sui singoli organi** senza sbirciare la GT.
- **Verifica con Test:** Creato `tests/test_evaluator.py` con 4 test dedicati:
  - `test_recall_never_exceeds_100_percent`: verifica che estrazioni con alias multipli dello stesso dataset abbiano Recall $\le 100\%$.
  - `test_evaluator_blind_selection_no_oracle_leakage`: verifica che tra più prompt regime venga scelto il candidato in base alle regole cieche e non per vicinanza al true Dice.
  - `test_macro_average_computed_blindly_when_no_overall_row`: verifica la sintesi macro su organi disaggregati in assenza della riga complessiva.
  - `test_dual_regime_ground_truth_map`: verifica l'estrazione coerente delle mappe GT per regime `primary` e `best`.

#### P0.2: Doppia Valutazione Esplicita (`Primary` vs `Best`) e Metriche Globali Pesate
- **File modificati:** [`src/agent_gmmis/extraction/evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py), [`scripts/run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py), [`src/agent_gmmis/extraction/run_logger.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/run_logger.py).
- **Risoluzione metodologica:**
  1. *Schema di valutazione duale:* [`ExtractionEvaluationReport`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py) calcola simultaneamente:
     - `dice_mae_primary`: errore rispetto a `results_primary` (regime conservativo 1-click / standard).
     - `dice_mae_best`: errore rispetto a `results_best` (regime di picco / bounding box).
     - `dice_mae_regime_agnostic`: metrica ausiliaria trasparente che minimizza la distanza dai due regimi ufficiali di Moglia et al.
  2. *Correzione del Survivorship Bias e del Denominatore:*
     - Il denominatore globale del Recall è ora fissato a tutti i 284 benchmark attivi (`total_gt_active_all`), impedendo che i paper falliti o andati in errore riducano artificiosamente il denominatore.
     - Calcolo del **`Micro-Averaged Benchmark MAE`** ($\sum |\text{pred} - \text{true}| / N_{\text{matched}}$) affiancato al Macro-MAE sui paper.
     - Aggiunto il **`Coverage-Adjusted Score`** ($\text{Recall} \times (1.0 - \text{Micro\_MAE})$) per penalizzare sia la bassa copertura sia l'errore numerico.
     - Aggiunta riga footer di aggregazione batch nella tabella di riepilogo in `benchmark_report.md` e persistenza di tutte le metriche pesate in `benchmark_results.json`.

#### P0.3: Correzione dell'Inversione Booleana `run_extraction.py` e Curation Single-Agent
- **File modificati:** [`scripts/run_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py), [`src/agent_gmmis/auditing/schemas.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/schemas.py), [`src/agent_gmmis/auditing/auditor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py), [`src/agent_gmmis/database/database_agent.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py), [`tests/test_database_agent.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_database_agent.py).
- **Risoluzione:**
  1. Invertita la condizione errata `if not is_dual:` alla riga 213 in `if is_dual:` e rimosso il ramo else distruttivo che sovrascriveva il `ConsensusAuditReport` chiamando `auditor.audit(extracted_data)`.
  2. Aggiunto il campo `validated_scores: List[RawExtractedScore]` a `AuditReport`, popolato da `AuditorAgent.audit()`.
  3. Sostituito il `pass` vuoto in `DatabaseAgent._extract_score_items()` per estrarre e curare i punteggi con status `single_agent_unconfirmed`.
  4. Testato con `test_single_agent_audit_report_curation` che verifica la transazione SQLite end-to-end su `AuditReport`.

---

### 4.2. Priorità P1 — Sblocco del Recall (Contesto PDF, Tabelle e Fallback)

#### P1.4: Rimozione del Collo di Bottiglia a 12.000 Caratteri e Prioritizzazione Pagine
- **File modificati:** [`src/agent_gmmis/extraction/extractor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py), [`tests/test_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_extraction.py).
- **Risoluzione:**
  1. `max_chars` è ora dinamico tramite helper `get_default_context_chars()`: 80.000 caratteri per Gemini 2.5 Flash, 42.000 per vLLM locale (superando il tetto hardcoded di 12.000 caratteri).
  2. Implementato `_score_page()` in `ExtractionAgent`: le pagine candidate non vengono più inserite in ordine cronologico lineare (che consumava tutto il budget sull'Introduzione), ma classificate per densità di valori decimali, parole chiave di segmentazione e presenza di tabelle Markdown.
  3. L'assemblaggio finale preserva l'ordine naturale di lettura delle sole pagine ad alto contenuto informativo.
  4. Coperto dal test `test_dynamic_context_budget_and_table_prioritization`.

#### P1.5: Estrazione Tabellare Strutturata Markdown con PyMuPDF
- **File modificati:** [`src/agent_gmmis/extraction/pdf_parser.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/pdf_parser.py).
- **Risoluzione:**
  1. Integrata l'API nativa `page.find_tables()` di PyMuPDF. Per ogni tabella rilevata nella pagina, viene generato il Markdown strutturato tramite `.to_markdown()` e appeso in testa al testo della pagina.
  2. Le pagine contenenti tabelle strutturate ricevono un boost prioritario (+25 punti) nell'indice di rilevanza per il context window dell'estrattore.

#### P1.6: Ripristino dei Fallback `RepoInspector` e `OutreachAgent` in Dual-Agent
- **File modificati:** [`src/agent_gmmis/auditing/auditor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py), [`scripts/run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py), [`scripts/run_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py), [`tests/test_consensus_auditing.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_consensus_auditing.py).
- **Risoluzione:**
  1. In `AuditorAgent.audit_consensus()`, la clausola bloccante `total_unique_benchmarks == 0` è stata sostituita da `concordant_count == 0 and paired_count == 0`. Se due agenti non hanno alcun accordo concordante né benchmark confrontabili e almeno uno segnala dati mancanti, il coordinatore non abortisce più ma attiva correttamente `RepoInspector` o `OutreachAgent`.
  2. Aggiunto lookup di fallback nel database SQLite (`db.get_model(model_id).code`) per recuperare i link ai repository GitHub censiti qualora non vengano estratti direttamente dal PDF.
  3. Aggiunti unit test specifici per il fallback dual-agent con rumore parziale (`test_audit_consensus_fallback_repo_inspection_with_unconfirmed_noise` e `test_audit_consensus_fallback_outreach_with_unconfirmed_noise`).

---

### 4.3. Priorità P2 — Igiene del Database (`DatabaseAgent`) e Risoluzione delle Entità

#### P2.7: Blocco dell'Inquinamento dello Schema SQLite, Deduplicazione ed Entity Resolution
- **File modificati:** [`src/agent_gmmis/database/database_agent.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/database/database_agent.py), [`scripts/run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py), [`scripts/run_database_curation.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_database_curation.py), [`tests/test_database_agent.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_database_agent.py).
- **Risoluzione tecnica dettagliata:**
  1. *Schema Guard contro l'esplosione di colonne:*
     - Aggiunta blacklist di organi anatomici (`ORGAN_BLACKLIST`: 40+ strutture anatomiche tra cui `Liver`, `Spleen`, `Aorta`, `Pancreas`, `Kidney`, ecc.).
     - Aggiunta blacklist di modalità radiologiche generiche (`MODALITY_BLACKLIST`: `CT`, `MRI`, `FLAIR`, `T1`, `T2`, `X-ray`, ecc.).
     - Aggiunta blacklist di segnaposto generici (`GENERIC_PLACEHOLDER_BLACKLIST`: `Dataset 1`, `Task 2`, `Test Set`, `Ours`, `Proposed`, ecc.).
     - Nuova funzione di validazione `_is_valid_new_benchmark_name()`: scarta automaticamente nomi troppo corti (<3 caratteri), puramente numerici, pattern `dataset \d+` o combinazioni organo/modalità generica.
     - **Regola di Promozione Restrittiva:** `ALTER TABLE ... ADD COLUMN` su tutte le 5 tabelle del database viene emesso **esclusivamente se**:
       - Il dataset non è privato (`_is_private_cohort == False`).
       - Lo status del punteggio è `CONCORDANT` (o derivato da accordo duale). **I punteggi `single_agent_unconfirmed` o `disputed` NON possono mai alterare lo schema del database**.
       - Il nome del dataset supera `_is_valid_new_benchmark_name()`.
     - Gli score che non soddisfano tali requisiti vengono reindirizzati in modo sicuro alla colonna di commento e provenienza (`results_best_comments`), proteggendo SQLite da schemi corrotti e impedendo runtime crash.
  2. *Normalizzazione Unicode NFKC e Risoluzione Identità Modello:*
     - Applicata normalizzazione `unicodedata.normalize("NFKC", text)` su tutti i nomi di modelli e dataset, risolvendo all'origine le ligature tipografiche (`Diﬀ-UNet` $\to$ `Diff-UNet`).
     - Creata la blacklist `GENERIC_MODEL_NAMES` (`ours`, `proposed`, `baseline`, `model`, `our method`, ecc.) per impedire l'inserimento di righe fittizie nella tabella `models`.
     - Algoritmo di risoluzione entità a 4 livelli in `_resolve_model_id()`:
       1. Lookup dell'`expected_model_id` passato dalla pipeline o dai metadati del paper.
       2. Match esatto o case-insensitive/NFKC sui modelli esistenti nel database.
       3. Match per titolo scientifico (`paper_title` vs `m.title` nel database).
       4. Fallback al primo candidato non generico.
  3. *Deduplicazione pre-UPDATE per cella `(model_id, canonical_name)`:*
     - Implementato `_deduplicate_score_items()`: se un report presenta punteggi multipli per lo stesso dataset (es. 13 organi distinti o più prompt regime), i dati vengono pre-aggregati prima di generare le query SQL.
     - Se è presente una riga di sintesi macro (`Overall`), viene selezionata quella come picco principale ed esclusi i singoli organi.
     - Se sono presenti solo organi disaggregati, viene calcolata la media macro sintetizzata.
     - Se nel gruppo è presente anche una baseline primaria standard (es. 1-click), viene associata al record per aggiornare `results_primary`, garantendo **al massimo un singolo `UPDATE` per cella** ed eliminando le sovrascritture distruttive nei commenti.
  4. *Sanitizzazione e SQL Escaping Robusto:*
     - Aggiunto `_escape_ident()` per sanitizzare tutti i nomi di tabelle e colonne racchiusi tra doppi apici (`"{esc_col}"`), raddoppiando i doppi apici secondo lo standard SQL ed eliminando caratteri di controllo, ritorni a capo e punti e virgola.
- **Verifica con Test:** Aggiunti 4 test di regressione in `tests/test_database_agent.py`:
  - `test_schema_guard_rejects_organs_modalities_and_placeholders`: verifica il blocco di organi, modalità e segnaposto (`ADD COLUMN` non emesso, reindirizzamento ai commenti).
  - `test_schema_guard_blocks_unconfirmed_single_agent_new_columns`: verifica che un punteggio single-agent unconfirmed non possa alterare lo schema SQLite.
  - `test_unicode_nfkc_and_generic_model_id_resolution`: verifica che `Diﬀ-UNet` aggiorni il record `Diff-UNet` e che `Ours` sia scartato a favore di `expected_model_id`.
  - `test_pre_update_deduplication_single_update_per_cell`: verifica che organi multipli e una riga Overall per BTCV producano esattamente un singolo statement `UPDATE "results_best"` con il valore Overall corretto.

---

### 4.4. Priorità P3 — Robustezza Dual-Agent, Guided Decoding, Sicurezza e Test

#### P3.8: Guided JSON Schema Decoding, Validatori Permissivi, Tolleranza ai Guasti e Deliberation Collision Fix
- **File modificati:** [`src/agent_gmmis/llm/client.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py), [`src/agent_gmmis/extraction/schemas.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py), [`src/agent_gmmis/extraction/dual_coordinator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py), [`tests/test_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_extraction.py).
- **Risoluzione tecnica:**
  1. *Guided Decoding su vLLM:* In `LLMClient._call_openai_compatible()`, quando si utilizzano backend locali (`mufasa`, `ollama`), viene passato `extra_body={"guided_json": raw_schema}`, vincolando la generazione di token alla grammatica dello schema JSON ed eliminando allucinazioni sintattiche di `Mistral-7B`. In caso di mancato supporto backend, interviene il fallback trasparente a `response_format={"type": "json_object"}`.
  2. *Validatori Pydantic Permissivi:*
     - Aggiunto `@field_validator("is_baseline_comparison", mode="before")` in [`RawExtractedScore`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py#L33): converte `None`, stringhe esplicative (`"Not applicable"`, `"none"`, `"no"`, `"false"`) o interi `0` in `False`, e stringhe `"true"`, `"yes"`, `"baseline"` in `True`.
     - Aggiunto `@field_validator("extracted_scores", mode="before")` in [`ExtractedPaperData`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/schemas.py#L97): scarta automaticamente elementi JSON incompleti o privi di `dataset_name`, evitando il fallimento globale della deserializzazione.
  3. *Graceful Degradation su Guasto Parziale:*
     - In [`DualExtractionCoordinator.extract_manuscript_parallel`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L96-L150), se un agente subisce un errore runtime o timeout ma l'altro completa con successo, il sistema non solleva più `RuntimeError`. Viene generato un fallback sicuro che consente all'agente superstite di procedere, con i punteggi etichettati come `SINGLE_AGENT_UNCONFIRMED`. `RuntimeError` viene sollevato unicamente se entrambi gli agenti falliscono.
  4. *Collision Fix nella Deliberazione:*
     - In `_find_matching_score_index`, il matching dei punteggi da aggiornare include ora `prompt_type` e `data_budget_or_setting` (es. `10% labeled` vs `100% labeled`), evitando che ri-valutazioni di split differenti sovrascrivano la riga sbagliata.
- **Verifica con Test:** Aggiunti 4 test in `tests/test_extraction.py` (`TestPrioritaP3Robustness`):
  - `test_pydantic_permissive_is_baseline_comparison`
  - `test_pydantic_filter_malformed_scores`
  - `test_dual_extraction_graceful_degradation_when_one_agent_fails`
  - `test_find_matching_score_index_budget_and_prompt_discrimination`

#### P3.9: Allineamento Dipendenze, Sicurezza TLS e Normalizzazione Outreach
- **File modificati:** [`pyproject.toml`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/pyproject.toml), [`src/agent_gmmis/auditing/repo_inspector.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/repo_inspector.py), [`src/agent_gmmis/auditing/outreach.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/outreach.py), [`tests/test_outreach.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_outreach.py).
- **Risoluzione:**
  1. Dichiarate esplicitamente le dipendenze `google-genai>=1.0.0` e `certifi>=2024.0.0` in `pyproject.toml`.
  2. Sostituito `ssl._create_unverified_context()` con `ssl.create_default_context(cafile=certifi.where())` in `RepoInspector.parse_github_url()`, ripristinando la corretta validazione dei certificati TLS verso redirect GitHub.
  3. Allineato l'indirizzo istituzionale dell'autore in `OutreachAgent._render_email_body()` e nei relativi test a `luca2.pezzoli@mail.polimi.it`.


---

## 5. Seconda Revisione (Audit Post-Implementazione): 3 Correzioni Chirurgiche Obbligatorie

**Stato:** Verifica del working tree dopo gli interventi della Sezione 4.  
**Esito generale:** Il ~90% delle modifiche (`P0` -> `P3`) è stato implementato in modo pulito e solido. Tuttavia, l'ispezione incrociata del diff ha evidenziato **3 bug collaterali (di cui 1 crash bloccante `TypeError` a fine benchmark)** introdotti dalle nuove modifiche e non coperti dai test attuali.

Di seguito sono riportate le **istruzioni puntuali (file, righe e codice esatto)** che l'agente esecutore deve applicare per chiudere definitivamente la revisione.

---

### 5.1. [RISOLTO E VERIFICATO - FIX-1 / BLOCCANTE] Crash `TypeError: unsupported format string passed to NoneType.__format__` a fine run (`evaluator.py`, `run_batch_benchmark.py`, `run_extraction.py`)

#### Causa del problema
1. In [`src/agent_gmmis/extraction/evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L187), `gt_keys` viene ora costruito come l'unione di `gt_map` (`primary`), `gt_primary_map` e `gt_best_map`:
   ```python
   gt_keys = list(dict.fromkeys(list(gt_map.keys()) + list(gt_primary_map.keys()) + list(gt_best_map.keys())))
   ```
2. Nel database SQLite (`data/gmmis_database.db`), **37 modelli su 40** (es. `MedSAM-2`, `SAM-Med3D`, `SegVol`, `SAM3D`, `HERMES`, `MultiTalent`) hanno dataset valorizzati in `results_best` ma a `NULL` in `results_primary`.
3. Quando si esegue il benchmark con `compare_source="primary"` (default) e l'estrattore estrae uno di questi dataset (es. `BraTS` per `MedSAM-2` o `BTCV` per `SegVol`), alle righe [`214-260` di `evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L214-L260) `true_dice = gt_map.get(match_gt_key)` è **`None`** e `diff` è **`None`**, per cui `comp_entry` viene aggiunto a `rep.score_comparisons` con:
   ```python
   "ground_truth": None,
   "diff": None,
   ```
4. Al termine della run sui 40 paper, durante la formattazione della tabella Markdown in [`scripts/run_batch_benchmark.py` (riga 699)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py#L699) e in [`scripts/run_extraction.py` (riga 354)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L354), l'istruzione:
   ```python
   [c["dataset"], c.get("prompt_type", "N/A"), f"{c['predicted']:.4f}", f"{c['ground_truth']:.4f}", f"{c['diff']:.4f}"]
   ```
   solleva **`TypeError: unsupported format string passed to NoneType.__format__`**, facendo crashare l'intero script prima del salvataggio di `benchmark_report.md` e `benchmark_results.json`.

#### Modifica Puntuale Richiesta
1. **In [`scripts/run_batch_benchmark.py` (righe 698-701)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py#L698-L701) e in [`scripts/run_extraction.py` (righe 353-356)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L353-L356):**
   Formattare in modo sicuro `ground_truth` e `diff` quando sono `None` (oppure mostrare il fallback al regime alternativo con annotazione):
   ```python
   comp_rows = [
       [
           c["dataset"],
           c.get("prompt_type", "N/A"),
           f"{c['predicted']:.4f}" if c.get("predicted") is not None else "N/A",
           f"{c['ground_truth']:.4f}" if c.get("ground_truth") is not None else (
               f"{c['ground_truth_best']:.4f} (best)" if c.get("ground_truth_best") is not None else "N/A"
           ),
           f"{c['diff']:.4f}" if c.get("diff") is not None else (
               f"{c['diff_best']:.4f} (best)" if c.get("diff_best") is not None else "N/A"
           ),
       ]
       for c in rep.score_comparisons
   ]
   ```
2. **In [`src/agent_gmmis/extraction/evaluator.py` (righe 261-265)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L261-L265):**
   Verificare che `unmatched_extracted_datasets` e `n_pred` non contino come "matched" sul `compare_source` attivo un dataset che esiste solo nell'altro regime, oppure assicurarsi che `score_comparisons` contenga sempre valori numerici validi o gestiti dai chiamanti.
3. **Test da aggiungere in [`tests/test_evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_evaluator.py):**
   Aggiungere un test in cui `db.get_scores_for_model` restituisce `{"BraTS": {"primary": 85.0, "best": 88.0}, "BTCV": {"primary": None, "best": 82.0}}`, valutare con `compare_source="primary"` un'estrazione che contiene sia `BraTS` sia `BTCV`, e verificare che la formattazione tabellare di `report.score_comparisons` non sollevi `TypeError` e che `report.ground_truth_datasets_count == 1` e `report.matched_datasets_count == 1`.

---

### 5.2. [RISOLTO E VERIFICATO - FIX-2 / ALTO IMPATTO SUL RECALL] Disallineamento tra `ExtractionAgent` (`42.000` char) e il Pre-Trimmer in `LLMClient` (`client.py:218-233`) che cancella le pagine centrali del paper

#### Causa del problema
1. In [`src/agent_gmmis/extraction/extractor.py` (righe 33-42)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L33-L42), `get_default_context_chars()` è stato giustamente alzato a **`42000` caratteri** per `("mufasa", "ollama", "vllm")`. Alla riga [`121`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L121), le pagine selezionate dallo scoring vengono riordinate in ordine crescente di pagina (`selected_pages = sorted(pages_to_include)`).
2. Tuttavia, in [`src/agent_gmmis/llm/client.py` (righe 218-233)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L218-L233) è rimasto invariato il vecchio blocco di pre-trimming:
   ```python
   total_chars = len(prompt) + (len(system_prompt) if system_prompt else 0) + len(schema_json)
   approx_tokens = int(total_chars / 2.0) + 400
   context_limit = 16384 if self.config.provider in ("mufasa", "ollama") else 32768

   required_headroom = max_needed_output + 300
   if approx_tokens > context_limit - required_headroom and len(prompt) > 4000:
       excess_tokens = approx_tokens - (context_limit - required_headroom)
       excess_chars = int(excess_tokens * 2.2) + 400
       mid = len(prompt) // 2
       half_trim = min(mid - 500, max(500, excess_chars // 2))
       prompt = prompt[: mid - half_trim] + "\n\n[... middle text trimmed to respect context limit ...]\n\n" + prompt[mid + half_trim :]
   ```
3. **Effetto distruttivo:**
   - La stima `total_chars / 2.0` sovrastima enormemente i token (su testo inglese/Markdown, 1 token $\approx 3.3 - 3.8$ caratteri).
   - Con `context_limit = 16384` e `required_headroom = 3800`, la soglia `context_limit - required_headroom` è `12584` token, che con `/ 2.0` corrisponde a soli **`~24.300` caratteri totali** (`~20.500` caratteri di `prompt`).
   - Di conseguenza, quando `extractor.py` costruisce un prompt di `42.000` caratteri con le pagine ordinate `[Page 1, Page 2, Page 5, Page 6, Page 7, Page 8, Page 10, Page 12]`, `client.py` **taglia via ~23.000 caratteri esattamente dalla metà centrale di `prompt` (`prompt[:mid - half_trim] + ... + prompt[mid + half_trim:]`)**!
   - Nella metà centrale di `selected_pages` si trovano **proprio le pagine 5, 6, 7 e 8 (le tabelle principali dei risultati sperimentali)**, mentre sopravvivono solo Pagina 1–2 in testa e Pagina 10–12 (spesso Ablation/Bibliografia) in coda!

#### Modifica Puntuale Richiesta
1. **In [`src/agent_gmmis/llm/client.py` (righe 218-235)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L218-L235):**
   - Sostituire il divisore pessimistico `2.0` con **`3.2`** (`approx_tokens = int(total_chars / 3.2) + 300`) e `excess_chars = int(excess_tokens * 3.2) + 300`. Con `3.2`, un contesto vLLM di `16384` token supporta fino a `~40.000` caratteri totali (`~36.000` di prompt) senza alcun taglio!
   - Qualora il pre-trimmer dovesse comunque intervenire su prompt giganteschi, **non tagliare la metà centrale del blocco `=== MANUSCRIPT CONTENT ===`**: tagliare la coda del solo `content` *prima* delle istruzioni finali del template utente, oppure lasciare che sia `_build_context_content` a rispettare il budget.
2. **In [`src/agent_gmmis/extraction/extractor.py` (righe 33-42)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L33-L42):**
   - Impostare `get_default_context_chars()` per `("mufasa", "ollama", "vllm")` a **`35000`** caratteri (che sommati a system prompt e schema JSON restano sotto la soglia di `12.584` token con divisore `3.2`), garantendo che **nessuna pagina selezionata da `_score_page()` venga mai mutilata a valle da `LLMClient`**.
3. **Test da aggiungere in [`tests/test_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_extraction.py):**
   Verificare che un `prompt` generato da `ExtractionAgent` con budget default per `mufasa` (`35000` char) passato a `LLMClient._openai_client.chat.completions.create` (mockato) **non** contenga la stringa `"[... middle text trimmed to respect context limit ...]"`.

---

### 5.3. [RISOLTO E VERIFICATO - FIX-3 / LOGICA DUAL-AGENT] Falso innesco di `NEEDS_REPO_INSPECTION` e Deliberazione su Agente Fallito durante la *Graceful Degradation* (`auditor.py`, `dual_coordinator.py`)

#### Causa del problema
1. **Deliberazione inutile su agente fallito ([`src/agent_gmmis/extraction/dual_coordinator.py`, righe 203-235](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L203-L235)):**
   Se in `extract_manuscript_parallel()` uno dei due agenti va in crash (es. Agente B fallisce e viene sostituito dal fallback vuoto con `extraction_notes="Agent B failed: ... Single-agent fallback to Agent A."`), in `extract_manuscript_with_deliberation()` tutti i punteggi dell'Agente A risultano `SINGLE_AGENT_UNCONFIRMED` (`unconfirmed_count > 0`). Il coordinatore entra quindi nel ciclo `for round_num in range(2, max_rounds + 1):` e **invoca ripetutamente `self.agent_b.reinterrogate_omission()` sull'Agente B che era appena andato in crash**!
2. **Falso `tables_missing = True` quando un solo agente estrae tabelle valide ([`src/agent_gmmis/auditing/auditor.py`, righe 1064-1088](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L1064-L1088)):**
   In `AuditorAgent.audit_consensus()` è stato scritto:
   ```python
   tables_missing = (
       report_a.status in (AuditStatus.NEEDS_REPO_INSPECTION, AuditStatus.NEEDS_OUTREACH)
       or report_b.status in (AuditStatus.NEEDS_REPO_INSPECTION, AuditStatus.NEEDS_OUTREACH)
       or (concordant_count == 0 and paired_count == 0)
   )
   ```
   Se l'Agente A estrae **5 benchmark canonici validi** dal PDF (`report_a.status == AuditStatus.VERIFIED`), ma l'Agente B fallisce (o non estrae nulla, per cui `report_b.status == NEEDS_REPO_INSPECTION`, `concordant_count == 0` e `paired_count == 0`), `tables_missing` diventa **`True`**!
   Di conseguenza:
   - `audit_consensus()` certifica erroneamente `final_status = AuditStatus.NEEDS_REPO_INSPECTION` (invece di `AuditStatus.FLAGGED_WARNING`), ignorando che l'Agente A ha già trovato le tabelle nel PDF!
   - Subito dopo, in [`scripts/run_batch_benchmark.py` (righe 281-283)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py#L281-L283), viene chiamato `repo_inspector.inspect_and_update()`, che (non trovando tabelle sul repo GitHub) **sovrascrive `audit_report` (di tipo `ConsensusAuditReport`) con un `AuditReport` single-agent**, facendo perdere la matrice di consenso nel JSON finale!
3. **Piccola omissione nel metodo single-agent `AuditorAgent.audit()` ([`src/agent_gmmis/auditing/auditor.py`, righe 561-573](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L561-L573)):**
   Alla riga 561 compare:
   ```python
   if len(scalar_scores) == 0 or has_cloned_template_hallucination or (not has_canonical_benchmark_table and extracted.code_repository_url):
   ```
   ma alla riga 573 l'`if` annidato controlla solo:
   ```python
   if (len(scalar_scores) == 0 or has_cloned_template_hallucination):
   ```
   dimenticando `or (not has_canonical_benchmark_table and extracted.code_repository_url)`. Se un agente estrae solo un dataset non canonico di ablation (come `"RandomAblationTask"` nel caso `BiomedParse`), `report_b.status` in single-agent rimane `VERIFIED` anziché `NEEDS_REPO_INSPECTION`.

#### Modifica Puntuale Richiesta
1. **In [`src/agent_gmmis/extraction/dual_coordinator.py` (subito dopo la riga 212)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L212):**
   Se uno dei due agenti è in modalità fallback single-agent (`"Single-agent fallback" in (extracted_a.extraction_notes or "")` oppure `"Single-agent fallback" in (extracted_b.extraction_notes or "")`), saltare i round di deliberazione e ritornare direttamente `(extracted_a, extracted_b, audit_report, deliberation_trajectory)`.
2. **In [`src/agent_gmmis/auditing/auditor.py` (riga 573)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L573):**
   Allineare la condizione interna alla riga 561:
   ```python
   if (
       len(scalar_scores) == 0
       or has_cloned_template_hallucination
       or (not has_canonical_benchmark_table and bool(extracted.code_repository_url or (self.db and self.db.get_model(extracted.proposed_model_id))))
   ):
   ```
3. **In [`src/agent_gmmis/auditing/auditor.py` (righe 1064-1070)](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L1064-L1070):**
   Definire `tables_missing` in modo che sia `True` **solo se nessuno dei due agenti ha estratto tabelle canoniche certificate (`VERIFIED`)**:
   ```python
   neither_agent_verified_tables = (
       report_a.status != AuditStatus.VERIFIED
       and report_b.status != AuditStatus.VERIFIED
   )
   tables_missing = (
       neither_agent_verified_tables
       and (
           report_a.status in (AuditStatus.NEEDS_REPO_INSPECTION, AuditStatus.NEEDS_OUTREACH)
           or report_b.status in (AuditStatus.NEEDS_REPO_INSPECTION, AuditStatus.NEEDS_OUTREACH)
           or (concordant_count == 0 and paired_count == 0)
       )
   )
   ```
   *(In questo modo: se `BiomedParse` ha 0 tabelle in A e solo `"RandomAblationTask"` non canonico in B, entrambi i report non sono `VERIFIED` e `tables_missing` è `True` $\to$ `NEEDS_REPO_INSPECTION`. Se invece in `SurvivorModel` l'Agente A ha estratto `BraTS: 0.91` dal `Table 1` del PDF ed è `VERIFIED` mentre l'Agente B è fallito, `neither_agent_verified_tables` è `False` e il report finale resta correttamente `FLAGGED_WARNING` con i punteggi `SINGLE_AGENT_UNCONFIRMED` preservati!).*
4. **Test da aggiungere in [`tests/test_consensus_auditing.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/tests/test_consensus_auditing.py):**
   Aggiungere `test_graceful_degradation_preserves_verified_agent_without_false_repo_inspection`: quando l'Agente A estrae un benchmark canonico valido (`dataset_name="BraTS"`, `table_or_section_ref="Table 1"`, `dice_score=0.91`) e l'Agente B è il fallback vuoto di graceful degradation, `auditor.audit_consensus(ext_a, ext_b)` deve restituire `status == AuditStatus.FLAGGED_WARNING` (e **non** `NEEDS_REPO_INSPECTION`) con `unconfirmed_count == 1`.

---

### 5.4. Documentazione di Implementazione e Collaudo Unitario (113/113 Test Passati)

Tutte le correzioni chirurgiche richieste nella Sezione 5 sono state implementate, verificate con nuovi test unitari dedicati e collaudate su tutta la suite pytest:

| ID Fix | Categoria | File Modificati | Test Aggiunti / Aggiornati | Esito |
| :--- | :--- | :--- | :--- | :--- |
| **5.1** | `FIX-1 / BLOCCANTE` | `evaluator.py`, `run_batch_benchmark.py`, `run_extraction.py` | `tests/test_evaluator.py::test_evaluator_and_table_format_with_null_primary_ground_truth` | **RISOLTO E VERIFICATO** |
| **5.2** | `FIX-2 / RECALL` | `llm/client.py`, `extraction/extractor.py` | `tests/test_extraction.py::test_dynamic_context_budget_and_table_prioritization`, `test_mufasa_default_budget_does_not_trigger_middle_trimming` | **RISOLTO E VERIFICATO** |
| **5.3** | `FIX-3 / DUAL-AGENT` | `extraction/dual_coordinator.py`, `auditing/auditor.py` | `tests/test_consensus_auditing.py::test_graceful_degradation_preserves_verified_agent_without_false_repo_inspection`, `test_deliberation_skips_when_single_agent_fallback_active` | **RISOLTO E VERIFICATO** |

#### Dettagli Tecnici delle Modifiche Applicate:
1. **Fix 5.1 (Formattazione e Valutazione su Ground Truth `None`):**
   - In [`scripts/run_batch_benchmark.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_batch_benchmark.py#L698-L715) e [`scripts/run_extraction.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/scripts/run_extraction.py#L353-L370), la stampa tabellare protegge `ground_truth` e `diff` con formattazione condizionale sicura, mostrando annotazioni trasparenti `(best)` o `(primary)` in caso di fallback tra regimi, oppure `"N/A"` se nullo.
   - In [`src/agent_gmmis/extraction/evaluator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/evaluator.py#L261-L267), aggiunto il conteggio `extra_regime_matched` per garantire che `n_pred` e precision non conteggino come match un benchmark assente nel regime di confronto attivo.
2. **Fix 5.2 (Allineamento Divisore Token e Salvaguardia Tabelle Centrali):**
   - In [`src/agent_gmmis/llm/client.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/llm/client.py#L218-L235), sostituito il divisore sovrastimato `2.0` con `3.2` (`approx_tokens = int(total_chars / 3.2) + 300`, `excess_chars = int(excess_tokens * 3.2) + 300`).
   - Se il prompt supera la capacità massima, il trimming agisce sul testo terminale delle appendici/ablazioni (`[... trailing manuscript text trimmed to respect context limit ...]`), salvaguardando integralmente le pagine centrali contenenti le tabelle sperimentali comparative.
   - In [`src/agent_gmmis/extraction/extractor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/extractor.py#L40-L41), impostato il budget di default a `35000` caratteri per provider con finestra a 16k (`mufasa`, `ollama`, `vllm`), garantendo che nessuna pagina selezionata venga tagliata.
3. **Fix 5.3 (Dual-Agent Graceful Degradation & Auditor Routing):**
   - In [`src/agent_gmmis/extraction/dual_coordinator.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/extraction/dual_coordinator.py#L212-L222), prima del ciclo di deliberazione viene verificata la presenza di `"Single-agent fallback"`; se attiva, la deliberazione iterativa viene saltata istantaneamente, evitando chiamate inutili all'agente in stato di errore.
   - In [`src/agent_gmmis/auditing/auditor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L561-L585), unificato il controllo di disponibilità codice (`has_repo = bool(...)`) e allineata la condizione annidata per diagnosticare correttamente dataset di sola ablation (`RandomAblationTask`) verso `CODE_REPO_AVAILABLE`.
   - In [`src/agent_gmmis/auditing/auditor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L1068-L1078), `tables_missing` viene condizionato su `neither_agent_verified_tables`. Se un agente ha estratto tabelle valide certificate (`VERIFIED`), il modello non viene erroneamente instradato a `NEEDS_REPO_INSPECTION`, ma certificato come `FLAGGED_WARNING` con punteggi `SINGLE_AGENT_UNCONFIRMED` intatti.
   - In [`src/agent_gmmis/auditing/auditor.py`](file:///Users/pezzoli/PoliMi/Thesis/AgentForGMMIS/src/agent_gmmis/auditing/auditor.py#L595-L606), le anomalie critiche (es. confusione del modello con baseline nota) hanno priorità assoluta di rejection su `CODE_REPO_AVAILABLE`, ad eccezione della sola allucinazione di template ripetuto su cui `RepoInspector` agisce come layer di recupero.

#### Risultati del Collaudo Unitario:
```bash
============================= test session starts ==============================
rootdir: /Users/pezzoli/PoliMi/Thesis/AgentForGMMIS
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.1.0, anyio-4.15.1
collected 113 items

tests/test_auditor.py ...............                                    [ 13%]
tests/test_consensus_auditing.py ...........................             [ 37%]
tests/test_database_agent.py ................                            [ 51%]
tests/test_db_layer.py .........                                         [ 59%]
tests/test_evaluator.py .....                                            [ 63%]
tests/test_extraction.py ................                                [ 77%]
tests/test_outreach.py ........                                          [ 84%]
tests/test_repo_inspector.py ..........                                  [ 93%]
tests/test_run_logger.py .......                                         [100%]

============================= 113 passed in 3.11s ==============================
```
Tutti i 113 test del framework passano al 100% senza alcuna regressione. Il codice è pronto per la revisione finale da parte dell'agente revisore senior.
