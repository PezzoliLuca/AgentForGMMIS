# 6 Novembre

## 1.  Logging e Tracciabilità delle Run

salvataggio dei log (`RunLogger`) ristrutturato

- **Struttura per run**: ogni esecuzione (locale o su Mufasa) genera una cartella con timestamp e configurazione in `logs/mufasa_runs/<timestamp>_<modelli>/`.
- **Salvataggio per singolo paper**: nella sottocartella `extractions/<model_id>/` vengono salvati:
  - I JSON estratti dall'Agente A e dall'Agente B al Round 1 e nei round successivi (`round_1/`, `round_2/`, ecc.).
  - Il file `deliberation_trajectory.json`, che registra come cambiano la concordanza e la differenza media tra i due modelli da un round all'altro.
  - Il piano del database (`curation_plan.json` e `curation.sql`) e le bozze di email per gli autori (`<model_id>_outreach.md` e `.json`).
- **Report**: a fine run vengono generati automaticamente il riepilogo Markdown (`benchmark_report.md`), il JSON completo con tutte le metriche (`benchmark_results.json`) e lo script SQL (`database_curation.sql`).
- **Sincronizzazione cluster**: aggiunto uno script per scaricare i log generati da Mufasa.

---

## 2. Criteri dell'Auditor fatto in overleaf

---

## 3. Loop con due agenti

estrazione tramite due modelli indipendenti (`ExtractionAgent A` e `ExtractionAgent B`) coordinati da `DualExtractionCoordinator`:

1. **Round 1 (Estrazione parallela)**: i due modelli analizzano il paper senza alcuna comunicazione.
2. **Allineamento**: l'Auditor normalizza i nomi dei dataset tramite `DatasetAliasResolver` (es. *Synapse* $\rightarrow$ *BTCV*), normalizza i data budget (es. `"100% labeled"` vs `"full training data"`) e accoppia i punteggi dei due agenti massimizzando la compatibilità su organo, regime di prompt e budget.
3. **Classificazione iniziale**:
   - `CONCORDANT`: entrambi hanno estratto il benchmark e $|S_A - S_B| \le 0.01$. Il valore finale è la media dei due.
   - `DISPUTED`: entrambi hanno estratto il benchmark ma $|S_A - S_B| > 0.01$.
   - `SINGLE_AGENT_UNCONFIRMED`: il benchmark è stato estratto da uno solo dei due modelli.
4. **Round successivi (fino a 3 round)**:
   - **nessun agente vede mai il numero estratto dall'altro**, per evitare che un modello si accodi passivamente al valore dell'altro
   - *Su un valore `DISPUTED`*: a entrambi viene comunicato SOLO che sul dataset c'è una differenza rispetto all'altro estrattore, chiedendo di rileggere la tabella puntando al valore massimo del modello proposto e citando riga e colonna.
   - *Su un valore `SINGLE_AGENT_UNCONFIRMED`*: al modello che non ha estratto il dataset $D$ viene chiesto di verificare se nel paper esiste effettivamente una valutazione 3D in Dice per $D$, oppure di confermare il motivo dell'esclusione (es. dataset 2D, baseline, o solo grafico).
   - se un modello ha estratto i singoli organi e l'altro solo l' `Overall`, viene inviata una richiesta mirata per estrarre la riga di sintesi (`Avg` / `Mean`).
5. **Early Stopping se**:
   - Tutte le dispute e omissioni risolte (`VERIFIED_BY_DELIBERATION`).
   - **Stallo**: se tra il round $k-1$ e il round $k$ nessuno dei due agenti modifica i propri punteggi, il loop si ferma subito per non sprecare chiamate API o gpu.
   - Raggiungimento del numero massimo di round: le divergenze non risolte restano marcate come `DISPUTED`.

### 3.2. Osservazioni: Qwen vs Gemini e stallo nei pareggi
- **Differenza di ricerca a parità di prompt**: usando lo stesso identico prompt, **Qwen tende a estrarre un valore per ciascun singolo organo** (es. tutti i 13 organi di BTCV), mentre **Gemini estrae quasi sempre solo il valore `Overall`**.
  - All'inizio questo falsava il calcolo della concordanza (1 match su 14 voci totali sembrava un 7% di accordo). Abbiamo risolto calcolando l'accordo solo sui benchmark estratti da entrambi (*Paired IAA*, disaccoppiato dai *Singleton*) e aggiungendo nell'evaluator la media aritmetica automatica sui singoli organi quando nella tabella del paper non c'è la colonna `Avg`.
- **Stallo per *Equilibrium* e possibile terzo modello "Arbitro"**:
  - Quando i due agenti entrano in disaccordo su un punteggio, durante il loop di deliberazione spesso **entrambi difendono la propria estrazione citando frasi o tabelle diverse dello stesso paper** (ad esempio uno prende la tabella del test set principale e l'altro una tabella di ablation o zero-shot), portando allo stop al round 1 o 2.
  - *Possibile sviluppo futuro*: introdurre un terzo modello leggero con ruolo di "Giudice", chiamato solo quando c'è uno stallo per fare il cross-referencing delle due tabelle citate dai due agenti e sbloccare questi pareggi.

---

## 4. Chain-of-Thought

Per forzare i modelli a ragionare prima di estrarre i numeri:
- inserito l'oggetto `thought_process` (`ExtractionThoughtProcess`) come **primo campo** dello schema JSON (`ExtractedPaperData`).
- Prima di poter compilare la lista `extracted_scores`, il modello è obbligato a compilare 5 campi concisi:
  1. `proposed_model_identification` (quale modello è proposto dal paper e quali sono i competitor da ignorare);
  2. `primary_benchmark_tables` (quali tabelle contengono i confronti principali rispetto alle ablation);
  3. `peak_performance_logic` (qual è il regime di prompt o configurazione che dà il valore massimo);
  4. `multitask_or_macro_averages` (dove si trova la colonna `Avg`/`Mean` rispetto ai singoli organi);
  5. `ambiguities_or_caveats`.
- Inoltre, dentro ogni singolo punteggio estratto è stato aggiunto il campo `derivation_rationale`, dove il modello deve indicare esplicitamente tabella, riga, colonna e tipo di prompt utilizzato.

---

## 5. Regola Peak Performance e Analisi del MAE rispetto alla Ground Truth

### 5.1. Passando da "1-click" a "Valore Massimo"
rimosso il vincolo del *1-point click* e istruito gli agenti a estrarre per ogni dataset il **valore massimo di Dice** raggiunto dal modello proposto (includendo *Bounding Box prompt*, *multi-click*, o *fine-tuning*).

Da quando c'è questo, **il MAE sul Dice rispetto alla Ground Truth primaria del database della survey è aumentato**.

### 5.2. Analisi delle discrepanze
L'analisi mostra che **circa il 70% dello scostamento sul MAE non dipende da errori dell'LLM**, ma da come sono stati compilati i dati nella survey originale (`docs/SURVEY_VS_PEAK_ANALYSIS.md`):

| Modello | Dataset | Valore DB (`primary`) | Valore Estratto dall'Agente | Motivo della discrepanza |
| :--- | :--- | :---: | :---: | :--- |
| **SAM-Med3D** | AMOS | `0.7541` | `0.8590` | Nel DB è stato registrato il valore *1-point prompt* (come annotato in `commentOnPaper`). Nel paper, con *Bounding Box prompt*, il modello arriva a `0.8590` ($\Delta = 0.1049$). |
| **SAM3D** | BTCV | `0.8220` | `0.9580` | Il DB riporta il setting a 1 punto (`0.8220`); l'agente estrae il picco con *Box prompt* (`0.9580`, $\Delta = 0.1360$). |
| **Med-SA** | SegTHOR | `0.8155` | `0.8720` | Il DB fissa il setting a 1 punto; l'agente estrae il valore massimo multi-point/box riportato nella stessa tabella. |
| **SAT** | TotalSegmentator | `0.7600` | `0.8840` | Il DB riporta la versione base/zero-shot; l'agente estrae la variante `SAT-Pro` post-finetuning presentata dagli autori. |
| **MoME** | ATLAS v2.0 | `0.6203` | `0.8008` | Nelle note del DB è scritto che `0.6203` è una **media aritmetica calcolata a mano dai curatori** tra diverse modalità/sequenze MRI (T1, T2, FLAIR) della Tabella IV. L'agente legge la tabella ed estrae la migliore sequenza reale a testo (`0.8008`). |
| **LHU-Net** | BTCV / Synapse | `0.8749` | `0.8783` | Tutti i modelli (Qwen, Gemini, Mistral, Codestral) estraggono `0.8783`, che è il numero effettivamente stampato nella Tabella 2 del paper (probabile refuso di trascrizione nel DB o versione preprint diversa). |

A questo si aggiungono due aspetti tecnici emersi sul database SQLite e sugli errori reali dei modelli (~30% dei casi):
1. **Valori `NULL` in `results_primary` vs `results_best`**: nel database originale, su **37 modelli su 40** diversi benchmark hanno il valore compilato in `results_best` ma lasciato a `NULL` in `results_primary` (196 valori totali in `primary` contro 289 in `best`). Ho modificato l'`ExtractionEvaluator` in modo che calcoli contemporaneamente sia il **MAE contro `primary`** sia il **MAE contro `best`** (oltre a un *Regime-Agnostic MAE* che verifica se il numero estratto coincide con almeno uno dei due ).
2. **Errori veri dell'LLM (corretti)**:
   - *Singolo organo scambiato per il dataset (`TP-Mamba`)*: all'inizio l'istruzione "cerca il massimo" induceva il modello a prendere il singolo organo col Dice più alto su BTCV (Fegato `0.972` o Milza `0.966`) invece della colonna `Avg` (`0.848`). Risolto vincolando `organ_or_task='Overall'` alla sola colonna `Avg`/`Mean` (ora su `TP-Mamba` il MAE è `0.0000`).
   - *Scambio di tabella in appendice (`BrainSegFounder`)*: l'agente prendeva i numeri dalla Tabella 8 in appendice (5-fold CV su BraTS, `0.9113`) e li attribuiva ad ATLAS v2.0 (il cui valore corretto in Tabella 6 è `0.7120`). L'Auditor lo aveva comunque intercettato come outlier (+20% sopra il SOTA storico di ATLAS), e ho aggiunto nel prompt il **`TABLE CAPTION & DATASET GROUNDING CHECK`** che vieta di assegnare un punteggio a un dataset se quel dataset non è nominato esplicitamente nell'intestazione o didascalia di quella specifica tabella.

---

## 6. Fatta una Revisione del Codice e Fix di bug (tramite LLM)

---

## 7. Valutazione DS4, Modelli Provati su Mufasa e Tabella di Confronto

### 7.1. Verifica su DwarfStar 4
Non è possibile utilizzare **DS4**, è progettato esclusivamente per **Mac con chip M** con memoria unificata
- Su Mufasa l'opzione corretta resta **vLLM** interamente in VRAM ($35\text{--}60\text{ token/s}$).

### 7.2. Configurazioni testate (40 paper su Mufasa)
testato 6 configurazioni diverse sugli stessi **40 paper**, sia in modalità **100% locale**, sia in modalità **Ibrida (Locale su Mufasa + Gemini 3.1 Flash-Lite)**:

1. **Dual-Local (Qwen 7B + Llama 3.1 8B)**: `Qwen2.5-7B-Instruct-AWQ` + `Meta-Llama-3.1-8B-Instruct-AWQ`
2. **Dual-Local (Qwen 7B + Mistral 7B)**: `Qwen2.5-7B-Instruct-AWQ` + `Mistral-7B-Instruct-v0.3-AWQ`
3. **Hybrid (Qwen 14B + Gemini)**: `Qwen2.5-14B-Instruct-AWQ` + `Gemini 3.1 Flash-Lite`
4. **Hybrid (Codestral 22B + Gemini)**: `Codestral-22B-v0.1-AWQ` (Mistral AI) + `Gemini 3.1 Flash-Lite`
5. **Hybrid (Mistral 7B + Gemini)**: `Mistral-7B-Instruct-v0.3` + `Gemini 3.1 Flash-Lite`
6. **Hybrid Reasoning (DeepSeek-R1 14B + Gemini)**: `DeepSeek-R1-Distill-Qwen-14B-AWQ` + `Gemini 3.1 Flash-Lite`

#### Tabella di comparazione

| Metrica | **1. Dual-Local**<br>`Qwen-7B + Llama-8B` | **2. Dual-Local**<br>`Qwen-7B + Mistral-7B` | **3. Hybrid**<br>`Qwen-14B + Gemini` | **4. Hybrid**<br>`Codestral-22B + Gemini` | **5. Hybrid**<br>`Mistral-7B + Gemini` | **6. Hybrid Reasoning**<br>`DeepSeek-R1-14B + Gemini` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tipo Architettura** | 100% Locale (20GB) | 100% Locale (20GB) | Ibrida (Mufasa+Cloud) | Ibrida (Mufasa+Cloud) | Ibrida (Mufasa+Cloud) | Ibrida (Mufasa+Cloud) |
| **Tempo Totale (40 paper)** | `50.8 min` (`76.3s`/p) | **`39.7 min` (`59.6s`/p)** | `67.3 min` (`101.0s`/p) | `83.9 min` (`125.8s`/p) | `72.3 min` (`108.5s`/p) | `68.0 min` (`102.0s`/p) |
| **Benchmark Trovati (`Primary`)** | `87 / 196` | `81 / 196` | `90 / 196` | `92 / 196` | `90 / 196` | **`94 / 196`** |
| **Benchmark Recall (`Primary`)** | `44.4%` | `41.3%` | `45.9%` | `46.9%` | `45.9%` | **`48.0%`** |
| **Micro-Averaged Dice MAE (`Primary`)** | `0.0546` | `0.0546` | `0.0176` | **`0.0106`** | `0.0179` | `0.0108` |
| **Micro-Averaged Dice MAE (`Best`)** | `0.0508` | `0.0568` | `0.0222` | **`0.0121`** | `0.0238` | `0.0152` |
| **Regime-Agnostic Micro MAE** | `0.0506` | `0.0509` | `0.0170` | **`0.0100`** | `0.0173` | `0.0109` |
| **Macro-Averaged Paper MAE (`Primary`)** | `0.0534` | `0.0511` | `0.0222` | **`0.0153`** | `0.0228` | `0.0157` |
| **Coverage-Adjusted Score** | `0.4196` | `0.3907` | `0.4511` | `0.4644` | `0.4510` | **`0.4744`** |
| **Paper con MAE = 0.0000 esatto** | `6` | `9` | `14` | `16` | **`17`** | `15` |
| **Paper Certificati `VERIFIED`** | `13 / 40` | `14 / 40` | **`16 / 40`** | `13 / 40` | `7 / 40` | **`16 / 40`** |
| **Confidenza Media Certificata** | `87%` | **`89%`** | `87%` | `86%` | `83%` | **`89%`** |
| **Nuove Colonne DB Proposte (post Schema-Guard)** | `36` | `25` | `47` | `46` | `30` | `44` |

*(Nota: rispetto alle run pre-revisione, la Recall sulle 5 configurazioni è salita dal range `31.4%–40.3%` a `41.3%–48.0%`, il Micro-MAE sulle configurazioni ibride è sceso fino a `0.0106` e l'esplosione di colonne SQLite spazzatura è scesa da `108–143` a sole `25–47` colonne filtrate).*

### 7.3. Caso specifico da discutere: il comportamento di Mistral vs Qwen su `SAM-Med2D`
sul paper **`SAM-Med2D`**, c'è una differenza sistematica:
- **Modelli europei (Mistral-7B e Codestral-22B)**: leggono che `SAM-Med2D` è un'architettura puramente 2D (che processa i volumi CT/MRI tagliandoli in fette 2D indipendenti) e, **in pieno accordo con Gemini**, mettono `is_3d = False`, **scartando il paper**.
- **Modelli cinesi (Qwen 7B in dual-local)**: vedendo che i dataset su cui viene testato il modello includono scansioni CT e MRI (e che l'Auditor ha la regola di salvataggio sui dataset canonici), non lo respingono in fase di screening ma **lo lasciano passare in stato `FLAGGED_WARNING`** (confidenza 74%–80%).
- **Anche `DeepSeek-R1-14B` + `Gemini`** lo blocca dal database per bassa confidenza (`56% < 60%`).
- **da chiarire**: `SAM-Med2D` era stato incluso nel database originale. Da decidere se per i prossimi paper di questo tipo vogliamo il comportamento di Mistral o quello degli altri.

---

## 8. Database Agent (`DatabaseAgent`)

implementato `DatabaseAgent`, gestisce l'inserimento e l'aggiornamento nel db

### 8.1. regole
1. **Filtro in ingresso**:
   - Se il report dell'Auditor è `REJECTED`, `NEEDS_REPO_INSPECTION` o `NEEDS_OUTREACH`, oppure ha una confidenza $< 0.60$, il paper viene bloccato e non modifica il database.
2. **Gestione consenso del singolo punteggio**:
   - **`CONCORDANT`** ($|\Delta| \le 0.01$): il valore viene inserito/aggiornato nella colonna numerica di `results_best` (`results_primary` se il modello è nuovo), registrando nella colonna commenti l'accordo tra i due agenti e il riferimento alla tabella.
   - **`DISPUTED`** ($|\Delta| > 0.01$): **la cella numerica non viene toccata** (resta `NULL` o mantiene il vecchio valore per non inserire numeri incerti), ma **viene scritto un commento dettagliato** in `results_best_comments` con i due valori estratti (`[DISPUTE BLOCKED] Model A: ... vs Model B: ...`).
   - **`SINGLE_AGENT_UNCONFIRMED`** **viene inserito solo se la confidenza certificata è alta ($\ge 0.85$)**, accompagnato da un avviso esplicito nel commento (`[Unconfirmed Single-Agent: ...]`). Se la confidenza è $< 0.85$, viene scartato.
3. **Regola tra nuovo valore e vecchio valore registrato**:
   - Quando nel database esiste già un valore per quella cella `(modello, dataset)` in `results_best`:
     - Se $\text{Dice}_{\text{nuovo}} > \text{Dice}_{\text{vecchio}}$: il valore numerico viene aggiornato al nuovo massimo e viene ricalcolato il guadagno percentuale $\Delta\%$ rispetto a `results_primary`.
     - Se $\text{Dice}_{\text{nuovo}} \le \text{Dice}_{\text{vecchio}}$: **non si fa nulla sul valore numerico** (il record precedente viene preservato) e il valore più basso viene solo annotato nei commenti come configurazione alternativa.
4. **Schema Guard**:
   - I dataset privati ospedalieri vengono bloccati e annotati solo in `commentOnPaper`.
   - Per i nuovi benchmark pubblici non presenti nelle colonne originali della survey, l'agente genera `ALTER TABLE ... ADD COLUMN` sulle 5 tabelle del DB **solo se** il punteggio è `CONCORDANT` e il nome non appartiene alla blacklist di singoli organi (`Liver`, `Spleen`, `Aorta`, ecc.) o di semplici modalità (`CT`, `MRI`, `T1`, `FLAIR`).
   - *Nota sui nuovi benchmark trovati*: nei 40 paper il sistema ha individuato diverse challenge pubbliche reali che non erano state messe come colonne nella survey originale (es. `IMed-361M`, `ULS23`, `CAS2023`, `SEG.A.2023`, `BraTS2020`, `ATLAS`, `RAVIR`, `JHH`, `Fed-IXI`). saranno giuste?

---

## 9. Interfaccia per Benchmark

Fatta una dashboard in Streamlit per lanciare i benchmark su mufasa
---

## 10. Scouter Base (`ScoutAgent`)

Realizzato e testato in modalità standalone la prima versione dello **`ScoutAgent`**.

### 10.1. 3 Pilastri di Ricerca e 4 filtri
Per evitare sia i limiti delle sole query statiche, sia l'irriproducibilità di una ricerca web libera fatta dall'LLM:
1. **Query booleane sistematiche** su 4 fonti: `arXiv`, `OpenAlex`, `EuropePMC/PubMed` e `Google Scholar` (con cache su disco e interruttore automatico se Google Scholar attiva un CAPTCHA).
2. interroga `OpenAlex` per trovare tutti i nuovi lavori che citano i paper chiave della survey.
3. **Agentic Query Planner**: prima della ricerca l'LLM genera query aggiuntive basate sugli ultimi modelli nel DB; dopo lo screening degli abstract, raccoglie i nomi di eventuali nuovi modelli 3D concorrenti citati dentro gli abstract e lancia ricerche mirate per trovarli.

**4 filtri**:
- **1 (Harvesting)**: raccolta multi-sorgente nel range di date indicato (`--since` / `--until`).
- **2 (Deduplicazione, Ciclo di Vita e Filtro Lessicale Zero-LLM)**:
  - Unifica i duplicati trovati su più database e controlla `database.sqlite`.
  - **Tracciamento `LIFECYCLE_UPGRADE`**: poiché l'86.7% (85 su 98) dei modelli nel DB della survey erano preprint `arXiv`, se lo Scouter trova lo stesso paper pubblicato su una rivista peer-reviewed con un nuovo DOI, non lo scarta come duplicato ma lo promuove come `LIFECYCLE_UPGRADE` per aggiornare i metadati di pubblicazione.
  - **Filtro lessicale**: scarta via regex (senza spendere chiamate LLM) le review/survey pure, i lavori esplicitamente 2D non radiologici (istologia WSI, retinografia, colonscopia, dermatoscopia) e i paper puramente comparativi (va bene?).
- **3 (Triage dell'Abstract: `YES`, `IMPLICIT_UNKNOWN`, `NO`)**:
  - *Problema*: in un abstract di 150–200 parole spesso gli autori non scrivono la parola *"Dice"* o non specificano se le immagini sono 3D. Usare un filtro rigido sull'abstract causerebbe falsi negativi.
  - *Soluzione*: ogni criterio viene valutato come `YES`, `IMPLICIT_UNKNOWN` (non specificato nell'abstract, ma compatibile $\rightarrow$ **promosso alla lettura del PDF**) o `NO` (violazione esplicita nell'abstract $\rightarrow$ **scartato subito**). Inoltre, se l'abstract cita un dataset 3D noto (`BraTS`, `BTCV`, `AMOS`, ecc.) o se usa solo un modello già esistente *off-the-shelf* (`nnU-Net` standard su coorte clinica), intervengono regole deterministiche di salvataggio o scarto.
  - Classifica i promossi in `TIER_1_GENERALIST` (foundation/promptable/VLM) e `TIER_2_BENCHMARK_SOTA` (architetture 3D specifiche su benchmark pubblici), salvando il ragionamento CoT in `scout_cot_trace.md`.
- **4 (Risoluzione PDF)**: recupera il link diretto al PDF (`arXiv`, `PMC`, o cerca il gemello preprint su arXiv se l'articolo su rivista è chiuso).

### 10.2. test sullo Scouter
1. **Benchmark su 40 Abstract controllati (`20 True Positives` + `20 Hard Negatives`)**:
   - Sia con `Gemini 3.1 Flash-Lite` in cloud sia su Mufasa con `Qwen2.5-14B-Instruct-AWQ`, lo Scouter ha ottenuto **100% Recall (`20/20`)** e **100% Specificity (`20/20`, F1 = 100%)**, con il filtro 2 che ha bloccato 15 dei 20 Hard Negatives e `Qwen-14B` che ha classificato correttamente `20/20` anche la distinzione tra Tier 1 e Tier 2.
2. **Backtest reale su Aprile–Maggio 2025**:
   - Mascherando dal database i modelli usciti da aprile 2025 in poi e interrogando le API live su Aprile–Maggio 2025 (326 paper unici trovati), lo Scouter ha **ritrovato 3 su 3 (100%) i modelli della survey usciti in quella finestra (`MedSAM2`, `CDPDNet`, `TAGS`)**, oltre a individuare nuovi modelli 3D usciti nello stesso periodo (`RadSAM`, `Memorizing SAM`, `Swin-BOB`).
3. **Verifica della catena a 3 filtri (`ScoutAgent` $\rightarrow$ `ExtractionAgent` $\rightarrow$ `AuditorAgent`)**:
   - Testato che i paper "trappola" con abstract vago promossi dallo Scouter come `IMPLICIT_UNKNOWN` (es. abstract generico il cui PDF completo rivela poi l'uso di soli dataset privati, immagini 2D retiniche o assenza della metrica Dice) vengono tutti bloccati al 100% (`REJECTED`) non appena `ExtractionAgent` e `AuditorAgent` leggono il PDF completo.

---

## 11. Domande

1. **Regole del `DatabaseAgent` (Dispute, Single-Agent e Politica Monotona)**:
   - Confermare la gestione attuale:
     - Se `DISPUTED` $\rightarrow$ non inserisco il numero in tabella ma metto solo il commento con i due valori;
     - Se `SINGLE_AGENT_UNCONFIRMED` $\rightarrow$ inserisco il numero solo se la confidenza è alta ($\ge 0.85$) aggiungendo un avviso nel commento;
     - Se il nuovo valore estratto è inferiore al vecchio valore già registrato (`new < old`) $\rightarrow$ non sovrascrivo il numero in `results_best` (lo annoto solo nei commenti). Va bene mantenere questa regola monotona e lasciare `results_primary` intoccabile come baseline storica della survey?
2. **"Valore Massimo" (Peak Performance / Bounding Box) vs Ground Truth**:
   - Visto che chiedendo agli agenti il valore massimo su modelli interattivi estraiamo i valori con Bounding Box (~86%–95%) mentre in `results_primary` della survey erano stati salvati i valori 1-click (~75%–82%) o medie fatte a mano, come preferisce presentare questo aspetto nella tesi?
3. **`SAM-Med2D` (Mistral + Gemini lo scartano vs Qwen lo tiene in warning)**:

4. **Gestione dei nuovi dataset pubblici trovati nei paper**:
   - Quando gli agenti estraggono risultati concordi su vere challenge pubbliche 3D che non erano tra le 33 colonne originali della survey (es. `IMed-361M`, `ULS23`, `CAS2023`, `BraTS2020`, `ATLAS`), il `DatabaseAgent` deve aggiungerle come nuove colonne del database? E per i paper che riportano solo i singoli organi senza media, confermiamo il calcolo della media aritmetica macro nel dataset invece di creare colonne per i singoli organi?
5. **Disponibilità di una GPU più potente (40gb) — CONFIGURATO IL 13 NOVEMBRE**:
   - Attivato l'accesso su `amoglia@10.79.23.96` con ambiente isolato in `/home/amoglia/lpezzoli/AgentForGMMIS` (`--qos=gpuheavy-40`, `--gres=gpu:40gb:1` su scheda `NVIDIA A100-SXM4-40GB`).
   - Collegata tramite symlink la cache dei modelli (`/home/lpezzoli/.cache/huggingface`, 118 GB) e il `venv` per non occupare la quota disco condivisa con Riccardo e Tommaso.
   - Sbloccata l'esecuzione **Dual-Local simultanea fino a 26 GB di pesi** (es. `Qwen-14B + DeepSeek-R1-14B` o `Codestral-22B + Qwen-14B` in parallelo) e aggiunto il supporto ai modelli **32B AWQ** (`Qwen2.5-32B-AWQ` e `DeepSeek-R1-32B-AWQ`), alzando il budget di contesto su Mufasa da `16,000` a `35,000` caratteri (`24k–32k` token).

### Sviluppi rimanenti da completare (To-Do):
- **Benchmark su GPU 40 GB**: `B1.1` completato, `B1.2` in corso (`Job 6348`), `B1.3`–`B3.3` in coda automatica (`scripts/mufasa/queue_feeder_mufasa.sh`)
- **Revisione soglie `VERIFIED` vs `FLAGGED_WARNING` e singleton multi-organo** *(da discutere dopo i benchmark)*
- **Collegamento tra Scouter e Estrazione**
- **Interfaccia e Chatbot**

---

# 13–17 Novembre — Estrazione Granulare Per-Organo, Fix Auditor e Primi Benchmark su GPU 40 GB

## 12. Aggiornamenti Architetturali Implementati (Moduli 1–4)

In seguito all'analisi di fattibilità descritta in [PIANO_IMPLEMENTAZIONE_E_BENCHMARK_40GB.md](PIANO_IMPLEMENTAZIONE_E_BENCHMARK_40GB.md), sono state implementate e testate (`154 passed` in `pytest`) le seguenti modifiche strutturali:

1. **Modulo 1 — Estrazione Granulare Per-Organo e `OrganAliasResolver`**:
   - Creato `OrganAliasResolver` in `src/agent_gmmis/extraction/alias_resolver.py` (~30 forme anatomiche canoniche su BTCV/Synapse/AMOS/FLARE22/BraTS/ACDC/SegTHOR/KiTS/LiTS e sinonimi `Dataset-Level`).
   - Espanso il budget contestuale su GPU A100 40 GB (`35.000` caratteri per manoscritto, `max_output_tokens` alzato a `4096` / `8192` per i modelli reasoning in `src/agent_gmmis/llm/registry.py` e `client.py`), rimosso il tetto di 25 score e introdotto il mandato esplicito per-organo nei prompt (`src/agent_gmmis/extraction/prompts.py`).
   - Chiarita la regola di inclusione 3D su `SAM-Med2D` (dataset radiologici 3D valutati per fette 2D inclusi, escludendo solo benchmark fotografici non-radiologici 2D come Kvasir/ISIC/DRIVE).
   - Aggiornata la deliberazione multi-round in `extractor.py` e `dual_coordinator.py` sostituendo `reinterrogate_granularity_macro` con `reinterrogate_organ_breakdown` (quando un agente estrae solo `Overall` mentre l'altro estrae i singoli organi, chiede in cieco i valori per singolo organo).

2. **Modulo 2 — Fix Auditor & Metriche IAA di Tesi**:
   - In `src/agent_gmmis/auditing/auditor.py`: esteso `KNOWN_3D_DATASETS_AND_MODALITIES` (inclusi `CrossMoDA`, `Totalsegmentator` sub-task, `ATLAS2023`, `ToothFairy`), aggiunto `is_non_3d_photographic_score`, propagato l'override `is_3d` in `audit_consensus`, introdotto il pre-consolidamento multi-prompt per `(dataset, organo, data_budget)` e corretto `normalize_data_budget` per uniformare sia le diciture full-supervision sia gli split standard (`Validation`, `Testing`, `5-fold CV`).
   - Aggiunto il calcolo completo delle metriche di Inter-Agent Agreement in `scripts/run_batch_benchmark.py` e `src/agent_gmmis/auditing/schemas.py`: `Paired IAA` (micro/macro), `Strict Total IAA`, `Exact IAA` ($|\Delta| \le 0.001$), `Inter-Agent MAE`, impatto della deliberazione (`Round-1` $\to$ `Finale`) e stratificazione GT (`MAE_GT | CONCORDANT` vs `NON-CONCORDANT`).

3. **Modulo 3 — Schema SQLite `results_organ_scores` & DatabaseAgent Granulare**:
   - Aggiunta in `src/agent_gmmis/data/db_manager.py` la tabella normalizzata `results_organ_scores` con chiave univoca `(model_id, dataset_name, organ_name)` e guardia preventiva `has_column` in `get_dataset_statistics` / `get_leaderboard_for_dataset` (per evitare errori SQLite su dataset non ancora presenti in `results_best`, come `CrossMoDA`).
   - Aggiornato `DatabaseAgent` per popolare `results_organ_scores` con ogni singolo organo e mantenere `results_best` allineata alla media dataset-level (calcolata deterministicamente dalle componenti per-organo se manca la riga `Overall`).

4. **Modulo 4 — Evaluator Deterministico & Tabelle Tesi (`T1`–`T5`)**:
   - Aggiornato `ExtractionEvaluator` (`src/agent_gmmis/extraction/evaluator.py`) con bridge deterministico media per-organo $\to$ GT.
   - Creato `scripts/generate_thesis_tables.py` (con fix di lettura sia su chiavi JSON flat sia annidate) per generare automaticamente le tabelle Markdown e LaTeX `T1`–`T5`.

---

## 13. Risultati Smoke Test (`Job 6201`) e Benchmark Completo `B1.1` (`Job 6202`, 40 Paper)

### 13.1. Smoke Test e Fix Rapidi (13 Novembre)
Il primo Smoke Test (`Job 6200`) aveva evidenziato due bug su dati reali poi corretti subito:
- `SAM-Med2D` andava in errore (`could not convert string to float: 'CrossMoDA'`) in `db_manager.get_dataset_statistics` perché SQLite valutava `"CrossMoDA"` come stringa letterale in assenza della colonna;
- `Swin UNETR` otteneva `0.0%` di concordanza perché Agent A aveva annotato `data_budget_or_setting = "Validation"` / `"Testing"` mentre Agent B `null`.

Dopo la correzione e un bonus matching `+3` su `table_or_section_ref`, nello **Smoke Test v2 (`Job 6201`)** `Swin UNETR` è passato a **`VERIFIED` (`conf=100%`, `6/6` Paired IAA `100%`, `Inter-Agent MAE = 0.0000`)** e `SAM-Med2D` ha completato con `Paired IAA = 71.4%` (`conf=88%`), popolando regolarmente `results_organ_scores`.

### 13.2. Confronto Pre vs Post su `B1.1` (`Qwen/Qwen2.5-14B-Instruct-AWQ` + `Gemini 3.1 Flash-Lite`, 40 Paper)

| Metrica | Pre-Fix *(29 Ott, 16GB/Overall)* | **Post-Fix (`B1.1` — `Job 6202`, 40GB/Per-Organo)** | Variazione |
| :--- | :---: | :---: | :---: |
| **Stabilità (Paper senza crash)** | `39 / 40` (`Merlin ❌`) | **`40 / 40` (`0` errori)** | **100% completati** |
| **GT Benchmark Recall (`Fixed GT = 196`)** | `90 / 196` (`45.9%`) | **`108 / 196` (`55.1%`)** | **`+9.2 pp` (`+18` benchmark)** |
| **Coverage-Adjusted Score (CAS)** | `0.4511` | **`0.5404`** | **`+19.8%` relativo** |
| **Macro-Averaged Paper MAE (`PRIMARY`)** | `0.0222` | **`0.0212`** | **`-0.0010`** |
| **Micro Paired IAA (`Concordant / Paired`)** | `69.1%` (`141/204`) | **`83.3%` (`219/263`)** | **`+14.2 pp`** |
| **Macro Paired IAA** | `73.8%` | **`88.1%`** | **`+14.3 pp`** |
| **Exact IAA ($|A - B| \le 0.001$)** | `62.7%` micro | **`78.3%` micro / `83.3%` macro** | **`+15.6 pp`** |
| **Deliberation Impact (Round 1 $\to$ Finale)** | — | **`63.4%` $\to$ `88.1%`** | **`+24.7 pp` (`86` voci risolte)** |
| **Dispute residue (`DISPUTED`)** | `63` | **`44`** | **`-30.2%`** |
| **Confidenza Media Certificata** | `87%` | **`93%`** (`39/40` curabili) | **`+6 pp`** |
| **Score granulari `results_organ_scores`** | `0` | **`734` INSERT SQL** | **Nuova tabella popolata** |

* **Report & Tabelle generate**: sincronizzati in `logs/mufasa_runs/20261113_162543_b1_1_qwen14b_gemini_40gb/` e [thesis_tables.md](logs/mufasa_runs/20261113_162543_b1_1_qwen14b_gemini_40gb/thesis_tables/thesis_tables.md) (`.tex` incluso).
* **Punti aperti da discutere in seguito**:
  - `AuditStatus`: alcuni paper con confidenza `96%–100%` (es. `STU-Net`) finiscono in `FLAGGED_WARNING` per una singola disputa secondaria o outlier;
  - `Strict Total IAA` (`30.2%`, `461` singleton): nei paper multi-organo densi (es. 24–41 righe), quando entrambi gli agenti estraggono organi specifici in numero diverso non scatta la re-interrogazione degli organi mancanti.

---

## 14. Completamento Prima Coda Notturna (`18 Novembre`), Scaling Law 32B e Coda Rilancio `v2`

La coda automatica notturna (`queue_feeder_mufasa.sh`) ha completato l'intera sequenza di sottomissioni alle **12:35 del 18 Novembre**, con tre risultati scientifici centrali per la tesi e l'individuazione di 3 bug tecnici sui run 100% locali/sequenziali subito risolti:

### 14.1. Risultati Chiave: Scaling Law (`7B → 14B → 32B`) e "Concordanza $\implies$ Verità"
* **Nuovi Best Assoluti (`Blocco 3 — 32B AWQ`)**:
  - **`B3.1` (`Qwen2.5-32B-Instruct-AWQ + Gemini 3.1 Flash-Lite`)**: **Recall `59.7%` (`117/196`)**, **CAS `0.5836`**, **Micro Paired IAA `89.7%` (`269/300`)**, **Macro Paired IAA `90.7%`**, **Exact IAA `84.0%`**, `Inter-Agent MAE = 0.0101`, **`21/40` paper `VERIFIED`**, **`398` INSERT in `results_organ_scores`** (`+18.7 pp` dalla deliberazione).
  - **`B3.2` (`DeepSeek-R1-Distill-Qwen-32B-AWQ + Gemini 3.1 Flash-Lite`)**: **Recall `57.7%` (`113/196`)**, **CAS `0.5613`**, **Micro Paired IAA `89.1%` (`228/256`)**, **Strict IAA `46.4%`** (massimo assoluto), **`20/40` paper `VERIFIED`**, `0` errori su 40 paper, **`305` INSERT in `results_organ_scores`** (`+24.4 pp`, 80 voci risolte dalla deliberazione).
* **Prova regina sul vecchio punto debole (`B1.4`: `Mistral-7B + Gemini`)**: Macro Paired IAA salita dal vecchio `32.5%` al **`74.9%`** (`+36.0 pp` in deliberazione, `0` errori su 40 paper, `11/40 VERIFIED`, Recall `46.9%`, CAS `0.4581`).
* **Riduzione errore con accordo in doppio cieco ($\text{MAE}_{\text{GT}} \mid \text{Concordant}$ vs $\text{Non-Concordant}$)**:
  - `B1.2 (Codestral-22B + Gemini)`: `0.0004` vs `0.0402` (`94×` più preciso)
  - `B2.1 (Qwen-14B + DeepSeek-14B)`: `0.0055` vs `0.0611` (`11.2×` più preciso)
  - `B1.4 (Mistral-7B + Gemini)`: `0.0085` vs `0.0280` (`3.3×` più preciso)
  - `B1.3 (DeepSeek-14B + Gemini)`: `0.0097` vs `0.0355` (`3.7×` più preciso)
  - `B3.1 (Qwen-32B + Gemini)`: `0.0129` vs `0.0298` (`2.3×` più preciso)
  - `B2.2 (Codestral-22B + Qwen-14B)`: `0.0209` vs `0.0737` (`3.5×` più preciso)
* **Efficacia Fix Retry Gemini**: dopo il push di `client.py` (17 Nov, `max_retries=8`), **`B1.3`, `B1.4` e `B3.2` hanno completato 40/40 paper senza un solo fallimento `503`**.

### 14.2. Bug Risolti dopo la Prima Tornata e Rilancio Coda `v2`
1. **Esportazione `MUFASA_MAX_MODEL_LEN` + stima token conservativa**: i run `dual_simultaneous` (`B1.5`, `B1.6`, `B2.1`, `B2.2`, `B2.3`) avviavano vLLM a `16384` token ma `client.py` leggeva `24576`, causando rifiuto sui ~15–20 paper lunghi (`Recall 12%–24%`). Risolto esportando `MUFASA_MAX_MODEL_LEN` in `scripts/mufasa/run_unified_benchmark_mufasa.sh` e usando `chars_per_token = 1.85` per `context_limit <= 16384` in `src/agent_gmmis/llm/client.py`.
2. **Serializzazione `RepoInspector` JSON**: corretto in `scripts/run_batch_benchmark.py` (causava `TypeError` su `BiomedParse` in `B3.1`/`B2.2` e `MEA M-SAM` in `B2.2`).
3. **Cleanup `VLLM::EngineCore` in `dual_sequential`**: corretto in `scripts/mufasa/run_unified_benchmark_mufasa.sh` (`pkill -9 -P` + `pkill -9 -f "VLLM::EngineCore|vllm serve"` + `sleep 10`) così lo Stage 2 di **`B3.3`** (`Qwen-32B + DeepSeek-32B`) trova tutta la VRAM libera.
4. **Filtro `discover_runs()` su run incomplete**: `src/agent_gmmis/ui/run_analyzer.py` ora salta directory prive di `benchmark_results.json`.
5. **Nota `B2.4` (`GLM-4-9B-Chat`)**: vLLM su Mufasa non supporta `--quantization bitsandbytes`.
6. **Coda `v2` rilanciata su Mufasa (`queue_feeder_mufasa.sh`)**: `B1.2_v2`, `B3.3_v2`, `B2.2_v2`, `B2.1_v2`, `B2.3_v2`, `B1.5_v2`, `B1.6_v2`. Tabelle comparative della prima tornata salvate in [logs/mufasa_runs/thesis_tables_40gb/thesis_tables.md](logs/mufasa_runs/thesis_tables_40gb/thesis_tables.md).



