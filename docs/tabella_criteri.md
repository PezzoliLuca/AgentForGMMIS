### 1. Criteri di Screening e Inclusione (Gatekeeping)
Il paper viene valutato rispetto a 4 criteri di inclusione:
* **(`is_3d`)**: ammessi esclusivamente task di segmentazione 3D.
  * *Regola di Override*: se l'estrattore indica `is_3d = False`, l'Auditor verifica se il dataset appartiene alla lista dei benchmark (`ACDC`, `BraTS`, `BTCV`, `AMOS`, `KiTS`, `LiTS`, `FLARE`, `TotalSegmentator`, `Synapse`, `ISLES`, `SegTHOR`). In caso positivo, l'Auditor sovrascrive il flag impostando `passed_screening = True`.
* **Modalità Radiologica (`is_radiological`)**: ammesse solo modalità radiologiche (CT, MRI, PET, Ultrasuoni, RX). 
* **Dataset Pubblici (`is_public_dataset`)**: ammessi esclusivamente benchmark pubblici.
* **Presenza Dice (`has_dice_metric`)**: obbligatoria la presenza del Dice Similarity Coefficient (DSC); scarto se il lavoro riporta unicamente altre metriche (IoU, ASSD, HD95) senza Dice.
* **Esito**: se anche uno solo dei criteri non è soddisfatto (e non sanato dall'override), lo stato viene impostato a `REJECTED` e la confidenza a `0.0`.

---

### 2. Vincoli Fisici, Numerici e Architetturali
Controlli su intervalli scalari, coerenza computazionale e compatibilità hardware:
* **Intervallo della Metrica**:
  * Regola di ammissibilità fisica: $\text{Dice} \in [0.0, 1.0]$. alrtrimenti esito `REJECTED`.
  * plausibilità: se $\text{Dice} \ge 0.999$, viene generato un `WARNING` per valore troppo alto
* **Range Parametri e Famiglia di Vision Backbone**:
  * Limiti assoluti di plausibilità: $0 < \text{Parameters (M)} \le 50.000\text{ M}$. Valori negativi, nulli o superiori a 50B vengono rifiutati (`REJECTED`).
  * Regole per Vision Backbone:
    * Se contiene `vit-b`, `vit base` o `vit-base`: range ammesso $[30.0, 350.0]\text{ M}$.
    * Se contiene `vit-l`, `vit large` o `vit-large`: range ammesso $[150.0, 700.0]\text{ M}$.
    * Se contiene `vit-h`, `vit huge` o `vit-huge`: range ammesso $[350.0, 1500.0]\text{ M}$.
    * Valori fuori intervallo generano un'anomalia di tipo `WARNING`.
* **Regola di Coerenza Hardware - VRAM**:
  * Se la stringa risorse dichiara una singola GPU e la VRAM estratta risulta $> 120\text{ GB}$, viene generato un `WARNING` per superamento del limite.
  * Valori di VRAM totale $\le 0\text{ GB}$ o $> 20.000\text{ GB}$ sollevano un `WARNING`.

---

### 3. Prevenzione della Confusione di Baseline e Auto-consistenza
Controlli per intercettare allucinazioni e scambi di riga nelle tabelle:
* **Verifica Competitor di Baseline**:
  * L'Auditor mantiene una lista deterministica di modelli di riferimento: `KNOWN_BASELINE_MODELS = {"nnu-net", "nnunet", "unetr", "swin-unetr", "swin unetr", "swinunetr", "transunet", "nnformer", "segresnet", "v-net", "u-net", "unet", "d-lka net", "med-sa", "sam3d"}`.
  * Se il modello estratto è presente in lista, l'Auditor esegue una verifica di compatibilità a 3 livelli:
    1. *Corrispondenza nel titolo*: verifica di sottostringa esatta o normalizzata nel titolo del paper.
    2. *Allineamento per acronimo o iniziali*: i token del modello corrispondono a prefissi delle parole del titolo o alle iniziali di parole consecutive.
    3. *Pattern esplicito di Self-Nomination*: presenza nell'abstract o nelle prime 3 pagine del nome tra parentesi subito dopo il titolo, oppure frasi quali `(?:we propose|introduce|present|develop|our model|method|framework) ... [nome_modello]`.
  * Se il modello è un noto competitor e non supera nessuno dei 3 livelli di compatibilità, l'estrazione viene respinta (`REJECTED`).
* **Rilevamento Allucinazioni**:
  * Se lo stesso identico punteggio Dice si ripete identico su $\ge 4$ combinazioni distinte di `(dataset, task)` nelle tabelle, l'Auditor identifica un'allucinazione di ripetizione e respinge l'estrazione (`passed_sanity = False`).
* **Verifica Punteggi Duplicati**:
  * I punteggi vengono raggruppati per quintupla: `(dataset_canonico, organ_or_task, prompt_type, table_ref, data_budget)`.
  * Vengono escluse le partizioni sperimentali note (stringhe contenenti `scarce`, `label`, `%`, `few`, `semi`, `ablation`, `round`, `shot`, `split`).
  * Regola di contraddizione: se per condizioni identiche risultano esattamente 2 valori distinti con differenza $|\Delta| > 0.03\text{ DSC}$, viene sollevato un `WARNING`.
* **Righe Baseline Non Etichettate**:
  * Se una riga di punteggio appartiene a un modello della lista baseline ma non ha il flag `is_baseline_comparison = True`, viene generato un `WARNING`.

---

### 4. Rilevamento Outlier su Distribuzione Storica (Knowledge Base SQLite)
Per ogni benchmark estratto, l'Auditor interroga il db (`min`, `max`, `mean`, `count`) dei modelli già presenti nel database. Vengono applicate 3 regole deterministiche:
1. **Pancreas**: se il nome del dataset o del task contiene `"pancreas"` e il punteggio è $\text{Dice} > 0.92$, viene generato un `OutlierDetectionResult`.
2. **Superamento Record**: se $\text{Dice} > \text{max\_storico} + 0.04$, il valore viene registrato come outlier.
3. **Minimo Anomalo**: se $\text{Dice} < 0.20$ ma distribuzione storica con $\text{min\_storico} > 0.40$, viene registrato come outlier.

---

### 5. Gestione Dati Mancanti e Macchina a Stati per le Deleghe
Quando un paper non presenta tabelle numeriche (o presenta solo allucinazioni da template):
1. **Scansione Strutturale**: l'Auditor analizza il testo del PDF cercando la presenza combinata di riferimenti a grafici (regex: `figure`, `fig`, `bar chart`, `radar plot`, `box plot`, `performance curve`) e di menzioni di metriche (`dice`, `dsc`, `mean dice`).
2. **Transizioni della Macchina a Stati**:
   * **Stato `NEEDS_REPO_INSPECTION`**: se non sono presenti tabelle numeriche nel PDF, ma è stato estratto un URL GitHub (`code_repository_url`) non ancora ispezionato. L'Auditor delega all'agente `RepoInspector` per ricercare file `.json`, `.csv`, `.tsv` o tabelle nel `README.md`.
   * **Stato `NEEDS_OUTREACH`**: se non sono presenti tabelle numeriche, non esiste un repository (o l'ispezione non ha prodotto dati), ma sono state individuate figure/grafici con metriche Dice. L'Auditor delega all'`OutreachAgent` per preparare la bozza di email da inviare agli autori.
   * **Stato `REJECTED` (`MISSING_TABLE_NO_DATA`)**: scatta se non sono presenti né tabelle numeriche, né grafici con metriche, né repository.

---

### 6. Arbitraggio Dual-Agent
L'estrazione viene eseguita in parallelo da due agenti indipendenti:
* **Risoluzione Canonica**: i nomi dei dataset estratti dai due modelli vengono normalizzati sui nomi standard tramite `DatasetAliasResolver` (es. *"Beyond The Cranial Vault"* e *"Synapse"* ricondotti entrambi al nome canonico *"BTCV"*).
* **Accoppiamento Punteggi**: l'Auditor accoppia le righe dei due agenti calcolando un punteggio di affinità basato su dataset canonico, equivalenza dei target anatomici (gestendo sinonimi standard: `overall`, `mean`, `average`), prompt type e data budget.
* **Regole di Classificazione dei Punteggi**:
  1. `CONCORDANT`: entrambi i modelli hanno estratto il punteggio e la differenza assoluta rispetta la soglia $|S_A - S_B| \le \epsilon$ con $\epsilon = 0.01$ il valore viene unificato come media: $S = (S_A + S_B) / 2$.
  2. `DISPUTED`: la differenza supera la tolleranza ($|S_A - S_B| > 0.01$) oppure un modello ha estratto un valore e l'altro no. Il punteggio viene impostato a `None` e viene sollevato il flag disputed.
  3. `SINGLE_AGENT_UNCONFIRMED`: benchmark identificato da uno solo dei due agenti. Viene conservato a database ma contrassegnato con peso di confidenza ridotto. (questo diventerà poi il loop di cui si parlava)
* **Consenso sui Parametri Architetturali**:
  * I parametri $P_A$ e $P_B$ sono considerati concordanti se la differenza rispetta il limite: $|P_A - P_B| / \max(P_A, P_B) \le 0.05$. In caso positivo, il valore risolto è la media; altrimenti lo stato è marcato come `DISPUTED`.
* **Metriche di Accordo**:
  * **Inter-Model Dice MAE**: media di $|S_A - S_B|$ calcolata sulle sole coppie di benchmark estratte da entrambi gli agenti.
  * **Concordance Rate**: rapporto tra numero di punteggi concordanti e numero totale di coppie di benchmark comuni.

---

### 7. Calcolo Deterministico della Confidenza e Stati di Certificazione
La confidenza parte da $1.0$ e subisce sottrazioni:
$$\text{Confidence} = 1.0 - (\text{Anomalie Critiche} \times 0.50) - \text{Penalità Warning} - \text{Penalità Outlier} - \text{Penalità Dispute}$$
* Penalità Warning: $-0.05$ per ciascun warning (cappata a un massimo di $-0.35$).
* Penalità Outlier: $-0.05$ per ciascun outlier (cappata a un massimo di $-0.20$).
* Penalità Discrepanze Survey: $-0.05$ per ciascuna discrepanza (cappata a max $-0.15$).
* Regole Dual-Agent:
  * Bonus Concordanza: $+0.03$ per ogni benchmark concorde (fino a un massimo di $+0.20$).
  * Penalità Dispute: proporzionale alla percentuale di dispute, fino a un massimo di $-0.25$.
  * Penalità singleton: $-0.01$ per ogni score presente su un solo agente (cappata a $-0.08$).
  * Disputa sui criteri di screening: $-0.15$; 
  * Disputa sul nome del modello: $-0.10$.
* Limiti: se è presente un'anomalia critica o lo screening è fallito, la confidenza non può superare $0.40$ (e scende a $0.0$ se fallisce lo screening).

Gli stati finali della macchina a stati dell'Auditor sono:
* `VERIFIED`: screening superato, vincoli fisici e architetturali rispettati, zero anomalie critiche, zero warning, zero outlier e zero dispute.
* `FLAGGED_WARNING`: screening e vincoli fisici rispettati, ma presenza di anomalie non bloccanti (warning su VRAM/parametri, outlier statistici o dispute di punteggio tra i due agenti).
* `NEEDS_REPO_INSPECTION`: assenza di tabelle numeriche nel PDF, ma presenza di un repository di codice GitHub da ispezionare.
* `NEEDS_OUTREACH`: assenza di tabelle numeriche e di codice sorgente, ma presenza di grafici/figure contenenti metriche; bozza di email.
* `REJECTED`: violazione di uno o più criteri di inclusione, violazione dei vincoli fisici (es. Dice fuori da $[0, 1]$), confusione con baseline competitor, template allucinato o totale assenza di dati quantitativi.