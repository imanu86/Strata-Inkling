# Inkling 2 Mini: Windows e GPU miste

Checkpoint del 21 novembre 2026. Questo documento distingue le verifiche completate dal lavoro ancora in corso.

## Perimetro

- Repository: https://github.com/imanu86/Strata-Inkling, branch `main`.
- Base: `kuraneko1/Strata`, commit `19e73bd6df9dc6bdf024716ed030eb369bdf16e8`.
- Motore Strata/Kuraneko: non è stato integrato codice Basalt.
- GitHub Actions disabilitato a livello repository su richiesta dell'owner (`actions/permissions.enabled=false`); verificati come `completed/cancelled` tutti i quattro run attivi: 44883794401, 44883794368, 44882714072 e 44882714025. I prossimi push non avviano quei workflow.
- Repository derivato con cronologia conservata. GitHub non ha creato un secondo fork nella stessa rete: la richiesta restituiva il fork Qwen già esistente.
- Il Daily Qwen è preservato. Il suo server sulla porta 8100 è stato fermato; il gateway preesistente non è stato modificato.
- Capacità del contesto: **262144**, anche quando il prompt di prova è corto. KV Q8.
- RTX 3080 Ti 12 GB: UUID `GPU-d945c2f8-983f-1aed-1e7d-5ba1510f13aa`.
- RTX 2080 Ti 22 GB: UUID `GPU-3ec189a9-0fe7-a128-af2c-8df8fb3e15c7`.
- RTX 3060: riservata al desktop Windows, esclusa dai processi di inferenza tramite UUID.

## Modello verificato

File: `D:\ds4_work\inkling2\models\Inkling-2-Mini-UD-Q3_K_XL.gguf`.

- Fonte: `unsloth/Inkling-2-Mini-GGUF`, revisione `ea0c79b2ecd73669e774c8519b79bfabfddbc027`.
- Dimensione: 13.631.778.016 byte.
- SHA-256 dell'intero file verificato: `f72f396d1b479b14d76346df80baa449c8b1f399653bff38fa4764b359a40048`.
- Tokenizer ufficiale: `thinkingmachines/Inkling-2-Mini`, revisione `ed40e7da625f594fc9c4990f5f66be24873c22d7`, salvato in `D:\ds4_work\inkling2\models\tokenizer` con manifest di provenienza.

La variante scelta è il file nella radice, architettura `inkling2`, con 24 layer fisici e 48 visite. `Mainline_Support` usa un'altra convenzione (`inkling2_loop`): non sostituirlo automaticamente. L'errore dell'infografica `blk.24.attn_q.weight not found` non basta a dichiarare corrotti questi pesi. L'hash verifica l'integrità del download, non la qualità numerica del motore.

La quantizzazione è mista: comprende NVFP4 e un tensore Q3_K (`blk.19.ffn_gate_up_exps.weight`). L'assenza di un dispatch GPU Q3_K impediva il caricamento. Il supporto CPU esisteva già; è stato aggiunto il percorso CUDA con un test indipendente di unpacking, dot e geometria degli esperti. NVFP4 usa anche kernel software compatibili con Turing e Ampere.

## Modifiche e verifiche

- Q3_K CUDA: aggiunti dot, `Fmt<11>` e dispatch GU/DOWN/MMVQ. Test CPU generic, AVX e AVX2 passati; test CUDA passati su 2080 Ti SM75 e 3080 Ti SM86. Errori relativi L1 degli esperti sintetici circa 1%, sotto soglia 3%; questa non è una validazione della qualità dell'intero modello.
- Sampler Inkling: `top_k=0` ora conserva l'intero vocabolario; prima applicava un limite nascosto di 256 candidati. Test CPU passato contando oltre 400 ID distinti su 512 estrazioni: un insieme fisso di 256 candidati non può superarlo.
- Frontend: `high=0.9`, `medium=0.7`, `minimal=0.1`; valori numerici finiti 0..0.99. Mapping Qwen preservato. Template ufficiale, blocchi `generation` e tool JSON coperti da fixture locale.
- Preset: temperature 1, top_p 1, top_k 0, min_p 0. Cambiare effort non costituisce un'ottimizzazione del motore e va misurato separatamente.
- Frontend: **73 test CPU passati**. Non è una prova del modello né della correttezza numerica CUDA.
- Tokenizer: il fork riutilizzava erroneamente il pretokenizer DeepSeek. Aggiunto il pattern ufficiale Inkling; parità HF esatta su 511 stringhe, 60 token speciali e prompt ufficiali. Sette test di parità passati; 21 test di regressione eseguiti, con tre saltati per oracle Qwen non disponibile. Pack esportato in `D:\ds4_work\inkling2\pack\tokenizer`.
- Build Windows **completata**: MSVC 19.44, CUDA 12.6, Ninja, CMake 4.4.2, architetture 75 e 86. Test CTest sampler/Q3 CPU 2/2 passati.
- La prima richiesta HTTP ha risposto correttamente `Roma`, ma riportava `finish_reason=length` a 84 token su limite 256. Corretti EOS/DONE: 46 test CPU passati e verifica HTTP reale riuscita con risposta `Roma` e `finish_reason=stop`.

## Baseline misurata

Baseline immutabile: `D:\ds4_work\inkling2\builds\single-gpu-baseline\strata-inkling2.exe`, commit `3835283751e520b13fdd84d556eb359840d9a65d`, SHA-256 `cc5b8be7d199db3121d6408e00e2b8790d644fa6993222d311618f1f76ef32ff`.

Sulla sola 2080 Ti: 3072/3072 esperti in VRAM, circa 17332 MiB usati dal processo/driver secondo nvidia-smi, 4144 MiB liberi dichiarati dal motore dopo il placement. Caricamento del processo circa 6,45 secondi. La 3060 resta sul desktop; la 3080 è libera.

Prompt italiano di 36 token, effort `none`, temperatura 1, top_p 1, top_k/min_p 0, seed 20261121, capacità 262144:

- Due richieste native da 32 token: 60,08 e 62,23 passi di decode/s (31 passi, primo token prodotto dal prefill). Test troncati durante il ragionamento; output identico con seed uguale.
- Richiesta da 64 token: 61,04 passi di decode/s; ancora troncata durante il ragionamento.
- Chat HTTP con limite 256: risposta finale `Roma`, 84 token complessivi, circa 1982 ms HTTP, 569,6 ms prefill e 1333,8 ms decode. La metrica API di 63 token/s conta i token emessi, non i soli forward di decode.

Sono prove brevi a prompt corto: **non** misure a contesto occupato di 256K, né confronti BAB o benchmark di qualità. Effort 0 non ha eliminato tutto il ragionamento in questa richiesta. Nessun ripristino SSD usato in queste prove brevi.

Artefatti: `D:\ds4_work\inkling2\runs\20261121-baseline2080-smoke`, `20261121-baseline2080-answer`, `20261121-http-first-answer.json`, `20261121-q3-parity`. Golden per la parità dello split: `20261121-baseline-logits\chunk64.f32` e `chunk1.f32`, 200058 logits ciascuno, medesimi 36 ID del report smoke e quattro token greedy. I due percorsi chunk hanno prodotto gli stessi quattro token; non si assume parità numerica tra chunk differenti.

## Split e banco aggiornato

Implementato `--layer-split-device N`, con `--split-core-graph` opzionale. Il device principale possiede visite 0..5 e 42..47 più head; il core possiede 6..41. Sul PC: 2080 principale, 3080 core, 3060 esclusa. I trasferimenti usano memoria host pinned portable, senza P2P. Il modello e la lettura GGUF sono condivisi; ogni runtime possiede pesi, KV, conv, stream, workspace e graph pertinenti.

Verifiche:

- Nuovo codice senza split, chunk64/chunk1: 200058 logits e quattro token greedy identici bit per bit alla baseline.
- Split logico sulla stessa 2080, eager e core graph: parità verificata.
- Split su 2080+3080, chunk1: logits identici bit per bit alla baseline, con e senza core graph.
- Sul prefill chunk64, il dual ha una differenza relativa L2 circa 1,015% rispetto al single. Lo stesso single differisce fra chunk64 e chunk1 di circa 0,625%. I percorsi batch/decode usano arrotondamenti diversi; non attribuire staticamente l'intera differenza. Disabilitare dequant wide/store Turing produce gli stessi logits dual: esclusa quella variante come causa. La parità bitwise del prefill batch su GPU diverse **non è dimostrata**.
- Checkpoint RAM/reset con core graph: A→B warm e B cold→A reset danno gli stessi ID. B warm riusa 36 token, cold/reset riusano zero. Artefatto `20261121-state-dual-graph`; non è un test SSD.
- Tutti i 3072 esperti sono residenti. A chunk64 il motore dichiara circa 830 MiB liberi sulla 3080 dopo il placement. Picchi nvidia-smi durante il BAB chunk1: circa 10452 MiB sulla 3080, 7120 MiB sulla 2080; la 3060 resta intorno a 1580 MiB per Windows. Singola 2080: circa 17282 MiB.

Binario split immutabile: `D:\ds4_work\inkling2\builds\split-core-graph-707f20e73\strata-inkling2.exe`, sorgente `707f20e738dd140a1c77210b2c531854daf16019`, SHA-256 `8917afc2e26d0bf6e8c479f49237b120a7ec631b3165424954b371c85a1def93`. La cartella contiene anche `BUILD.json` con provenienza e verifiche.

Il primo BAB da 64 token dava circa +2%; l'utente ha richiesto di passare a **128..1000 token**. Banco successivo: prompt narrativo italiano di 99 token, stessa sequenza di ID, seed 20261121, temperature1/top_p1/top_k0/min_p0/effort none, capacità262144 e chunk1 su tutti i passaggi. Le tre lunghezze cambiano a caldo, senza ricaricare il modello. Il processo si riavvia solo fra B/A/B per cambiare la disposizione GPU.

| Output realmente generato | B1: dual graph | A: sola 2080 | B2: dual graph | Guadagno medio B/A |
|---:|---:|---:|---:|---:|
| 128 | 63,47 t/s | 60,87 t/s | 62,79 t/s | +3,7% |
| 512 | 63,29 t/s | 57,68 t/s | 63,18 t/s | +9,6% |
| 1000 | 62,04 t/s | 57,57 t/s | 61,84 t/s | +7,6% |

Sono passi di decode `(token emessi - 1) / decode_s`; primo token nel prefill. Tutte le risposte raggiungono il limite richiesto (`finish=length`); gli ID sono identici fra A e i due B a ciascuna lunghezza. Artefatti `20261121-long-bab-B1`, `20261121-long-bab-A`, `20261121-long-bab-B2`. Nessun confronto equivalente con HTTP completo o con 256K token già occupati. Il contesto usato cresce da 99 a circa 1099 token; la capacità resta 256K.

Riproduzione: `tools/inkling_smoke.py --gpu-uuid <2080_UUID> --core-gpu-uuid <3080_UUID> --split-core-graph --chunk 1 --lengths 128 512 1000 --prompt <prompt_del_report> --artifact-dir <nuova_cartella>`. Per A omettere core/graph e scegliere il binario baseline immutabile con `--exe`. `tools/inkling_state_parity.py` verifica lo stato con la stessa selezione UUID.

## Chat disponibile

Server dual precedentemente verificato su `http://127.0.0.1:8100/`, API `/v1`. All'avvio del banco lungo del 21 novembre la porta risultava libera, nessun processo Inkling e zero VRAM sulle due GPU di inferenza. Ripristinare il server al termine delle prove con il binario validato. Configurazione precedente: `D:\ds4_work\inkling2\config\inkling-dual-graph.json`. Usa capacità 262144, KV Q8, prefill chunk64, 2080 outer/head e 3080 core. La 3060 rimane esclusa tramite UUID. Default: effort high=0.9, temperature1/top_p1/top_k0/min_p0.

Verifica HTTP reale: domanda 17+25, override effort medium, limite512; risposta `42`, 78 token complessivi inclusi quelli di ragionamento, `finish_reason=stop`. Prefill962,7ms e decode1162,2ms riportati dal motore. È una verifica funzionale separata dal BAB, non un benchmark confrontabile. Artefatto: `D:\ds4_work\inkling2\runs\20261121-http-dual-verified.json`.

Per riavviare quando la porta è libera, dalla cartella `terminal` del repository:

```powershell
python -m serve.server --engine strata --config D:/ds4_work/inkling2/config/inkling-dual-graph.json --host 127.0.0.1 --port 8100
```

Corretto anche il messaggio di avvio: il banner generico attribuiva head e draft all'ultima GPU; Inkling mantiene la head sulla 2080 e qui non usa draft. Verificato il testo con la configurazione effettiva e preservato il banner Qwen. Nessuna modifica ai kernel in questa correzione.

## Confronto con il Daily Qwen

Non abbiamo ancora un confronto Qwen/Inkling sul medesimo banco. La configurazione Daily è `D:\ds4_work\strata\Strata\strata-02131-4c-daily-schermo3060.json`: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`, due peer FUSED, MTP `--spec 4` con draft `rt_it48k`. Il log Daily conferma il kernel persistente per finestra. Il vantaggio MTP va quantificato con accettazioni effettive, non dedotto dalla sola configurazione.

Dai metadata GGUF: Qwen48 layer, hidden2560, FFN640, top10 su512 esperti più shared640; Inkling48 visite, hidden2304, FFN1152, top8 su128. I tre prodotti FFN applicano circa 2,60 miliardi di coefficienti/token in Qwen (shared inclusi), contro 3,06 miliardi in Inkling: circa **18% in più** per Inkling. È solo il lavoro FFN teorico, non i parametri attivi complessivi né una spiegazione quantitativa del rapporto di velocità. Inkling aggiunge loop/control/hyperconnection e attivazione SITU; occorre misurare il costo.

Analisi Nsight Systems2024.5.1 completata in `D:\ds4_work\inkling2\runs\20261121-nsys-inkling\decode-analysis.md` e `.json`. Esclusi loader/prefill, isolati 127 passi decode dopo 99 token prefill chunk1; 128 output greedy identici. Il confine usa la fine degli argmax: include il piccolo readback finale del prefill ed esclude quello dell'ultimo decode.

| Tempo profilato per passo | Sola 2080 | Dual |
|---|---:|---:|
| Finestra decode | 15,849 ms | 14,210 ms |
| Kernel core | 10,144 ms | 7,257 ms |
| Kernel outer/head | 3,648 ms | 4,758 ms |
| Copie GPU, senza sovrapposizione | 0,0084 ms | 0,0356 ms |
| Nessuna attività GPU tracciata | 2,048 ms | 2,159 ms |

Il core risparmia 2,887 ms/passo, outer/head ne assorbe 1,110 ms. La causa di questa regressione non è identificata: clock, power e overhead del profiler restano confondenti, non è un bug dimostrato. Le copie aumentano di 27,2 microsecondi/passo; i gap ai confini core aumentano di 242,2 microsecondi, comprendendo copie e dispatch/sync: non sommarli di nuovo. Le attese CPU nelle API si sovrappongono ai kernel e non sono overhead aggiuntivo puro. Queste tracce non sostengono una diagnosi di rallentamento principalmente dovuto alle copie PCIe.

Una leva motivata per il prossimo giro: CUDA Graph per le due tranche outer. Restano 394 lanci kernel/passo; 0,860 ms/passo di inattività GPU coincide con queste chiamate. È un margine da verificare con parità e BAB, non un guadagno promesso né una spiegazione della regressione outer.

Limiti: una cattura per configurazione, binari diversi, nessun profilo Qwen. Nsight segnala driver CUDA 13.3 non supportato e usa librerie trace 12.6. Le misure sotto profiler non sostituiscono il BAB esterno e non vanno estrapolate ai suoi t/s.

Per questa shell era assente `PROCESSOR_ARCHITECTURE`. Configurare con `$env:PROCESSOR_ARCHITECTURE='AMD64'` dopo aver caricato `Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64`: senza questa variabile CMake rilevava CPU sconosciuta e ggml sceglieva il backend generico. La configurazione corretta rileva x86 e AVX2.

Comando di configurazione:

```powershell
cmake -S engine -B build-windows-inkling -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=75;86' -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-windows-inkling --target strata-inkling2 inkling_q3_parity inkling_sampling_test --parallel 4
```

## Prossimi passi

### Aggiornamento DSpark e contesti lunghi (in corso)

Trovato il draft ufficiale DSpark a tre layer, incluso nel rilascio Thinking Machines. Scaricato e verificato il GGUF Unsloth Q8 pronto: `D:\ds4_work\inkling2\mtp-research\dspark-Inkling-2-Mini-Q8_0.gguf`, 402278560 byte, revisione `ea0c79b2ecd73669e774c8519b79bfabfddbc027`, SHA256 `2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773`. Riferimenti e confronto CPU sui 54 tensori con il BF16 ufficiale in `mtp-research\research-report.md`. Il confronto dei pesi non è una validazione end-to-end del draft.

Kuraneko ha aggiunto DSpark nei commit `b946633f7` e `0222f8159`, successivi alla nostra base. Il porting è in `ee9bc61c2`: preserva split, top_k0 sull'intero vocabolario e campionamento con accettazione p/q. Il draft legge gli input delle visite virtuali 5,23,41 e la norma finale48; i layer fisici sono un'altra numerazione. Build CUDA Windows e prove qui descritte completate; HIP e gli altri backend non sono stati compilati. Il draft resta sperimentale e spento nel preset operativo.

Durante la revisione è emerso un difetto preesistente nel prefill batch oltre il primo giro della cache SWA: scrivere T righe in un ring da 1024 sovrascrive alcuni KV ancora necessari alle prime query del batch. La correzione amplia la capacità a `1024+tmax-1`, lasciando la finestra di attention a 1024. Il BAB precedente a chunk1 non attraversa questo caso.

Test GPU numerico indipendente `inkling_swa_ring`: passati 28 casi per GPU su 2080 e 3080. Il controllo negativo con la capacità precedente riproduce l'errore (fino a 0,06146 nel caso sintetico); la capacità corretta ha errore massimo 2,98e-8. Log in `runs\20261121-swa-ring`.

Per misurare i contesti senza attendere DSpark, creato un candidato separato dai sorgenti committati `2d898a16e`, con la sola correzione della capacità SWA: `D:\ds4_work\inkling2\builds\swa-corrected-baseline-2d898a16e\build\strata-inkling2.exe`, SHA256 `16e96652fdb000e51911c22c27b303009080738628072ea1b9ff4152909d129f`. È una cartella normale, senza branch/worktree; `SOURCE.json` conserva la modifica e gli hash. Build completata, logits sul prompt corto identici bit per bit al precedente dual sia a chunk1 sia a chunk64. Non include MTP.

Banco completato: `tools/inkling_context_bench.py`, 10000/20000/40000 input esatti, chunk64, 512 output, capacità262144, KV Q8, effort none e campionamento T1/P1/K0/minP0. Prompt sintetici diversi all'inizio per impedire il riuso di prefissi; un solo processo per i tre punti. Nessun ripristino SSD. Artefatti `D:\ds4_work\inkling2\runs\20261121-context-baseline-swa-fixed`.

| Input | Prefill | Velocità prefill | Decode di 511 passi |
|---:|---:|---:|---:|
| 10000 | 27,723 s | 360,71 t/s | 57,95 t/s |
| 20000 | 60,087 s | 332,85 t/s | 53,56 t/s |
| 40000 | 137,190 s | 291,57 t/s | 47,70 t/s |

Tutte le risposte hanno generato 512 token e terminato per limite; riuso zero. Non è un banco di qualità. Durante il decode: VRAM stabile a 10546 MiB sulla 3080 e 7174 MiB sulla 2080; RAM disponibile almeno 65,7 GiB. La 3080 registra 1965 MHz a 10K e 1935 MHz a 20K/40K; la 2080 oscilla nelle letture fra 1080 e 1875 MHz. Questo è un possibile fattore da controllare nei confronti futuri, non una causa dimostrata del calo.

Build DSpark completata e test CPU sampling passato. La verifica senza draft ha rilevato una variazione dei logits introdotta dall'aggiunta delle trace ai kernel shortconv. Il codice ora registra la history con kernel separati che copiano soltanto i dati e lascia invariata l'aritmetica originale. Con draft caricato, 200058 logits iniziali sono bit-identici al riferimento, sia a chunk1 sia a chunk64 (`20261121-dspark-first-parity`). Il controllo RAM warm/cold/reset passa (`20261121-dspark-state`); non costituisce una prova SSD.

Primo BAB DSpark spec3, prompt99/chunk64, capacità256K, campionamento T1/P1/K0/minP0 e output realmente generati:

| Output | B1 DSpark | A senza draft | B2 DSpark |
|---:|---:|---:|---:|
| 128 | 41,94 t/s | 61,50 t/s | 45,17 t/s |
| 512 | 33,16 t/s | 63,54 t/s | 33,90 t/s |
| 1000 | 29,15 t/s | 61,24 t/s | 29,34 t/s |

Sono token generati durante il decode, escluso il primo prodotto dal prefill; con MTP non equivalgono al numero di forward. Artefatti `20261121-dspark-bab-B1/A/B2`. Questo candidato è più lento e non va promosso come miglioramento. Inoltre B1/B2 divergono già agli indici5/18/18, pur con seed uguale. La revisione ha trovato un race concreto nell'iniezione KV del draft: due warp riutilizzavano lo stesso array shared per RMS, massimo K e massimo V senza proteggere tutte le letture. La correzione separa i tre array; build completata SHA256 `e859541df5b459556e246a9c58da7ab222b862ea6bf3bc611d78197e82502b4e`, preservata in `builds\dspark-racefixed-e859541d`.

Ripetizione completa `20261121-dspark-racefixed-bab-B1/A/B2`, stessi input, sampling e binario su tutti i passaggi:

| Output | B1 DSpark corretto | A senza draft | B2 DSpark corretto | Accettati / proposti in ciascun B |
|---:|---:|---:|---:|---:|
| 128 | 47,44 t/s | 62,03 t/s | 47,43 t/s | 68 / 172 |
| 512 | 40,47 t/s | 64,92 t/s | 40,68 t/s | 228 / 843 |
| 1000 | 30,94 t/s | 60,83 t/s | 31,12 t/s | 306 / 2073 |

B1 e B2 producono ID identici a tutte le lunghezze, con gli stessi contatori di accettazione. Il difetto di ripetibilità non si presenta più in questo banco. L'MTP resta più lento, quindi rimane opzionale e spento nel preset operativo. Non è ancora dimostrata la parità end-to-end del draft Q8 con l'implementazione BF16 ufficiale.

Banco greedy con1100 input,128 output e spec1/3/7 (`20261121-dspark-width-parity`, ripetuto dopo la correzione in `20261121-dspark-racefixed-width-parity`): tutte le varianti divergono dal decode normale all'indice60. Spec1/spec3/spec3 dopo STOP danno gli stessi128 ID; spec7 diverge da loro più avanti. STOP ferma a16 token, ritorna il contratto nativo `finish=cancel` e il processo resta disponibile; la prima versione del banco attendeva erroneamente `stop`, ora corretto. Il reset dopo STOP conserva gli ID. La differenza dal decode normale resta segnalata, non ignorata: il codice usa proiezioni BF16/FP32 e attention differenti fra T1 e T>1.

Oracolo `20261121-dspark-row0-oracle`: ricostruito il prefisso con1100 input a chunk64 e59 token forzati a T1. Dallo stesso checkpoint di entrambi i runtime, eseguiti T1/2/4/8 senza chiamare draft, inject o commit_verify. T1 predice150759 con margine0,01889 sul secondo150078; T2/T4 mantengono lo stesso argmax, ma RMS dei logits0,05916 e differenza relativa L2=0,6493%; T8 RMS0,06967 e L2=0,7647%. Cambiando soltanto i sette token futuri di T8, la riga0 resta bit-identica. È dimostrata una differenza numerica prima del rollback e la causalità in questa prova; l'inversione del banco completo non è riprodotta dallo stesso stato. L'accumulo di differenze lungo il decode è un'ipotesi, non una causa già isolata. CLI diagnostica `--verify-row0-oracle <8 ID> --oracle-force-t1 59`, inattiva nel normale uso.

### CUDA Graph dei layer esterni: guadagno verificato

Commit `2a6825a1e`, flag opzionale `--split-outer-graph`, solo split a T1 senza draft e con tutti gli esperti esterni residenti. Cattura visite0..5 e42..47; head, embedding, aggiornamento posizione e trasferimenti fra GPU restano fuori. Il preset precedente non attiva il flag automaticamente.

Binario testato: `D:\ds4_work\inkling2\builds\outer-graph-eb41ea7f\strata-inkling2.exe`, SHA256 `eb41ea7f38cd74a046255f579613a17b039b0ed406f725651945e76502de65ee`. Logits200058 bit-identici al riferimento dual corretto sia chunk1 sia64, quattro token greedy identici (`20261121-outer-graph-parity`). Warm/cold/reset con128 output massimi passa (`20261121-outer-graph-state`). Preset `D:\ds4_work\inkling2\config\inkling-dual-outer-graph.json`, con questo binario, capacità256K, KV Q8 e MTP spento. Dopo la diagnosi MTP il server è stato riavviato e verificato via HTTP; dettagli dello stato operativo in fondo.

BAB senza profiler, stesso binario con flag acceso/spento, prompt99, chunk64, output128/512/1000, stesso sampler e seed:

| Output | B1 outer graph | A solo core graph | B2 outer graph | Guadagno medio |
|---:|---:|---:|---:|---:|
| 128 | 75,81 t/s | 61,87 t/s | 70,07 t/s | +17,9% |
| 512 | 72,90 t/s | 64,43 t/s | 71,92 t/s | +12,4% |
| 1000 | 68,52 t/s | 61,84 t/s | 68,42 t/s | +10,7% |

Tutti gli ID emessi e i contatori hit/lookup coincidono fra A/B1/B2. Artefatti `20261121-outer-graph-bab-B1/A/B2`. Il breve128 mostra variabilità fra i due B; evitare di scegliere il solo valore migliore. Nel decode1000: VRAM3080=10546 MiB;2080=7174 MiB in A,7176 nei B; RAM disponibile almeno66,19 GiB. Clock2080 non fissato, intervalli1080..1605 MHz;3080=1950..1965 MHz. Temperature3080 crescono46..56 gradi attraverso il BAB; entrambi i B restano più rapidi. Nessuna regolazione di clock/potenza applicata.

### Diagnosi MTP in parallelo

Su richiesta dell'owner il lavoro MTP continua mentre si consolida il percorso senza draft. Un worker Sol possiede `inkling2_dspark.cuh` e il comparatore CPU; un altro possiede `inkling2_engine.cu`. Build e prove GPU sono coordinate in sequenza; nessun comparatore CPU pesante durante i banchi prestazionali.

Dump diagnostico one-shot opt-in tramite `INKLING2_DSPARK_DUMP_PREFIX` (directory nuova, mai sovrascritta) e `INKLING2_DSPARK_DUMP_CALL` (indice zero). Disattivato senza variabile. Capture `mtp-research\capture-call0-20261121`,92 file/8,8 MB, pos99, anchor200001. La richiesta da128 token produce gli stessi ID del BAB senza dump: non usare i suoi tempi come benchmark.

Comparatore `mtp-research\dspark_stage_oracle.py --no-fold --threads 8`: stadi locali ricostruiti con gli stessi input catturati e pesi GGUF. Max differenza relativa L2 locale0,01463% nel layer0.wo; resto≤0,000399%. Head nativa ricostruita con RMS circa2,7e-6 e distanza totale di variazione≤1,02e-7. Rimuovere soltanto Q8_1 dalla head cambia TV al massimo0,002493; nessun argmax diverso nelle tre proposte. RNN BF16: stato diverso dello0,24..0,30%, effetto sull'accettazione non ancora dimostrato. Call0 è quasi deterministica e riguarda framing iniziale, quindi non rappresenta la prosa lunga. Prossima cattura in fase narrativa.

**Head target verificata Q5_K**,302,21 MiB nelle200058 righe utilizzate. Il draft Q8 non rende Q8 anche la head target. La precedente stima interna467 MiB era errata; non usarla. Un eventuale fold BF16 rank256 occuperebbe97,68 MiB più bias, ma non è implementato e il solo risparmio di traffico non prova un guadagno complessivo. Il riferimento target BF16 originale non è disponibile: un confronto folded sul GGUF dequantizzato è un proxy, non parità ufficiale end-to-end.

Prima profilazione Nsight sampled tramite protocollo interattivo non completata: `20261121-nsys-dspark-sampled512` non contiene una generazione valida. Il processo di quella prova è stato fermato e chiuso; preservata la traccia come tentativo fallito. È in preparazione un wrapper Python per separare il protocollo del server dal forwarding del profiler. Non interpretare la traccia fallita come misura MTP.

1. Profilare il costo MTP e confrontare gli stadi nella prosa; scegliere l'ottimizzazione sulla componente misurata.
2. Approfondire il prefill batch prima di dichiarare parità su GPU miste; non cambiare quantizzazione senza una prova controllata.
3. Ripristino SSD completo da portare prima dei banchi a contesti lunghi. Il checkpoint RAM salva SWA/convoluzione e usa KV globale ancora residente: **non è un ripristino autonomo da SSD**.
4. Valutare i worker esperti separati dopo il consolidamento del percorso a strati; non sono implementati in questo prototipo Inkling.

Binario split immutabile, verifica HTTP dual, banco 128/512/1000 e contesti occupati10/20/40K completati. Prima di una nuova prova GPU usare il server esistente oppure scaricarne il modello, evitando due processi concorrenti. Il multichat resta da eseguire.

La ricostruzione della chat Claude precedente è in `C:\Users\imanu\Documents\Codex\2026-10-07\rip\work\TERRA_RIASSUNTO_SESSIONE_CLAUDE_STRATA_2026-11-21.md`. Il Daily a due server usava worker esperti persistenti, non due repliche HTTP. La versione a strati ripartiva prefill e decode; per Inkling occorre rispettare le 48 visite e il relativo stato KV, senza copiare alla cieca i meccanismi Qwen.

Non sono state aperte PR né inviati messaggi esterni per questo lavoro.

### Profilazione MTP completata e verifiche successive

La cattura valida è `runs\20261121-nsys-dspark-python512`: Nsight avvia Python, che gestisce il protocollo del processo nativo figlio. `command.json` conserva il comando; `capture.nsys-rep`, `capture.sqlite`, `analysis.py/json/md` conservano traccia e analisi riproducibile. La richiesta512 produce gli stessi ID della baseline DSpark, con228 proposte accettate su843. Non usare i tempi sotto profiler come banco prestazionale: la traccia serve a ripartire il costo. La precedente cattura interattiva resta fallita.

- Finestra decode12294,34 ms; verify target8887,3 ms, **72,29%**. GPU attiva in unione9280,85 ms; idle3013,5 ms.
- Head delle843 proposte573,6 ms, **4,67%**. Il fold della head è quindi una priorità bassa, anche se risparmia banda.
- DFlare438,8 ms; intervalli host dopo readback draft1262,0 ms e dopo verify805,4 ms. Senza stack CPU/NVTX non attribuire questi intervalli a una singola funzione.
- Nei kernel verify: BF16 magma sulla2080 vale1193,16 ms/4512 chiamate; GU+DOWN NVFP4 sulla3080 valgono1515,19 ms. I MoE sono già raggruppati e processano più righe: non descrivere un nuovo kernel come introduzione del batching.
- 282 round, con prefissi accettati `{0:145,1:74,2:35,3:28}`. Accettazione condizionata alla proposta raggiunta:48,58%,45,99%,44,44% rispettivamente. Non è corretto attribuire il problema soltanto al terzo token.1125 righe verificate per510 avanzamenti reali.

Capture narrativa `mtp-research\capture-call400-pq-20261121`: pos742, tre proposte e logits P completi prima del commit. Le prime due proposte sono accettate; overlap P/Q99,58%,89,79%,4,08%. Alla terza, P assegna99,705% alla virgoletta tipografica `“`, Q soltanto3,928%; Q preferisce la virgoletta ASCII. È uno scarto ampio in quel punto, non un quasi pareggio numerico. La ricostruzione CPU locale con gli stessi input e pesi quantizzati riproduce gli stadi e la head; cambiare soltanto Q8_1 della head o rounding RNN ha effetto piccolo nei due campioni osservati. Questo non esclude differenze accumulate nell'intero draft.

Completato l'oracolo con pesi draft BF16 ufficiali e matematica FP32. Commit diagnostico `e1b4fe46d`; build immutabile `builds\dspark-history-f857764b`, SHA256 `f857764baf46321b1385584fd4307a1bfe9de122bcac3dabf6fc8c7152b43405`. Build CUDA e CTest sampler passati. `INKLING2_DSPARK_DUMP_HISTORY=1`, insieme a prefix/call, conserva i quattro tap dei soli token committati; gap, limite4096 e restore senza history invalidano il replay. Il percorso normale non alloca né copia questa history.

Capture `mtp-research\capture-call400-history-20261121`, raccolta in `runs\20261121-dspark-dump-call400-history`: tutti i1000 ID coincidono con il BAB corretto;306 accettati su2073 proposti. History completa742 righe, generazione1, errore0, quattro file da6838272 byte. I tempi della cattura diagnostica non sono benchmark.

`mtp-research\dspark_full_cpu_oracle.py` ricostruisce tutte le KV e l'intera testa DFlare+RNN senza quantizzazione draft/KV. Anchor, feature target, ID precedenti e head target Q5_K restano fissati. Replay riuscito in7,03 s su8 thread CPU, CUDA non inizializzata. Le sette mask sono esattamente quelle ufficiali; ricalibrazione head nativa RMS1,13..1,33e-6 e TV≤4,84e-7.

| Proposta | Overlap P/Q nativa | Overlap con pesi BF16 e calcolo FP32 |
|---:|---:|---:|
|0|99,575944%|99,576083%|
|1|89,786025%|89,779205%|
|2|4,080842%|4,357531%|

Gli argmax delle tre Q non cambiano. La terza proposta recupera solo0,277 punti percentuali: eliminare la quantizzazione draft/KV non risolve questo rifiuto. La differenza accumulata degli output DFlare arriva a0,942% L2, dell'hidden normalizzato a1,690%. È una controprova controllata locale, **non** parità con un runtime BF16 ufficiale e non un tasso di accettazione dell'intera corsa. La precisione del target e delle sue feature resta fuori dal confronto. Rapporti e logits controfattuali sono nella capture, con nomi `full-fp32-bf16-weights-oracle.json` e `.q.f32`.

Due tentativi completati, entrambi sperimentali e non promossi:

| Output | BF16 GEMV per riga B1 / A / B2 | Spec1 B1 / spec3 A / spec1 B2 |
|---:|---:|---:|
|128|53,68 /46,02 /50,49 t/s|47,11 /47,42 /49,52 t/s|
|512|38,97 /40,59 /38,50 t/s|39,24 /40,54 /39,45 t/s|
|1000|29,42 /30,94 /29,06 t/s|33,77 /31,07 /33,79 t/s|

Artefatti `20261121-sm75-row-gemv-bab-B1/A/B2` e `20261121-dspark-spec1-bab-B1/A/B2`. GEMV per riga cambia l'aritmetica del verify (input FP32 invece di arrotondato BF16) e il percorso generato: accettati71/212/271 contro68/228/306, quindi il delta temporale non isola soltanto il costo del kernel. Spec1 aiuta il solo punto1000 e resta sotto il percorso senza draft. B1/B2 sono ripetibili in entrambi i banchi, ma non producono gli stessi ID di A.

Completato anche `20261121-sm75-multi-gemv-bab-B1/A/B2`, unico lancio GEMV con più righe:128=50,70/45,93/54,75;512=38,84/40,67/38,97;1000=29,70/31,03/29,79 t/s. B1/B2 e la variante per riga producono esattamente gli stessi ID. L'oracolo T2/4/8 dà le stesse metriche numeriche della variante per riga, senza migliorare la corrispondenza con T1. Anche questa variante perde sui test lunghi. Entrambi i flag BF16 e i relativi dispatch sono stati rimossi dal codice corrente; restano binari, patch e metadati nelle cartelle `builds\sm75-row-gemv-54716e4e` e `builds\sm75-multi-gemv-1026b00f`, senza perdere la riproducibilità.

Analisi statica dei graph verify: il core alla larghezza massima è già catturato; routing MoE e scratch potrebbero supportare altri graph, ma l'intero budget verify non-kernel è circa815 ms su12294 ms. Eliminarlo idealmente tutto varrebbe circa+7,1% complessivo, prima dei costi inevitabili: non risolverebbe il divario osservato. Segnalazione di bookkeeping da approfondire: nel replay core gli incrementi host di `hy_gpu` della cattura non risultano compensati. Non usare quel contatore per dedurre un numero assoluto di operazioni GPU; non è una differenza dei logits.

Upstream controllato fino a `602dbbc59179be8133d5468da77225df4fe176cd`, senza merge. Il nuovo WMMA FlashAttention e MoE BM64 richiedono T>=16, quindi non accelerano direttamente il verify T2/4/8. Il solo cambiamento `rel_table` otto righe/CTA raggiunge quel percorso; beneficio non ancora misurato. Per un futuro port prefill occorre preservare la correzione SWA, la scelta per-device e verificare i nuovi arrotondamenti FP16. Non importare automaticamente nuovi chunk o kernel sul presupposto che migliorino il decode.

### Controllo separato in inglese e stato operativo

Tradotta in inglese la stessa richiesta narrativa, senza sostituire il banco italiano. Prompt80 token, chunk64, capacità262144, output128/512/1000, T1/P1/K0/minP0, seed20261121. Stesso binario immutabile `dspark-history-f857764b` in B1/A/B2. B attiva DSpark spec3; A usa il percorso senza draft con core e outer graph. Il flag outer graph è richiesto in tutti i passaggi, ma il motore lo esclude esplicitamente con DSpark: si confrontano i preset utilizzabili, non un kernel isolato.

| Output | B1 DSpark | A senza draft | B2 DSpark | Accettati / proposti nei B |
|---:|---:|---:|---:|---:|
|128|47,27 t/s|72,39 t/s|47,74 t/s|69 /169|
|512|42,38 t/s|70,66 t/s|42,30 t/s|239 /810|
|1000|38,88 t/s|67,90 t/s|38,88 t/s|448 /1649|

Artefatti `20261121-dspark-english-bab-B1/A/B2`. Tutte le risposte raggiungono la lunghezza richiesta; B1/B2 hanno ID identici. L'inglese migliora il punto1000 rispetto all'italiano, ma DSpark resta nettamente più lento del percorso senza draft sullo stesso prompt. Non estendere questa conclusione a tutti i domini o a tutte le quantizzazioni.

Telemetria dell'intera esecuzione, inclusi caricamento e generazione: nei B picco VRAM3080=10576 MiB e2080=8114 MiB; in A10546/7176 MiB.3060=1579 MiB per Windows. RAM disponibile almeno65,67 GiB; clock non bloccati (2080 fino a1905 MHz nei B,1380 in A). Nessuna compilazione, oracolo CPU pesante o altra generazione concorrente durante il BAB. Le curve complete sono nei rispettivi `telemetry.jsonl`.

Alle10:42 locali del21 novembre avviato il server su `http://127.0.0.1:8100`, API `/v1`, dal preset `inkling-dual-outer-graph.json`. Core e outer graph entrambi eligible,3072/3072 esperti residenti,2080 outer/head e3080 core,3060 esclusa. MTP resta spento nel preset più veloce verificato; il supporto sperimentale e i relativi artefatti sono conservati. Daily Qwen e gateway8787 invariati.

Verifica HTTP effettiva `20261121-http-outer-graph-verified.json`: richiesta con limite256, risposta finale `Roma`,69 token complessivi, `finish_reason=stop`,36 token input,1,745 s HTTP. È uno smoke funzionale con arresto naturale, non un benchmark da69 token. Il server espone capacità262144; non significa che siano stati misurati262144 token occupati. I contesti10/20/40K sopra sono misure precedenti all'ottimizzazione outer graph.

Resta da isolare la causa dell'aderenza limitata del draft al target in questo carico; il replay completo esclude un grande recupero dalla sola rimozione della quantizzazione draft nel campione controllato. La precisione del target, le sue feature e la parità greedy T1/batch restano aperte. Non presentare l'MTP come accelerazione già ottenuta. Gli esperimenti BF16 perdenti sono rimossi dal codice corrente; nessuna PR o comunicazione esterna inviata.

### Ricognizione estesa dei motori e dei fix, 21 novembre

Ricontrollati live Kuraneko `602dbbc59179be8133d5468da77225df4fe176cd` e Niko `3142b66b96e9c0e775e30f620b11b191a1427c2a`. La ricerca comprende anche NInfer-all, Xolotl, llama.cpp mainline e Unsloth, i veri PR Mini di SGLang/vLLM, HF ufficiale e Unsloth. Nessun merge generale, cambio motore o nuovo branch. Le prestazioni degli altri autori non sono misure sulla nostra macchina.

| Fonte primaria fissata | Riscontro e applicabilità |
|---|---|
| [NInfer-all](https://github.com/iamwavecut/ninfer-all/blob/395ce8b85627bcb2d53a1a81623088eab755443c/docs/inkling2.md) | L'autore misura AR88,9→111,4 t/s su3090 raggruppando controller, dividendo depth mix e fondendo routing. Sampled essay DSpark0,99..1,05×; greedy essay1,14× e Python1,41×. IQ4_XS/F16KV/singola3090: non confrontare direttamente quei112 t/s con i nostri70. Inkling non supporta lì KV quantizzata. La nostra implementazione ha già grouping MoE, norm cached e alcuni kernel fusi: occorre confrontare i singoli percorsi. |
| [NInfer controller](https://github.com/iamwavecut/ninfer-all/blob/395ce8b85627bcb2d53a1a81623088eab755443c/src/models/inkling2/executor.cpp#L1804) | Gap locale verificato: Q/K/V T1 eseguono in sequenza tre volte prep/down/up+combine con scratch condiviso. Sono raggruppabili senza cambiare somme o pesi. Il controller MLP dipende dall'attenzione e resta separato. |
| [Xolotl attention multicolonna](https://github.com/Indras-Mirror/Xolotl/commit/9d222fa3c42f04b57329d07537055c7102e58257) e [MMVQ interleaved](https://github.com/Indras-Mirror/Xolotl/commit/17a85465d94d321274002a3a5493e84c37cdf982) | Riuso KV/pesi fra query T1..8. Candidati per verify; per Inkling servono relative bias, sink, SWA e layout Q8 corretti. Non portati né misurati. Il suo DSpark sampled non è un sostituto pronto del nostro rejection sampling. |
| [llama.cpp32704](https://github.com/ggml-org/llama.cpp/pull/32704), `4030448224768f5c83b98e70aef81111b1037039` | Fix aperto per scale NVFP4 perse nei passaggi ripetuti. Audit Strata: ogni blob porta le scale per esperto; tutte le visite riusano il medesimo PhysLayer e tail, copiati integralmente anche fra tier. Non trovato l'analogo del bug; è una verifica statica, non PPL misurata. |
| [llama.cpp32724](https://github.com/ggml-org/llama.cpp/pull/32724), `85fd12a7b4ab60c35039c3362487d2fe1bc6dedc` | Proposta chiusa senza merge: KV/conv delle visite sul device dei pesi condivisi. Il nostro split fa già questa colocazione; i risultati su due T4 non implicano un guadagno aggiuntivo qui. |
| [Unsloth390](https://github.com/unslothai/llama.cpp/pull/390), `8aa2f23f6bbe0b2d23eb3ff676f318f96111d591`; [394](https://github.com/unslothai/llama.cpp/pull/394), `ffa1b96a80b1a4d30be7ee9a663005eb4c8e9915` | Fix ancora aperti: tap per visita, fuse F32, limitare ricorrenza alle proposte richieste e un prodotto head per passo. Il nostro mapping e numero di passi risultano già corretti.394 misura77,3 t/s contro60,3 AR su3090Ti, ma con temperatura0/draft2, non il nostro samplerT1.390 dichiara stessa acceptance con draftBF16/Q8. |
| [SGLang Mini51926](https://github.com/sgl-project/sglang/pull/51926), `b736e624fec767d239a49b581578fe2da45154dc`; [vLLM Mini69028](https://github.com/vllm-project/vllm/pull/69028), `fc1443152c0eb655f5c2762f56b540bebe93b196` | I veri sorgenti Mini erano ancora PR aperti, non bastava il precedente riferimento main flat. Audit tap5/23/41/48, anchor, mask, previous-ID e softcap confermato. Differenza reale nei controller: RMS e down sono arrotondatiBF16 nel riferimento; Strata conservaFP32 e usa fast_tanh/FMA differenti. Impatto sull'accettazione ancora non misurato. |

Riletti i due pesi norm target ufficiali tramite piccoli range HTTP: `embed_norm→token_embd_norm` e `norm→output_norm` sono bit-identici dopo upcast. Anche norm final draft e norm f_rnn sono mappate correttamente. Nessuna head Mini alternativa trovata nelle fonti controllate; i file Unsloth targetQ3 e draftQ8 mantengono gli hash già usati. Mancano confronti pubblici controllati di acceptance Mini Q3/Q4; non confondere KLD dei logits con fedeltà delle hidden feature.

La documentazione [vLLM](https://vllm-project.github.io/2026/11/20/inkling2-mini.html) descrive inoltre `thinkingmachines/TML-d24-varloop-research`, checkpoint distinto addestrato a variare i passaggi. Uno o due passaggi riducono il lavoro, cambiando il compromesso qualità/velocità; DSpark richiede i tre passaggi standard. Non è corretto tagliare i loop del nostro checkpoint fisso. È una strada architetturale separata, non implementata né scaricata. Il [produttore](https://thinkingmachines.ai/news/inkling-2-mini/) riporta5,4B parametri attivi e48 visite da24 layer memorizzati: la dimensione su disco non misura il lavoro per token.

Esaminati anche TokenSpeed, inkling-turbo e inkling2-exl3-spark: le fonti trovate riguardano rispettivamente soprattutto hardware datacenter/Inkling precedente, attention con limiti specifici di piattaforma, e Inkling2Small su dueDGXSpark. Nessuna accelerazione Mini SM75/86 pronta dimostrata da quelle pagine. La ricerca non costituisce un censimento esaustivo di ogni fork.

### Prove derivate dalla ricognizione

Il [percorso BF16 Tensor Core di Niko](https://github.com/Niko1221/Strata/commit/4dbc4c8b82b24ae5dbbba256fde4ff1e3d4a61ec) era già compilato e opt-in per SM75 con `STRATA_BF16_TC=1`. A lascia la variabile assente, preservando il default SM86; B1/B2 la impostano1, sempre nuovi processi. Binario immutabile `dspark-history-f857764b`, stessa configurazione italiana spec3 del banco precedente.

Il controllo chunk1/T1 conserva tutti i200058 logits. Il prefill64 cambia i logits rispetto al riferimento dual corretto: L2 relativa0,52565%, quattro token greedy iniziali invariati. Nell'oracolo1100+59 T1 forzati cambia anche l'argmax; la coda futura resta causalmente indipendente. Nessun nonfinito. Non è un'ottimizzazione bitwise.

| Output | B1 BF16_TC | A default | B2 BF16_TC | Accettati B / A |
|---:|---:|---:|---:|---:|
|128|47,34|47,48|47,56 t/s|69 /68|
|512|35,35|40,52|35,39 t/s|185 /228|
|1000|28,43|31,09|28,49 t/s|249 /306|

Artefatti `20261121-bf16tc-parity`, `20261121-bf16tc-row0-oracle-B1/A`, `20261121-bf16tc-bab-B1/A/B2`. Le traiettorie e l'accettazione cambiano, quindi il delta non isola il solo costo GEMM. Il risultato operativo peggiora: flag non promosso. Nessuna modifica di sorgente necessaria per questo tentativo.

Completato e pubblicato su main il candidato `--batch-qkv-control-t1`, commit `6ff2e48e8`: tre kernel raggruppano Q/K/V al posto di nove, mantenendo i corpi aritmetici Strata. Non copia codice NInfer; ne usa l'idea come riferimento.24 visite con messaggio producono144 nodi graph in meno per forwardT1, non144 lanci CPU evitati. Flag impostato prima di init/capture, scratch separato preallocato, default/MLP/T>1 invariati. Revisione statica indipendente passa, inclusi alias SWA e device ownership.

Build immutabile `builds\qkv-control-0dc78e64`, SHA256 `0dc78e6451c3047665319b136217320005badddfe32e642f65d5b88fa94f736f`. Build CUDA e CTest `inkling_sampling_test` passati; sette test Python del banco passati. Test standalone `engine/tests/test_inkling2_qkv_control_t1.cu`:108 casi su ciascuna GPU, prep/down/output identici bitwise eager e tre replay graph, eps1e-6, scratch avvelenati con NaN e controllo nonfiniti. Tutti i200058 logits sono identici al riferimento **dual** corretto, sia chunk1 sia64, con flag spento e acceso. Non usare il confronto storico single-GPU riportato dal vecchio helper come esito di questa parità. Artefatti `20261121-qkv-control-unit`, `-parity-A/B`, `-state`.

Stato RAM warm/cold/reset passato, riuso36 token contro0 nel cold e reset; non è ripristino SSD. BAB con99 token input, T1/P1/K0, seed20261121, core+outer graph, senza draft, produce tutti gli ID identici in A/B1/B2 e tutte le lunghezze richieste:

| Output | B1 QKV raggruppati | A default | B2 QKV raggruppati | Delta media B / A |
|---:|---:|---:|---:|---:|
|128|74,18 t/s|75,62 t/s|76,02 t/s|−0,68%|
|512|74,69 t/s|72,60 t/s|74,32 t/s|+2,62%|
|1000|71,42 t/s|69,33 t/s|71,94 t/s|+3,39%|

Artefatti `20261121-qkv-control-bab-B1/A/B2`, riepilogo verificabile `20261121-qkv-control-summary.json`. Velocità calcolate come `(generated−1)/decode_seconds`, separatamente dai tempi completi nativi; nessuna misura HTTP in questa tabella. Guadagno piccolo e coerente nei due B lunghi, nessun vantaggio chiaro a128; non generalizzarlo a tutti i prompt o a256K occupati. Picco VRAM3080=10546 MiB,2080=7176 MiB,3060=1579 MiB; RAM disponibile almeno65,95 GiB. Clock non bloccati, nessuna build o prova concorrente. La capacità resta262144, il contesto effettivo di questo banco è99.

BAB QKV con DSpark spec3 completato sul medesimo prompt:128=47,82/46,82/46,26;512=40,38/40,71/40,41;1000=30,88/31,13/30,84 t/s, ordine B1/A/B2. Tutti gli ID coincidono; acceptance invariata68/172,228/843,306/2073. Nessun vantaggio nel verify MTP; il raggruppamento T1 rimane un miglioramento del preset senza draft. Artefatti `20261121-qkv-mtp-bab-B1/A/B2` e `20261121-qkv-mtp-summary.json`.

Completata anche la verifica derivata dai veri sorgenti Mini: due arrotondamenti BF16 del controller (`casts`) e, separatamente, cast più proiezioni up separate, `tanhf` e fusione FP32 senza FMA (`official-merge`). Quest'ultima non isola soltanto FMA. Non è una conversione dell'intero target in BF16 e conserva l'ordine di riduzione Strata. Default invariato e batching QKV scavalcato nei due opt-in.

Snapshot `builds\controller-precision-6ff2e48e8-20261121`, exe SHA256 `55bd1c3d5c034bddcb426c691990ac07157b64f21f735569f28956b07f514d27`. Build CUDA e CPU sampler passati; review indipendente senza blocker. Test GPU su SM75/86 passa cast RNE anche ai limiti e10 casi forma/alias con tre replay graph. Tutti i200058 logits del default rimangono identici al riferimento dual a chunk1/64. Oracolo1100+59T1: nessun nonfinito, coda futura indipendente; L2 relativa T4/T1 passa da0,6493% native a1,1696% casts e0,7826% official-merge. Non risolve quindi la differenza fra forme del verify.

| Modalità B | Output | B1 / A native / B2, t/s | Accettati / proposti B | Accettati / proposti A |
|---|---:|---:|---:|---:|
|casts|128|46,00 /44,05 /45,08|66 /177|68 /172|
|casts|512|32,81 /40,52 /32,76|160 /1047|228 /843|
|casts|1000|28,47 /30,78 /28,50|250 /2241|306 /2073|
|official-merge|128|49,14 /48,11 /47,17|71 /163|68 /172|
|official-merge|512|35,33 /40,55 /35,24|190 /960|228 /843|
|official-merge|1000|28,97 /30,84 /28,99|266 /2196|306 /2073|

Stesso banco italiano, spec3, seed20261121, capacità262144, output128/512/1000. Tutte le lunghezze raggiunte; B1/B2 identici, B diversi da A. I cast aggiungono kernel: i tempi non isolano il solo effetto numerico. Anche l'accettazione però peggiora nei due test lunghi, sulle traiettorie prodotte. Nessuna promozione; patch e test conservati nello snapshot, sorgente operativo ripristinato a `6ff2e48e8` dopo verifica di recuperabilità e hash. Artefatti `20261121-controller-precision-unit`, `-native-parity`, `-row0-native/casts/official-merge`, `20261121-controller-casts-bab-B1/A/B2`, `20261121-controller-official-merge-bab-B1/A/B2`, riepilogo `20261121-controller-precision-summary.json`. Non dedurre da questa prova che la fedeltà dell'intero target BF16 sia irrilevante.

Audit dell'adattività NInfer: una policy può scegliere K0/1/2/3 dalla storia senza cambiare rejection sampling p/q. Tuttavia il transformer draft continua a elaborare block8 anche con K1; si risparmiano head/Markov e verify, non tutto il draft. Oggi il graph verify copre solo la larghezza massima: diminuire K a caldo può ricadere su eager. Il fallback con DSpark attivo conserva inoltre cattura/iniezione feature e costa più dell'AR puro. Un confidence floor non deve filtrare i token già estratti conservando la q originale: introdurrebbe bias. Servono cache graph per larghezza e misura del costo completo prima di implementare la policy. Nessuna adattività introdotta in questa fase.

### Baseline llama.cpp richiesta dall'owner

Prima di questa fase non era stato eseguito un confronto reale con llama.cpp sulla macchina. Il runtime compatibile individuato è la [release ufficiale Unsloth b12588-mix-c0c3de8](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8), sorgente `883286076efe5b85fddb31d9cc7035a6b698ba9b`, pacchetto Windows CUDA12 older con SM75/86. Archivio SHA256 `7adbcfbb6b4980632885767b914468e2c1a560b485ce4be2dfbf432458508138`; exe `llama-baseline/prebuilt-b12588-cuda12-older/llama-server.exe`, SHA256 `bcbc39f0a6f31f5de251be7e21d5e45bdeb5e80aeac33568e086c8a80b35793a`. Provenienza, sorgente, licenza e comandi conservati in `D:/ds4_work/inkling2/llama-baseline/PROVENANCE.json` e `PREPARED.txt`. Nessuna build locale o modifica del GGUF.

Il nuovo head upstream della PR32704, `608e3703fd15ecb989d96747b28fa98320c5cc4f`, richiede metadati `inkling2_loop` diversi dal nostro GGUF legacy `inkling2`,48 visite/24 blocchi tensor. Non è stato usato per un benchmark caricando male i pesi. Il runtime Unsloth scelto incorpora i cambiamenti389/390 e carica le scale prima di condividere i layer; gli stati KV/conv delle visite seguono il proprietario fisico.

Il vero `tensor split` è esplicitamente rifiutato per Inkling da questa revisione. `row` compare nella CLI, ma il backend CUDA non implementa il buffer split: prova reale `20261121-llama-row-unsupported`, avvio fallito con `device CUDA0 does not support split buffers`. Non riportare un numero di prestazioni per row e non equipararlo a tensor parallelism. Anche l'help che colloca KV/intermediates su `--main-gpu` in row è obsoleto in questo sorgente.

Harness esterno `D:/ds4_work/inkling2/llama-baseline-bench.py`, con versioni precedenti salvate e hash. Usa `/completion`, gli stessi99 ID interi del banco Strata, sampler T1/P1/K0/minP0, seed20261121, penalità neutre, nessun draft, output128/512/1000, un solo slot e `cache_prompt=false`. Verifica99 input, riuso0, capacità effettiva262144 e ID output nel vocabolario non padded200058. Il `predicted_per_second` nativo llama usa già `(predicted_n−1)/predicted_ms`; il numeratore coincide con Strata, senza garantire identici confini del timer. Tempi HTTP conservati separatamente. KV entrambe a8 bit ma rappresentazione diversa: llama Q8_0 scala per32 valori, Strata Q8 perhead128.

Primo confronto `20261121-llama-layer-bab-B1/A/B2`: llama layer65/35, ordine2080/3080, main0, chunk64, fit off e flash attention on. B1/B2 hanno output identici e raggiungono tutte le lunghezze. Velocità llama B1=17,36/15,22/14,71 e B2=15,37/15,14/14,63 t/s; Strata A=75,51/75,13/72,39. B1 è un pilot con telemetria ogni0,5s; A/B2 ogni2s. Non presentare questo primo giro come controllo perfettamente uniforme della telemetria.

Quel65/35 distribuisce le25 unità fisiche, incluso l'output: fisici0..16 sulla2080,17..23+head sulla3080. Ben33 delle36 visite del core rimangono sulla2080. Non è il placement Strata, e le prestazioni non sono un limite teorico delle schede. Il codice CUDA supporta NVFP4/Q3_K anche su SM75/86; non è emerso un fallback CPU obbligato per quei pesi. Input embedding CPU previsto, head sulla3080. I clock osservati sono variabili; nessun lock o modifica della potenza.

Provata una seconda configurazione stock con `GGML_CUDA_DEVICES=3`: device virtuali2080/3080/2080, `--device CUDA0,CUDA1,CUDA2 --tensor-split 6,12,7`. Riproduce fisici0..5 e18..23+head sulla2080,6..17 sulla3080, con stato colocato. Non coinvolge la3060. A chunk64 fallisce la riserva dei buffer di calcolo sulla3080:3814 MiB con pipeline,3257,53 MiB nel retry senza pipeline. A chunk32 fallisce ancora:1908,65/1629,23 MiB. Artefatti `20261121-llama-virtual-layer-bab-B1` e `20261121-llama-virtual32-bab-B1`: sono avvii falliti, nessun dato prestazionale.

Il sorgente materializza bias relativi F32 di dimensione `n_kv × T ×24`; alla riserva262144 un buffer vale1536 MiB aT64,768 aT32,384 aT16. La catena get_rows/cont può richiederne due: spiega plausibilmente gran parte dello scratch osservato, senza essere una misura diretta dell'allocatore. Con KV Q8 il fastpath speciale banded è escluso e viene usata l'attenzione ordinaria. A chunk16 il caricamento riesce dopo disattivazione automatica della pipeline, mantenendo256K/Q8. Il BAB `20261121-llama-virtual16-bab-B1/A/B2` è completato: usa16 in entrambi i motori, telemetria2s e identici altri parametri sopra.

| Output effettivo | B1 llama, decode | A Strata, decode | B2 llama, decode | Tempi completi B1 / A / B2 |
|---:|---:|---:|---:|---:|
|128|14,06 t/s|78,11 t/s|14,72 t/s|10,250 /2,531 /9,844 s|
|512|14,48 t/s|76,37 t/s|14,72 t/s|36,234 /7,219 /35,625 s|
|1000|14,65 t/s|72,60 t/s|14,66 t/s|69,375 /14,312 /69,500 s|

Tutte le risposte raggiungono il limite; B1/B2 hanno ID identici. I tempi completi includono HTTP per llama e protocollo nativo per Strata: non sono una misura isolata del server HTTP. Riepilogo `20261121-llama-versus-strata-summary.json`. Il chunk16 serve a rendere possibile il confronto llama; non viene promosso come migliore prefill Strata rispetto al64 operativo.

Llama arriva a12080 MiB sulla3080 e9668 MiB sulla2080, con margine VRAM minimo sulla3080. Strata nello stesso banco raggiunge10526/7150 MiB.3060=1579 MiB per Windows, esclusa da entrambi. RAM disponibile minima67,45 GiB nei B,65,91 GiB in A. I clock sono variabili e differiscono fra carichi; i range completi sono nel riepilogo e nelle curve. Nessun blocco clock, build o altra generazione concorrente. La pipeline llama viene disabilitata automaticamente in entrambi i B; questi numeri descrivono le configurazioni effettivamente caricabili, non un limite teorico di llama o delle GPU. Non misurati200 t/s né vantaggi MTP in questa baseline.

### Stato finale di questa verifica

Sorgente su main: kernel QKV verificato `6ff2e48e8`; esperimenti controller precision rimossi e recuperabili dai rispettivi snapshot. Il banco Strata accetta ora anche chunk16/32, mantenendo64 come default; sette test Python passati. Nessuna PR o messaggio esterno inviato.

Server riavviato come processo separato nascosto, tramite `Start-Process`, su `http://127.0.0.1:8100`, preset nuovo `D:/ds4_work/inkling2/config/inkling-dual-qkv-graph.json`. Usa il binario immutabile QKV `0dc78e64`, chunk64, capacità262144, Q8, core+outer graph, MTP spento. Il precedente preset outer graph è conservato. Health conferma `loaded=true`; richiesta HTTP con limite256 si arresta naturalmente con `Roma`,36 token input e69 output,1,900 s completi: smoke funzionale, non benchmark breve. PID e log in `runs/20261121-http-qkv-graph-operational/launch.json`, esito in `verified.json`. Il server foreground precedentemente osservato alle10:42 non era più presente al controllo iniziale di questa fase; causa non accertata. Daily Qwen e gateway8787 invariati.
