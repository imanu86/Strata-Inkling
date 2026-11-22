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

### Stato finale della verifica llama

Sorgente su main: kernel QKV verificato `6ff2e48e8`; esperimenti controller precision rimossi e recuperabili dai rispettivi snapshot. Il banco Strata accetta ora anche chunk16/32, mantenendo64 come default; sette test Python passati. Nessuna PR o messaggio esterno inviato.

Server riavviato come processo separato nascosto, tramite `Start-Process`, su `http://127.0.0.1:8100`, preset nuovo `D:/ds4_work/inkling2/config/inkling-dual-qkv-graph.json`. Usa il binario immutabile QKV `0dc78e64`, chunk64, capacità262144, Q8, core+outer graph, MTP spento. Il precedente preset outer graph è conservato. Health conferma `loaded=true`; richiesta HTTP con limite256 si arresta naturalmente con `Roma`,36 token input e69 output,1,900 s completi: smoke funzionale, non benchmark breve. PID e log in `runs/20261121-http-qkv-graph-operational/launch.json`, esito in `verified.json`. Il server foreground precedentemente osservato alle10:42 non era più presente al controllo iniziale di questa fase; causa non accertata. Daily Qwen e gateway8787 invariati.

### Prototipo di esperti replicati, lavoro autorizzato il 21 novembre

L'owner ha richiesto un prototipo con tutti i 3072 esperti sulla 2080 Ti e copie degli esperti più usati sulla 3080 Ti. Il calcolo degli esperti selezionati deve poter procedere sulle due GPU contemporaneamente, con assegnazione esclusiva di ogni contributo e ricongiungimento prima del layer successivo. La duplicazione dei pesi non elimina le dipendenze tra attention, router e passaggi ripetuti del core.

Base di recupero pubblicata: `7e4665778b164bc16c986938db7a7c37cad3abdf`, checkout operativo su main. Implementazione opt-in compilata CUDA12.6/SM75/SM86: `D:/ds4_work/inkling2/replica-prototype-20261121/build/strata-inkling2-replica.exe`, SHA256 `35c9cc2117680d9931a6df85737eb60b1b009950ec21d3ac075c3d3309aa4493`. Default invariato, MTP e layer graph non supportati nel prototipo. Il server operativo è stato fermato soltanto per le prove GPU e ripristinato al termine con il binario QKV e il preset sopra. Non è stata cambiata la configurazione Daily Qwen.

Piano di verifica: copie hot determinate da frequenze reali durante warmup, aggiornamento soltanto fra richieste, capacità262144/KVQ8, MTP spento e 3060 esclusa. Prima verificare residenza, calcolo una volta sola, risultati e sovrapposizione; poi confronto breve con la baseline. Il banco completo mantiene output128/512/1000 e distingue decode dai tempi complessivi. Il primo prototipo non promette accelerazioni né replica la cache KV su entrambe le schede.

Controlli completati: test CPU della selezione hot e otto test CPU del banco passati; test CUDA grouping/ownership/scatter passati su entrambe le GPU. Default nuovo contro binario QKV: tutti i200058 logits bit-identici a chunk1 e64. Replica con528 hot contro vecchio layer split: chunk1 identico; chunk64 differisce dell'1,8216% in L2 relativa, argmax invariato. Non presentare il nuovo placement come bit-identico al vecchio a chunk64.

Controllo discriminante a chunk64, stessi99 ID e stessa build: hot0 contro hot2048MiB con dense sulla3080 produce tutti i200058 logits bit-identici. Tutto sulla2080 contro dense3080/esperti2080/hot0 differisce dell'1,76989% in L2 relativa, argmax invariato. Il cambio di GPU dei calcoli densi e i diversi percorsi batch cuBLAS sono coerenti con la differenza osservata; non attribuirla quantitativamente a un singolo kernel. Artefatti `replica-bench/{regression-old-logits,regression-new-default-logits,prototype-hot-logits,controlled-hotzero-chunk64,controlled-all2080-chunk64}` e JSON dei confronti. Nei primi report logits l'RSS del processo non era campionato: il valore zero è da leggere come sconosciuto; il banco è stato corretto per le prove successive.

Il warmup del banco usa una richiesta distinta di98 ID, divergente dal prompt canonico dopo12 ID. Prima di ciascuna misura si esegue quel warmup da128 output; la richiesta misurata resta di99 ID, senza riuso del prefisso, con output128 iniziale. Il prototipo aggiorna gli hot al reset, non durante il decode. Il codice v1 è pubblicato in `201805793`; il banco con otto test CPU in `b0ed23232`.

Pilot senza profiler, output128 effettivi: C con hot0=18,7334 t/s; B1 con hot2048MiB=20,2726; A vecchio layer split=79,6508; B2 hot2048=19,6891. C/B1/B2 hanno gli stessi128 ID; A/B differiscono in92 posizioni, prima differenza all'indice34. Il confronto con A non è una parità bitwise né un isolamento del solo parallelismo. B1/B2 hanno528 hot,53286 contributi locali e33498 remoti nella richiesta completa, comprendente prefill; nessun contributo duplicato/mancante nel piano host. Picchi B:2080=11818MiB,3080=7812MiB; A=7176/10546MiB;3060=1578MiB. La soglia di rallentamento è fallita: non eseguite le lunghezze512/1000. Riepilogo `replica-bench/pilot128-summary.json`.

Unica variante successiva: budget richiesto8192MiB, limitato automaticamente alla memoria disponibile dopo dense/KV/scratch/riserva.1320 hot,5241927680 byte allocati alle copie,3080 picco10826MiB. Logits a chunk64 e128 ID prodotti sono identici alla replica hot0. Decode20,9560 t/s,6,468 s complessivi;76668 contributi locali e10116 remoti (88,34% locali nella richiesta comprendente prefill). Anche questa configurazione resta nettamente più lenta dello split operativo. Artefatti `replica-bench/maxhot8192-{logits,run128}`.

Profilazione separata Nsight Systems2024.5.1 completata su quest'ultima configurazione; i tempi profilati non sono usati come benchmark. SQLite `replica-bench/nsys-maxhot128-20261121.sqlite`, analisi `nsys-maxhot128-analysis.json`. In una finestra di2 secondi interna al decode: GPU0/2080 occupata da kernel per303,5ms, GPU1/3080 per725,9ms; sovrapposizione dei kernel esperti86,3ms in4180 intervalli. Il calcolo simultaneo è quindi osservato. Nella stessa finestra la CPU spende circa1075ms in147693 chiamate di lancio kernel e580ms in9166 `cudaStreamSynchronize`. Queste durate diagnostiche non sono una previsione di guadagno e non vanno sommate ai tempi GPU.

Variante asincrona pubblicata in `785a510a3`, opt-in `--expert-replica-async`: eventi fra GPU sostituiscono le attese CPU per layer e i contatori vengono scaricati al reset/fine richiesta. Binario immutabile `build-async/strata-inkling2-replica-async.exe`, SHA256 `9b3582ebaa1df2aa58fe8e12292d7528ef47c14e66afec5f2f2e48f1cdc6196f`. Il test CUDA esegue40 generazioni con T variabile e un solo slot pinned, verificando tutti gli output dopo un'unica attesa CPU; passato in entrambe le direzioni fra le schede. Passano anche frequenze, conteggi hot/cold/ID non validi e reset dei contatori di richiesta senza azzerare lo storico.

Regressione del default e confronto async/sync: tutti i200058 logits bit-identici a chunk1 e64. BAB128, stessa nuova build, stessi hot massimi e flag QKV in tutti i passaggi: B1 async23,5621 t/s, A sync22,0679, B2 async22,6471. Tutti i128 ID coincidono fra i tre passaggi. Media async23,1046, circa+4,7% sul singolo A, con variabilità fra i B: non è una stima statistica robusta. Resta circa il29% dei79,6508 t/s del preset operativo e non passa la soglia per continuare con512/1000. Picchi2080/3080=11818/10826MiB,3060=1578MiB. `join_ms=null` nella modalità async significa non misurato, non zero costo. Riepilogo `replica-bench/v2-bab128-summary.json`.

Seconda traccia separata: `replica-bench/nsys-v2-async-maxhot128.sqlite`, analisi riproducibile con `replica-prototype-20261121/verification/summarize_nsys.py`. Ultimi2 secondi di decode profilato:63 `cudaStreamSynchronize` per0,13ms,124541 lanci kernel per1020ms e18142 `cudaMemcpyAsync` per596ms. Sovrapposizione dei kernel esperti4,77ms in447 intervalli. La rimozione delle attese CPU esplicite non elimina il costo del dispatch e delle copie. Le due tracce differiscono anche per il flag QKV e non costituiscono una misura isolata di una sola modifica; nessun confronto Windows/Linux è stato eseguito.

Ultimo riordino verificato e pubblicato in `9c13ae066`: accodare gli esperti locali subito dopo l'evento di input pronto, prima di accodare il ritorno remoto che può trattenere l'host. Solo sei righe spostate, senza cambiare calcoli, buffer o eventi. Binario finale `build-async-early/strata-inkling2-replica-async-early.exe`, SHA256 `83ad859cbbf0e862253ecd73395a87217f7e6ff699f2c8f68022f112614e6a64`. Tutti i200058 logits a chunk1/64 e i128 ID prodotti coincidono con la variante precedente. BAB128: B1 nuovo23,5017, A precedente22,8283, B2 nuovo24,6133 t/s; tempi completi5,750/5,906/5,500 s. Media nuovo24,0575 t/s, circa+5,38% sul singolo A, ancora con variabilità e nettamente sotto il preset operativo. Riepilogo `replica-bench/early-bab128-summary.json`. Nessuna ulteriore variante, prova lunga o nuova profilazione del riordino finale.

Stato finale: prototipo funzionante e opt-in, non promosso. Tiene3072 esperti sulla2080 e1320 copie hot sulla3080 nel budget massimo provato; dense e KV sono sulla3080. Il refresh degli hot avviene al reset del contesto/protocollo: il banco invia `RESET` esplicito dopo il warmup. Non è ancora un'assegnazione opportunistica basata sulla GPU libera, né un refresh automatico durante una risposta o un riuso del prefisso. La3060 resta fuori dal runtime CUDA. La capacità262144 è allocata, ma queste prove usano99 token effettivi di input e128 di output: non sono test con256K occupati.

Riproduzione del banco: `tools/inkling_replica_bench.py`, modello originale Q3_K_XL in D, flag conservati in `D:/ds4_work/inkling2/replica-bench/v2-maxhot-qkv-async-args.json`. Ordine CUDA visibile2080,3080; flag `--expert-dense-device 1 --expert-replica-device 0 --expert-hot-mib 8192 --expert-replica-async --batch-qkv-control-t1`, oltre a capacità262144/KVQ8/chunk64 impostati dal banco.8192 è un limite richiesto, non la quantità garantita: l'allocatore lo riduce dopo aver riservato dense, KV, workspace e margine. Prima di ripetere, fermare il server che usa le stesse GPU e scegliere una nuova cartella artefatti. I test CPU, i test CUDA standalone e le build immutabili sono conservati; nessun backend HIP/SYCL o altra GPU è stato verificato per questa feature CUDA.

Server finale ripristinato il21 novembre alle13:19UTC, processo Python30408 e motore29700, sul preset operativo `inkling-dual-qkv-graph.json` e porta8100. Health `loaded=true`, capacità262144; smoke HTTP reale: risposta `Roma`, `finish_reason=stop`,36 input e69 output,1,704 s completi. È solo una verifica funzionale, non un benchmark da69 token. Prova salvata in `replica-prototype-20261121/verification/operational-restored-final/verified.json`. Nessun processo di benchmark/profilazione resta attivo. GitHub Actions confermato disabilitato; nessuna PR o comunicazione esterna inviata.

### Ripresa del 21 novembre, sera: quant, NInfer e KV remota

Il ripristino delle13:19UTC sopra è storico. I processi Inkling sono stati poi fermati per il banco separato L2 Qwen; alle18:34UTC nessun server Strata/Inkling/llama/NInfer era attivo. La3080Ti e la2080Ti risultavano libere, mentre la3060 serviva il desktop. Il binario di riferimento QKV mantiene SHA256 `0dc78e6451c3047665319b136217320005badddfe32e642f65d5b88fa94f736f`. Nessuna modifica ai file operativi Daily Qwen.

L'owner ha chiesto di provare il calcolo principale sulla3080 e gli esperti eccedenti sulla2080, chiedendo anche quale implementazione avesse raggiunto112t/s. Quel risultato è NInfer-all su una sola3090, UD-IQ4_XS, F16KV, senza DSpark,128 output;96,7t/s dopo10K occupati. La raccolta primaria è [NInfer 8c6eed17](https://github.com/iamwavecut/ninfer-all/blob/8c6eed17e44e41041a96a8323fa41c7acca4a0c0/docs/inkling2.md#across-cards). Non è una misura locale né una promessa per dueGPU eterogenee.

Ispezione dell'header del Q3 locale:734 tensori, senza leggere i pesi per questa analisi. Nel core fisico6–17,11 gate/up e tutti12 down sono NVFP4; il gate/up del blocco12 è IQ3_XXS. Il94,44% dei coefficienti esperti core è dunque già NVFP4. L'unico tensore Q3_K è il gate/up del blocco19, fuori dal core; head ed embedding sono Q5_K. Il nome UD-Q3_K_XL non descrive un modello uniformemente a3bit. Nessun A/B locale fra target Q3/Q4/IQ3_XXS è stato eseguito.

La prova NInfer è preparata in una cartella normale `D:/ds4_work/inkling2/ninfer-trial-20261121`, senza branch/worktree nuovo. Runtime Windows CUDA12.9 multiSM della release2026-11-21.4, build `bd8195f42e87`, archivi verificati contro SHA256 ufficiali. Il GGUF Q3 esistente è convertito suCPU, solo testo, senza draft/vision e senza requantizzazione. Artifact `models/Inkling-2-Mini-UD-Q3_K_XL-text.ninfer`:13.650.004.224byte, SHA256 `48ee0e1fc25b86e1b0a8c32cb78e421d491d8991acdd9b8ecdd3e7b1166903e0`; tutte48 banche esperti confrontate byte per byte con l'originale. Provenienza in `setup/artifact-provenance.json`.

NInfer EP tiene dense, attention, KV, head e draft sul primo device; divide gli esperti in intervalli contigui disgiunti. Non sceglie gli hot perGPU e non offre qui un draft-device o una KV remota. Il Q3 convertito contiene11,36GiB di esperti e1,327GiB nonexpert; un esperto per layer pesa90,89MiB in totale. La sola KV full F16 a262144 richiede8GiB sulla primaria, prima di ring, workspace e graph. Il piano iniziale conservativo `--tensor-split 1,127` è soltanto un preflight, non la configurazione finale né una capienza verificata. GPU ordinate mediante CUDA_VISIBLE_DEVICES3080,2080;3060 esclusa. Prefill NInfer minimo128, diverso dal64 della baseline: confronto di preset caricabili, non isolamento del solo kernel.

Banco esterno in `bench/`: prompt canonico99 ID congelato, sampler T1/P1/K0/minP0, seed20261121, capacità262144, DSpark spento. Piano: prima pilot128, poi512/1000 solo se utile. Tempi di decode e HTTP distinti, nessun ripristinoSSD attribuito a Inkling.

Preflight GPU concluso: NInfer carica realmente262144 di capacità con F16KV, dense/KV sulla3080 ed esperti0 sulla3080/1–127 sulla2080, per ciascun layer. Runtime registra8410MiB di KV e410MiB di workspace; forza chunk512 anche se richiesto128. `/tokenize` restituisce esattamente i99 ID canonici. Dopo il warmup interno di inizializzazione e la cattura graph, `cudaMemGetInfo` registra492MiB liberi contro1259MiB liberi riportati da nvidia-smi. La differenza di contabilità è osservata, ma la causa driver non è stabilita. Non aumentata la quota esperti: il valore CUDA era già inferiore al margine di banco512MiB. Il caricamento è riuscito; non descriverlo come un OOM o supporto GPU assente.

Blocco del confronto equivalente trovato prima di generare: NInfer `src/runtime/contract/sampling.cpp:56` risolve `top_k=0` in20, e `include/ninfer/ops/sampling.h:92` limita la pipeline a20 candidati. Non riproduce il full-vocabulary K0 del banco Strata. Nessuna richiesta generativa, nessun pilot128 o t/s misurato; nessuna sostituzione implicita con K20 o greedy. Il processo del preflight è stato terminato. Esito completo `bench/preflight-result.json`, con log, tokenizzazione e statistiche in `bench/preflight-1-127/`. Il vincolo del sampler riguarda questo confronto T1; non invalida di per sé i benchmark greedy pubblicati da NInfer.

L'owner ha proposto UD-IQ3_XXS e KV o MTP sulla2080. UD-IQ3_XXS pesa11,66GB su disco, ma non è stato scaricato: conKV sulla3080 non basta per contenere256K. In Strata, a262144/chunk64, KVQ8 occupa esattamente4.452.514.560byte:8 visite full=4,0625GiB e40 SWA=0,08423GiB. Il draft attuale segue `m.main_device`, senza selettore separato.

Audit del vero Daily `kv_stream`: il target Qwen conserva laKV completa in RAM pinned e usa cache VRAM CLOCK delle pagine selezionate dall'indexer QSA; il drafter usa un ring della propria finestra. Config Daily:262144/int8/kv-resident32768, log3,09GiB pinned e circa396MiB di payload cacheQSA. Non è uno streamingSSD. Inkling ha full attention senza quell'indexer: un cache-ring host non evita di leggere tutta laKV delle8 visite full. Non trasferire il beneficio Qwen a Inkling senza una diversa implementazione e misura.

La candidata architettura KV remota mantiene sulla2080 laKV delle8 full attention e ne esegue lì il calcolo; proiezioni, controller, MoE e40SWA rimangono sulla3080. Stima da forme esistenti:373.504byte di attivazioni andata/ritorno per token, circa729,5KiB contando entrambi i passaggi del relay pinned. Traffico indipendente dalla lunghezza del contesto, ma8 dipendenze round-trip. Serve nuovo codice per ownership KV, stato/checkpoint e graph per segmenti o multidevice. Non implementata né misurata; non è sufficiente spostare un puntatore o abilitare i graph della replica eager.


### 21 novembre, campionamento GPU a vocabolario completo

Nel protocollo nativo AR il flag `--gpu-full-vocab` estrae il token sulla GPU della head dopo il primo token della risposta. Con il preset T1/P1/K0/minP0 e penalità neutre evita la copia di 200.058 logits e il calcolo delle probabilità sulla CPU a ogni passo. Il seed e il generatore casuale sono quelli del server. Il primo token usa ancora il sampler CPU. Gli altri parametri di campionamento e DSpark seguono i percorsi esistenti. La somma parallela ha un ordine diverso dalla CPU, quindi la parità va riferita alle prove eseguite.

Banco del 21 novembre su Q3_K_XL. Il confronto cambia un solo flag nello stesso binario, con 2080 Ti outer/head, 3080 Ti core, capacità 262.144, KV Q8, chunk64, MTP spento e seed 20261121. Ogni richiesta misurata legge da zero i 99 ID canonici dopo un warmup distinto. Decode calcolato come (output − 1) / secondi di decode. Tempi completi dal protocollo nativo.

| Output | GPU B1, t/s | CPU A, t/s | GPU B2, t/s | Variazione media dei B su A | Secondi completi B1 / A / B2 |
|---:|---:|---:|---:|---:|---:|
| 128 | 83,51 | 77,77 | 84,30 | +7,9% | 1,828 / 1,969 / 1,828 |
| 512 | 82,63 | 74,75 | 82,99 | +10,8% | 6,468 / 7,141 / 6,438 |
| 1.000 | 79,06 | 71,53 | 78,59 | +10,2% | 12,938 / 14,281 / 13,000 |

Tutti gli ID coincidono fra i tre passaggi a ciascuna lunghezza. Altri due seed, 20261122 e 20261123, coincidono per tutti i 128 token confrontati. I numeri descrivono questa singola chat, con una sola misura A. Report e log locali nella cartella `D:/ds4_work/inkling2/astra-placement-bench`, riepilogo `sampler-bab-summary.json`.

Preset locale pronto in `D:/ds4_work/inkling2/config/inkling-dual-qkv-gpu-sampler.json`. Build CUDA 12.6 SM75/SM86 del solo sampler in `builds/full-vocab-0eb965c1/strata-inkling2.exe`, SHA256 `0eb965c15c47570c62fe29d90bf9710e318d6a649f723c8e920f521a393187af`.


### KV sulla 2080 Ti e prefisso locale sulla 3080 Ti

La full attention può ora girare sulla 2080 Ti accanto alla sua KV Q8. Il flag `--split-full-attn-remote` si aggiunge al normale layer split. `--full-kv-local-prefix N` conserva sulla 3080 le prime N posizioni delle visite full assegnate ai suoi blocchi. Quando un chunk supera N, l'engine copia le sole posizioni già scritte attraverso RAM pinned e ricattura il graph. Da quel momento invia le attivazioni alla 2080 per ciascuna visita full interessata. Le SWA conservano il proprio ring locale. `--split-prelude-core` sposta anche i blocchi fisici 0–5 sulla 3080, che così esegue le visite virtuali 0–41. La 2080 esegue le visite 42–47 e la head. Capacità 262.144, chunk64, 3060 dedicata al desktop.

**Correttezza.** Lo spostamento della sola full attention sulla 2080, con la ripartizione originaria dei blocchi, ha prodotto 200.058 logits identici al riferimento a chunk1 e chunk64 e gli stessi 128 token generati. Ha lasciato circa 3,9 GiB liberi sulla 3080. Aggiungendo lo spostamento dei sei blocchi iniziali, i logits a chunk1 coincidono. A chunk64 l'argmax coincide, la differenza L2 relativa è 1,69% e 92 dei 128 ID campionati differiscono dal vecchio layout. Nelle prove del passaggio di soglia, mantenendo fisso il nuovo placement, N=0/64/128 hanno prodotto gli stessi 128 ID, due volte ciascuno dopo RESET. N=64 attraversa la soglia in prefill, N=128 in decode. Artefatti in `astra-placement-bench/prefix-boundary-v2`.

**Placement provati.** I valori seguenti sono pilot da 99 token di input e 128 di output. Alcuni pilot si sono sovrapposti alle compilazioni CPU. Il confronto controllato del sampler è nella tabella precedente.

| Modello e disposizione | Sampler | Decode t/s |
|---|---|---:|
| Q3, sei visite full del core spostate sulla 2080, prefisso locale 0 | CPU | 68,62 |
| Q3, 20 blocchi fisici sulla 3080 alternati ai quattro blocchi full sulla 2080 | CPU | 52,49 |
| Q3, 18 blocchi contigui sulla 3080, 6 sulla 2080, prefisso 8192 | GPU | 81,25 |
| IQ3_XXS, stessi 18 + 6 e prefisso 8192 | GPU | 85,61 |
| IQ3_XXS, 23 blocchi sulla 3080, ultimo blocco e head sulla 2080, prefisso 8192 | GPU | 79,24 |

IQ3_XXS verificato su D, SHA256 `bd7ef0865adf119b5183e45e741c629bafa649769b33b5ebf689bb4bcc4690d6`, 11.663.309.504 byte. Rispetto al Q3, nel core fisico 6–17 cambiano le 12 matrici gate/up. Le 12 down restano NVFP4. L'IQ3_XXS con tutti i 24 blocchi sulla 3080, prefisso 8192 e head sulla 2080 si arresta al controllo di residenza al pilot con chunk64. Nel caricamento diagnostico separato con chunk1 servono 10.298.982.400 byte per gli esperti, contro un budget allocatore di 9.831.448.576 byte dopo le riserve e lo staging previsti per il percorso ibrido. Quel divario include la politica dell'allocatore. Con il blocco 23 sulla 2080, i 3.072 esperti entrano e i logits chunk1 coincidono con il riferimento IQ3 su singola 2080 per tutte le 200.058 voci. Sul pilot chunk64 restano 486 MiB liberi sulla 3080 dopo il placement. Il prototipo a 23 blocchi è recuperabile in `C:/Users/imanu/Documents/Codex/inkling-iq3-allblocks-lab`. Il preset per la chat resta il Q3 con GPU sampler.

Con 10.000 ID occupati e 128 generati, stesso prompt freddo e Q3, la configurazione originale misura 68,65 t/s con sampler CPU e 75,13 t/s con sampler GPU. Tutti gli ID coincidono. Il prefill dura 27,571 / 27,584 s e la richiesta nativa completa 29,422 / 29,281 s. Con il placement 18 + 6 e il sampler GPU, lo stesso confronto a contesto fisso misura 70,69 t/s con soglia 8192 (KV già migrata) e 72,31 t/s con soglia 16384 (KV ancora locale). Gli ID sono identici fra queste due soglie. I rispettivi prefill durano 29,357 / 29,502 s. È una singola coppia di osservazioni. Report in `astra-placement-bench/context10k-*`.

Sul preset Q3 scelto per la chat, mantenendo 128 output, prompt freddo e zero riuso:

| Input realmente occupato | Prefill | Prefill t/s | Decode t/s | Richiesta nativa |
|---:|---:|---:|---:|---:|
| 10.000 | 27,58 s | 362,5 | 75,13 | 29,28 s |
| 20.000 | 60,31 s | 331,6 | 69,49 | 62,16 s |
| 40.000 | 137,33 s | 291,3 | 60,84 | 139,45 s |

I tre prompt provengono da `runs/20261121-context-baseline-swa-fixed/inputs.json`. Riepilogo e telemetria in `astra-placement-bench/context-summary.json`. Nessun restore SSD. Nel BAB corto, picco 10.546 MiB sulla 3080 Ti (PCIe 4.0 x4) e 7.176 MiB sulla 2080 Ti (PCIe 3.0 x16). La 3060 serve Windows.

Verifica su due turni della chat con migrazione KV. Con soglia 37 e prompt A da 36 token, la generazione attraversa la soglia. Il turno B riusa i 36 token di A e genera gli stessi ID della richiesta B a freddo. A dopo reset coincide con A iniziale. Report in `astra-placement-bench/prefix-warm-state/report.json`. Per ripeterla, `tools/inkling_state_parity.py` accetta `--full-kv-local-prefix 37 --split-prelude-core` oltre ai normali flag del layer split.

Preset alternativo riproducibile in `D:/ds4_work/inkling2/config/inkling-dual-kv2080-prefix8192.json`. Binario immutabile SHA256 `61891828ad9b898963605ff1f21a333694c81d8af919e2af9359da739345a109`. Per il preset più veloce usare la configurazione `inkling-dual-qkv-gpu-sampler.json` indicata sopra. Dalla directory `terminal` del checkout:

```powershell
python -m serve.server --engine strata --config D:/ds4_work/inkling2/config/inkling-dual-qkv-gpu-sampler.json --host 127.0.0.1 --port 8100
```

Verifica HTTP finale del preset chat su porta 8100. Il primo turno ha risposto «Roma» in 1,422 s completi, con 31 token di input e 78 generati inclusi i token di ragionamento. Il secondo ha risposto «Parigi» in 1,391 s, con 54 token di input di cui 31 riusati e 105 generati. Entrambi hanno terminato con stop. I limiti di output erano 128 e 512. Uno script preliminare aveva alterato gli accenti del prompt Windows: la prova finale usa il testo UTF-8 verificato. Report `astra-placement-bench/http-full-vocab-final/report.json`. Il server di prova è stato chiuso. Verifica finale: 0 MiB sulla 3080 Ti e sulla 2080 Ti, 1.641 MiB sulla 3060.

## DSpark adattivo e confronto con la RTX 3090, 21 novembre 2026

Implementazione CUDA su `main` in `ac69771a3`. Il preset con DSpark carica la testa Unsloth Q8_0 sulla 2080 Ti. Il target Q3 usa il layer split 2080 Ti + 3080 Ti, con la 3060 dedicata al desktop. Le opzioni sono indipendenti dal preset Qwen.

- `--dspark-gpu-pq` campiona le proposte, calcola il rapporto di accettazione e campiona la distribuzione residua sulla GPU. Il percorso qualificato usa temperatura 1, top-p 1, top-k 0, min-p 0 e penalità neutre. Gli altri sampler continuano a usare il verificatore precedente.
- `--dspark-outer-graph` consente il replay dei blocchi esterni anche per la verifica speculativa, mantenendo i tap e i dati richiesti per il rollback.
- `--dspark-adaptive` sceglie il metodo del round successivo dai millisecondi per token già committato. Mantiene la testa sincronizzata anche durante il decode singolo e torna periodicamente a misurarla. La policy adattiva richiede il sampler GPU a vocabolario completo. Con temperatura zero il preset segue la verifica greedy fissa.
- `--split-depth-mix` distribuisce il calcolo delle somme pesate dei residui su più blocchi CUDA. Si può usare anche a draft spento.

### Chat italiana, Q3 e capacità di 262.144 token

Prompt storico di 99 ID, reasoning high, warmup distinto, seed 20261121, chunk 64, KV Q8. Ogni richiesta misurata ha letto tutto il prompt e prodotto la lunghezza indicata. Nessun ripristino SSD. Le cifre sono token/s di decode, pari a (token generati meno uno) diviso il tempo di decode. Le quattro configurazioni usano lo stesso eseguibile. Le due repliche adattive racchiudono la prova senza draft con depth mix.

| Modalità | 128 output | 512 output | 1.000 output |
|---|---:|---:|---:|
| Senza MTP, sampler GPU precedente | 83,56 | 80,62 | 78,31 |
| Senza MTP, depth mix distribuito | **88,58** | **82,08** | **79,27** |
| DSpark fisso K2, p/q GPU, graph, depth mix | 67,50 | 51,54 | 40,08 |
| DSpark adattivo, prima replica | 79,44 | 80,20 | 78,04 |
| DSpark adattivo, seconda replica | 78,94 | 80,35 | 77,83 |

Le due repliche adattive hanno gli stessi ID e contatori. A 1.000 token il primo arriva dal prefill. Dei 999 successivi, 973 arrivano da round singoli e 26 da 17 round speculativi, con 9 proposte accettate su 34. Forzando K2 si accettano 308 proposte su 1.379. Le due configurazioni senza draft producono gli stessi ID a tutte e tre le lunghezze. Il guadagno del depth mix è più marcato sul test corto e il dato a 1.000 deriva da una sola coppia.

Nel BAB separato che isola il solo p/q GPU a K2 fisso, i tre passaggi producono gli stessi ID e le stesse accettazioni (67/118, 215/592, 308/1379).

| Output | GPU B1 | CPU A | GPU B2 |
|---:|---:|---:|---:|
| 128 | 64,30 | 51,87 | 64,43 |
| 512 | 49,63 | 40,96 | 49,77 |
| 1.000 | 38,83 | 32,75 | 38,86 |

Con **10.000 ID realmente occupati**, zero riuso e 128 output, il confronto nella medesima build ha dato 74,22 t/s senza MTP e 65,19 con DSpark adattivo (12/20). Prefill rispettivamente 27,68 e 27,81 s. Richiesta nativa intera 29,39 e 29,77 s. Nel BAB a contesto corto, la telemetria di B1 ha misurato picchi di 10.574 MiB sulla 3080 Ti (PCIe 4.0 x4) e 8.124 MiB sulla 2080 Ti (PCIe 3.0 x16).

### Prosa greedy e confronto NInfer

Sulla 3090 NInfer pubblica 113,5 t/s senza draft e 126,1 con DSpark, 2,03 token per round, con UD-Q3_K_XL in una singola prova da 384 output. Il dato spesso citato di 112 t/s riguarda un altro test e una quantizzazione diversa. [Misure NInfer per quantizzazione](https://github.com/iamwavecut/ninfer-all/blob/8c6eed17e44e41041a96a8323fa41c7acca4a0c0/docs/inkling2.md#smaller-quantizations).

Sul nostro prompt inglese sulla stampa di Gutenberg, 42 ID formattati, greedy e 384 output, si misurano **82,23 t/s senza draft e 58,52 con K2 fisso** (180/404). Il testo proviene da `tools/bench/refbench.py` di NInfer, ma l'identità con il prompt della misura 3090 non è verificata. Differiscono anche runtime, GPU, placement e configurazione della KV. Nel vecchio binario gli stessi test danno 80,19 e 56,31. Ciascuna modalità nuova riproduce tutti i 384 ID della sua corrispondente precedente. Greedy con e senza draft divergono dal token 121 in entrambe le build. Il verificatore multi-riga ha quindi ancora un problema aperto di equivalenza con il target a una riga su questo banco.

### Configurazioni e verifiche

Binario `D:/ds4_work/inkling2/builds/mtp-adaptive-ad37bc62/strata-inkling2.exe`, SHA256 `ad37bc62db7e19063dcb7e73ff2bb9894e99c5b4693f935fe5fd8380cd070523`. Preset per la chat più veloce sul banco italiano `D:/ds4_work/inkling2/config/inkling-dual-depth-gpu-sampler.json`. Per la chat con la testa attiva, dalla directory `terminal` dello stesso checkout:

```powershell
python -m serve.server --engine strata --config D:/ds4_work/inkling2/config/inkling-dual-mtp-adaptive.json --host 127.0.0.1 --port 8100
```

HTTP reale in due turni sul preset adattivo: «Roma» in 1,406 s completi, 83 token di completamento; «Parigi» in 0,875 s, 61 token, 31 dei 54 token di input riusati. Entrambe le risposte terminano con stop. Il log nativo conferma 9/14 e 8/14 proposte accettate. Il processo di prova è stato chiuso dopo l'unload.

Directory degli artefatti `D:/ds4_work/inkling2/astra-mtp-bench/`. Confronto completo in `release-*/report.json`, p/q isolato in `pq-clean-*/report.json`, contesto in `ctx10k-*/report.json`, prosa in `prose-*/report.json` e chat in `http-mtp-checked/report.json`. Ogni report nativo contiene gli ID, le opzioni e i tempi. I pilot con compilazione concorrente sono esclusi.

CTest su Windows 2/2. Eseguibili CMake `inkling_gpu_pq` e `inkling_depth_split` ricompilati dai sorgenti e passati su entrambe le schede: 97 casi di campionamento e 96 casi bit-identici di depth mix per GPU. Suite Python mirata: 33 test passati, 6 saltati e 102 subtest passati.

### Controlli QKV multi-riga nella verifica

Il prototipo in `C:/Users/imanu/Documents/Codex/inkling-mtp-verify-qkv-lab` applica le tre operazioni del controller Q/K/V usate a T1 a più righe indipendenti durante la verifica. Sul prompt greedy da 384 token il BAB dà **65,88 / 58,81 / 65,80 t/s**. I due B accettano 184/395 e A accetta 180/404. B coincide con il target singolo per i primi 166 token, A per i primi 120. Il testo cambia, quindi la differenza di velocità comprende anche il diverso lavoro di generazione. In italiano, con K2 fisso, il prototipo misura **72,70 / 55,01 / 42,13 t/s** a 128/512/1.000, contro 67,50 / 51,54 / 40,08 del preset fisso. Le prove italiane del prototipo sono pilot singoli. Il preset pubblicato usa il verificatore precedente; il prototipo e i log completi sono recuperabili in `astra-mtp-bench/control-*`. Alla fine delle prove la porta 8100 era libera, con 0 MiB usati sulle due GPU di inferenza.

## Ricerca dei 150 t/s in chat singola e verifica Kuraneko Round 5

21 novembre 2026. Dati di decode su Inkling-2-Mini. Le cifre pubblicate da altri sono separate dalle misure sulla workstation.

**Risultati pubblicati.** Project Maya con una 3090 e UD-Q4_K_XL riporta 123/129 t/s sulla prosa (plain/DSpark), 123/161 sul codice con finestra fissa da 5 token e 121/211 in editing. La sua finestra dedicata verifica 5 posizioni in 15,6 ms, contro 40,8 passando per forward; un token plain richiede 7,8 ms. Il drafter della release richiede una sola GPU. La PR Windows #1106 misura 110-113 sulla prosa e 114-173 sul codice, greedy, thinking disattivato, una sola 3090, una corsa per build. NInfer con IQ4_XS riporta 154,5 sul task Python greedy e 136,5 a temperatura 1. Il port Ampere vLLM con checkpoint originale e sette proposte riporta 91 senza DSpark e 112-147 con DSpark, con capacità 131072. [Maya #1095](https://github.com/mw00/project-maya/pull/1095), [Windows #1106](https://github.com/mw00/project-maya/pull/1106), [NInfer](https://github.com/iamwavecut/ninfer-all/blob/8c6eed17e44e41041a96a8323fa41c7acca4a0c0/docs/inkling2.md), [vLLM Ampere](https://github.com/Hiutaky/vllm/blob/6e9d1cce3e/INKLING2-RTX3090.md).

**Strata.** Le PR aperte #6506 e #6520 sono due implementazioni distinte. Nel sorgente #6506 a 5b272b2b60 la configurazione looped con draft viene rifiutata. #6520 carica il checkpoint ufficiale e riporta circa 92 t/s residenti su 3090; al momento della verifica era draft e monogpu. La segnalazione comunitaria sullo scambio delle norme del drafter è stata controllata: il nostro codice applica già output_norm al blocco finale e dspark_norm alle feature Markov. [PR #6506](https://github.com/Niko1221/Strata/pull/6506), [PR #6520](https://github.com/Niko1221/Strata/pull/6520), [segnalazione](https://github.com/Niko1221/Strata/pull/6506#issuecomment-7068154732).

**Kuraneko.** Dal precedente controllo 602dbbc591 a 74bbd802af ci sono 15 commit. A 9056d598 la full attention Q8 ha un percorso dedicato per T=4, una posizione anchor e tre proposte. Su V100 Q4_K_M, greedy, 64 output, la serie pubblicata passa da 68,7 a 100,2 t/s al contesto realmente occupato di 262144; a 512 input passa da 130,5 a 130,2. [Tabella upstream](https://github.com/kuraneko1/Strata/blob/74bbd802af/docs/bench-results/2026-11-22-v100-inkling2-800k/README.md).

**Prototipo locale.** Dal commit 9056d598 sono stati estratti il kernel verify4 e la riduzione degli split. Nel microbenchmark, con KV sintetica identica e scale diverse per testa, si riduce l'errore cambiando l'ordine della dequantizzazione: prima si recupera l'intero con segno, poi si moltiplica per la scala. A 10k la differenza L2 relativa dal percorso corrente passa dal 2,43% a circa 0,0255%. La versione corretta impiega 162 microsecondi contro 1779 sulla 2080 Ti, 92 contro 1414 sulla 3080 Ti per questa sola full attention, con 24 teste Q e 8 KV. Nel motore la prova usa questa versione corretta, solamente durante la verifica a quattro righe; stesso binario per A e B, stessa configurazione Q3, KV Q8, chunk64, due GPU, capacità 262144. Tutti i 3072 esperti sono residenti. Il banco usa warmup distinto e zero riuso di input; nessun restore SSD.

| Prova, MTP fisso K3 | B1 nuovo | A corrente | B2 nuovo |
|---|---:|---:|---:|
| 10.000 input reali, 128 output, t/s | 50,11 | 43,30 | 49,74 |
| 99 input, 512 output, t/s | 40,68 | 50,98 | 40,44 |

Nelle due prove i B hanno gli stessi ID tra loro. Rispetto ad A la prima differenza è al token 19 sui 10k e al 56 sul prompt corto. Le accettazioni al contesto lungo sono 67/174 contro 71/163, sul corto 157/1056 contro 228/843. I delta end-to-end includono la diversa sequenza generata. Il kernel resta nel laboratorio.

**Codice ed editing sulla build pubblicata ad37bc62 e sul prototipo QKV.** Il task Python LRU con TTL proviene da tools/bench/refbench.py di NInfer; l'identità con la singola corsa da 154,5 non è documentata. Il task di editing locale chiede una sola sostituzione in una nota tecnica. Prompt e confronto integrale con il testo atteso sono negli artefatti. Greedy, KV Q8, seed e lunghezze massime invariati fra varianti. Con effort high tutti i primi 384 token del task Python sono reasoning; 81,15 t/s senza DSpark e 68,86/68,30 con K4. Anche con il solo effort none, a 384 token il parser conta zero caratteri visibili sul task Python. La serie seguente precompila esplicitamente i due identificatori di apertura del testo visibile per misurare il codice e il documento emessi. Le cifre escludono il prefill. Le righe control-B1/B2 usano il prototipo QKV multi-riga della sessione precedente, control-A la build pubblicata. Nel confronto control il task di codice usa K4 e l'editing usa K7. K4 e K7 sono il numero di proposte oltre l'anchor.

| Prova | Token emessi | Decode t/s | Accettati/offerti | Termine |
|---|---:|---:|---:|---|
| code-visible-AR | 384 | 81.46 | 0/0 | length |
| code-visible-control-A | 384 | 81.43 | 260/491 | length |
| code-visible-control-B1 | 384 | 87.95 | 262/484 | length |
| code-visible-control-B2 | 384 | 87.86 | 262/484 | length |
| code-visible-K4-B1 | 384 | 81.46 | 260/491 | length |
| code-visible-K4-B2 | 384 | 81.51 | 260/491 | length |
| edit-visible-AR | 206 | 85.69 | 0/0 | stop |
| edit-visible-control-A | 206 | 156.80 | 178/189 | stop |
| edit-visible-control-B1 | 206 | 159.30 | 178/189 | stop |
| edit-visible-control-B2 | 206 | 159.00 | 178/189 | stop |
| edit-visible-K4-B1 | 206 | 124.51 | 164/168 | stop |
| edit-visible-K4-B2 | 206 | 124.42 | 164/168 | stop |
| edit-visible-K7 | 206 | 157.55 | 178/189 | stop |

Editing completo e conforme al testo atteso (spazi esterni ignorati): AR sì, control-A sì, control-B1 sì, control-B2 sì, K4-B1 sì, K4-B2 sì, K7 sì.

I report con tempi di prefill, wall time, conteggi e ID dei token sono in D:/ds4_work/inkling2/astra-mtp-bench/. La sottocartella repo-scan-2046 contiene summary.json, le sorgenti del microbenchmark, il binario di laboratorio e gli script per ripetere le prove. La build del Daily e i suoi preset restano quelli del precedente checkpoint.

### Preset operativo per l'editing diretto su HTTP

Il preset separato `D:/ds4_work/inkling2/config/inkling-dual-edit-visible-k7.json` usa il binario pubblicato ad37bc62 e il normale endpoint `/v1/chat/completions`. Imposta temperatura zero, effort none e sette proposte. Una copia del tokenizer in `D:/ds4_work/inkling2/pack/tokenizer-edit-visible-v1` usa gli stessi file di vocabolario e tokenizzazione del preset. Il solo template, su un turno utente senza strumenti e a effort zero, aggiunge i due marcatori `<|message_model|><|content_text|>` al prompt. Questo avvia direttamente la risposta visibile. Controllati cinque casi di rendering, inclusi effort high, strumenti e render senza apertura generazione. Il chiamante può richiedere altri valori di temperatura ed effort; i dati sotto sono per i default del preset.

Banco HTTP B-A-B con un processo per passaggio, stesso binario e stesso template. La richiesta di editing seguiva un warmup distinto. Il client ha omesso temperatura ed effort, ha permesso fino a 1.000 token e ha ricevuto il documento completo con stop naturale in tutti i casi. Input 262 ID, zero riuso al turno misurato, 206 token emessi. Verificata la corrispondenza integrale del testo, ignorando solo gli spazi esterni. Al secondo turno il server ha risposto «archive directory». La metrica storica del banco usa 205 intervalli di decode; il server divide 206 token per lo stesso tempo.

| Passaggio | Decode banco t/s | API t/s | HTTP completo | Proposte accettate/offerte |
|---|---:|---:|---:|---:|
| B1, DSpark K7 | 160.55 | 161.3 | 1.938 s | 178/189 |
| A, senza MTP | 88.56 | 89.0 | 3.063 s | 0/0 |
| B2, DSpark K7 | 160.52 | 161.3 | 1.938 s | 178/189 |

Con il template ordinario e K7 il modello svolge anche il ragionamento: stessa modifica esatta, 524 token generati in 5,25 s HTTP completi, 114,23 t/s secondo la metrica del banco. Senza MTP, 524 token in 6,875 s HTTP e 84,92 t/s. Il distinto guadagno dovuto al template e quello dovuto alla speculazione sono quindi misurabili.

Per il lancio dalla directory terminal del fork Inkling:

```powershell
python -m serve.server --engine strata --config D:/ds4_work/inkling2/config/inkling-dual-edit-visible-k7.json --host 127.0.0.1 --port 8100
```

Il task Python con apertura visibile dà 81,46 t/s senza draft e 81,46/81,51 a K4 sulla build pubblicata, con tutti i 384 ID identici. Le 260 proposte accettate implicano 3,11 token per round e circa 38,2 ms per round compresa la proposta. A parità di accettazione servirebbero 20,8 ms per 150 t/s. La traccia più pertinente da portare da Maya è il suo verificatore dedicato alle finestre corte, con graph per larghezza e rollback delle convoluzioni in un lancio. [Codice del verificatore Maya](https://github.com/mw00/project-maya/blob/v1.1.72/src/ik2/ik2_main.cu#L2490).

I processi HTTP di prova sulla porta 18166 sono stati scaricati e terminati. Preset originari e daily Qwen restano disponibili.

## 22 novembre 2026 — Inkling sulle richieste del Daily Qwen

I 160,55 t/s della prova precedente misurano la riproduzione di un documento fornito in input con una sostituzione. Per la velocità di conversazione usiamo il banco rapido del Daily Qwen, le sue richieste Standard-v1 e le richieste comunità archiviate in D1schermo3060. Le lunghezze nominate dai test si riferiscono al tokenizer di Qwen. Per Inkling riportiamo anche i token effettivi. Il Daily Qwen è il riferimento storico a tre GPU.

Su Inkling UD-Q3_K_XL il motore Strata usa la 3080 Ti per il core a tre passaggi e la 2080 Ti per gli strati esterni e la proiezione finale. KV Q8, capacità 262144, prefill a blocchi di 64. Il banco salva stato RAM e GPU, corpi delle richieste, risposte SSE, contatori motore e tempi HTTP. La 3060 gestisce il desktop. Per il banco rapido abbiamo eseguito direttamente le funzioni chat e agente di tappa4c. Per Standard-v1 abbiamo riprodotto le nove richieste della sezione prestazioni con il sampling originale. Per comunità abbiamo usato gli stessi messaggi e nonce del Daily, temperatura zero, tre repliche del riferimento, cap 256 e contatore di riuso zero. Decode comunità = token generati / secondi motore; i tempi HTTP includono il prefill.

| Banco | Decode Qwen storico, t/s | Input effettivo Inkling | Decode Inkling, t/s | Prefill Inkling, t/s | HTTP completo Inkling |
|---|---:|---:|---:|---:|---:|
| Rapido, chat (9 richieste) | 92 | 42 | 87,80 | 284,50 | 1,31 s |
| Standard-v1 short | 113 | 42 | 80,55 | 267,67 | 6,53 s |
| Standard-v1 4k | 105 | 3423 | 76,10 | 377,27 | 15,85 s |
| Standard-v1 16k | 100 | 13775 | 69,15 | 351,44 | 46,65 s |
| Comunità 4096 | 94,71 | 3213 | 81,19 | 390,11 | 11,43 s |
| Comunità 32768 | 101,14 | 24965 | 67,28 | 321,60 | 81,48 s |
| Comunità 128000 | 109,66 | 97793 | 43,11 | 215,56 | 459,75 s |

Nel banco rapido agente i turni successivi al primo danno 78,7 t/s di decode contro i 102 t/s del riferimento Qwen. Il profilo agente misura una sessione multi-turno con strumenti e riuso del prefisso.

La chat rapida conta nove risposte con cap 100 come previsto da quel banco. Le richieste Standard-v1 hanno prodotto 512 token ciascuna. Quattro risposte su nove contengono testo visibile. Nella serie comunità standard Inkling può esaurire i 256 token nel canale di ragionamento anche con effort none. Sullo stesso corpus comunità 4K abbiamo quindi provato anche un'apertura esplicita del canale testo: aggiunge solo due token di controllo, poi il modello scrive liberamente la spiegazione del codice. Tutte e tre le risposte hanno prodotto 256 token visibili. Decode mediano 80,77 t/s senza draft e 43,69 con DSpark K7. Tempo HTTP completo rispettivamente 11,46 e 14,17 secondi.

### Maya e residenza degli esperti

Maya originale rifiuta i GGUF locali UD-Q3_K_XL e UD-IQ3_XXS alla lettura delle quantizzazioni miste. Nel laboratorio abbiamo aggiunto i decoder mancanti, verificati contro GGUF, e portato due primitive CUDA su SM75 per la 2080 Ti. La build locale usa CUDA 12.6 e include SM75 e SM86. Corretto anche il prefill breve per rispettare l'assegnazione degli esperti alle GPU. Sullo stesso file Q3, a 262144 di capacità con KV FP8, tutte le prove seguenti usano gli stessi 3215 ID di input e generano 256 token visibili. Maya attribuisce il primo token al prefill e misura i successivi 255 nel decode.

| Maya Q3, banco comunità 4K | Prefill t/s | Decode t/s | Miss esperti prefill / decode | Processo completo |
|---|---:|---:|---:|---:|
| 3080 Ti, esperti in RAM e 2,9 GiB di cache VRAM, una prova | 27,7 | 13,61 | 158332 / 21145 | 138,72 s |
| 3080 Ti + 2080 Ti, mediana di tre prove | 127,0 | 19,97 | 0 / 0 | 45,81 s |
| 2080 Ti, esperti precaricati, una prova | 120,3 | 35,46 | 0 / 0 | 41,28 s |

Con due GPU il warmup colloca 572 esperti sulla 3080 Ti e 2500 sulla 2080 Ti; con la sola 2080 Ti ne carica 3072. Entrambe usano IK2_HOST=mmap: il GGUF è mappato e le pagine possono rimanere nella cache di Windows. Il runtime evita una copia pinned integrale dedicata e registra zero trasferimenti di pesi dopo il warmup nelle prove residenti. In dual-GPU il runtime ha segnalato peer access disabilitato: le attivazioni passano per l'host. La differenza fra dual e singola include anche kernel e modalità di esecuzione. Tempi Maya del processo intero includono il caricamento; i tempi HTTP Strata si riferiscono al server già carico. Per IQ3 abbiamo verificato il caricamento e il primo decode, senza misurarne la velocità.

Sulla singola 2080 Ti il percorso CUDA Graph resta disattivato. Nel Q3 locale, 14 dei 24 layer hanno tipi di quantizzazione diversi fra proiezioni Q, K, V, rel e gate e il loader lascia le proiezioni separate. Il percorso veloce richiede matrici fuse. Un candidato per la prossima prova è catturare le proiezioni separate già presenti e verificare la parità degli ID. La PR 6506 contiene il graph per esperti residenti e staging delle attivazioni su host, ma il suo controllo per due GPU non verifica la presenza di quelle matrici fuse. Dettagli in maya-turing-ep-1122/dense-graph-audit.json e REPORT.md.

### Kernel, draft e repository

Il kernel Q8 full-attention T=1 di Kuraneko (23125db1) è stato portato isolatamente mantenendo lo split e i 12 blocchi del profilo locale. Nel B–A–B comunità 4K, con tre repliche per braccio, le mediane sono 80,81 / 81,19 / 80,82 t/s. Nel B–A–B comunità 32K si misurano 67,33 / 67,28 / 67,38 t/s, rispettivamente una, tre e una replica. L'output greedy cambia fra A e B. Il kernel usa aritmetica e ordine delle riduzioni differenti; nel vettore sintetico a cancellazione elevata il massimo scarto assoluto è 0,000108 e lo scarto L2 relativo arriva all'8,75%.

Il prototipo di folding della proiezione del drafter ha effettivamente attivato la matrice FP32 aggiuntiva da 201 MiB. Sul primo prompt comunità 4K visibile produce gli stessi 256 ID del K7 ordinario, con 160 proposte accettate su 658. Entrambi divergono dal greedy senza draft al token 159. Il test diagnostico nativo misura 46,69 t/s per K7 e 47,63 con fold, contro 78,23 senza MTP. Le varianti restano nel laboratorio.

Upstream Niko main 6e38f588a6. La PR 6506 al commit c7f7583a91 introduce decode resident su due GPU e verifica speculativa multi-token. L'autore riporta 80,7 t/s greedy e 62,8 con draft su RTX 5080 + 4060 Ti. La PR 6520 al commit 48febba2bb propone una strada distinta con i pesi ufficiali e offload su CPU per schede da 8–16 GB. Kuraneko main 0f98bdc839 include anche un ring di streaming per il prefill ibrido. [PR 6506, misure](https://github.com/Niko1221/Strata/pull/6506#issuecomment-7069371155) · [PR 6520](https://github.com/Niko1221/Strata/pull/6520) · [Benchmark Kuraneko](https://github.com/kuraneko1/Strata/blob/0f98bdc839/docs/bench-results/2026-11-22-rtx4070-inkling2-262k/README.md).

Risultati e runner sulla workstation in D:/ds4_work/inkling2/qwen-comparison-20261122, maya-qwen-bench-1122 e maya-turing-ep-1122. I prototipi CUDA si trovano in kernel-standard-v1-1122 e dspark-folded-1122.

### Parallelismo interno ai tre giri

Inkling esegue gli strati fisici da 6 a 17 per tre volte. All'inizio dei giri 2 e 3 ripristina lo stato che entrava nel core. Per ciascuno dei 12 strati legge gli stati intermedi e quello finale del giro precedente e ricava quattro messaggi con pesi appresi. I messaggi modulano Q, K, V e il ramo MoE. Nel ramo MoE il controllo precede la selezione degli 8 esperti. La temperatura entra nel campionamento del prossimo token, dopo che il target ha calcolato i logits. [Architettura ufficiale](https://huggingface.co/thinkingmachines/Inkling-2-Mini#3-model-properties) · [Core del nostro fork](https://github.com/imanu86/Strata-Inkling/blob/54f713b6d/engine/src/inkling2/inkling2_engine.cu#L2206).

Il prototipo loop-message-bank-1122 prepara i quattro messaggi di ciascuno dei 12 strati insieme all'inizio del giro, sulla GPU che possiede il core. Tutti i parziali e le uscite coincidono bit per bit nei test sintetici SM75 e SM86. Sul primo prompt visibile coincidono tutti i 256 ID e nelle tre repliche HTTP coincidono i testi a ogni cambio del kernel. Mediane B–A–B di 81,12 / 80,84 / 81,50 t/s. Il beneficio osservato su questo carico resta inferiore all'1%.

Abbiamo misurato l'effetto dei messaggi su un router reale, strato fisico 6, per la stessa generazione libera da 256 token. Il controllo in parallelo ricostruisce tutti gli otto ID nell'ordine corretto in 510 decisioni su 510, e il motore emette gli stessi 256 ID del riferimento. Sostituendo il messaggio col corrispondente del token precedente, sui 254 confronti possibili per giro coincidono in media 4,44 esperti su 8, sia nel giro 2 sia nel giro 3. L'intero insieme coincide una sola volta per giro. Omettendo il contributo del messaggio le intersezioni medie scendono a 3,99 e salgono a 4,65 rispettivamente. Si tratta di perturbazioni locali con tutta la storia esatta. Non sono state misurate la qualità generativa di tre rami indipendenti o una fusione finale addestrata. I dati sono in qwen-comparison-20261122/idcheck-true-shadow/routing-summary.json.

Infine abbiamo ripartito fra GPU i gruppi di teste di una vera full attention Q8 a T=1. Il core ha sei visite a questo tipo di attenzione. Il microbenchmark usa stati sintetici, KV storica residente e copie reali delle attivazioni via host. Le uscite sono bit per bit identiche per tutti i tagli. A posizione 97793, un'attenzione costa 1,261 ms sulla 3080 Ti sola; 1,461 ms con sei gruppi sulla 3080 Ti e due sulla 3060; 1,748 ms su 3080 Ti e 2080 Ti (5+3); 1,754 ms con tutte e tre (4+3+1). Con la terza scheda si è allungata l'attesa del risultato. Questi sono tempi del singolo strato con copie e sincronizzazioni. Proiezioni e MoE sono esclusi. Dettagli in loop-head-shard-1122/summary.csv.


### Dipendenza causale del messaggio fra giri

Sul Daily comunità 4K con apertura visibile, stesso prefisso di 3215 ID, greedy e 128 token generati, abbiamo modificato unicamente il passaggio 1→2 alla posizione 3215. Quattro esecuzioni isolate comprendono controllo, azzeramento del solo messaggio MoE dello strato fisico 6, azzeramento di tutti i 48 messaggi ricevuti dal giro 2 e replica di controllo. Il prefisso e i vettori 1→2 prima dell'intervento sono identici. Il diagnostico legge i messaggi 2→3 effettivi del motore sullo stesso token. Nessun tempo di questa build diagnostica viene usato come benchmark.

Per token e passaggio il banco contiene due stati normalizzati da 2304 float, uno intermedio e quello finale. I dodici strati leggono ciascuno quattro miscele dei due vettori per Q, K, V e MoE. I coefficienti di miscela sono prodotti da softmax dei prodotti scalari con le query apprese. I quattro segnali modulano gli ingressi dei rispettivi moduli. Azzerare un messaggio conserva il controllo appreso e gli fornisce un vettore nullo.

| Intervento sul giro 2 | Vettori 2→3 cambiati | Scarto L2 complessivo rispetto al controllo | Primo ID generato diverso (base 1) |
|---|---:|---:|---:|
| Replica di controllo | 0/48 | 0% | nessuno su 128 |
| Solo messaggio MoE dello strato 6 azzerato | 48/48 | 4,60% | 123 |
| Tutti i 48 messaggi azzerati | 48/48 | 33,98% | 34 |

Le modifiche mostrano causalità interna per questo token. Il primo token scelto dopo l'intervento rimane uguale, anche nel caso più forte. La configurazione proposta con tre rami perturbati in parallelo e un giro finale di fusione resta una diversa architettura da valutare. Per isolare l'effetto della diversità servono anche due giri senza rumore e tre rami identici con la stessa fusione. Artefatti locali in D:/ds4_work/inkling2/loop-message-causality-1122/result.json e mix.json.

## Qualità della fusione di tre rami su Inkling-2-Mini

22 novembre 2026. Prototipo isolato su RTX 3080 Ti e 2080 Ti con gli stessi pesi UD-Q3_K_XL. Tre esecuzioni del primo giro ricevono gli stessi token e conservano cache indipendenti. La prima usa lo stato iniziale originale, le altre due lo perturbano in direzioni opposte. Per ciascun token si mediano i due vettori di memoria corrispondenti dei tre rami. Il giro finale legge i vettori fusi con il proprio controllore appreso e il preambolo originale. La temperatura di campionamento non interviene. Il controllo a due giri esegue il primo e il terzo giro virtuale, saltando quello intermedio.

Due ampiezze provate sullo sviluppo e una selezionata prima del test (0,12 volte la RMS dello stato, seed 1122). Sul controllo con tre rami identici, i punteggi per posizione coincidono byte per byte con quelli di due giri, sui tre esempi di verifica. Il banco ha controllato l’argmax del modello originale rispetto al binario di riferimento sui tre esempi.

| Configurazione | ARC-Challenge /48 | MMLU /48 | Totale /96 | Argmax libero corretto /96 | Perplessità WikiText ↓ |
|---|---:|---:|---:|---:|---:|
| Un giro | 10 | 9 | 19 | 7 | 17,63 |
| Due giri, primo e finale | 14 | 15 | 29 | 29 | 12,85 |
| Tre rami fusi, poi giro finale | 13 | 16 | 29 | 29 | 12,82 |
| Inkling originale, tre giri | 44 | 37 | 81 | 80 | 7,16 |

Domande brevi dei test pubblici ARC-Challenge e MMLU, selezionate per hash, con risposta diretta fra quattro lettere, senza ragionamento esplicito. Perplessità su 32 estratti indipendenti di WikiText-2, 48 token valutati per estratto. I risultati descrivono questi campioni. [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) · [MMLU](https://huggingface.co/datasets/cais/mmlu) · [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)

Rispetto a un giro la fusione corregge 13 domande e ne perde 3 (McNemar esatto bilaterale p = 0,021). Rispetto a due giri corregge 1 domanda e ne perde 1 (p = 1). Rispetto ai tre giri originali corregge 5 domande e ne perde 57. Tutte e cinque le correzioni si ottengono anche con due giri semplici, nell’ordine iniziale. La sola nuova risposta giusta rispetto a due giri riguarda il minimo intero che contiene la cifra 9 e il cui reciproco ha un decimale finito (4096). La risposta persa riguarda il gruppo dell’argon (18).

Su tutte e cinque le correzioni rispetto all’originale abbiamo ripetuto le quattro presentazioni cicliche delle opzioni, mantenendo la chiave corretta. I conteggi per domanda comprendono l’ordine iniziale.

| Domanda | Un giro /4 | Due giri /4 | Fusione /4 | Originale /4 |
|---|---:|---:|---:|---:|
| Origine del primo apparecchio acustico | 1 | 1 | 1 | 1 |
| Recupero di energia in frenata | 1 | 2 | 2 | 0 |
| Percentuale di tracce occupate su disco | 1 | 1 | 1 | 2 |
| Consegna di quattro pacchi | 1 | 1 | 1 | 2 |
| Campo di spezzamento e campo di 60 elementi | 1 | 1 | 1 | 0 |

Su queste cinque domande fusione e due giri danno lo stesso esito di correttezza in tutte e 20 le presentazioni. La fusione sceglie A in 67 delle 96 domande del test, dove A è corretta 19 volte.

ARC-AGI-1 è un test separato di trasformazioni di griglie. Sono stati scelti sei fallimenti documentati di Inkling XHigh (e6de6e8f, a8610ef7, bf32578f, 8597cfd7, 5207a7b5, ce039d91) e tre successi (66e6c45b, 00576224, be03b35f). Nel test locale tutte e quattro le configurazioni ottengono 0/9 nel formato griglia diretto. Sui sei fallimenti pubblici sia la fusione sia l’originale producono una cifra incompatibile con la soluzione. Il protocollo XHigh pubblico usa condizioni diverse. [Risultati per singola griglia](https://arcprize.org/results/thinky-inkling-2-mini)

Su 12.698 posizioni, la distanza L2 relativa mediana dei messaggi interni rispetto al ramo neutro vale 8,25% e 8,37% per i due rami perturbati e 3,31% per la media. La fusione attenua buona parte della variazione e sul campione muove soltanto due esiti di correttezza rispetto a due giri. I rami sono stati eseguiti in sequenza per misurare la qualità. La latenza su tre GPU resta da misurare.

Banco e risultati per domanda sulla workstation in `D:/ds4_work/inkling2/fusion-eval-1122` (protocollo, campioni, result.json, item-results.json, grid-certified.json, permutation-results.json, diagnostics.json). Patch e binario in `D:/ds4_work/inkling2/branch-fusion-quality-1122`.

## Verifica speculativa su finestre congelate di cinque token

22 novembre 2026. RTX 3080 Ti e 2080 Ti, terza GPU dedicata al desktop. Stesso eseguibile per tutti i bracci, GGUF UD-Q3_K_XL, drafter DSpark Q8_0, KV Q8 di capacità 262144 e split per strati. Tutti i 3072 esperti sono residenti. B usa due modifiche ispirate a Maya. La full attention Q8 legge K/V una volta per cinque posizioni causali e il ripristino delle convoluzioni dopo un rifiuto usa un solo lancio per GPU. Il controllo R abilita solo il ripristino. La verifica del modello completo mantiene gli stessi strati ed esperti. Si cambiano i flag al lancio del processo per ricatturare i CUDA Graph. [Verificatore Maya](https://github.com/mw00/project-maya/pull/1095)

Prefissi storici da 42 ID (prosa), 99 (italiano) e 10000 (contesto realmente occupato). Abbiamo registrato le finestre greedy del percorso ordinario, ciascuna formata da un token iniziale e quattro proposte reali. Per ciascun prefisso si riproducono le prime 40 finestre con tre ripetizioni. Lo stato canonico avanza con i token committati nel tracciato, uno per volta. Per ogni ripetizione si ripristinano su entrambe le GPU la posizione, le convoluzioni e gli slot della KV circolare sovrascrivibili. La verifica include il backup speculativo, l’intero forward e la lettura delle cinque scelte. Il commit comprende il ripristino in caso di rifiuto. I tempi escludono drafter, prefill, ripristino del banco e controlli di parità. Le prime quattro finestre sono escluse soltanto dalle statistiche di tempo. Mediana delle tre ripetizioni per finestra e successivamente mediana sulle 36 finestre.

| Prefisso | Verifica B1 / A / B2, ms | Ripristino sui rifiuti B1 / A / B2, ms | Verifica + commit B1 / A / B2, ms |
|---|---:|---:|---:|
| prosa | 32,45 / 32,35 / 32,62 | 0,08 / 1,06 / 0,08 | 32,52 / 33,15 / 32,70 |
| italiano | 32,97 / 33,18 / 33,04 | 0,08 / 1,07 / 0,08 | 33,02 / 34,10 / 33,12 |
| 10k | 43,43 / 52,62 / 43,68 | 0,08 / 0,88 / 0,08 | 43,49 / 53,57 / 43,76 |

| Prefisso | Scelte diverse B/A su 200 | Scelte diverse rispetto a cinque passaggi singoli A / B | Finestre con decisione di accettazione diversa B/A su 40 | Token di controllo diversi dopo il rifiuto B/A |
|---|---:|---:|---:|---:|
| prosa | 1 | 0 / 1 | 0 | 0/32 |
| italiano | 0 | 2 / 2 | 0 | 0/35 |
| 10k | 1 | 3 / 2 | 0 | 0/33 |

Il solo ripristino R coincide con A in tutte le 600 scelte e nei 100 token di controllo dopo il rifiuto. Le cinque quantità di proposte accettate (da zero a quattro) sono tutte rappresentate. Per il candidato combinato, le due repliche B coincidono tra loro in tutte le scelte. Questi controlli misurano ID e decisioni, senza implicare identità bit per bit di tutta la cache.

### Generazione sul banco storico

Stesso binario e stessi ID dei prompt, warmup distinto, nessun riuso del prompt e prefill ricalcolato. Server nativo del banco Claude/Strata. Prosa greedy 384 token, italiano a temperatura 1 fino a 512, contesto 10k a temperatura 1 fino a 128. I tempi di decode sono (token generati meno uno) / secondi di decode. Il report generation.json comprende separatamente prefill e tempo totale della richiesta nativa.

| Prefisso | B1 t/s | A t/s | B2 t/s | Senza draft t/s | Solo ripristino R t/s | Primo ID diverso B1/A (base 1) |
|---|---:|---:|---:|---:|---:|---:|
| prosa | 60,48 | 55,56 | 60,57 | 81,62 | 56,85 | 121 |
| italiano | 38,75 | 48,80 | 38,57 | 82,52 | 49,59 | 29 |
| 10k | 48,52 | 43,10 | 48,59 | 76,37 | 43,86 | 18 |

Le due repliche B producono la stessa sequenza su ciascun prompt. Il confronto B/A diverge agli ID di output 121, 29, 18 rispettivamente, contando da 1. I tempi della generazione libera includono quindi anche un numero diverso di accettazioni e una diversa continuazione. R riproduce tutti gli ID di A su tutti e tre i prompt.

A contesto breve la maggior parte del risparmio nel replay viene dal ripristino delle convoluzioni. A 10k la variante combinata risparmia circa 10 ms per finestra e vince su tutte le 36 finestre cronometrate. In generazione il decode diretto è il più veloce su tutti e tre i prompt. Il candidato resta nel laboratorio.

Laboratorio in `D:/ds4_work/inkling2/verify-window-1122`. In `bank/` si trovano i token e le proposte. In `results/` le ripetizioni, gli ID e lo stato GPU. In `generation/` si trovano le richieste indipendenti del banco storico. Riepiloghi in `results.json` e `generation.json`, codice del candidato in `candidate/candidate.patch`. SHA256 del binario di misura `fdec2bf4553ca392c8344ea7d293c91d7b84823bb0da8af99e28942a288d9d8a`.
