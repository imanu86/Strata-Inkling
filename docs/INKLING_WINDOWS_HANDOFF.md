# Inkling 2 Mini: Windows e GPU miste

Checkpoint del 21 novembre 2026. Questo documento distingue le verifiche completate dal lavoro ancora in corso.

## Perimetro

- Repository: https://github.com/imanu86/Strata-Inkling, branch `main`.
- Base: `kuraneko1/Strata`, commit `19e73bd6df9dc6bdf024716ed030eb369bdf16e8`.
- Motore Strata/Kuraneko: non è stato integrato codice Basalt.
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

Server dual verificato e lasciato in ascolto su `http://127.0.0.1:8100/`, API `/v1`. Configurazione: `D:\ds4_work\inkling2\config\inkling-dual-graph.json`. Usa il binario split immutabile, capacità 262144, KV Q8, prefill chunk64, 2080 outer/head e 3080 core. La 3060 rimane esclusa tramite UUID. Default: effort high=0.9, temperature1/top_p1/top_k0/min_p0.

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

1. Eventuale ottimizzazione outer con CUDA Graph, seguendo la misura Nsight; verificare parità e ripetere il BAB 128/512/1000 senza profiler.
2. Approfondire il prefill batch prima di dichiarare parità su GPU miste; non cambiare quantizzazione senza una prova controllata.
3. Ripristino SSD completo da portare prima dei banchi a contesti lunghi. Il checkpoint RAM salva SWA/convoluzione e usa KV globale ancora residente: **non è un ripristino autonomo da SSD**.
4. Valutare i worker esperti separati dopo il consolidamento del percorso a strati; non sono implementati in questo prototipo Inkling.

Binario split immutabile, verifica HTTP dual e banco 128/512/1000 completati. Prima di una nuova prova GPU usare il server esistente oppure scaricarne il modello, evitando due processi concorrenti. I test a contesto lungo occupato e multichat restano da eseguire.

La ricostruzione della chat Claude precedente è in `C:\Users\imanu\Documents\Codex\2026-10-07\rip\work\TERRA_RIASSUNTO_SESSIONE_CLAUDE_STRATA_2026-11-21.md`. Il Daily a due server usava worker esperti persistenti, non due repliche HTTP. La versione a strati ripartiva prefill e decode; per Inkling occorre rispettare le 48 visite e il relativo stato KV, senza copiare alla cieca i meccanismi Qwen.

Non sono state aperte PR né inviati messaggi esterni per questo lavoro.
