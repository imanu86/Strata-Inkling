# Inkling 2 Mini sulla sola RTX 3060 12 GB

22 novembre 2026. Windows, monitor collegato alla RTX 3060 durante le prove. Modello UD-Q3_K_XL, cache KV Q8, generazione autoregressiva, capacità 262.144, chunk 64. Il ring GPU da 40.960 token contiene entrambi i prompt e la loro continuazione di 256 token. Il motore conserva la storia globale anche in RAM. Le GPU aggiuntive sono escluse dalla maschera CUDA del processo.

Il motore colloca 2176/3072 esperti sulla GPU (8,40 GiB). Il percorso ibrido usa sette worker CPU per gli esperti fuori VRAM. Nel test preliminare su 1.024 token il profilo automatico è arrivato primo tra quattro budget. Il primo avvio automatico del banco HTTP era stato interrotto dopo circa 44 s di lettura; i tempi sotto vengono da un nuovo processo e comprendono solo richieste complete.

Le due richieste HTTP sono le prime repliche originali del banco comunità Inkling da 20k e 40k, verificate byte per byte. Ogni misura della 3060 è una singola richiesta a server caldo con contatore di riuso zero. Per confronto sono riportate le corrispondenti prime repliche archiviate dello split di riferimento 2080 Ti + 3080 Ti. Il primo testo SSE include il ragionamento. Il decode comunità conta i token generati divisi per i secondi di decode motore.

| GPU | Input | Output | Prefill t/s | Decode t/s | Primo testo SSE (s) | HTTP completo (s) |
|---|---:|---:|---:|---:|---:|---:|
| 3060 | 20000 | 256 | 48,35 | 43,69 | 413,77 | 419,58 |
| Split | 20000 | 256 | 386,98 | 84,12 | 51,76 | 54,77 |
| 3060 | 40000 | 256 | 46,93 | 40,55 | 852,51 | 858,77 |
| Split | 40000 | 256 | 330,36 | 79,20 | 121,21 | 124,42 |

Sul test nativo di scrittura della funzione Python free_windows (136 token di input, cap 1.000) il profilo scelto ha generato 321 token in 9,28 s compreso il prefill a motore caldo. Il decode nativo misura i 320 forward dopo il primo token e registra 48,13 t/s. La funzione supera 305/305 controlli. Nel banco comunità le richieste con testo visibile sono 0/2.

Config riproducibile, log, programmi generati, controlli e risposte SSE in `D:/ds4_work/inkling2/rtx3060-single-1122`. La build usa la correzione degli esperti CPU SITU e la patch host-KV del relativo handoff. Il preset locale è `inkling-3060-tested.json`. La configurazione operativa Qwen è esterna a questo test.
