## Banco comunità Strata a 20.000 e 40.000 token

22 novembre 2026. Richieste HTTP seriali con un server già carico, un warmup separato, un nonce per replica, temperatura zero e massimo 256 token. Il prompt chiede di spiegare un modulo Python sintetico. I due target aggiungono punti al banco originale del Daily che usa normalmente 4.096, 32.768 e 128.000. A parità di lunghezza ciascun tokenizer determina la porzione di modulo inclusa. Le due righe sui messaggi Qwen usano invece gli stessi byte della prima richiesta del Daily a ciascuna lunghezza.

| Configurazione | Target | Token di input | Repliche | Prefill t/s | Decode t/s | Primo testo SSE (s) | HTTP completo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.8 Flash Next, Daily | 20000 | 20000 | 3 | 1408,16 | 114,63 | 14,24 | 16,47 |
| Qwen 3.8 Flash Next, Daily | 40000 | 40000 | 3 | 1595,80 | 111,09 | 25,13 | 27,41 |
| Inkling split, riferimento | 20000 | 20000 | 3 | 385,39 | 84,09 | 51,98 | 55,00 |
| Inkling split, riferimento | 40000 | 40000 | 3 | 330,00 | 78,95 | 121,30 | 124,52 |
| Inkling split, candidato | 20000 | 20000 | 3 | 404,72 | 83,82 | 49,49 | 52,52 |
| Inkling split, candidato | 40000 | 40000 | 3 | 342,70 | 78,80 | 116,84 | 120,06 |
| Inkling split sui messaggi Qwen | 20000 | 15275 | 1 | 397,53 | 84,96 | 38,51 | 41,50 |
| Inkling split sui messaggi Qwen | 40000 | 30473 | 1 | 351,93 | 81,03 | 86,69 | 89,82 |

Le colonne mostrano mediane. I token/s di decode usano i token generati divisi per i secondi di decode del motore, secondo la formula originale. Il primo testo SSE include il ragionamento emesso. Il tempo HTTP include il prefill. Il banco archivia ogni risposta e controlla i token effettivi. Per Inkling sono stati adattati il pre-tokenizzatore, gli ID speciali e il valore di effort usati per contare l’input nel template, in accordo con il server. La capacità Inkling è 262.144 token. La 3080 Ti ospita il core ricorrente e la 2080 Ti gli strati esterni. Il Daily Qwen usa il proprio preset a tre GPU con MTP K4. Inkling usa generazione AR.

- Inkling split, riferimento: 6 richieste, 6 con il solo canale di ragionamento, 0 token di prefisso riusati.
- Qwen 3.8 Flash Next, Daily: 6 richieste, 0 con il solo canale di ragionamento, 0 token di prefisso riusati.
- Inkling split, candidato: 6 richieste, 6 con il solo canale di ragionamento, 0 token di prefisso riusati.
- Inkling split sui messaggi Qwen: 2 richieste, 2 con il solo canale di ragionamento, 0 token di prefisso riusati.

Sullo stesso identico corpo HTTP il candidato conserva il testo generato del riferimento in 6/6 richieste. I flag usati sono riportati nel file selected.json.

Il candidato aggiunge una cache FP16 per le matrici dense dequantizzate durante il prefill. Sulla traccia preesistente i kernel di dequantizzazione occupavano complessivamente 479 ms su 6.988 ms per 49 blocchi di prefill. La cache mantiene la stessa suddivisione delle GEMM. Un secondo flag sperimentale cambia la proiezione Q8 del decode. A 129 token tutte le quattro combinazioni dei due flag hanno riprodotto bit per bit i 200.058 logits. Sul banco nativo corto (3.215 token di input e 256 generati) sono stati misurati entrambi i flag e i singoli interventi.

| Prova corta | Prefill (t/s) | Decode forward (t/s) | ID identici al controllo |
|---|---:|---:|---|
| Cache FP16 + kernel Q8, B1 | 469,63 | 86,87 | sì |
| Controllo | 446,31 | 86,62 | sì |
| Cache FP16 + kernel Q8, B2 | 473,38 | 87,33 | sì |
| Sola cache FP16 | 472,49 | 86,75 | sì |
| Solo kernel Q8 | 445,13 | 87,77 | sì |

Per riprodurre il candidato, la patch split-speed-combined.patch si applica dopo verified-next.patch nella serie di fusioni CUDA documentata nell’handoff. SHA-256 del riferimento `848d9297342299ae25c5d1ae9566f5950a5946f6b78d5073b754f18a85eb7e9f`, candidato `582b9473176ca389f3dbfd97d07afc7a37d14565e75abd5714f442fc5629ec72`. Nei test HTTP la sola cache FP16 è abilitata. benchmark-counter.patch documenta l’adattamento del contatore nel generatore originale. Per Qwen si usa il generatore originale senza patch.

Misure grezze, config, stato macchina, richieste, SSE e log motore sono sotto `D:/ds4_work/inkling2/split-speed-1122`. I file JSON del riepilogo conservano i tempi per replica e gli hash SHA-256 delle richieste e delle risposte.
