# Inkling SQ10 ufficiale su Windows — 23 novembre 2026

Il pacchetto ufficiale funziona con capacità 262.144 e DSpark sulla workstation. Sulla sola RTX 3080 Ti 12 GB colloca 2.653 esperti in VRAM e 419 in RAM; sulla RTX 2080 Ti 22 GB tutti i 3.072 esperti restano in VRAM. La 2080 è più veloce nel banco SQ10. Il controllo Astra AR duale dello stesso giorno resta più veloce sul fixture comunità breve, mentre SQ10/DSpark sulla 2080 vince sul piccolo problema Python. Non è una misura generale della qualità.

## Provenienza e isolamento

- Runtime portabile: [Niko1221/Strata v0.2.20](https://github.com/Niko1221/Strata/releases/tag/v0.2.20), `strata-inkling 0.1.66`, sorgente `2e87c08db18a`, pacchetto Windows CUDA12 compilato con CUDA12.9. SHA256 archivio `afa34ae5af2c11b9bbdf259d94554c3fcdb43b8640c15a6c5d4fa0b435950421`, verificato contro `SHA256SUMS.txt` della release.
- Modello: [strata-engine/Inkling-2-Mini-Strata](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata/tree/04996a1e2c45009b6a04f2af126cb6bdb2ba3587), revisione `04996a1e2c45009b6a04f2af126cb6bdb2ba3587`, `Inkling-2-Mini-SQ-10.strata`, 10.471.526.400 byte. SHA256 completo verificato: `ac4d452ad03e4930abd9d46f9c5eec22b2f75a1da45a1e49bd24ccd58b065934`.
- Artefatti, manifest, runtime, script e log: `D:/ds4_work/inkling2/sq10-official-1123`. Nessun installer, updater o ambiente globale modificato. Daily Qwen e gateway intatti.
- GPU selezionate tramite UUID nei soli processi figli. RTX 3060 riservata al desktop. Nessun processo di inferenza estraneo terminato. Al termine 3080 e 2080 sono tornate a zero MiB e tutti i processi del banco sono chiusi.

## Protocollo

Tutte le misure qui sono **native tramite pipe, non HTTP**. Prefill, decode e tempo completo della richiesta sono separati. `READY 262144` è verificato a ogni avvio: questa è la capacità, non un test con 256K token occupati. Il massimo input completato è 40.000.

Configurazione ufficiale invariata: `--parallel 1 --gpus 1 --chunk 64 --kv k8v8 --kv-offload auto`, DSpark `--draft dspark`, controllo AR `--draft off`. Greedy: temperatura 0, top_p 1, top_k 1, min_p 0, seed 20261121. Nel runtime ufficiale top_k 0 diventa1: non equivale al campionamento sull'intero vocabolario.

Il B–A–B su 3080 usa processi nuovi: DSpark acceso, AR, DSpark acceso. Prima di ogni richiesta misurata c'è un warmup distinto da 128 token; riuso del prompt misurato sempre0. Il prompt Python usa gli stessi 136 ID archiviati, effort0 e prefisso del canale risposta. Limite1000, termina spontaneamente a 295 token SQ10 o321 Astra. Il fixture comunità breve usa gli stessi 3.215 ID e 256 token di output. Nessun benchmark da 64 token.

Prefill t/s = token nuovi / secondi di prefill. Output t/s = `(token emessi - 1) / secondi di decode`, perché il primo arriva dal prefill. Con DSpark questo è throughput di output, **non** il numero di forward del target. Nei JSON grezzi il nome storico `decode_forward_tok_s` conserva questa formula.

## Risultati brevi

| Configurazione | Caso | Input / output | Prefill t/s | Output t/s | Richiesta nativa s | DSpark accettati/offerti |
|---|---|---:|---:|---:|---:|---:|
| SQ10 3080 B1 | Python | 136 / 295 | 208,40 | 68,78 | 4,953 | 233/405 |
| SQ10 3080 A, AR | Python | 136 / 295 | 204,23 | 16,01 | 19,047 | 0/0 |
| SQ10 3080 B2 | Python | 136 / 295 | 206,50 | 69,00 | 4,937 | 233/405 |
| SQ10 3080 B1 | Comunità | 3215 / 256 | 249,65 | 51,85 | 17,859 | 155/344 |
| SQ10 3080 A, AR | Comunità | 3215 / 256 | 250,50 | 14,12 | 30,953 | 0/0 |
| SQ10 3080 B2 | Comunità | 3215 / 256 | 249,70 | 51,63 | 17,891 | 155/344 |
| SQ10 2080, DSpark | Python | 136 / 295 | 296,04 | 119,80 | 2,922 | 233/386 |
| SQ10 2080, DSpark | Comunità | 3215 / 256 | 306,08 | 74,10 | 13,985 | 149/381 |
| Astra AR duale, stesso giorno | Python | 136 / 321 | 348,37 | 93,41 | 3,828 | 0/0 |
| Astra AR duale, stesso giorno | Comunità | 3215 / 256 | 446,52 | 86,01 | 10,171 | 0/0 |

Il controllo Astra riusa senza modifiche `verify-next-1122/final-coding-AR/setup.json`, con GGUF UD-Q3_K_XL e split2080+3080. SHA256 eseguibile verificato: `848d9297342299ae25c5d1ae9566f5950a5946f6b78d5073b754f18a85eb7e9f`. SQ10 e GGUF hanno quantizzazioni diverse; la disposizione delle GPU cambia. I tempi non provano equivalenza numerica fra questi motori.

Tutte le cinque risposte Python misurate superano305/305 controlli funzionali, inclusi iteratori monouso. I295 ID SQ10 coincidono fra B1/A/B2 e la2080. Nel fixture comunità i 256 ID B1/B2 coincidono, ma divergono da AR all'indice51; la2080 diverge dalla 3080 all'indice11, contando da zero. Le risposte restano sensate; non sono confronti a identico testo generato. Non è dimostrato che DSpark sia lossless su ogni prompt.

## Pilot lungo

Stessi corpi delle prime richieste archiviate del banco comunità, effort0,256 output, riuso0. Il coordinatore ha riprodotto tutti gli ID con il tokenizer custom storico e lo stesso `ChatTemplate`, non soltanto il conteggio. SHA256 degli ID decimali separati da virgole:20k `9f45118fcf3fc1413b10cefe03f6a3d44630f10ce82f23639b1204af4c5b25f8`;40k `8982a0cc83ccd06add60561f0d22a337193cee33f1c036dd3f35253f7d2adba5`.

| GPU SQ10 / DSpark | Input / output | Prefill t/s | Output t/s | Richiesta nativa s | DSpark accettati/offerti |
|---|---:|---:|---:|---:|---:|
| 3080 Ti | 20000 / 256 | 189,15 | 26,69 | 115,375 | 181/345 |
| 2080 Ti | 20000 / 256 | 269,55 | 61,01 | 78,469 | 151/404 |
| 2080 Ti | 40000 / 256 | 238,10 | 58,68 | 172,390 | 167/364 |

Una sola richiesta completa per combinazione: non sono mediane. A20k le sequenze2080/3080 divergono all'indice4. I256 token restano nel ragionamento e non completano la spiegazione finale richiesta. La correttezza funzionale305/305 riguarda soltanto Python breve.

Sulla3080 il runtime stock mantiene40 visite SWA in VRAM e usa, per le 8 globali,15.360 posizioni in VRAM più 246.784 in RAM mappata (3.976,5 MiB host dichiarati). È il suo percorso ufficiale, distinto dal KV ring del banco Astra già implementato e misurato. Sulla2080 non compare host offload: il log mostra la crescita elastica della KV globale fino 40.960 posizioni su GPU e nessun avviso di disattivazione DSpark. I contatori accettati/offerti confermano che DSpark opera in ogni richiesta completata.

Il40k sulla 3080 è stato **interrotto dal coordinatore per priorità e costo della prova**, non per OOM o crash: ultimo progresso22.400/40.000,146,7704s di prefill, ultimo chunk 50,9t/s. Nessuna metrica finale40k 3080 viene riportata. Log parziale e `ABORTED_BY_COORDINATOR.json` conservati.

Monitor ogni 2s con RAM, RSS, tutte le GPU e link PCIe. Sotto carico la3080 usa PCIe4x4, la2080 PCIe3x16; non ci sono segni di pressione della RAM di sistema. Il confronto20k 2080/3080 cambia anche hardware e residenza degli esperti: non isola il costo della KV in RAM. Il calo dopo15.360 posizioni sulla 3080 coincide con l'accesso mappato, senza una verifica causale B–A–B dell'offload.

## Riproduzione e limiti

Nel percorso D del banco: `trial.py` esegue smoke128 e Python completo; `bab.py` esegue B–A–B e accetta `--mode single|long|astra`, `--gpu <UUID>`, `--out <cartella nuova>`; `report.py` produce riepiloghi. Esempi effettivamente usati:

```powershell
python bab.py
python bab.py --mode single --gpu GPU-3ec189a9-0fe7-a128-af2c-8df8fb3e15c7 --out single2080
python bab.py --mode astra --out astra-control
python bab.py --mode long --gpu GPU-3ec189a9-0fe7-a128-af2c-8df8fb3e15c7 --out long2080
```

Log nativi, setup, ID/rendering, warmup, risposte, score e monitor sono nelle rispettive sottocartelle. `results.json` qui contiene i numeri compatti; non duplica i lunghi array di token. Nessuna prova HTTP o nuovo test Daily Qwen, nessuna build o nuova ottimizzazione in questa sessione. Il confronto misura il runtime stock; il banco KV ring precedente resta intatto.
