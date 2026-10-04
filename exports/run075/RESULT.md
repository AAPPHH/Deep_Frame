# 0,75-mm-Lauf mit Lastmodell (SIMP+MMA, Mehrgitter) auf der RTX 4080: nicht gerechnet

Stand 04.10.2026, ca. 14:50. **Der Hauptlauf wurde nicht gestartet.** Schon die Vorprobe hat den Arbeitsspeicher des Rechners erschöpft, bevor die erste Auswertung lief. Der Engpass ist also der Host-RAM, nicht die GPU. Damit gibt es keinen Rohkörper, keine v3b-Rekonstruktion, keine Bewertung, kein Vergleichsbild und keine STLs. Das Ziel ist verfehlt und wird so berichtet.

## Wartebedingung

Der Akku-Testlauf ist fertig, Commit 46abec1 in `Deep_Frame-layout` mit `exports/layout/RESULT.md`. Bei der Prüfung um 14:38 lief kein `density_simp`-Job aus `Deep_Frame-layout` mehr. Danach wurde die Probe eingereicht.

## Raster

- Domäne voll 136 × 128 × 32 mm, gleich wie bei simp_mma_cov3 (4/3 mm: 102 × 96 × 24).
- `shape` [182, 170, 48] wie bei den DGX-Läufen, also 0,747 × 0,753 × 0,667 mm. Halbdomäne 91 × 170 × 48 = 742 560 Zellen gegenüber 117 504 bei 4/3 mm (6,3-fach). Das ergibt rund 2,31 Mio. Freiheitsgrade gegenüber 378 300.
- Die Formulierung rechnet mit anisotroper Zellgröße: Elementmatrix je Achse, Zellvolumen als Produkt, Schattenmaß über die Zellmittelpunkte. `auto` wählt Mehrgitter, weil 0,753 mm unter 1,1 mm liegt.
- Start: Entwurf `simp_mma_cov3_opt/fine/design.npz` (4/3 mm), auf 0,75 mm prolongiert, ohne Grobstufe.

## Vorprobe

Konfiguration `probe.json`: 3 MMA-Iterationen auf der letzten Stufe β = 64, Speicher jede Sekunde abgetastet (`probe/memory.jsonl`). Eingereicht als Ray-Job `density_simp_1mm` mit 8 Kernen, 28 GB RAM und 14 GPU-GB; `density_simp` deklariert nur 8 GB RAM. Job raysubmit_EpLacDRLAS5TQ1PX.

| Größe | Leerlauf (erste 10 s) | Spitze | Stand bei Abbruch (483 s) |
|---|---|---|---|
| GPU dediziert (Windows-Zähler, Prozess) | 0,21 GB | 8,36 GB | 8,36 GB |
| GPU geteilt (Windows-Zähler, Prozess) | 0,07 GB | 0,13 GB | 0,13 GB |
| nvidia-smi gesamt | 0,89 GB | 9,09 GB | 9,09 GB |
| Host-RSS | 0,61 GB | **34,6 GB** | 34,3 GB |
| Prozess-Commit (Task-Manager) | | **50,7 GB** | |
| freier Host-RAM | | **0,8 GB** | |

Verlauf: Das RSS stieg im Aufbau in rund 55 s von 0,6 auf 34 GB, die GPU dabei auf 8,4 GB. Danach gab es über 7 min keinen Fortschritt, keine einzige MMA-Iteration, und der Rechner lagerte aus. Die Probe wurde abgebrochen, nur dieser eigene Job, um die anderen Teams nicht weiter zu bremsen. `ray job stop` beendete den Python-Kindprozess nicht. Er wurde von Hand beendet (PID 28304, Kommandozeile geprüft), danach waren wieder 35,8 GB frei.

GPU: Bis zum Abbruch lief kein Auslagern in den geteilten Speicher, geteilt +0,06 GB über dem Leerlauf. Die 8,4 GB dediziert fielen aber schon vor der ersten Auswertung an. Bei 4/3 mm lag die gesamte Spitze bei 4,71 GB. Ob die Karte die Auswertung trägt, ist deshalb offen.

## Ursache (vorläufig, nicht vermessen)

Den Aufbau Schritt für Schritt zu vermessen war in dieser Sitzung nicht möglich, weil das Berechtigungssystem die weitere Code-Inspektion blockiert hat. Vermutet werden die CPU-seitigen Montageindizes aus dem Aufbau von `HexElasticity`: `rows`/`columns` als int64 über alle aktiven Elemente × 576, das sind bei rund 0,7 Mio. Elementen etwa 6,5 GB. Dazu kommt die Filtermatrix mit Radius 2,84 mm (≈ 3,8 Zellen, rund 230 Nachbarn je Zelle, etwa 170 Mio. Einträge) samt COO-Zwischenstufen. Beides wächst linear mit der Zellzahl und wird im Mehrgitterpfad wahrscheinlich nicht gebraucht.

## Was fehlt für den Lauf

1. Den Aufbau mit RSS nach jedem Schritt vermessen, sowohl Domänenaufbau als auch DensityMap und HexElasticity.
2. Die Montageindizes im Mehrgitterpfad nicht anlegen, nur bei cuDSS/SuperLU. Den Filter speichersparend aufbauen, entweder blockweise oder als separable Faltung auf der GPU.
3. Danach die Probe wiederholen (`probe.json`) und erst bei RSS < 28 GB und GPU dediziert < 14 GB den Lauf `run.json` starten.

## Vorbereitet

- `tools/formulation_study.py frame_mma` kann jetzt Folgendes:
  - `mma.start`: Feinstufe aus einem fertigen Feinlauf starten, prolongiert, ohne Grobstufe;
  - `mma.memory_s`: Speicherabtastung über den ganzen Lauf, laufend nach `memory.jsonl` geschrieben, Leerlauf-Basis und Spitzen je Phase in `memory.json` und `info.json`;
  - `mma.calibration`: Kalibrierfaktor überschreiben, für einen Nachkalibrierlauf.
- `compose` zeigt jetzt auch Panels aus `comparison.previous`, damit 4/3 mm v3b neben 0,75 mm v3b und ManaFly erscheint.
- `run.json` ist fertig: 0,75 mm, Start simp_mma_cov3, Körper roh und v3b (b74dcf2), STLs nach `Deep_Frame-neural/exports/simp_mma_cov_075{,_v3b}`, Bild `exports/run075/compare.png`.
