# DGX-Übergabe, 5. Oktober 2026

Auftrag: über GitHub wechseln und große Dateien auf der DGX neu rechnen. `main` enthält die integrierten Änderungen; alle Arbeitsbranches bleiben erhalten. Kein lokales Dichtefeld, STL, Netz oder alter Worktree wird für den Neustart benötigt. Die eingefrorenen skalaren Referenzwerte stehen in `docs/validation/formulation_reference.json` und der Konfiguration. Eine neue ManaFly-Auswertung oder Vergleichsbilder benötigen zusätzlich die ursprüngliche ManaFly-Geometrie; diese fremde Referenz lässt sich aus dem Projektcode nicht neu erzeugen.

## Umgebung

Im Repository-Hauptverzeichnis arbeiten. Vorhandene passende Umgebungen weiterverwenden; bei neuer Installation Python 3.12 und die gepinnten Requirements verwenden. CalculiX, die Gmsh-Systembibliotheken und PrusaSlicer müssen auf Linux verfügbar sein. Auf Ubuntu heißen die entsprechenden Pakete `calculix-ccx`, `libglu1-mesa`, `libgl1`, `libopengl0` und `prusa-slicer`.

```sh
git clone https://github.com/AAPPHH/Deep_Frame.git
cd Deep_Frame
git switch main
git pull --ff-only
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-all.txt -r requirements-gpu.txt -r requirements-ray.txt
export DEEP_FRAME_MACHINE=dgx
export RAY_PYTHON="$PWD/.venv/bin/python"
export DEEP_FRAME_PYTHON="$RAY_PYTHON"
export DEEP_FRAME_GPU_PYTHON="$RAY_PYTHON"
export CALCULIX_PATH="$(command -v ccx)"
export DEEP_FRAME_SLICER="$(command -v prusa-slicer)"
nvidia-smi
```

`tools/compute.py` hat ein DGX-Profil für 128 CPUs, acht GPUs mit jeweils 80 GiB, 768 GiB Ray-Arbeitsspeicher und 8 GiB Object Store. Das Profil muss zur tatsächlichen DGX passen. Einen gesunden bestehenden Ray-Head weiterverwenden; bei einem neuen Head:

```sh
python tools/compute.py head dgx
```

`gpu_a100` reserviert eine ganze GPU, 80 GiB GPU-Speicher, 46 GiB Host-RAM und 16 CPUs. `CUDA_VISIBLE_DEVICES` wird von Ray gesetzt; kein eigenes Überschreiben. Das GPU-Monitoring funktioniert jetzt unter Linux, misst Prozessspeicher und das zugewiesene CUDA-Gerät und erfindet keinen Windows-Shared-Speicherwert. Nach 30 Sekunden und dann jede Minute Auslastung und Speicher prüfen. Nach einem Neustart der Shell die obigen Umgebungsvariablen erneut setzen.

Vor den Rechnungen:

```sh
python tools/compute.py suite -- python -m pytest -q
python tools/compute.py gpu_a100 -- python -m pytest tests/test_topology_problem.py tests/test_topology_multigrid.py tests/test_formulation_study.py -q
```

## Frisch rechnen

SIMP/MMA mit gemeinsamer Lastkovarianz, freiem Akku und freier Kamera. Keine Neural-AL-Rechnung und keine Rückkehr zu einzelnen Steifigkeitsconstraints. Startdichte 0,5; Montagebereiche bleiben erhalten und verbotene Zellen leer. Der Stand-Fix und der Kabel-Mindestbreitenconstraint bleiben für diese Referenzläufe ausgeschaltet. Keine Grenzwerte lockern.

```sh
python tools/compute.py gpu_a100 -- python tools/formulation_study.py frame_mma docs/validation/dgx/free_43_coldstart.json
python tools/compute.py gpu_a100 -- python tools/formulation_study.py frame_mma docs/validation/free_075_dgx_probe_cfg.json
```

Die 4/3-mm-Konfiguration rechnet grob und fein, speichert `exports/runs/free_layout4_opt/fine2/{design.npz,result.json,density_half.npz}` und exportiert den Rohkörper nach `exports/runs/free_layout4_opt/free_layout4/`. Die drei Iterationen der 0,75-mm-Probe verwenden das neu erzeugte 4/3-mm-Design und explizit cuDSS. Erst anhand von `exports/runs/free_075_probe/fine/result.json`, `iterations.jsonl` und `memory.json` prüfen, ob die Probe erfolgreich und innerhalb des reservierten Speichers war. Ein erfolgreicher Prozess allein belegt keine Designkonvergenz. Danach:

```sh
python tools/compute.py gpu_a100 -- python tools/formulation_study.py frame_mma docs/validation/free_075_dgx_cfg.json
```

Der Vollrun schreibt nach `exports/runs/free_075_opt/`, durchläuft Beta 8/16/32/64 und exportiert nach `free_075/`. Der Export verwendet das zum Lauf passende feinere Raster, nicht das alte 4/3-mm-Raster. MMA, Neural und OC verwenden die gemeinsamen Continuation-Regeln: Machbarkeit und tatsächliche Designänderung, Mindestiterationen und reguläre Stufenbudgets. Bloßer Massestall in einer Zwischenstufe beendet einen gesunden Lauf nicht. Nichtkonvergenz und Best-feasible-Fallback bleiben ausdrücklich erkennbar.

## Rekonstruktion, Evaluation und Kabel

Für den frisch erzeugten 4/3-mm-Körper:

```sh
python tools/compute.py geometry -- python run.py stage exports/layout/free_contact_reconstruction_dgx.json
python exports/layout/postprocess_free.py stage
python tools/evaluate_frame.py run exports/layout/free_contact_evaluation_dgx.json
python tools/compute.py cpu -- python exports/layout/postprocess_free.py contacts
python tools/compute.py cpu -- python exports/layout/postprocess_free.py layout
python tools/compute.py reconstruction -- python tools/reconstruction_study.py cables docs/validation/dgx/cables_free_layout4_v3c.json
```

Die Rekonstruktion speichert ihren Domänenvertrag unter `exports/runs/free_layout4_v3_contact/reconstruction/domain.json` und das passende Dichtefeld mit Masken daneben in `density_fine.npz`; die Kabelkonfiguration verwendet genau diesen Vertrag. Ungleiche Zellabstände werden vor der Splinerekonstruktion anhand physischer Zellmitten auf ein isotropes Raster übertragen, ohne die Geometriehöhe zu strecken. Das Optimierungsraster bleibt unverändert. `stage` kopiert nur das neue STL an den vereinbarten Auswertungspfad. Die Evaluation orchestriert ihre Ray-Unterjobs selbst und darf nicht zusätzlich in einen ressourcenhaltenden Ray-Job eingepackt werden. Kontakte und Layout benötigen Rechenjobs. Falls nur die Feinstufe gerechnet wurde: zuerst `python tools/compute.py geometry -- python exports/layout/postprocess_free.py export` ausführen.

Für 0,75 mm sind eigene Eingaben vorbereitet; die größere Rekonstruktion erhält 26 GiB Host-RAM:

```sh
python tools/compute.py reconstruction -- python run.py stage docs/validation/dgx/free_075_reconstruction.json
python exports/layout/postprocess_free.py stage docs/validation/dgx/free_075_postprocess.json
python tools/evaluate_frame.py run docs/validation/dgx/free_075_evaluation.json
python tools/compute.py cpu -- python exports/layout/postprocess_free.py contacts docs/validation/dgx/free_075_postprocess.json
python tools/compute.py cpu -- python exports/layout/postprocess_free.py layout docs/validation/dgx/free_075_postprocess.json
```

## Befunde und nächste Aufgaben

Die belegten alten Messungen sind keine Ergebnisse der neuen DGX-Rechnung. Details: [4/3-mm-Bericht](../exports/layout/RESULT_free.md), [0,75-mm-Diagnose](validation/run075_dgx_handoff.md), [Kabelstatus](validation/dgx/cables_status.md).

- Kamerakontaktfix: lokale Anschlussflächen erhalten. Alter reparierter Körper 15,6706 g, f1 669,565 Hz, Armsteifigkeit 23,635 N/mm; die zugehörigen kleinen Messbelege liegen unter `exports/layout/`.
- Stand und Wandregel bleiben verfehlt. Alter reparierter Körper: Standreserve −4,443 mm, tiefe Wandfraktion 1,364 % und größte Komponente 47,654 mm³. Unveränderte Wandregel: Öffnung r=1 mm, Voxel ≤0,1 mm, Tiefe ≥0,45 mm, Fraktion ≤0,5 %, keine Komponente >5 mm³ und keine betroffene Motorzone.
- Lokaler 0,75-mm-Multigrid-Lauf scheiterte am Twist-RHS: relatives Residuum 4,1995e−8 bei Soll 1e−8, null MMA-Iterationen. Die getestete Projektionskorrektur und optionale Trace-Instrumentierung sind auf `codex/finish-075` gesichert; sie sind kein bestandener Vollrun. Die DGX-Eingaben wählen deshalb ausdrücklich `cuda_cudss`. Dessen Speicherbedarf wird erst auf der DGX nachgewiesen.
- Sigma/Kamera/Twist auf dem korrigierten Körper vollständig nachweisen; Kalibrierabweichungen berichten und gegebenenfalls einen getrennten Nachkalibrierlauf rechnen.
- Layout mit dem neuen Körper erneut optimieren. Alter reparierter Körper: Schwerpunkt y=1,0454 mm, daher ±1-mm-Nachweis noch offen. Die allgemeine Laufvorgabe bleibt CoG ±5 mm. Ein neuer Layoutvorschlag verlangt neue passende Anschlussgeometrie.
- Kabel am neuen Körper prüfen: alle fünf Pfade, Kontinuität, Propseitenprüfung, Bündel, Mindestbreiten, vier Ansichten und Querschnitte. Wandregel und FEA vor/nach Kanälen vergleichen; alte Kanäle verletzten die Wandregel deutlich.
- Datenblatt einschließlich Dynamikzeile, Neun-Kriterien-Auswertung und vier maßstabsgleiche Vergleichsansichten erstellen: v3c | Schienen-v3b | ManaFly sowie 0,75 mm | 4/3 mm | ManaFly. Fehlende Referenzkörper müssen separat bereitgestellt oder aus den zugehörigen alten Anforderungen neu gerechnet werden.

Feste Bauteilentscheidungen: WHOOP-Stackpfosten 4,5 mm, M2 von oben; HDZero Lux 4:3-FOV 126°/94°; Boden des Designraums z=−4 mm. Kabel-/Stand-Fixes getrennt bewerten, bevor sie in die nächsten Optimierungsanforderungen eingeschaltet werden.

Große Ergebnisse bleiben unter den ignorierten Exportverzeichnissen. Kleine nachvollziehbare Messbelege und Codeänderungen committen. Keine alten Ergebnisdateien als neuen Nachweis übernehmen. Windows-Jobs wurden für die Übergabe beendet; die DGX-Rechnungen sind noch nicht ausgeführt.
