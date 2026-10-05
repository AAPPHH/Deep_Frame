# DGX-Übergabe, 5. Oktober 2026

Auftrag: über GitHub wechseln und große Dateien auf der DGX neu rechnen. `main` enthält die integrierten Änderungen; alle Arbeitsbranches bleiben erhalten. Kein lokales Dichtefeld, STL, Netz oder alter Worktree wird für den Neustart benötigt. Die eingefrorenen skalaren Referenzwerte stehen in `docs/validation/formulation_reference.json` und der Konfiguration. Eine neue ManaFly-Auswertung oder Vergleichsbilder benötigen zusätzlich die ursprüngliche ManaFly-Geometrie; diese fremde Referenz lässt sich aus dem Projektcode nicht neu erzeugen.

## Umgebung

### Cluster auf Node 20 und 21

Der am 5. Oktober eingerichtete Cluster verwendet Node 20 als CPU-Head (`192.168.2.20:6380`, Jobs/Dashboard `http://192.168.2.20:8266`) mit 80 CPUs und 768 GiB Ray-RAM. Node 21 stellt zusätzlich 32 CPUs, eine A100 mit 80 GiB und 256 GiB Ray-RAM bereit. Der andere Ray-Cluster auf Node 21 läuft weiter; Deep_Frame verwendet einen eigenen temporären Pfad und eigene Ports. Die zwei A40 auf Node 20 werden nicht eingeplant: dort fehlen `libcuda.so.1` und das Kernelmodul `nvidia_uvm`, der installierte vGPU-Hosttreiber erlaubt derzeit keinen CUDA-Lauf.

```sh
bash tools/ray_nodes.sh start  # nur wenn der eigene Head noch nicht läuft
source .wf/ray_nodes/env.sh
bash tools/ray_nodes.sh status
.venv/bin/python tools/compute.py suite -- .venv/bin/python -m pytest -q
.venv/bin/python tools/compute.py gpu_a100 -- .venv/bin/python -m pytest tests/test_topology_problem.py tests/test_topology_multigrid.py tests/test_formulation_study.py -q
.venv/bin/python tools/compute.py gpu_a100 -- .venv/bin/python tools/formulation_study.py frame_mma docs/validation/dgx/free_43_coldstart.json
```

Das Profil `node20_21` bindet CPU-Jobs an Node 20 und GPU-Jobs an Node 21. `gpu_a100` reserviert eine ganze GPU, 80 GiB GPU-Ressource, 46 GiB Host-RAM und 16 CPUs. Verschachtelte Jobs erhalten die Clusteradresse. `DEEP_FRAME_GPU_RESERVED` hält die tatsächliche GPU-Reservierung für Unterprozesse fest; GPU-Tests in CPU-Jobs werden vor einer CUDA-Treiberabfrage übersprungen, ohne `CUDA_VISIBLE_DEVICES` zu überschreiben. Die größere 0,75-mm-Probe muss ihren tatsächlichen Speicherbedarf zunächst nachweisen.

Die gemeinsame Python-3.12-Umgebung liegt in `.venv`; `requirements-all.txt` enthält auch die für Berichte benötigte Matplotlib-Installation. Die Linux-Pins von cuBLAS und CUDA Toolkit stimmen mit PyTorch 2.9/cu129 überein. Node 21 hatte Kernel-Treiber 580.159.03, aber Userspace-Bibliotheken 580.178.04. Das exakte Paket `libnvidia-compute-580_580.159.03-1ubuntu1_amd64.deb` aus dem [NVIDIA-Ubuntu-22.04-Repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-compute-580_580.159.03-1ubuntu1_amd64.deb) wurde nach SHA256-Prüfung nur nach `.wf/nvidia_node21` entpackt (SHA256 `a6d1c47bc34ff9aeadafc39f4203f95e87e6d4d408629685014eee5296a1aa86`). `env.sh` setzt diese Bibliotheken für Deep_Frame; Systeminstallation und geladener Treiber bleiben unverändert. Nach einem System-Treiberwechsel muss diese Bibliotheksversion erneut zum geladenen Kernelmodul passen.

CalculiX 2.17, PrusaSlicer 2.4 und deren fehlende Bibliotheken liegen ohne Systeminstallation unter `.wf/sysroot`. Zum Wiederaufbau auf Ubuntu 22.04 die folgenden Pakete mit `apt-get download` herunterladen und jeweils mit `dpkg-deb -x DATEI .wf/sysroot` entpacken: `calculix-ccx libspooles2.2 libarpack2 liblapack3 libblas3 prusa-slicer libboost-log1.74.0 libboost-filesystem1.74.0 libboost-locale1.74.0 libboost-regex1.74.0 libboost-chrono1.74.0 libopenvdb8.1 libilmbase25 libtbb2 libnlopt0 libglew2.2 libwxbase3.0-0v5 libwxgtk3.0-gtk3-0v5 liblog4cplus-2.0.5 libblosc1 libnotify4 libsnappy1v5`. `env.sh` setzt die zugehörigen Bibliotheks- und Programmpfade. Der optionale PaStiX-Test wird übersprungen, wenn CalculiX ausdrücklich meldet, dass dieser Backend nicht einkompiliert wurde; die SPOOLES-Physiktests bleiben verpflichtend. Der Null-RHS-Regressionstest rechnet für seinen unveränderten 1e−12-Vergleich ausdrücklich mit Float64 und einem strengeren Residuum von 1e−12; der Produktionssolver und dessen Genauigkeitsvorgaben werden nicht geändert. Head-/Worker-Protokolle und Startprozess-IDs liegen unter `.wf/ray_nodes`; **kein globales `ray stop` auf Node 21 verwenden**, da dort der andere Cluster läuft.

Die ursprüngliche Referenz ist lokal unter `Examples/Frames/ManaFly3/Manafly+3inch+BETA+V4+Frame.stl` vorhanden. Für neue Messungen zuerst Orientierung und Aufbereitung nachvollziehbar neu erzeugen; die alten Windows-Auswertungspfade sind kein neuer Nachweis.

Beide vollständigen Linux-Testsuiten sind bestanden: CPU auf Node 20 **837 bestanden, 21 übersprungen**; GPU auf Node 21 **856 bestanden, zwei übersprungen**. Die anfänglichen sechs CUDA-Fixture-Fehler im CPU-Lauf, ihre Korrektur und die anschließenden grünen Prüfungen sind in [node20_21_validation.json](validation/dgx/node20_21_validation.json) nachvollziehbar festgehalten. Kein Genauigkeitsgrenzwert des Produktionssolvers wurde gelockert.

Am 5. Oktober um 09:51 MESZ wurde der frische 4/3-mm-Lauf als Ray-Job `raysubmit_sUwR5QggA7Ju1X9Z` gestartet. Die lokale Steuerung `.wf/ray_nodes/continue_handoff.py` wartet auf ihn und führt danach die Rekonstruktion, Evaluation, Kontakt-/Layoutprüfung und Kabelrechnung auf CPU-Ressourcen sowie die 0,75-mm-Probe und gegebenenfalls den Vollrun auf der A100 aus. Aktueller Status und Job-IDs: `.wf/ray_nodes/pipeline/status.json`; Steuerungsprotokoll: `.wf/ray_nodes/pipeline_controller.log`; Hardwaremessungen: `.wf/ray_nodes/pipeline/hardware.jsonl`. Den Controller nicht doppelt starten; vor einem Neustart seine Prozess-ID und die bereits eingetragenen Ray-Jobs prüfen. Er setzt einen konvergierten 4/3-mm-Feinlauf und drei erfolgreiche 0,75-mm-Probeiterationen mit gemessenen Spitzen innerhalb der Reservierung voraus. Ein fehlgeschlagener Evaluations-Unterjob verhindert die unabhängigen Kontakt-, Layout- und Kabeldiagnosen nicht. Die abschließenden Vergleiche, Nachkalibrierung und Berichte brauchen weiterhin eine Prüfung der neu entstandenen Ergebnisse.

Die Original-ManaFly-Geometrie wurde um 180° um Z gedreht und mit der Unterseite auf Z=0 gesetzt; vier neue Ansichten liegen unter `exports/runs/reference_manafly_original`. Herkunft und Messwerte stehen in [manafly_node20_21.json](validation/dgx/manafly_node20_21.json). Die Original-STL ist nicht wasserdicht; sie ersetzt deshalb nicht die reparierte Geometrie der eingefrorenen mechanischen Referenz. Diese Einzelbilder verwenden automatische Bildausschnitte. Maßstabsgleiche Vergleichsansichten benötigen später einen gemeinsamen Bildausschnitt.

### Einzelne DGX mit A100

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

Große Ergebnisse bleiben unter den ignorierten Exportverzeichnissen. Kleine nachvollziehbare Messbelege und Codeänderungen committen. Keine alten Ergebnisdateien als neuen Nachweis übernehmen. Windows-Jobs wurden für die Übergabe beendet; die Neuberechnung auf Node 20/21 läuft seit dem oben dokumentierten Start. Ein gestarteter Lauf belegt noch kein fertiges oder druckbares Design.
