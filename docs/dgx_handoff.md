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
.venv/bin/python tools/compare_pipelines.py docs/validation/dgx/ground15_compare.json
```

Das Profil `node20_21` bindet CPU-Jobs an Node 20 und GPU-Jobs an Node 21. `gpu_a100` reserviert eine ganze GPU, 80 GiB GPU-Ressource, 46 GiB Host-RAM und 16 CPUs. Verschachtelte Jobs erhalten die Clusteradresse. `DEEP_FRAME_GPU_RESERVED` hält die tatsächliche GPU-Reservierung für Unterprozesse fest; GPU-Tests in CPU-Jobs werden vor einer CUDA-Treiberabfrage übersprungen, ohne `CUDA_VISIBLE_DEVICES` zu überschreiben. Die größere 0,75-mm-Probe muss ihren tatsächlichen Speicherbedarf zunächst nachweisen.

Die gemeinsame Python-3.12-Umgebung liegt in `.venv`; `requirements-all.txt` enthält auch die für Berichte benötigte Matplotlib-Installation. Die Linux-Pins von cuBLAS und CUDA Toolkit stimmen mit PyTorch 2.9/cu129 überein. Node 21 hatte Kernel-Treiber 580.159.03, aber Userspace-Bibliotheken 580.178.04. Das exakte Paket `libnvidia-compute-580_580.159.03-1ubuntu1_amd64.deb` aus dem [NVIDIA-Ubuntu-22.04-Repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-compute-580_580.159.03-1ubuntu1_amd64.deb) wurde nach SHA256-Prüfung nur nach `.wf/nvidia_node21` entpackt (SHA256 `a6d1c47bc34ff9aeadafc39f4203f95e87e6d4d408629685014eee5296a1aa86`). `env.sh` setzt diese Bibliotheken für Deep_Frame; Systeminstallation und geladener Treiber bleiben unverändert. Nach einem System-Treiberwechsel muss diese Bibliotheksversion erneut zum geladenen Kernelmodul passen.

CalculiX 2.17, PrusaSlicer 2.4 und deren fehlende Bibliotheken liegen ohne Systeminstallation unter `.wf/sysroot`. Zum Wiederaufbau auf Ubuntu 22.04 die folgenden Pakete mit `apt-get download` herunterladen und jeweils mit `dpkg-deb -x DATEI .wf/sysroot` entpacken: `calculix-ccx libspooles2.2 libarpack2 liblapack3 libblas3 prusa-slicer libboost-log1.74.0 libboost-filesystem1.74.0 libboost-locale1.74.0 libboost-regex1.74.0 libboost-chrono1.74.0 libopenvdb8.1 libilmbase25 libtbb2 libnlopt0 libglew2.2 libwxbase3.0-0v5 libwxgtk3.0-gtk3-0v5 liblog4cplus-2.0.5 libblosc1 libnotify4 libsnappy1v5`. `env.sh` setzt die zugehörigen Bibliotheks- und Programmpfade. Der optionale PaStiX-Test wird übersprungen, wenn CalculiX ausdrücklich meldet, dass dieser Backend nicht einkompiliert wurde; die SPOOLES-Physiktests bleiben verpflichtend. Der Null-RHS-Regressionstest rechnet für seinen unveränderten 1e−12-Vergleich ausdrücklich mit Float64 und einem strengeren Residuum von 1e−12; der Produktionssolver und dessen Genauigkeitsvorgaben werden nicht geändert. Head-/Worker-Protokolle und Startprozess-IDs liegen unter `.wf/ray_nodes`; **kein globales `ray stop` auf Node 21 verwenden**, da dort der andere Cluster läuft.

Die ursprüngliche Referenz ist lokal unter `Examples/Frames/ManaFly3/Manafly+3inch+BETA+V4+Frame.stl` vorhanden. Für neue Messungen zuerst Orientierung und Aufbereitung nachvollziehbar neu erzeugen; die alten Windows-Auswertungspfade sind kein neuer Nachweis.

Beide vollständigen Linux-Testsuiten sind bestanden: CPU auf Node 20 **837 bestanden, 21 übersprungen**; GPU auf Node 21 **856 bestanden, zwei übersprungen**. Die anfänglichen sechs CUDA-Fixture-Fehler im CPU-Lauf, ihre Korrektur und die anschließenden grünen Prüfungen sind in [node20_21_validation.json](validation/dgx/node20_21_validation.json) nachvollziehbar festgehalten. Kein Genauigkeitsgrenzwert des Produktionssolvers wurde gelockert.

Der am 5. Oktober um 09:51 MESZ gestartete alte 4/3-mm-Lauf und seine Steuerung wurden auf Nutzerauftrag gestoppt. Diese Ausgaben erfüllen die danach präzisierten Kamera- und Akku-Anforderungen nicht und werden nicht fortgesetzt. Der anschließende ground15-Lauf (`exports/runs/ground15_comparison/status.json`) ist beendet (Steuerung 15:13 MESZ, 13:13 UTC); beide Routen sind gescheitert, siehe unten.

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

Aktueller Auftrag vom 5. Oktober: **beide Pipelines parallel weiterentwickeln**, SIMP/MMA und neuronale Fourier-MLP-Reparametrisierung auf derselben `TopologyProblem`-Formulierung. Die neuronale Route ist ein Adam/Augmented-Lagrangian-Optimierer in den Netzparametern (Logit-Schranke 4 mit Box [0, 1] wie SIMP, Lernrate 0,01 × 0,7^Stufe, Strafgewichte je Bedingung 1 → 10, Multiplikatoren und Strafgewichte über β-Stufen erhalten, Schrittweite auf die MMA-Bewegungsgrenze begrenzt, Startanpassung 800 Schritte mit RMSE-Grenze 0,05), siehe [neural_topology.md](neural_topology.md). Am Kragträger-Benchmark erreicht sie 0,585 g gegenüber 0,567 g bei MMA, endet aber `not_converged_stalled`, weil das gemeinsame Konvergenzkriterium (Entwurfsänderung < 10⁻³ und Parameterraum-KKT-Residuum) nicht erreicht wird ([neural_al_cantilever.json](validation/neural_al_cantilever.json)); „beide Läufe konvergiert“ ist für die neuronale Route damit derzeit nicht zu erwarten. Beide Routen starten vom gemeinsamen Startfeld aus `frame_seed`: gleichmäßige Dichte 0,5 mit `body_start` auf dem Feingitter, deterministisch aus Konfiguration und Problemdefinition (`exports/runs/ground15_start`, SHA256 5a219e13…, Problem-Fingerabdruck 3c6bf87a…, auf Node 21 zweimal byte-gleich erzeugt; Aufzeichnung [ground15_start.json](validation/dgx/ground15_start.json)). Es ist ein neutraler, unzulässiger Startpunkt (238,6 g, Verletzung 19,7), kein Entwurf. Die Steuerung `tools/compare_pipelines.py` erzeugt und prüft das Startfeld selbst: fehlt `design.npz`, startet sie `frame_seed` über Ray, prüft dann SHA256 (5a219e1365c7cdfe55e75ac027e1821e3dca750b9ce91509ec428a93e59f39c5) und Problem-Fingerabdruck und übergibt beiden Routen denselben `start_sha256`; ein abweichender Hash oder eine fehlende `result.json` bricht vor dem Start ab. Ein manueller `frame_seed` ist nicht mehr nötig. Für einen Neustart braucht [ground15_compare.json](validation/dgx/ground15_compare.json) ein neues `state_directory`, neue `mma.root` und `resume: false`; ein unsauberer Arbeitsbaum (auch unversionierte Dateien) blockiert jeden neuen Produktionsjob, nur das Wiederaufnehmen bereits aufgezeichneter Jobs ist erlaubt. Das frühere Grob-SIMP-Startfeld 85f48b9b… (`exports/runs/ground15_seed`, Grob-Lauf `not_converged_diverged`) wird nicht mehr verwendet. Der alte Lauf `raysubmit_sUwR5QggA7Ju1X9Z` und sein Controller sind auf Nutzerauftrag gestoppt; seine Ergebnisse sind historische Zwischenstände. Die folgenden alten 4/3- und 0,75-mm-Konfigurationen bleiben als historische Referenz erhalten und werden nicht automatisch weitergerechnet.

Definition des gescheiterten ground15-Laufs (wird ersetzt, siehe unten): 15° Nase nach unten, Linsenabstand zur tiefsten Stelle des gesamten modellierten bestückten Copters, eine entsprechende geneigte Materialbegrenzung, zusätzlicher Linsenabstandsanteil im Optimierungsziel sowie lokale Kameraschutzfläche. Kameraanschlüsse, Schutz und FOV bleiben Optimierungsbedingungen. Die Anschlussstruktur entsteht aus dem freien Dichtefeld.

Akku in dieser gescheiterten Definition (entfällt künftig): Der Akku wird ideal angepresst. Das Gummi hat **keine eigene Geometrie, Steifigkeit oder Vorspannung in der Rechnung**. Die geschlossenen Kontaktfedern der Unterseite übertragen weiterhin die Akku-Flug- und Crashwrenches einschließlich Kipp- und Abhebelasten in den Frame. Die Führungsseiten gehören nicht zu diesem Lastinterface. Die Einlegehilfe ist auf eine 2,5-mm-Zone mit aufgeweitetem oberem Eintritt begrenzt; obere Verriegelungen und Käfige sind ausgeschlossen. Die Optimierung fordert Kontaktfläche an der dem Akku nächsten Rasterlage sowie eine Verbindung für kleine Handhabungslasten. Kein vorgezeichneter Clipkörper wird eingefügt.

Konfigurationen für den Neustart, abgeleitet aus denen dieses Laufs: [klassisch](validation/dgx/ground15_simp.json), [neural](validation/dgx/ground15_neural.json), [Steuerung](validation/dgx/ground15_compare.json). Der gescheiterte Lauf rechnete zusätzlich mit `battery.positioning_aid` (Einlegehilfe); dieser Schlüssel ist entfernt, die Dateien reproduzieren den damaligen Lauf daher nicht mehr. Seine Definition bleibt über den Fingerabdruck in ground15_validation.json und den Git-Stand vor bf28b94 nachvollziehbar. Beide Routen nutzen `TopologyProblem`, dieselben Domänenmasken, Kontakte, Lasten und Bedingungen. Die Proben speichern die vollständige Definition und ihren SHA256-Fingerabdruck. Die Steuerung verglich dabei nur den Fingerabdruck der Probedefinition, nicht den der Produktionsdefinition. Die drei Probeiterationen haben je 16,65 GiB GPU-Speicher und unter 9,4 GiB Host-RAM benötigt; sie sind ein numerischer Nachweis, kein abgenommenes Design. 416 gezielte CPU-Tests und die fünf Prüfungen der zuletzt präzisierten Funktion und neuralen Kettenableitung sind bestanden. Nach der Export- und Abnahmekorrektur bestanden zusätzlich 254 gezielte Tests (ein GPU-Test im CPU-Profil übersprungen), anschließend 211 Export-/Abnahme-/Fortsetzungstests sowie sieben gezielte Steuerungsprüfungen. Die erneut gestartete vollständige CPU-Suite ist unter `raysubmit_degyqR51kAMAUfAK` mit **860 bestandenen und 21 übersprungenen Tests in 866,12 s** abgeschlossen. Die zuletzt ergänzten fünf Abnahmeprüfungen für endliche Mechanikwerte sowie Standreserve/Propfreiraum sind ebenfalls bestanden. Nachweise: [ground15_validation.json](validation/dgx/ground15_validation.json).

Die beiden 4/3-mm-Feinrouten liefen ab 13:05 MESZ parallel auf Node 21 und sind **ohne Konvergenz beendet** (neural 15:08, SIMP 15:13 MESZ). Beide Ray-Jobs sind SUCCEEDED, beide Zweige stehen auf FAILED; die abhängige Rekonstruktion, Abnahme, Kabel- und 0,75-mm-Schritte wurden nicht ausgeführt:

- SIMP/MMA (`raysubmit_FtAAQ9pueLzpBwnN`): `not_converged_diverged_best_feasible` nach 494 Iterationen. Feldmasse 19,67 g (Zwischenfeld; erodiert 11,86 g, dilatiert 32,46 g). Das daraus exportierte Roh-STL hat 13,73 g und **drei getrennte Körper**; die exakte Boolesche Exportprüfung ist nicht bestanden.
- Neural (`raysubmit_SUaC833pWQi2DaiL`): `not_converged_stalled_best_feasible` nach 582 Iterationen. Feldmasse 37,20 g (Zwischenfeld; erodiert 24,10 g, dilatiert 51,51 g). Das Roh-STL hat 33,02 g und einen Körper, aber keinen durchgehenden Untergurt.

Die Ergebnisse sind **nicht verwendbar**, weder als Design noch als Initialisierung. Ursachen laut Prüfung: Die Kameraschutzbedingung (0,29 im Crashfall vorn) erzwingt Material unter der Linse, der zulässige Zusatzspalt von 3 mm lässt den Frame unter die Kamera fallen, der Linsenabstandsterm im Ziel ist verzerrt und unwirksam, Führungszonen und Handhabungslasten destabilisieren SIMP, der Optimierer wählt das beste zulässige Feld ohne Stationaritätsprüfung und schaltet Stufen am Iterationslimit auch unzulässig weiter, die neurale Route setzt Multiplikatoren, Strafe und Adam je β-Stufe zurück und hat weder Logitgrenze noch Schrittweitenabfall. Die Machbarkeitsmessung des Grobrasters steht weiterhin in [ground15_camera_headroom.json](validation/dgx/ground15_camera_headroom.json).

Es folgen eine korrigierte Problemdefinition und ein Neustart: Kameraneigung 15° (bei 15° Fluglage netto 0°), Kamera so tief wie möglich relativ zur tiefsten Stelle des bestückten Copters, Akku ideal angepresst ohne Gummimodell und ohne Einlegehilfe, Kameraschutz-/Crashreserve-Bedingung zurückgestellt, Korrekturen an Optimierer und neuraler Route. Erst nach Konvergenz beider Routen folgen Rekonstruktion, Abnahme, Kabelkanäle und 0,75 mm. Offen vor dem Produktionslauf: Die aktiven Nachgiebigkeitsgrenzen `crash_front`/`crash_below` (1,5 × `CAMERA_SUPPORT.crash_reference` 195,27 / 685,22 N mm) wurden mit dem entfernten Akku-Kontaktmodell gemessen; sie sind mit ideal angepresstem Akku neu zu messen, oder die zurückgestellte Crashreserve-Neuauslegung muss diese beiden Bedingungen ausdrücklich mit abschalten.

Der Export speichert den gemeinsamen Domänenvertrag mit den tatsächlichen Masken. Rohkörper und Rekonstruktion wenden die geneigte Bodenbegrenzung und die Einlege-Keep-outs an. `tools/functional_geometry_review.py` prüft ausschließlich ein vom jeweiligen Pipeline-Lauf protokolliertes und tatsächlich rekonstruiertes STL. Es misst den Linsenabstand am bestückten Modell, Kamerafreiraum und lokalen Schutz, Akkuauflage (Mindestfläche `BATTERY_SUPPORT.min_area_mm2`), Standsicherheit auf der Auflage und das senkrechte Einsetzen; Einlegehilfe und Führungsprüfungen sind entfernt. Ergänzend wird das rekonstruierte STL auf dem gemeinsamen FE-Raster mit der ideal angepressten Akkuauflage geprüft. Die unabhängige FEA, Wandregel und Kabelauswertung bleiben erforderlich. Die Fortsetzungssteuerung prüft nach den Kabelkanälen erneut alle fünf Pfade, Kontinuität, Propabschluss, Mindestbreiten, die Kamera-/Akku-Funktionen, gemeinsame Mechanik und unabhängige Evaluation einschließlich Wandregel. Vorher-/Nachher-Berichte, maßstabsgleiche Vieransichten und numerische Vergleiche werden gespeichert. Ein nicht erreichbarer OCP-Viewer wird als offener Anzeigeschritt protokolliert und hält die Rechnung nicht an. Die Steuerung kann unter einer Dateisperre an bestehende Ray-Job-IDs wieder anknüpfen, ohne doppelte Produktionsläufe zu starten.

Die Kameralinse ist im vorhandenen Bauteilmodell der Mittelpunkt der Kamerafront. Der optische Versatz ist bislang nicht am realen Lux-Bauteil gemessen; diese Modellannahme bleibt in jedem Messbericht sichtbar. Ein Optimiererstatus `converged` ersetzt keinen Funktionsnachweis.

OCP erhält anschließend nur den besten tatsächlichen Pipeline-Frame neben der ursprünglichen ManaFly-Geometrie in einer normalen, undurchsichtigen Standardansicht. Die separate alte Führungszeichnung ist kein neues Ergebnis und wird nicht weiterentwickelt.

## Zeitplan

Der frühere Zeitplan mit erwarteten Ergebnissen ab 16:05 MESZ ist hinfällig. Ein neuer Zeitplan folgt nach dem Neustart mit korrigierter Definition. GitHub-Push ist autorisiert, scheitert in dieser Umgebung derzeit aber an der fehlenden Repository-Authentifizierung.

## Historische Rekonstruktionsbefehle

Die folgenden Befehle beziehen sich auf die frühere Problemdefinition. Für die aktuelle Fortsetzung übernimmt `tools/compare_pipelines.py` Rekonstruktion, Funktionsprüfung, gemeinsame Mechanik, unabhängige Evaluation, Kabelprüfungen vorher/nachher und die bedingte 0,75-mm-Rechnung aus den neuen Konfigurationen. Die alten Pfade unten dürfen nicht als Eingang oder Abnahmenachweis des neuen Auftrags verwendet werden.

Früherer 4/3-mm-Körper:

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

Große Ergebnisse bleiben unter den ignorierten Exportverzeichnissen. Kleine nachvollziehbare Messbelege und Codeänderungen committen. Keine alten Ergebnisdateien als neuen Nachweis übernehmen. Windows-Jobs wurden für die Übergabe beendet; die alte Neuberechnung auf Node 20/21 wurde für die Funktionskorrektur angehalten; die beiden ground15-Feinrouten sind bis 15:13 MESZ ohne Konvergenz gescheitert. Derzeit gibt es keinen gültigen neuen Frame.
