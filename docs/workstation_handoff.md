# Uebergabe an die Workstation

Phase 1 ist ein geeigneter abgeschlossener Zwischenstand. Der eingefrorene Quellstand ist [Tag `phase1-workstation-2026-09-30`](https://github.com/AAPPHH/Deep_Frame/tree/phase1-workstation-2026-09-30), Commit `938bbdfc42a8ca04887fdc494fd6b0c4b5cb6061`. Der vollstaendige akzeptierte Lauf heisst `2cd7bcbc15b67fc7`.

**Freigabestatus:** 124 Tests bestanden; ein gueltiger Kandidat `default_t00`, fuenf verworfene Kandidaten; alle vorab festgelegten Geometrie-, Fertigungs- und mechanischen Grenzen erfuellt. 45 SIMP-Updates enden am Iterationslimit, ohne formale Dichtekonvergenz. Der akzeptierte Frame wiegt 58.5318 g. Details stehen im Quellpaket unter `docs/topology_phase1.md`; im aktuellen Stand ist dieser Bericht Teil von [topology_pipeline.md](topology_pipeline.md#phase-1-freie-klassische-3d-topologieoptimierung). Dieser Stand wird zur Uebergabe nicht weiter optimiert.

## Paket und Integritaet

Das lokale Archiv liegt unter `exports/topology/handoff/Deep_Frame-phase1-938bbdf-workstation.zip`, daneben die Datei `.zip.sha256`. Das grosse Archiv bleibt durch die bestehende `.gitignore` ausgeschlossen. Das Paket enthaelt:

- Ein eigenstaendig klonbares Git-Bundle des exakten Tags, einschliesslich versionierter Quellen, Dokumentation, kompakter Abnahmebelege und Git-Historie.
- Die 26 Python-Module aus der Run-Provenienz zusaetzlich mit ihren urspruenglichen Bytes unter `source/run_source_bytes/deep_frame/`, damit auch historische Zeilenenden geprueft werden koennen.
- Den unveraenderten kompletten akzeptierten Run: 89 Dateien mit insgesamt 365026154 Bytes, einschliesslich STEP/STL, NPZ, Inputs, Manifest, Baseline, aller sechs Kandidaten, Pareto-Daten und roher FEA-Netze/Eingaben/Logs/Ergebnisse.
- Die beobachteten Python-Paketversionen der getesteten Windows-Umgebung als Umgebungsnachweis, keine portable transitive Lock-Datei.
- `HANDOFF_MANIFEST.json`, `SHA256SUMS.txt` und einen eigenstaendigen Python-Pruefer. `.venv`, native Solver-Executables, Zugangsdaten und alte rohe Probelaufverzeichnisse sind nicht enthalten.

Die folgenden Befehle gelten im entpackten Paketverzeichnis. Python 3.11 oder neuer und Git muessen vorhanden sein. Entpacken oder Hashpruefung startet keinen Solver.

```powershell
python verify_handoff.py
git clone --branch phase1-workstation-2026-09-30 source/Deep_Frame-938bbdf.bundle Deep_Frame
git -C Deep_Frame switch -c workstation-phase1
git -C Deep_Frame remote set-url origin https://github.com/AAPPHH/Deep_Frame.git
git -C Deep_Frame rev-parse HEAD
```

Die letzte Ausgabe muss `938bbdfc42a8ca04887fdc494fd6b0c4b5cb6061` sein. Unter Linux funktionieren dieselben Git-Befehle; fuer den Pruefer gegebenenfalls `python3` verwenden. Die ZIP-Datei selbst vor dem Entpacken mit `Get-FileHash -Algorithm SHA256` gegen die mitgelieferte `.zip.sha256` vergleichen; unter Linux ist `sha256sum -c Deep_Frame-phase1-938bbdf-workstation.zip.sha256` moeglich.

## Ergebnisse uebernehmen

Unter Windows, weiterhin im entpackten Paketverzeichnis:

```powershell
New-Item -ItemType Directory -Path Deep_Frame/exports/topology/phase1 -Force
Copy-Item -LiteralPath runs/2cd7bcbc15b67fc7 -Destination Deep_Frame/exports/topology/phase1 -Recurse
Set-Location Deep_Frame
```

Unter Linux:

```sh
mkdir -p Deep_Frame/exports/topology/phase1
cp -a runs/2cd7bcbc15b67fc7 Deep_Frame/exports/topology/phase1/
cd Deep_Frame
```

In diesem frisch geklonten Ziel darf noch kein gleichnamiger Run liegen. `HANDOFF_MANIFEST.json` enthaelt die portablen Paketpfade. Die 88 relativen Artefaktverweise des Run-Manifests werden gegen dessen neuen Verzeichnisort aufgeloest. Ergebnisse und Geometrien sind so ohne erneute FEA lesbar.

Historische absolute Windows-Pfade in `manifest.run_dir`, `fea.artifacts`, `mesh_request.json` und Solver-Provenienz bleiben bewusst unveraendert. **Keine JSON-Suchen-und-Ersetzen-Aktion und kein Ueberschreiben der Inputs:** Das wuerde ihre gespeicherten Hashes brechen. Das alte maschinenspezifische `latest.json` wird nicht uebertragen; fuer den archivierten Lauf direkt dessen Manifest verwenden.

Die historische Run-Provenienz nennt Commit `f816fd9`; danach kamen Dokumentations- und Abnahmebelege hinzu. Alle 26 archivierten Python-Module sind nach Normalisierung der Zeilenenden mit dem eingefrorenen Tag `938bbdf` identisch; ihre Originalbytes stimmen mit allen Run-Quellhashes ueberein. Beim geprueften frischen Windows-Klon aenderte `core.autocrlf=true` die Zeilenenden von `components.py` und `model.py`, sodass nur 24 von 26 rohen Klon-Hashes direkt uebereinstimmten. Dies bestaetigt die notwendige Trennung zwischen einem identischen Git-Stand und einer identischen Run-Provenienz. Das spaeter verfasste Uebergabedokument liegt zusaetzlich ausserhalb des Bundles im Paket.

## Neue Umgebung unter Windows

Getestet wurde Python 3.13.5 unter Windows 11/AMD64 mit CalculiX 2.22. Eine neue virtuelle Umgebung wird lokal erstellt:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-all.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-visualization.txt
.\.venv\Scripts\python.exe tools/install_calculix.py
.\.venv\Scripts\python.exe -B -m pip check
.\.venv\Scripts\python.exe -B -c "import build123d,gmsh,numpy,scipy,trimesh; print('imports ok')"
.\.venv\Scripts\python.exe -B -c "from deep_frame.fea_config import FEA_CONFIG; from deep_frame.integration import solver_identity; print(solver_identity(FEA_CONFIG['settings']))"
```

Der vorhandene Windows-Installer laedt das dokumentierte CalculiX-2.22-Archiv und prueft dessen Hash. Die letzten drei Befehle pruefen Pakete, native Imports und Solveridentitaet; sie starten keine neue Optimierung oder FEA. OCP CAD Viewer in VS Code ist fuer interaktive Ansichten erforderlich, fuer die Rechenpipeline nicht. Die Befehle gelten fuer den eingefrorenen Tag `phase1-workstation-2026-09-30`; im aktuellen Stand lautet die Identitaetspruefung `from deep_frame.config import FEA_CONFIG; from deep_frame.fea import solver_identity`.

## Linux-Hinweise

Linux wurde fuer diesen Abnahmestand noch nicht getestet. Python 3.13 mit venv-Unterstuetzung und einen nativen `ccx` bereitstellen; den Windows-Installer hier nicht ausfuehren. Ubuntu/Debian bieten beispielsweise `calculix-ccx`; fuer das Gmsh-Wheel koennen OpenGL/GLU-Laufzeitbibliotheken erforderlich sein. Beispiel fuer vorhandenes Python 3.13:

```sh
sudo apt-get update
sudo apt-get install git python3-venv libgl1 libglu1-mesa calculix-ccx
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-all.txt
.venv/bin/python -m pip install -r requirements-visualization.txt
.venv/bin/python -B -m pip check
.venv/bin/python -B -c "import build123d,gmsh,numpy,scipy,trimesh; print('imports ok')"
.venv/bin/python -B -c "from deep_frame.fea_config import FEA_CONFIG; from deep_frame.integration import solver_identity; print(solver_identity(FEA_CONFIG['settings']))"
```

`ccx` wird im PATH gefunden; alternativ `CALCULIX_PATH` auf das lokale Binary setzen. Die Distributionsversion kann abweichen: [Ubuntu 24.04 liefert beispielsweise CalculiX 2.21](https://packages.ubuntu.com/noble/calculix-ccx), der abgenommene Windows-Lauf verwendete 2.22. Native Downloads/Quellen nennt die [CalculiX-Projektseite](https://www.dhondt.de/); Plattform- und Python-SDK-Hinweise stehen bei [Gmsh](https://gmsh.info/). Die Import- und Identitaetspruefung entscheidet, ob die neu installierte Plattform bereit ist.

## Wiederaufnahme und naechster Arbeitsschritt

Eine neue Maschine erzeugt normalerweise einen **neuen Run-Fingerprint**, auch bei unveraenderten Parametern: Solverpfad und Binary-Hash, Python-Version, Plattform, Pakete und tatsaechliche Quellbytes gehen in die Provenienz ein. Git-Zeilenenden koennen sich zwischen Windows und Linux ebenfalls unterscheiden. Exaktes Resume ist nur bei identischer Provenienz und intakten Artefakten moeglich; ein neuer Run ist sonst das vorgesehene Verhalten. Den archivierten akzeptierten Lauf als unveraenderte Referenz behalten.

Nach dem Umzug zuerst die vollstaendige Testsuite bewusst starten (`.venv/Scripts/python.exe -m pytest -q` unter Windows, `.venv/bin/python -m pytest -q` unter Linux); sie enthaelt echte Solvertests. Danach ist der naechste fachliche Schritt eine feinere Designraumdiskretisierung, eine separate FEA-Netzkonvergenzstudie und mehr SIMP-Iterationsbudget. Lastfaelle, Vergleichsgrenzen und Referenz dabei explizit festhalten. Einstellungen stehen in den vorhandenen Dicts in `deep_frame/topology_config.py`, `deep_frame/fea_config.py` und `run_topology.py`; der bewusste Start erfolgt mit der jeweiligen venv-Python-Datei und `run_topology.py`. Im aktuellen Stand liegen diese Dicts als `TOPOLOGY_CONFIG` und `FEA_CONFIG` in `deep_frame/config.py` sowie als `PIPELINE_CONFIG` in `deep_frame/topology_pipeline.py`; der Start erfolgt dort mit `run.py topology`. Weder dieses Paket noch seine Pruefung startet diese Arbeiten automatisch.

## Abgeschlossene Workstation-Fortsetzung

Die [Workstation-Etappe vom 30.09.2026](validation/workstation_20260930.md) dokumentiert die neu aufgesetzte Umgebung mit 151 bestandenen Tests, die Wiederherstellung aus den intakten versionierten Referenzbelegen, 300 SIMP-Updates, das feinere 8/3-mm-Raster, getrennte FEA-Netz- und Solverstudien sowie einen neu akzeptierten 50.95298-g-Frame. Der eingefrorene Run und seine Abnahmebelege bleiben unveraendert. Das grosse Archiv war auf der Workstation nicht vorhanden; neu erzeugte Daten besitzen eigene Provenienz. Fuer neue FEA-Laeufe verwenden die dort gezeigten Befehle ausdruecklich SPOOLES mit einem Thread, nachdem der native PaStiX-Pfad wiederholt mit einem Windows-Heapfehler abbrach. Die Fortsetzung dokumentiert ebenso die nicht erreichte Dichte-/Rasterkonvergenz und alle sechs verworfenen Extraktionen des feineren Felds.

Die spaetere [GPU-Fortsetzung](validation/workstation_gpu.md) belegt echte FP64-GPU-Faktorisierung, 23.75-fache Beschleunigung einer vollstaendigen Drei-Update-SIMP-Vergleichsfolge und das Erreichen des unveraenderten Dichteaenderungskriteriums nach 708 Updates in 9.60 Minuten. Dies ist keine neue Geometrie-/Mechanikabnahme; diese Phasen bleiben getrennt. Die CPU-Standardeinstellung bleibt erhalten.
