# Deep_Frame

Parametrischer, einteilig additiv gefertigter 2,5-Zoll-FPV-Frame und klassische freie 3D-Topologieoptimierung. Python mit build123d, Gmsh, CalculiX und OCP CAD Viewer. Konfigurationen sind einfache Dictionaries; die parametrische v0 bleibt als unabhaengige Vergleichsgeometrie erhalten.

## Start unter Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-all.txt
.\.venv\Scripts\python.exe tools/install_calculix.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run.py topology
```

Der vorhandene CalculiX-Installer und alternative Solverpfade sind in [docs/fea.md](docs/fea.md) beschrieben. Die Tests enthalten echte Solverlaeufe. Fuer die parametrische Referenz mit installiertem OCP CAD Viewer in VS Code dient `run.py frame`; `run.py smoke` prueft die grundlegende Viewer-/STL-Kette.

## Freie Struktur statt vorgegebener Arme

Die Phase-1-Pipeline startet mit einer uniformen Dichte in einem 3D-Quader. Feste Motor-, Akku-, Elektronik-, Kamera- und Anschlussbereiche sowie verbotene Komponenten- und Montageraume beschraenken das Volumen. Ein vollstaendiger Hex8-SIMP-Solver mit Dichtefilter optimiert die raeumliche Materialverteilung unter mehreren Lastfaellen. Es wird keine v0-Armgeometrie als Saat oder Verbindungsmaske verwendet.

Die Pipeline erzeugt automatisch BRep-Solids, prueft Einteiligkeit, Komponentenfreiraum und geometrische Fertigungsgrenzen und verifiziert gueltige Kandidaten mit der bestehenden Gmsh-/CalculiX-Kette. Masse, Steifigkeit, Verschiebung, Vergleichsspannung und Eigenfrequenzen werden unter denselben Randbedingungen gegen v0 bewertet. Die 37-g-Akkumasse ist im unabhaengigen FEA-Modell enthalten.

Die methodischen und physischen Einstellungen liegen in `TOPOLOGY_CONFIG` von [deep_frame/config.py](deep_frame/config.py); Suchplan, Vergleichsgrenzen und Ausgabepfad in `PIPELINE_CONFIG` von [deep_frame/topology_pipeline.py](deep_frame/topology_pipeline.py), das `run.py topology` als Kopie startet. Rohdaten stehen unter `exports/topology/phase1/`; `latest.json` verweist auf den aktuellen Lauf. Wiederaufnahme prueft Eingaben und Artefakthashes. Aenderungen von Material, Lasten, Domain, Optimierungs-/FEA-Settings oder Quellcode erzeugen getrennte Datensaetze.

Die isotrope PA6-CF-Annahme, grobe Diskretisierung, statischen Crash-Ersatzlasten und vorgegebenen Komponentenpositionen begrenzen die Aussagekraft. Ein gueltiger Rechenlauf ist kein gemessener Festigkeitsnachweis eines realen Drucks. Fuer Solverabbrueche oder nicht herstellbare Formen werden ausdruecklich negative Datenpunkte gespeichert.

## Dokumentation

| Thema | Dokument |
|---|---|
| Oeffentliche Geometrie-/FEA-/Optimierungs- und Domain-/Generator-Schnittstellen | [interfaces.md](docs/interfaces.md) |
| Designraum, Anschluesse, verbleibende Formannahmen, Rekonstruktion und Fertigungschecks | [topology_geometry.md](docs/topology_geometry.md) |
| SIMP, Hex8, optionaler GPU-Loeser, Gradienten- und Balkenpruefungen | [topology_optimization.md](docs/topology_optimization.md) |
| Vollautomatischer Lauf, Akzeptanz, Pareto, Datensatz, Vergleichsbilder und Phase-1-Abnahme | [topology_pipeline.md](docs/topology_pipeline.md) |
| Parametrischer Frame v0, Komponenten und Geometriechecks | [frame.md](docs/frame.md) |
| Unabhaengige FEA, Materialannahmen und Frame-Integration | [fea.md](docs/fea.md) |
| Parametrische Optuna-Optimierung | [optimization.md](docs/optimization.md) |
| Armattan-Recherche und v0-Herkunft | [armattan_research.md](docs/armattan_research.md) |
| Uebergabe an die Workstation | [workstation_handoff.md](docs/workstation_handoff.md) |

Die optionalen Vergleichsbilder benoetigen `requirements-visualization.txt`. Die vollstaendige Rechenpipeline und ihre Exporte benoetigen keinen interaktiv geoeffneten Viewer.

Lizenz: [CERN-OHL-P-2.0](LICENSE).
