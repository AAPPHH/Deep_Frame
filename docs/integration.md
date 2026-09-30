# Frame-v0-Integration und Probelauf

Die zentrale `deep_frame.config.CONFIG` referenziert jetzt Geometrie, Komponenten, Material, `fea`, `integration`, `optimization` und `optimization_search_space`. Das bestehende Smoke-Dict mit Laenge, Breite, Dicke und STL-Pfad bleibt erhalten. `reference_parameters()` entfernt den Smoke-Path und liefert eine unabhaengige JSON-faehige Kopie. Geometriechecks und FEA verwenden dieselbe zentrale Materialdichte; widerspruechliche Dichten werden abgelehnt.

`prepare_frame_case(parameters, fea_config=None, integration_config=None)` erzeugt ausschliesslich JSON-faehige FEA-Eingaben: Material, Akku-Punktmasse, vier Lastfaelle, Solvereinstellungen und Modellannahmen. `FrameEvaluator(reference_parameters, ...)` kapselt die festen Berechnungseinstellungen und ist als `evaluator(parameters)` direkt an Optuna uebergebbar. Bei jedem Kandidaten entstehen Solid, Motorpositionen, Selektoren und Akku-Schwerpunkt aus dessen Parameter-Dict. Die Referenzgeometrie wird dabei nicht wiederverwendet. Der Optimierungskern ruft den Geometrievalidator weiterhin vor der FEA auf.

## Berechnungsmodell

Der Akku kommt aus dem platzierten Komponentenmodell mit **37 g** und tatsaechlichem Schwerpunkt **(0, 0, 34,5) mm** bei v0. Sein MASS-Element wird an ein **4 mm breites Querband bei y = 0** auf den beiden Deckrails gekoppelt. Die Box reicht seitlich von -20,1 bis 20,1 mm und vertikal von 28,99 bis 29,01 mm; nur dort vorhandene Netzknoten werden gekoppelt. Das Deckenfenster liefert keine Knoten. Diese begrenzte Region wird starr, der restliche Frame bleibt verformbar. Es gibt keine eigene Rotationstraegheit des ausgedehnten Akkus und keine zusaetzlichen Motor-/Elektronikmassen. Die FEA-Masse ist daher Frame plus Akku; die vollstaendige Baugruppenmasse bleibt im Geometriecheck ausgewiesen.

| Lastfall | Last und Auswertung | Einspannung |
| --- | --- | --- |
| `arm_tip` | 1 N nach -z auf Oberseite des vorderen linken Motorpads; Zielsteifigkeit = Kraft / mittlerer Weg in Kraftrichtung | Unterseite des zentralen Basisrands, x/y jeweils ±18 mm; der 20 × 20 mm Ausschnitt bleibt leer |
| `battery_impact` | 10 g als statisches Aequivalent: 0,037 kg × 9,80665 m/s² × 10 = 3,6284605 N nach -z auf das Akku-Querband | Vier Motorpad-Unterseiten |
| `camera_side` | 5 N nach +x auf oberen Bereich beider Kamerakaefigwaende | Vier Motorpad-Unterseiten |
| `modes` | Erste sechs positive elastische Eigenfrequenzen ohne Vorlast | Vier Motorpad-Unterseiten |

Kraefte, g-Faktor, Patchbreite, Selektortoleranz und relative Fixture-Ausdehnungen stehen in `integration_config.py`. Sie sind **angenommene Vergleichsbedingungen**, keine Armattan-Daten, gemessenen Crashkraefte oder Flugrandbedingungen. Die Fixierung an den Motorpads entspricht einem idealisierten Pruefstand. Die statische Akku-Ersatzlast berechnet keine Aufprallzeit, Energieabsorption oder Bruchgrenze. Der obere Kamerabereich umfasst im Default die oberen 25 % der Kaefighoehe und die mittleren 50 % seiner Laenge. Boxen fuer Last und Einspannung werden je Kandidat aus Radstand, Layout, Padgroesse, Armhoehe und Kaefig-/Deckabmessungen neu berechnet.

Material bleibt der in [fea.md](fea.md) dokumentierte isotrope PA6-CF-Ersatz: E = 4430 MPa, nu = 0,30, rho = 1,09 g/cm³, vorlaeufige Bambu-Datenblattreferenz. Netzmaximum 3 mm, Minimum 0,5 mm, C3D10, ein Gmsh-Thread und zwei CalculiX-Threads. Die in [optimization.md](optimization.md) begruendeten fuenf relativen Constraints und beide Startkandidaten bleiben unveraendert.

## Installation, Aufruf und Fortsetzung

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-all.txt
.\.venv\Scripts\python.exe tools/install_calculix.py
.\.venv\Scripts\python.exe run_optimization.py
```

Der Runner verwendet ein einfaches `RUN_CONFIG`-Dict: vier Trials, persistente Studie `frame-v0-probe-v1`, SQLite unter `exports/optimization/probe/`, FEA-Artefakte unter `exports/fea/optimization_probe/`. OCP CAD Viewer muss fuer die abschliessende Kandidatenanzeige auf dem konfigurierten Port laufen; `show_viewer=False` ermoeglicht einen Lauf ohne Anzeige. Ein Wiederaufruf fuegt vier weitere Trials hinzu und verwendet dieselbe gespeicherte Referenz. Fuer eine neue unabhängige Abnahme sind neuer Studienname und Datenbankdatei einzutragen. Die beiden Startkandidaten sind weiterhin vollstaendig festgelegt: Armhoehe/Breite 4,2/6,5 mm und 4,0/6,7 mm.

`evaluation_id` ist ein SHA-256-Vertrag ueber Material, Last-/Kopplungsdefinitionen, Referenzpunktmasse und Selektoren, physische Netz-/Solvereinstellungen, erkannte CalculiX-Version und Binary-Pruefsumme, Gmsh/build123d/OCP-Versionen sowie die relevanten Implementierungsdateien. Dateizeilenenden werden vor dem Quellcode-Hash normalisiert. Worktree-, Solver- und Ausgabepfade gehen nicht in diesen Vertrag ein; ein portabler Pfadwechsel alleine erzeugt keine neue physische Referenz. Die identische Studie kann deshalb mit demselben Programmstand und denselben Werkzeugen in einem anderen Worktree fortgesetzt werden. Eine neue Solverversion, anderes Material, andere Lasten oder veraenderte Physik verlangt einen neuen Studiennamen. Der Codevertrag umfasst `integration.py`, `fea.py`, `fea_mesh.py`, `frame.py`, `components.py`, `geometry.py` und `checks.py`.

`result.json`, `pareto.json`, `pareto.csv` und SQLite bleiben als Laufartefakte lokal. Die spaetere Abnahme wird als vollstaendige JSON-Momentaufnahme und CSV unter `docs/validation/` versioniert. Der Frame-Runner exportiert ebenfalls lokal; ausschliesslich sein erzeugtes `exports/frame_v0.json` ist zusaetzlich ignoriert. Bereits versionierte JSON-Referenzen und Abnahmen bleiben sichtbar.

## Pruefung

Die Integrationstests pruefen echte CAD-Ueberschneidungen aller Fixture-, Last- und Punktmassenboxen, den Akku-Schwerpunkt, die 10-g-Kraft, Kandidatenabhaengigkeit der Selektoren, unveraenderte Eingaben, Dict-Serialisierbarkeit und den portablen Auswertungsvertrag. Ein Callback-Test prueft die Verdrahtung und Fehlerweitergabe; er ersetzt keine FEA-Abnahme. Der echte v0-Aufruf mit allen vier Lastfaellen und 37-g-Punktmasse ist die Referenzauswertung des anschliessenden Vier-Trial-Probelaufs. Die reale CalculiX-Balkenabnahme bleibt Bestandteil der Gesamtsuite.
