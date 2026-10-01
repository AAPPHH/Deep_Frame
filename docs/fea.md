# FEA mit Gmsh und CalculiX

`deep_frame.fea.evaluate(solid, material, point_masses, load_cases, settings)` implementiert den Vertrag in [interfaces.md](interfaces.md). Alle Eingaben bleiben unveraendert. Das Ergebnis ist ein JSON-faehiges Dict; ungueltige Eingaben liefern `status="invalid"`, Werkzeugfehler `status="failed"`. Fehlende Messwerte bleiben `null`. Solverfehler werden nicht durch analytische Ersatzwerte ueberdeckt.

## Installation und Betrieb

Ergaenzend zur vorhandenen Python-Umgebung:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-fea.txt
.\.venv\Scripts\python.exe tools/install_calculix.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_fea.py
```

Der Installer laedt das vom [CalculiX-Autor angebotene Windows-Archiv](https://www.dhondt.de/calculix_2.22_4win.zip), prueft SHA256 `a1f91281944c96d6cd914cc020421e8ae65973b3e15d055dc63a3e3e3066d281` und entpackt ausschliesslich in `.venv/calculix`. Verwendet wird `ccx_static.exe` Version 2.22; das eigenstaendige Binary benoetigt hier keine weitere Installation. Die [offizielle Downloadseite](https://www.dhondt.de/) bietet weitere Plattformen und neuere Versionen. Keine Solver-Binaries werden versioniert.

Suchreihenfolge: `settings.solver_path`, Umgebungsvariable `CALCULIX_PATH`, `ccx`/`ccx_static`/`ccx_dynamic` im PATH, danach `sys.prefix/calculix/**/ccx_static.exe`. Ein ausdruecklich angegebener fehlender Pfad ist ein Fehler. Auf anderen Betriebssystemen muss ein passendes CalculiX-Binary installiert werden; der mitgelieferte Download ist fuer Windows. Das Add-on pinnt Gmsh 4.15.2.

`settings.linear_solver` erlaubt die explizite Auswahl `"SPOOLES"` oder `"PASTIX"` fuer statische und modale Rechnungen. Fehlend oder `None` behaelt das native Standardbackend bei; es gibt keinen automatischen Fallback. Das gewaehlte Backend muss im Binary enthalten sein. `settings.threads` setzt die OMP-, CCX-Stiffness-, CCX-Results- und CCX-Equation-Solver-Threadvariablen sowie `NUMBER_OF_CPUS`; geerbte CCX-Overrides werden dabei ueberschrieben. Ergebnisse enthalten `linear_solver` und `solver_threads`. Neue Workstation-Verifikationen setzen nach dokumentierten nativen Abbruechen ausdruecklich `linear_solver="SPOOLES", threads=1`. Diagnose, Grenzen dieser Beobachtung und vollstaendige Aufrufbeispiele stehen in [workstation_solver_diagnosis.md](validation/workstation_solver_diagnosis.md). Die vorhandenen Defaults und historischen Abnahmeresultate wurden nicht umgestellt.

Jede Auswertung legt ein eigenes Verzeichnis unter `settings.work_dir` an. Dort liegen STEP, Gmsh-Netz, CalculiX-Eingaben, Ergebnisse, Logs und `result.json`. `exports/fea/` ignoriert diese grossen Laufartefakte. Die Programme oeffnen keine Konsolenfenster. Der Netzgenerator laeuft als eigener Prozess `python -m deep_frame.fea <request>`. Netzgenerator und jeder Solverlauf haben getrennte Timeouts, standardmaessig 180 s. Es gibt keine eigenen Status-Prints.

## Material, Einheiten und Vernetzung

`FEA_CONFIG` in [config.py](../deep_frame/config.py) ist ein unabhaengiges Config-Dict. Referenz sind Bambu PA6-CF-Daten: Dichte **1.09 g/cm3**, Zugmodul XY **4430 +/- 310 MPa**, Zugmodul Z **2170 +/- 230 MPa**. Der vorlaeufige isotrope Modul ist 4430 MPa. **Poissonzahl 0.30 ist eine unbelegte Modellannahme**. Das echte Filament bleibt offen. Diese Daten beschreiben keine beliebige Druckorientierung; besonders die Z-Steifigkeit wird mit dem XY-Ersatzmaterial ueberschaetzt. Quelle: [Bambu PA6-CF Technical Data Sheet v2](https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf).

Intern gilt mm-N-s mit Tonnen: `density_g_cm3 * 1e-9` ergibt t/mm3, `mass_g * 1e-6` Tonnen. Die ausgegebene Gesamtmasse ist CAD-Volumen mal Dichte plus uebergebene Punktmassen. Dies ist die Masse des FEA-Modells; sie ist von der vollstaendigen Baugruppenmasse der Geometriechecks zu unterscheiden.

Die Geometrie gelangt per STEP in Gmsh/OpenCASCADE. Es wird genau ein Volumen mit quadratischen Tetraedern C3D10 vernetzt. Gmsh schreibt die Abaqus-Knotenreihenfolge selbst. Positive minimale Jacobi-Determinanten und zusammenhaengende Elementknoten werden geprueft. Grundlage: [offizielles Gmsh-Handbuch](https://gmsh.info/doc/texinfo/).

`mesh_size_mm=3.0` ist die maximale Zielgroesse, kein Versprechen identischer Kantenlaengen. Kleine Bohrungen erzwingen kleinere Elemente; `mesh_min_size_mm=0.5`, `mesh_curvature_points=12`. `element_order=2` ist verpflichtend. `mesh_threads=1` vermeidet parallele Netzvariationen, `threads=2` begrenzt die Solverparallelitaet. Dies ist ein Screening-Netz; eine Frame-Netzkonvergenzstudie bleibt vor belastbaren Spannungsaussagen erforderlich. Maximalspannungen sind Werte an Integrationspunkten, keine ausgerundeten oder extrapolierten CAD-Kerbspannungen.

## Selektoren, Lasten und Punktmassen

Box-Selektoren waehlen alle enthaltenen Netzknoten. Jede Fixture fixiert deren drei Translationen. Mindestens drei nicht kollineare Fixture-Knoten sind erforderlich; ein leerer Selektor, Last auf fixierten Knoten oder Ueberlappung von Fixture und Punktmassenkopplung fuehrt zum Fehler. Freie Flugmoden ohne Fixture werden in dieser Version nicht berechnet. Alle Lastfaelle erhalten eigene Solverjobs, damit keine vorherige Last oder Randbedingung unbemerkt weiterwirkt.

Eine Kraft ist die **Gesamtkraft** auf die Region. Sie wird gleichmaessig nach Knotenanzahl verteilt. Das ist keine flaechentreue Druckintegration; bei ungleichmaessigen Netzen aendert sich die resultierende Flaechenlastverteilung. Die Aufteilung und Selektorknotenzahlen stehen im Ergebnis. Pro Last wird die mittlere Knotenverschiebung in Kraftrichtung ausgewertet; `stiffness_n_per_mm = |F| / u_directional`. Das Gesamtziel entstammt dem expliziten `settings.stiffness_load_case`, der genau eine Last besitzen muss.

Fuer den Frame sind folgende definierte Vergleichslasten vorgesehen; deren Zahlen sind **Modellannahmen**, keine gemessenen Crashdaten:

| Fall | Kraft | Fixture |
|---|---|---|
| `arm_tip` | 1 N nach unten auf ein freies Motorpad | Zentraler Basisbodenrand/AIO-Bereich; Zentralausschnitt liefert keine Knoten |
| `battery_impact` | `battery_mass_g / 1000 * 9.80665 * g_factor`, z.B. 10 g = 3.6284605 N bei 37 g | Vier Motorpad-Unterflaechen |
| `camera_side` | 5 N seitlich auf den oberen Kamerakaefig | Vier Motorpad-Unterflaechen |
| `modes` | Keine Vorlast, erste sechs Moden | Vier Motorpad-Unterflaechen |

Der Integrationsadapter leitet die konkreten Boxen aus den Frameparametern ab. Alle Kraefte, der g-Faktor und die Fixturekoordinaten bleiben aenderbar. Der Akku-Aufprall ist eine aequivalente **statische** Traegheitslast. Keine Kontaktzeit, Aufprallgeschwindigkeit oder Energieabsorption wird daraus abgeleitet.

Eine Punktmasse ist ein CalculiX-MASS-Element an `position_mm`. `*RIGID BODY` koppelt die angegebene Patchregion an diesen Referenzpunkt. Damit geht der Abstand des Akkuschwerpunkts zum Deck in die Kinematik ein. **Die gewaehlte Patchregion wird starr und versteift lokal den Frame.** Dies ist die Annahme einer starren Akkuauflage, kein Nachweis eines realen LiPo-Verbunds. Fuer v0 eignet sich ein begrenztes, parametrierbares 4-mm-Querband bei y=0 auf den Deckrails; nicht die gesamte Deckflaeche. Mindestens drei nicht kollineare Patchknoten sind erforderlich. Mehrere Punktmassen duerfen keine Knoten teilen. Modelliert wird keine eigenstaendige Rotationstraegheit des ausgedehnten Akkus. Der Rest der Elektronik und Motoren ist nur enthalten, wenn weitere Punktmassen uebergeben werden.

Die nativen MASS-, RIGID-BODY-, FREQUENCY- und Ausgabekarten sind im [offiziellen CalculiX-Handbuch](https://github.com/Dhondtguido/CalculiX/blob/master/doc/CalculiX.tex) beschrieben. Der Adapter meldet positive elastische Frequenzen oberhalb des konfigurierbaren 0.1-Hz-Grenzwerts; verworfene Nullmoden werden gezaehlt. Negative Eigenwerte oder fehlende Frequenzen sind Fehler.

## Reale Solver-Abnahme

`tests/test_fea.py` startet echte Gmsh- und CalculiX-Prozesse; fehlende Tools lassen diese Tests fehlschlagen. Es gibt keine Solver-Mocks und kein stilles Skip. Der Balken ist **80 x 8 x 4 mm**, an x=0 auf der ganzen Querschnittsflaeche eingespannt, am anderen Ende mit 1 N nach z belastet. C3D10-Zielgroesse **2 mm**; E=4430 MPa, rho=1090 kg/m3, nu=0.30. Der Balken ist in der Biegerichtung schlank (L/h=20).

Die unabhaengige analytische Berechnung erfolgt in SI: `I = b*h**3/12`, `delta = F*L**3/(3*E*I)` und `f1 = beta1**2/(2*pi)*sqrt(E*I/(rho*A*L**4))`. Ohne Endmasse gilt beta1=1.875104... . Mit Endmasse M wird die kleinste positive Nullstelle von `1 + cos(beta)*cosh(beta) + mu*beta*(cos(beta)*sinh(beta) - sin(beta)*cosh(beta))` mit `mu=M/(rho*A*L)` verwendet. Der Test loest sie durch Bisektion; er benutzt keinen FEA-Messwert als Referenz.

| Messung, erster bestaetigter Lauf | Analytisch | CalculiX | Relativer Fehler |
|---|---:|---:|---:|
| Endverschiebung ohne Endmasse | 0.902934537 mm | 0.897371365 mm | -0.6161 % |
| Erste Frequenz ohne Endmasse | 203.539589 Hz | 204.230000 Hz | +0.3392 % |
| Erste Frequenz mit 1.5 g Endmasse | 113.868491 Hz | 114.235500 Hz | +0.3223 % |
| Balkenmasse | 2.790400 g | 2.790400 g | Rundung |

Die Abnahmegrenze betraegt 3 % fuer Verschiebung und Frequenz. Sie beruecksichtigt die 3D-Einspannung, Schub-/Querkontraktionseffekte gegenueber Euler-Bernoulli und das endliche Netz. Die beobachteten Fehler liegen unter 1 %. Die Endmasse senkt die Frequenz, erhoeht die Gesamtmasse genau um 1.5 g und aendert die statische Steifigkeit am schmalen Balkenende um weniger als 0.5 %. Sieben FEA-Tests bestanden; Laufzeit im ersten vollstaendigen Testlauf 7.53 s. Kleine Netz-/Solverabweichungen zwischen Plattformen sind moeglich.

Ein zusaetzlicher vorlaeufiger Frame-Smoke mit 3-mm-Netz lief in **53.87 s** mit allen drei statischen Lasten und Modalanalyse: 45491 Tetraeder, 83406 Knoten, Akku-Patch mit 58 Knoten. Dieser Lauf beweist die Werkzeugkette fuer komplexe Framegeometrie. Seine Zahlen ersetzen nicht den spaeteren integrierten v0- und Optimierungsbericht.

Offen bleiben reale Materialkennwerte und Druckorientierung, Netzkonvergenz am Frame, Sensitivitaet gegen Fixture und Akku-Patchbreite, Feuchte, Porositaet/Infill, grosse Verformung, Plastizitaet, Bruch, reale Crashtransienten und Bauteildaempfung. Die lineare Rechnung vergleicht parametrische Entwuerfe unter den genannten Randbedingungen; sie ist keine Crashfreigabe.

## Frame-v0-Integration und Probelauf

Die zentrale `deep_frame.config.CONFIG` referenziert jetzt Geometrie, Komponenten, Material, `fea`, `integration`, `optimization` und `optimization_search_space`. Das bestehende Smoke-Dict mit Laenge, Breite, Dicke und STL-Pfad bleibt erhalten. `reference_parameters()` entfernt den Smoke-Path und liefert eine unabhaengige JSON-faehige Kopie. Geometriechecks und FEA verwenden dieselbe zentrale Materialdichte; widerspruechliche Dichten werden abgelehnt.

`prepare_frame_case(parameters, fea_config=None, integration_config=None)` erzeugt ausschliesslich JSON-faehige FEA-Eingaben: Material, Akku-Punktmasse, vier Lastfaelle, Solvereinstellungen und Modellannahmen. `FrameEvaluator(reference_parameters, ...)` kapselt die festen Berechnungseinstellungen und ist als `evaluator(parameters)` direkt an Optuna uebergebbar. Bei jedem Kandidaten entstehen Solid, Motorpositionen, Selektoren und Akku-Schwerpunkt aus dessen Parameter-Dict. Die Referenzgeometrie wird dabei nicht wiederverwendet. Der Optimierungskern ruft den Geometrievalidator weiterhin vor der FEA auf.

### Berechnungsmodell

Der Akku kommt aus dem platzierten Komponentenmodell mit **37 g** und tatsaechlichem Schwerpunkt **(0, 0, 34,5) mm** bei v0. Sein MASS-Element wird an ein **4 mm breites Querband bei y = 0** auf den beiden Deckrails gekoppelt. Die Box reicht seitlich von -20,1 bis 20,1 mm und vertikal von 28,99 bis 29,01 mm; nur dort vorhandene Netzknoten werden gekoppelt. Das Deckenfenster liefert keine Knoten. Diese begrenzte Region wird starr, der restliche Frame bleibt verformbar. Es gibt keine eigene Rotationstraegheit des ausgedehnten Akkus und keine zusaetzlichen Motor-/Elektronikmassen. Die FEA-Masse ist daher Frame plus Akku; die vollstaendige Baugruppenmasse bleibt im Geometriecheck ausgewiesen.

| Lastfall | Last und Auswertung | Einspannung |
| --- | --- | --- |
| `arm_tip` | 1 N nach -z auf Oberseite des vorderen linken Motorpads; Zielsteifigkeit = Kraft / mittlerer Weg in Kraftrichtung | Unterseite des zentralen Basisrands, x/y jeweils ±18 mm; der 20 × 20 mm Ausschnitt bleibt leer |
| `battery_impact` | 10 g als statisches Aequivalent: 0,037 kg × 9,80665 m/s² × 10 = 3,6284605 N nach -z auf das Akku-Querband | Vier Motorpad-Unterseiten |
| `camera_side` | 5 N nach +x auf oberen Bereich beider Kamerakaefigwaende | Vier Motorpad-Unterseiten |
| `modes` | Erste sechs positive elastische Eigenfrequenzen ohne Vorlast | Vier Motorpad-Unterseiten |

Kraefte, g-Faktor, Patchbreite, Selektortoleranz und relative Fixture-Ausdehnungen stehen in `INTEGRATION_CONFIG` von `deep_frame/config.py`. Sie sind **angenommene Vergleichsbedingungen**, keine Armattan-Daten, gemessenen Crashkraefte oder Flugrandbedingungen. Die Fixierung an den Motorpads entspricht einem idealisierten Pruefstand. Die statische Akku-Ersatzlast berechnet keine Aufprallzeit, Energieabsorption oder Bruchgrenze. Der obere Kamerabereich umfasst im Default die oberen 25 % der Kaefighoehe und die mittleren 50 % seiner Laenge. Boxen fuer Last und Einspannung werden je Kandidat aus Radstand, Layout, Padgroesse, Armhoehe und Kaefig-/Deckabmessungen neu berechnet.

Material bleibt der [oben](#material-einheiten-und-vernetzung) dokumentierte isotrope PA6-CF-Ersatz: E = 4430 MPa, nu = 0,30, rho = 1,09 g/cm³, vorlaeufige Bambu-Datenblattreferenz. Netzmaximum 3 mm, Minimum 0,5 mm, C3D10, ein Gmsh-Thread und zwei CalculiX-Threads. Die in [optimization.md](optimization.md) begruendeten fuenf relativen Constraints und beide Startkandidaten bleiben unveraendert.

### Installation, Aufruf und Fortsetzung

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-all.txt
.\.venv\Scripts\python.exe tools/install_calculix.py
.\.venv\Scripts\python.exe run.py optimization
```

Der Runner verwendet ein einfaches `RUN_CONFIG`-Dict: vier Trials, persistente Studie `frame-v0-probe-v1`, SQLite unter `exports/optimization/probe/`, FEA-Artefakte unter `exports/fea/optimization_probe/`. OCP CAD Viewer muss fuer die abschliessende Kandidatenanzeige auf dem konfigurierten Port laufen; `show_viewer=False` ermoeglicht einen Lauf ohne Anzeige. Ein Wiederaufruf fuegt vier weitere Trials hinzu und verwendet dieselbe gespeicherte Referenz. Fuer eine neue unabhängige Abnahme sind neuer Studienname und Datenbankdatei einzutragen. Die beiden Startkandidaten sind weiterhin vollstaendig festgelegt: Armhoehe/Breite 4,2/6,5 mm und 4,0/6,7 mm.

`evaluation_id` ist ein SHA-256-Vertrag ueber Material, Last-/Kopplungsdefinitionen, Referenzpunktmasse und Selektoren, physische Netz-/Solvereinstellungen, erkannte CalculiX-Version und Binary-Pruefsumme, Gmsh/build123d/OCP-Versionen sowie die relevanten Implementierungsdateien. Dateizeilenenden werden vor dem Quellcode-Hash normalisiert. Worktree-, Solver- und Ausgabepfade gehen nicht in diesen Vertrag ein; ein portabler Pfadwechsel alleine erzeugt keine neue physische Referenz. Die identische Studie kann deshalb mit demselben Programmstand und denselben Werkzeugen in einem anderen Worktree fortgesetzt werden. Eine neue Solverversion, anderes Material, andere Lasten oder veraenderte Physik verlangt einen neuen Studiennamen. Der Codevertrag umfasst `deep_frame/fea.py` und `deep_frame/frame.py`; darin sind Integration, FEA, Vernetzung, Frame, Komponenten, Geometrie und Checks zusammengefasst. Die Zusammenfassung der Module aenderte diese Quellcodehashes und damit `evaluation_id`; die bestehende Studie `frame-v0-probe-v1` ist deshalb nur mit dem frueheren Programmstand fortsetzbar, ein neuer Lauf braucht einen neuen Studiennamen.

`result.json`, `pareto.json`, `pareto.csv` und SQLite bleiben als Laufartefakte lokal. Die spaetere Abnahme wird als vollstaendige JSON-Momentaufnahme und CSV unter `docs/validation/` versioniert. Der Frame-Runner exportiert ebenfalls lokal; ausschliesslich sein erzeugtes `exports/frame_v0.json` ist zusaetzlich ignoriert. Bereits versionierte JSON-Referenzen und Abnahmen bleiben sichtbar.

### Pruefung

Die Integrationstests pruefen echte CAD-Ueberschneidungen aller Fixture-, Last- und Punktmassenboxen, den Akku-Schwerpunkt, die 10-g-Kraft, Kandidatenabhaengigkeit der Selektoren, unveraenderte Eingaben, Dict-Serialisierbarkeit und den portablen Auswertungsvertrag. Ein Callback-Test prueft die Verdrahtung und Fehlerweitergabe; er ersetzt keine FEA-Abnahme. Der echte v0-Aufruf mit allen vier Lastfaellen und 37-g-Punktmasse ist die Referenzauswertung des anschliessenden Vier-Trial-Probelaufs. Die reale CalculiX-Balkenabnahme bleibt Bestandteil der Gesamtsuite.
