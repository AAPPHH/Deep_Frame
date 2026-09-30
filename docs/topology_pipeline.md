# Automatische Phase-1-Pipeline

`run_topology.py` startet mit `reference_parameters()` und dem einfachen Dict `SETTINGS` die vollstaendige klassische Pipeline. Ausfuehrung im Repository nach Installation von `requirements-all.txt` und Bereitstellung des vorhandenen CalculiX-Solvers:

```powershell
.\.venv\Scripts\python.exe run_topology.py
```

`deep_frame.topology_pipeline.run_topology(parameters, settings)` erzeugt die Domain, berechnet die unveraenderte parametrische v0 unter den gemeinsamen Vergleichsrandbedingungen, optimiert die uniforme freie 3D-Dichte, rekonstruiert Kandidaten, prueft Geometrie/Fertigung und bewertet nur gueltige Kandidaten mit der vorhandenen gmsh/CalculiX-FEA. Der Runner gibt nur Status, ausgewaehlte ID, Pareto-IDs und Manifestpfad aus. Der Viewer wird fuer einen unbeaufsichtigten Rechenlauf nicht vorausgesetzt.

Die oeffentlichen Callbacks `domain_builder`, `generator`, `reconstructor`, `validator`, `evaluator` und `baseline_builder` sind austauschbar. Der FEA-Callback behielt die bereits vorhandene Signatur `(solid, material, point_masses, load_cases, settings) -> dict`. Andere klassische oder spaetere ML-Generatoren koennen denselben Domain-/Dichtevertrag bedienen. Diese Phase importiert kein ML-Modell.

## Festgelegte Suche und Akzeptanz

Der Standardsuchplan optimiert einmal mit der jeweiligen Domain-Konfiguration und untersucht danach die vorab in `PIPELINE_CONFIG` festgelegten Dichteschwellen `0.20, 0.25, 0.30, 0.35, 0.40, 0.50`. Der breitere Schwellenplan entstand aus dem dokumentierten 20-Iterations-Pilotfeld: Hohe Schwellen trennten notwendige Preserves; 0,20 ergab einen verbundenen Kandidaten innerhalb des Massenlimits. Mechanische Grenzwerte wurden dadurch nicht veraendert. Weitere unabhaengige Dichtelaeufe lassen sich mit benannten `optimizer_variants` und ihren Settings-Ueberschreibungen automatisieren.

Die generische, protokollierte Reparatur diagonaler Voxelberuehrungen ist ausdruecklich aktiviert und auf 40 ergaenzte Zellen begrenzt. Sie erfindet keine Verbindungsarme zwischen getrennten Pflichtbereichen. Das Entfernen unverbundener Inseln ist standardmaessig deaktiviert. Genaue Reparatur- und Volumendaten gehoeren zum Rekonstruktionsbericht jedes Kandidaten.

Jeder mechanisch akzeptierte Kandidat muss zuerst alle exakten CAD-/Fertigungschecks bestehen. Danach gelten gegen v0 unveraendert:

| Kennwert | Phase-1-Grenze |
|---|---:|
| Frame-Masse ohne Akku-Punktmasse | hoechstens 2,0 x v0 |
| Armsteifigkeit | mindestens 0,5 x v0 |
| Erste elastische Eigenfrequenz | mindestens 0,7 x v0 |
| Maximale statische Verschiebung | hoechstens 2,0 x v0 |
| Maximale statische Vergleichsspannung | hoechstens 2,0 x v0 |

Diese Grenzen sind explizite technische Vergleichsschranken fuer den Methodennachweis. Sie sind keine Flugfreigabe und keine Festigkeitszulassung eines realen Drucks. Zu schwere Kandidaten werden nach exaktem CAD-Massenvergleich vor der teuren FEA verworfen.

V0 und Kandidat erhalten dieselben Materialdaten, denselben 37-g-Akku, dieselben physikalischen Lastselektoren und dieselben Netzeinstellungen. Die gemeinsame Armfixierung liegt unter den vier AIO-Anschluessen. Der Vergleich umfasst alle vier vorhandenen physikalischen Faelle: Armkraft, Akku-Aufprall-Ersatzlast, Kameraseitenschlag und Modalanalyse. Die zusaetzlichen Anschluss-Prooflasten bleiben vollstaendig in Domain und SIMP-Historie gespeichert; ihre erforderliche geometrische Anbindung wird zusaetzlich vor der FEA geprueft. Ein fehlender statischer oder modaler Vergleichsfall macht den Kandidaten ungueltig, selbst wenn ein Auswerter `status: ok` meldet.

Die Pareto-Front beruecksichtigt Frame-Masse, Armsteifigkeit, erste Eigenfrequenz, maximale Verschiebung und Vergleichsspannung. Nur Kandidaten, die alle Grenzen erfuellen, gehoeren zur Front. Fuer eine einzelne deterministische Auswahl minimiert die Pipeline `Massenverhaeltnis + 1/Steifigkeitsverhaeltnis + 1/Frequenzverhaeltnis + Verschiebungsverhaeltnis + Spannungsverhaeltnis`. Die einzelnen Kennwerte und die gesamte Pareto-Front bleiben erhalten. Eine einelementige Pareto-Front ist zulaessig.

## Reproduzierbare Daten und Wiederaufnahme

Jeder physisch unterschiedliche Lauf liegt unter `exports/topology/phase1/<erste-16-Zeichen-des-Input-SHA256>/`. Die vollstaendige SHA256-Identitaet wird aus Parametern, Domain, Masken, Solver-/Rekonstruktions-/FEA-Settings, Quellenhashes, Callbackidentitaet und Toolversionen gebildet. Ein veraenderter Selektor, eine neue Preserve-Maske, eine andere Materialdichte oder ein anderer Code erzeugt einen anderen Lauf. Der Git-Commit wird dokumentiert; reine Merge-Commit-Aenderungen bei identischen Codehashes erzwingen keine inhaltlich unnoetige Neuberechnung.

Vor der ersten FEA werden `inputs.json` und `domain_masks.npz` geschrieben. `inputs.json` ist unveraenderlich; eine nachtraegliche Manipulation fuehrt zum Fehler. `manifest.json` ist das fortgeschriebene Statusjournal und verweist auf:

- Vollstaendige Parameter, generische Regionsprimitive, Materialien, Punktmassen, alle Lastfaelle, Fertigungsbedingungen, Seed, Einheiten, Tool-/Sourcehashes und Gitterlayout.
- `baseline/fea.json`, v0-STEP/STL und die vorhandenen Rohdateien der unabhaengigen FEA.
- `optimizer/<Variante>/fields.npz` mit physischer Dichte, Designvariablen und Allowed-/Preserve-/Forbiddenmasken; `result.json` mit Settings, Iterationshistorie, Compliancewerten, Konvergenz und Numerikdiagnostik.
- `candidates/<ID>/record.json`, STEP/STL, exakte Geometrie-/Fertigungsscreens, Rekonstruktionsreparaturen, FEA-Resultate und Rohdateien sowie die dimensionslosen v0-Vergleiche.
- `pareto.json`, `pareto.csv`, ausgewaehlte Kandidaten-ID und auch gescheiterte oder ungueltige Designs mit Fehlerphase und Ursache.

`latest.json` im Ausgabestamm zeigt auf das aktuelle Manifest. Alle Artefakte tragen Dateihashes. Wiederaufnahme benutzt vorhandene Ergebnisse nur bei identischen Inputs und intakten Artefakten; beschaedigte Felddaten oder fehlende FEA-Dateien werden neu berechnet. Ein fehlgeschlagener, bereits protokollierter Kandidat wird bei identischer Konfiguration nicht in einer Endlosschleife erneut versucht. Andere Settings oder deaktiviertes `resume` starten eine neue Auswertung.

Die Rohdaten sind lokal ignoriert und fuer einen spaeteren Trainingsdatensatz bereits nach Domain, Generatorzustand, akzeptierten/abgelehnten Formen und unabhaengigen mechanischen Labels getrennt. Vollstaendige Solverartefakte werden nicht als grosse Rohdateien in Git aufgenommen; kompakte Abnahmeberichte und Visualisierungen koennen gezielt in `docs/validation` versioniert werden.

Die Pipeline-Tests verwenden gezielte injizierte Auswerter, um Resume, Datenintegritaet, unveraenderte Eingaben, Paretoauswahl, Fehlermodi und harte Pre-FEA-Gates isoliert zu pruefen. Sie ersetzen nicht den gesonderten realen Phase-1-Lauf mit freier SIMP-Synthese und CalculiX-Verifikation.
