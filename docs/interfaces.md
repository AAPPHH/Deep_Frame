# Verbindliche Schnittstellen v1

Dieser Vertrag wird vor Beginn der parallelen Implementierung eingefroren. Die Implementierungen arbeiten in eigenen Git-Worktrees. Python-Code bleibt ohne Kommentare, Docstrings, argparse und unnoetige Prints. Konfigurationen sind einfache Dictionaries. Keine Geometrie eines Referenzrahmens wird kopiert.

## Einheiten und Daten

- Geometrie und Verschiebungen: mm; Kraefte: N; Spannung und Elastizitaetsmodul: MPa; Zeit: s; Frequenz: Hz.
- Oeffentliche Massen: g; Dichte: g/cm3; Massentraegheit: g mm2.
- CalculiX verwendet intern das konsistente mm-N-s-System mit Masse in Tonnen. Die FEA implementiert und testet diese Umrechnung ausdruecklich.
- Koordinaten: x seitlich, y nach vorne, z nach oben. Z=0 ist die unterste Druckflaeche des Frames.
- Ergebnis-Dicts enthalten ausschliesslich JSON-faehige Werte. Nicht berechenbare Werte sind null, niemals NaN oder erfundene Null-Ergebnisse.
- Parameter-Dicts werden von einer Auswertung nicht veraendert. Trial-Parameter werden tief kopiert.

## A: Geometrie und geometrische Checks

`deep_frame.frame.build_geometry(parameters: dict) -> build123d.Solid`

`parameters` enthaelt mindestens `frame` und `components`, entsprechend dem zentralen Config-Dict. Die Funktion liefert ausschliesslich den einteiligen Frame ohne Komponenten. Ungueltige Parameter, getrennte Solids oder ungueltige Geometrie fuehren zu ValueError.

`deep_frame.frame.validate_geometry(parameters: dict) -> dict`

Das Ergebnis enthaelt `passed: bool`, `violations: list[str]` und `checks: dict`. Es prueft mindestens einzelnen geschlossenen Solid, Montagepositionen, Kollisionen und erforderliche Freiraeume. Geplante Auflagekontakte werden ausdruecklich bezeichnet; positive Durchdringungsvolumina werden niemals als Auflagekontakt freigegeben. Die Funktion ruft keine FEA auf.

`deep_frame.frame.reference_parameters() -> dict`

Liefert eine unabhaengige, JSON-faehige Kopie der v0-Referenzparameter aus der zentralen Config. Quellen und Annahmen werden im Repo dokumentiert. Der Adapter darf JSON-faehige Metadaten enthalten, aber keine Shapes oder Paths.

Komponenten bleiben ueber `build_component_prototypes(config)` und `build_components(config, placements=None)` verfuegbar. Sie liefern Shapes, Massen, Schwerpunkt- und Montagekoordinaten. Die Referenzakku-Punktmasse betraegt 37 g; ihre Position ist der tatsaechliche Schwerpunkt des platzierten Akku-Platzhalters.

## B: FEA

`deep_frame.fea.evaluate(solid, material: dict, point_masses: list[dict], load_cases: list[dict], settings: dict) -> dict`

`material` hat `young_modulus_mpa`, `poisson_ratio`, `density_g_cm3`; optionale Felder sind `name`, `source` und `assumptions`. Isotropie ist eine explizite Modellannahme und keine Aussage ueber die reale Druckanisotropie.

`settings` enthaelt mindestens `mesh_size_mm`, `num_modes`, `work_dir`, `solver_path`. Weitere Netzparameter sind erlaubt und werden dokumentiert. Externe Werkzeuge werden mit begrenzter Laufzeit aufgerufen; fehlende Werkzeuge oder Solverfehler werden als fehlgeschlagene Auswertung kenntlich gemacht.

Ein Selektor ist `{kind: "box", min_mm: [x,y,z], max_mm: [x,y,z]}` und waehlt Netzknoten in einem Quader. Kein ausgewaehlter Knoten ist ein Fehler, kein stiller Lastverzicht.

Eine Punktmasse hat `name`, `mass_g`, `position_mm` und `attachment_region` (Selektor). Der FEA-Adapter koppelt sie nachvollziehbar an diese Region. Koppelart und Einfluss auf die Steifigkeit werden dokumentiert. Eine optionale Point-Mass-Alternative fuer den Balkentest darf an einen explizit ausgewiesenen Knoten koppeln.

Jeder Lastfall hat `name`, `analysis` ("static" oder "modal") und `fixed_regions` (Liste von Selektoren). Statische Lastfaelle enthalten `loads`, eine Liste aus `{region: Selektor, force_n: [Fx,Fy,Fz]}`. `force_n` ist die Gesamtkraft auf die Region und wird nicht pro Knoten vervielfacht. Modale Randbedingungen werden ausdruecklich festgehalten; starre Nullmoden werden nicht als erste elastische Eigenfrequenz ausgegeben.

Der Integrationsadapter bildet Armspitzenkraft, Akku-Aufprall als `mass_g / 1000 * 9.80665 * g_factor` und seitlichen Kameraschlag auf diese statischen Lastfaelle ab. Der Bezugsweg fuer Steifigkeit ist die Verschiebung in Kraftrichtung am belasteten Gebiet des bezeichneten Steifigkeitslastfalls.

Das Ergebnis hat stets folgende Schluessel:

```json
{
  "status": "ok",
  "mass_g": 0.0,
  "eigenfrequencies_hz": [],
  "max_displacement_mm": 0.0,
  "max_von_mises_mpa": 0.0,
  "stiffness_n_per_mm": 0.0,
  "load_cases": {},
  "diagnostics": [],
  "artifacts": {}
}
```

Die Nullen sind Schema-Beispiele, keine zulassigen Ersatzwerte fuer fehlgeschlagene Berechnungen. `status` ist "ok", "invalid" oder "failed". `mass_g` umfasst Frame und uebergebene Punktmassen; Einzelanteile werden zusaetzlich ausgewiesen. `load_cases` liefert pro Fall mindestens Verschiebung und Vergleichsspannung bei statischen Faellen beziehungsweise Frequenzen bei Modalfaellen. Gesamtmaxima beziehen sich auf die statischen Lastfaelle. `stiffness_n_per_mm` bezieht sich auf `settings.stiffness_load_case`; die genaue Auswertung wird zusaetzlich pro Lastfall ausgegeben.

Der analytische Kragbalkentest prueft Durchbiegung und erste Eigenfrequenz mit expliziter Geometrie, Randbedingung, SI-Umrechnung, Netzfeinheit und begruendeter Toleranz. Kein Mock ersetzt diesen Solver-Abnahmetest.

## C: Optimierung

`deep_frame.optimization.optimize(reference_parameters: dict, search_space: dict, evaluator, validator, settings: dict) -> dict`

`evaluator(parameters: dict) -> dict` liefert das FEA-Ergebnisschema. Es kann durch eine analytische Balkenbewertung ersetzt werden. Der Optimierer importiert weder Geometrieaufbau noch FEA-Implementierung und erhaelt keine Shapes.

`validator(parameters: dict) -> dict` liefert das Geometrie-Validierungsschema. Er wird zusammen mit den harten Druckregeln VOR dem Aufruf des Evaluators ausgefuehrt. Ungueltige Designs erreichen die FEA nicht.

`search_space` verwendet Punktpfade, z.B. `frame.arm_height_mm`, mit `{low, high, step}` fuer numerische Parameter. Diskrete Alternativen koennen ueber `choices` dargestellt werden. Nicht optimierte Werte bleiben unveraendert.

Ziele: Masse minimieren, Steifigkeit und erste elastische Eigenfrequenz maximieren. Relative Constraints werden einmal gegen eine erfolgreich ausgewertete Referenz festgelegt und als gemessener Wert, Grenzwert und passed-Status gespeichert. Die Referenz darf nicht nachtraeglich durch einen besseren Kandidaten ersetzt werden.

Harte Druckregeln enthaelt `settings.printability`: `nozzle_width_mm`, `minimum_wall_nozzles`, sowie explizite Parameterpfade fuer Wandstaerken und Querschnittspaare mit ihren `minimum_area_mm2`. Diese Regeln werden nicht allein aus Parameterbezeichnungen erraten. Minimale Wandstaerke = Duesenbreite mal geforderte Anzahl Bahnen. Mindestquerschnitte an Arm-/Einspannstellen werden separat geprueft.

Weitere Settings: `n_trials`, `seed`, `storage` (persistente Optuna-Datenbank), `study_name`, relative Constraint-Faktoren und `output_dir`. Deterministische Startkandidaten sind erlaubt, muessen aber unveraendert durch Validator, FEA und Constraints laufen. Ein fehlgeschlagener Solverlauf ist niemals ein gueltiger Pareto-Kandidat.

Ergebnis: JSON-faehiges Dict mit Referenzergebnis, Studie, Trial-Zaehlern, gueltiger Pareto-Front, Constraint-Werten und Artefaktpfaden. Pareto-Export als JSON oder CSV. Viewer-Anzeige ist ein separater Adapter, der Parametersets an die Geometriefunktion uebergibt; der Optimierungskern bleibt austauschbar.

Der analytische Abnahmetest definiert ein Problem mit bekanntem Optimum bzw. bekannter Pareto-Menge und prueft die gefundene Loesung mit fester Seed/Toleranz. Ein zweiter Test beweist, dass Designs unter der Mindestwandstaerke den Evaluator nie aufrufen.

## Worktrees, Ownership und Integration

- A: Referenzrecherche, Komponenten, Frame, geometrische Checks, Geometrieadapter, zugehoerige Tests und Dokumente; zentrale bestehende Config.
- B: `deep_frame/fea.py` oder Paket `deep_frame/fea/`, FEA-Config als separat importierbares Dict, `requirements-fea.txt`, Solverbeschaffung/-dokumentation, FEA-Tests.
- C: `deep_frame/optimization.py` oder Paket `deep_frame/optimization/`, Optimierungs-Config als Dict, `requirements-optimization.txt`, Optimierungstests und Viewer-Kandidatenadapter.
- Jede Arbeit erfolgt auf eigenem Branch/Worktree; die Ausgangs-.venv ist kein geteiltes Installationsziel. A darf sie unveraendert fuer vorhandene Pakete verwenden. B und C nutzen eigene Umgebungen.
- Kein Subagent pusht oder merged eigenstaendig nach main. Er erstellt klar abgegrenzte Schritt-Commits und meldet Testnachweise sowie Commit-IDs.
- Der Orchestrator uebernimmt Commit-/Worktree-/Merge-/Push-Koordination und delegiert jede Implementierung sowie Integrationskorrektur.
- I1: Branches integrieren und gemeinsame Tests ausfuehren; Korrekturen an die zustaendigen Subagents delegieren.
- I2: Agent implementiert den Adapter von v0-Parametern zu Solid, Akku-Punktmasse und FEA-Lastfaellen, dann zum Evaluator.
- I3: Agent fuehrt wenige echte FEA-Trials aus, exportiert die Pareto-Front und verifiziert mindestens ein gegen v0 verbessertes Ziel ohne Constraint-Verletzung. Gelingt das nicht, wird die Ursache behoben oder als offener Abnahmepunkt berichtet; Ergebnisse werden nicht erfunden.

## Ausgangsstand und offene Annahmen

Checkpoint `7a61253` ist ein Zwischenstand, keine fertige Frame-Abnahme. Rohrecherche: 36 Micro- und 39 grosse/historische Eintraege in `docs/research`; `frame_seed.json` enthaelt einen noch nicht final integrierten 135-mm-Entwurf.

XT30, Balancer und VTX-Antenne haben im Designraum keine Schnittstelle mehr (Gummiband, frei platziert); nur der parametrische v0-Referenzrahmen enthaelt noch Steckeraufnahmen und Antennenbohrung. Die Frame-Motorbohrungen verwenden bisher ein Quadrat, waehrend Komponenten bereits zwei Lochbilder unterstuetzen.

Belegte Motorreferenz: GEPRC GR1105, Durchmesser 14.2 mm, Hoehe ab Montageflaeche inkl. oberer Welle 14.6 mm, Masse 5.9 g inkl. abgebildeter Kabel. Zeichnung zeigt vier M2-Bohrungen auf 9-mm-Lochkreis, kein Quadrat mit 9-mm-Seiten. Quellen: https://geprc.com/product/gep-gr1105-motor/ , https://geprc.com/wp-content/uploads/2019/05/22-6199766706.jpg , https://geprc.com/wp-content/uploads/2019/05/22-8095453337.jpg . Die Nutzerfrage zu dieser Unterscheidung ist noch offen; beide Lochbilder bleiben parametrierbar und die getroffene Default-Annahme wird kenntlich gemacht.

PA6-CF-Dichte als vorlaeufige Herstellerreferenz: Bambu Lab 1.09 g/cm3, https://eu.store.bambulab.com/it-it/products/pa6-cf?from=home_web . Festigkeit/Modul benoetigen eine gesonderte belegte Annahme. Unbekanntes Motormodell, reales Filament, Druckanisotropie, tatsaechliche AIO-Stackhoehe, Schrauben und reale Crashrandbedingungen bleiben dokumentierte Unsicherheiten.

## Freie Topologieoptimierung: Schnittstellen v1

Dieser additive Vertrag erweitert die obigen Schnittstellen; die funktionierende v0-Geometrie und die bestehende FEA bleiben unveraendert als Referenz und unabhaengiger Pruefer. Die fachlichen Python-Module bleiben ohne Kommentare, Docstrings, argparse und unnoetige Prints. Alle fachlichen Konfigurationen sind Dictionaries. Separate Workstation-Studientools unter `tools/` besitzen je Befehl ein einfaches Default-Dict; ausdrueckliche Rechenbudgets und neue Ausgabepfade stehen in einer optionalen JSON-Datei, deren Schluessel, Typen und Grenzen vor dem Start geprueft werden. Sie veraendern diesen fachlichen Vertrag nicht.

### Methode und Verantwortlichkeiten

Phase 1 verwendet klassische dreidimensionale, gefilterte SIMP-Dichteoptimierung auf regulaeren Hex8-Zellen. Die Startdichte ist im freien Bereich gleichfoermig. Mehrere statische Lastfaelle bestimmen die Materialverteilung; Eigenfrequenzen, Spannungen, Verschiebungen, Masse und Steifigkeit werden bei der unabhaengigen Kandidatenbewertung mit der vorhandenen gmsh/CalculiX-Pipeline gemeinsam geprueft. Die Akzeptanz darf nicht allein auf SIMP-Ersatzkennwerten beruhen. Eine diskrete Kandidatenauswahl kann die gemeinsam formulierten Ziele und Constraints verwenden; Gradienten aller Ziele sind keine Voraussetzung dieses Vertrags.

Es gibt keine v0-Armmaske, keine X-Frame-Saat und keine vorgeschriebenen Verbindungen zwischen Anschluessen. Komponentenpositionen, Anschlussflaechen, aeusserer Bauraum und Fertigungsgrenzen sind die ausdruecklich verbleibenden menschlichen Formannahmen. `build_frame` und `build_geometry` duerfen bei Designraumerzeugung und freier Optimierung nicht aufgerufen werden.

### Designraum

`deep_frame.topology_geometry.build_design_domain(parameters: dict) -> dict`

`parameters` enthaelt die vorhandenen Komponenten-, Material- und Lastfallparameter sowie optional `topology`. Die Funktion kopiert Eingaben und liefert:

- `schema_version`: `deep-frame-topology-domain-v1`.
- `grid`: `origin_mm` ist die untere aeussere Ecke, `spacing_mm` ist immer eine Liste `[hx,hy,hz]`, `shape` ist `[nx,ny,nz]`, `axis_order` ist `xyz`, `order` ist `C`. Zellmittelpunkt `(i,j,k)` ist `origin + spacing * ([i,j,k]+0.5)`. C-Flatten: z laeuft am schnellsten. Gitterknoten besitzen entsprechend `(nx+1,ny+1,nz+1)`.
- `allowed`, `preserve`, `forbidden`: bool-NumPy-Arrays mit genau `grid.shape`. `allowed` umfasst freie und feste Materialzellen; `preserve` ist Teilmenge von `allowed`; `forbidden` ist das Komplement von `allowed`. Freie Zellen sind `allowed & ~preserve`. Vorgeschriebene Dichte ist 1 in Preserve und 0 in Forbidden. Subvoxel-Bohrungen werden zusaetzlich geometrisch exakt verarbeitet.
- `regions`: Liste benannter generischer raeumlicher Primitive mit `name`, `role` (`preserve`, `forbidden`, `allowed`) und `purpose`. Box: `kind: box`, `min_mm`, `max_mm`; achsparalleler Zylinder: `kind: cylinder`, `center_mm`, `radius_mm`, `height_mm`, `axis` (`x`, `y`, `z`). `center_mm` liegt im geometrischen Mittelpunkt. Primitive sind JSON-faehig. `rasterize: false` kennzeichnet subvoxelgenaue Ausschnitte, die nur bei Rekonstruktion ausgeschnitten werden; keine still verschwundenen Bohrungen.
- `material`, `point_masses`, `load_cases`, `fea_settings`: bestehender FEA-Vertrag, unveraendertes Einheitensystem. Die vorhandenen drei mechanischen Lastfaelle plus Modaltest bleiben enthalten. Zusaetzliche benannte Anschlusslastfaelle sind erlaubt, wenn geringe Lasten fuer sonst lastfreie notwendige Anbauteile deklariert werden.
- `manufacturing`: mindestens `nozzle_width_mm`, `minimum_wall_nozzles`, `minimum_feature_mm`, `minimum_attachment_area_mm2`, `supports_allowed`, `build_direction`. Supportfreiheit wird nicht behauptet. Diskrete Connectivity, Featuregroesse und lokale Anschlussquerschnitte sind vor FEA zu pruefen.
- `metadata`: JSON-faehige Entstehungsdaten mit freiem/festem/verbotenem Zellvolumen und Preserve-Anteil, Komponentenpositionen, Annahmen und Referenzbezug. Kein Shape in diesem Dict.

Die Zellen beschreiben die Optimierungsdiskretisierung; `regions` bleiben die genaue geometrische Wahrheit fuer Bauraumgrenzen, Bauteilfreiheit, Anschluesse und Montagezugang. Konservative Rasterabweichungen werden dokumentiert. Thin-Box-FEA-Selektoren gelten in physischen mm; der interne Hex8-Adapter darf diese fuer Knotenselektion um hoechstens eine halbe Gitterweite erweitern und muss die Erweiterung speichern. Eine leere Auswahl ist ein Fehler. Finale FEA nutzt die exakten Selektoren und die rekonstruierte Geometrie.

### Optimierer und Felddaten

`deep_frame.topology_optimization.optimize_topology(domain: dict, settings: dict) -> dict`

Fuer beobachtbare laengere Studien akzeptiert der vorhandene Optimierer zusaetzlich das optionale Keyword `progress_callback`. Es erhaelt nach jeder ausgewerteten Iteration und der abschliessenden Feldauswertung eine tiefe Kopie des jeweiligen Historieneintrags. Der Callback erhaelt keine veraenderbare Solverreferenz; ein Regressionstest prueft die unveraenderte Dichte gegen einen Lauf ohne Callback. Das Pipeline-Generatorprotokoll bleibt `(domain, settings)`. Das Journal ist ein Fortschrittsnachweis und kein Checkpoint fuer das Fortsetzen einer angefangenen Dichteiteration.

Das Ergebnis hat `status` (`ok`, `invalid`, `failed`), `density` (NumPy-Array), JSON-faehige `summary`, `history`, `diagnostics`. Optional weitere Felder sind explizit zu benennen. Status `ok` beschreibt einen abgeschlossenen Dichteoptimierungslauf, noch keine mechanisch akzeptierte Geometrie. Der Optimierer darf weder v0-Geometrie noch Rekonstruktion oder CalculiX importieren. Austauschbare Generatoren koennen denselben Feld-/Domain-Vertrag verwenden.

Settings umfassen `volume_fraction`, `filter_radius_mm`, `penalization`, `min_stiffness_ratio`, `max_iterations`, `change_tolerance`, `move_limit`, `projection_beta`, optional `case_weights`. Referenzlasten werden einzeln normiert, damit N/N-mm-Skalen keinen Lastfall versehentlich ausschalten. Iterationshistorie speichert insbesondere Volumen, Zielfunktion, Einzelcompliances, Aenderung und Konvergenzstatus. Fehlgeschlagene Loesungen werden nie als gueltige Nullwerte gespeichert.

### Rekonstruktion, Bewertung und Speicherung

`deep_frame.topology_geometry.reconstruct_topology(domain: dict, density, settings: dict) -> build123d.Solid`

`deep_frame.topology_geometry.validate_topology(solid, domain: dict, settings: dict) -> dict`

Der Rekonstruktionsadapter kann einen anderen belegten klassischen Algorithmus verwenden, muss aber automatisch ohne manuelles Nachmodellieren genau einen gueltigen geschlossenen Solid erzeugen. Er vereinigt erforderliche exakte Preserve-Primitiven und schneidet exakte Forbidden-Primitiven einschliesslich Bohrungen aus. Es darf keine von Hand eingezeichneten Ersatzarme oder Verbindungsrippen geben. Getrennte notwendige Preserve-Anschluesse duerfen nicht still entfernt werden. Fehler werden als nachvollziehbare Invalid-/Failed-Kandidaten gespeichert.

Die Validierung liefert `passed`, `violations`, `checks`. Pflichtchecks: ein Solid, Preserve-Abdeckung, Forbidden-Ueberdeckung, Komponenten-/Propellerfreiraum, Mindestfeature-/Duesenverhaeltnis und Anschlussquerschnitte. Die genaue Fertigungspruefmethode sowie moegliche Rastergrenzen werden dokumentiert. Nur gueltige Kandidaten erreichen `deep_frame.fea.evaluate`.

Persistenz enthaelt mindestens JSON-Manifest mit Domain ohne Arrays, Material, BCs, Lasten, Massen, Parametern, Seeds, Methode, Versionen/Hashes, Iterationshistorie und Kandidatenstatus; NPZ mit `density`, `allowed`, `preserve`, `forbidden`; resultierende STEP/STL; FEA-Ergebnis-Dicts und Solverartefakte; Vergleich zur unveraenderten v0. NPZ-Feldlayout steht im Manifest. Nicht erfolgreiche Designs bleiben als gekennzeichnete Datenpunkte erhalten. Ein reproduzierbarer Run benoetigt keinen manuellen CAD-Eingriff.

### Arbeitspakete und Akzeptanz

- A: dieser Vertrag, Designraum (heute in `topology_geometry.py`), `TOPOLOGY_CONFIG` (heute in `config.py`), Domain-Tests/-Dokumentation.
- B: `topology_optimization.py`, klassische Hex8/SIMP-Implementierung, analytische und Gradient-/Maskentests, Methodendokumentation.
- C: `topology_geometry.py`, Rekonstruktion, Fertigungs-/Geometrievalidierung, generischer Evaluator und Persistenz/Run-Adapter, unabhaengiger FEA-Test und Vergleich.
- Root orchestriert und integriert; jede Umsetzung findet in eigenen Branches/Worktrees statt.

Mindestens eine automatisch erzeugte, geometrisch und mechanisch gueltige Struktur muss sichtbare freie Lastpfade haben. Ein Vergleich zu v0 berichtet Masse, Steifigkeit, Maximalspannung, Maximalverschiebung und Eigenfrequenzen unter denselben physikalischen Last-/Materialannahmen. V0 bleibt Referenz; ein anderer Randbedingungsadapter muss auf beide Geometrien angewendet und begruendet werden. Ein besserer Einzelwert allein erfuellt Phase 1 nicht.
