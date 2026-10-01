# Parametrische Optimierung

`deep_frame.optimization.optimize(reference_parameters, search_space, evaluator, validator, settings)` arbeitet ausschliesslich mit Dictionaries. Der Kern importiert weder build123d noch FEA. Der Austausch zwischen Geometrie, Solver und Optimierer folgt [interfaces.md](interfaces.md). Die Beispielkonfiguration steht in `OPTIMIZATION_CONFIG` und `SEARCH_SPACE` von `deep_frame/config.py`; `requirements-optimization.txt` fixiert Optuna 5.0.0. Installation in der jeweiligen Umgebung:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optimization.txt
.\.venv\Scripts\python.exe -m pytest tests/test_optimization.py -q
```

Ein Evaluator liefert das FEA-Ergebnisschema mit Masse in g, Steifigkeit in N/mm und ausschliesslich positiven, aufsteigend sortierten elastischen Eigenfrequenzen in Hz. Nicht berechenbare Ergebnisse, Nullmoden, NaN, Solverfehler und Exceptions ergeben keinen gueltigen Kandidaten. Die Referenz muss alle Vorpruefungen bestehen und erfolgreich ausgewertet werden; andernfalls wird vor den Trials mit einer Exception abgebrochen.

## Ablauf und feste Referenz

Die Reihenfolge ist fuer Referenz und jeden Kandidaten dieselbe: explizite Druckregeln, Geometrievalidierung, Evaluator, relative Constraints. Druck- oder Geometrieverletzungen fuehren zu einem pruned Trial, bevor der Evaluator aufgerufen wird. Gescheiterte Auswertungen bekommen den Optuna-Status `FAIL`. Erfolgreich berechnete, aber relativ unzulaessige Designs bleiben als Messdaten erhalten und werden von der gueltigen Pareto-Front ausgeschlossen.

Die drei Ziele sind `mass_g` minimieren sowie `stiffness_n_per_mm` und `first_frequency_hz` maximieren. NSGA-II verwendet Seed 42 und standardmaessig acht Individuen pro Population. Die nativen benannten Constraints werden ueber `trial.set_constraint()` gesetzt; positive Residuen sind Verletzungen. Das Ergebnis enthaelt alle Vorpruefungen, Messwerte, feste Referenzwerte, Faktoren, absolute Grenzwerte und Entscheidungen je Trial. Die exportierte Front wird zusaetzlich aus ausschliesslich erfolgreichen, gueltigen Trials berechnet. Ein Pareto-Kandidat ist nicht zwingend besser als die Referenz in allen Zielen; `improved_objectives` nennt ausschliesslich seine streng verbesserten Ziele. Die Referenz ist getrennt gespeichert und wird nicht als neuer Trial ausgegeben.

Die folgenden Faktoren sind vor dem Probelauf festgelegte Entwurfsentscheidungen, keine Materialkennwerte oder Armattan-Vorgaben:

| Constraint | Grenze gegen die unveraenderte Referenz |
| --- | --- |
| Masse | maximal 1,05 |
| Steifigkeit | mindestens 0,95 |
| Erste elastische Eigenfrequenz | mindestens 0,95 |
| Maximale Verschiebung | maximal 1,05 |
| Maximale Vergleichsspannung | maximal 1,10 |

Damit ist ein kleiner Massenzuwachs fuer bessere Steifigkeit erlaubt. Die Grenzwerte werden weder aus dem besten Trial noch nachtraeglich aus dem Ergebnis abgeleitet. Alle fuenf Faktoren muessen explizit vorhanden und positiv sein. Bei einem Referenzwert null ist der absolute Grenzwert ebenfalls null; eine nicht definierte Ratio wird als JSON-null gespeichert. Fuer Masse, Steifigkeit und Frequenz ist bereits die Referenz zwingend positiv.

## Harte Druckregeln und Suchraum

`printability.nozzle_width_mm` mal `minimum_wall_nozzles` definiert die minimale Wandstaerke. Der vorlaeufige Default ist 0,4 mm mal vier Bahnen = 1,6 mm. Die expliziten `wall_thickness_paths` sind `frame.minimum_wall_mm`, `frame.base_thickness_mm`, `frame.deck_thickness_mm` und `frame.arm_height_mm`. Jede in der Geometrie neu eingefuehrte duennere Wand braucht ebenfalls einen Eintrag oder einen Geometriecheck; der Optimierer errät keine Materialstaerken aus Feldnamen.

`clamp_sections` benennt Rechteckquerschnitte mit `width_path`, `height_path` und `minimum_area_mm2`. Der Default prueft Arm-Breite mal Arm-Hoehe gegen 20 mm². Das ist eine vorlaeufige konstruktive Mindestgrenze, kein Festigkeitsnachweis fuer PA6-CF. Ausschnitte, Kerben oder nicht rechteckige Querschnitte muessen vom Geometrievalidator zusaetzlich abgesichert werden. Die gegenwaertige Regel setzt einen massiven rechteckigen Armquerschnitt voraus.

`search_space` enthaelt Punktpfade und `{low, high, step}` oder diskrete `{choices: [...]}`. Numerische Vorschlaege werden als Floats uebergeben. Nicht optimierte Felder bleiben unveraendert. Default-Suchraum: Armhoehe 3,8 bis 4,4 mm in 0,2-mm-Schritten und Armbreite 6,3 bis 6,9 mm in 0,2-mm-Schritten. Die beiden vollstaendig festgelegten Startkandidaten sind 4,2 × 6,5 mm und 4,0 × 6,7 mm. Sie durchlaufen saemtliche Pruefungen; sie sind keine vorab gueltigen Ergebnisse. Weitere Trials werden vom Sampler gewaehlt. Sechs Trials demonstrieren nur die integrierte Funktion, keine konvergierte Formoptimierung.

## Persistenz und Wiederaufnahme

Standardziel ist `sqlite:///exports/optimization/study.sqlite3`, Studienname `frame-v0`. Unter `output_dir` entstehen `result.json`, `pareto.json` und `pareto.csv`. JSON bewahrt die vollstaendigen Parameter-Dicts, Evaluator-Ergebnisse und Constraint-Pruefungen; CSV enthaelt Kennwerte sowie JSON-Spalten fuer Parameter und Constraints. Lokale Solver-/Studienartefakte unter `exports/optimization/` werden nicht automatisch versioniert. Eine fuer die Abnahme bestimmte Momentaufnahme kann gezielt an einem anderen Ort gespeichert werden.

Beim Wiederaufruf desselben Studiennamens werden `n_trials` weitere Trials hinzugefuegt und die einmal gespeicherte Referenzauswertung wiederverwendet. Ein SHA-256-Vertrag bindet Referenzparameter, Suchraum, Druckregeln, relative Constraints, Seed, Populationsgroesse und `evaluation_id`. Aenderungen verlangen einen anderen Studiennamen bzw. eine neue Datenbank. `evaluation_id` muss auch Material, Lasten, Punktmassen, Netz- und Solver-Einstellungen eindeutig kennzeichnen; der aufrufende Adapter ist dafuer verantwortlich. Es ist absichtlich unzulaessig, eine bestehende Studie durch veraenderte Randbedingungen aufzuwerten.

Der Seed macht einen ununterbrochenen Lauf mit gleichem Trial-Budget reproduzierbar. Die Datenbank speichert Trials, aber nicht den laufenden Zufallszahlengenerator. Wiederaufgenommene Laeufe sind daher nicht als identisch zur ununterbrochenen Zufallsfolge zu verstehen. Gleichzeitige Aufrufe auf dieselbe Studie werden nicht unterstuetzt; parallele Solverarbeit braucht separate Studien oder eine kuenftige koordinierte Ausfuehrung.

## Viewer und analytische Abnahme

`deep_frame.optimization.show_candidates(candidates, build_geometry, show=None, viewer_settings=None)` ist ein separater Adapter im selben Modul; `ocp_vscode` wird erst in dieser Funktion importiert. Er rekonstruiert ausschliesslich gueltige Parameter-Dicts ueber die injizierte Geometriefunktion und zeigt sie mit Trial-Namen im OCP CAD Viewer an. Der Kern bleibt ohne Viewer-Abhaengigkeit. Fuer einen direkten Vergleich gleicher Koordinaten koennen die einzelnen Trial-Objekte im Viewer ein- und ausgeblendet werden.

Der Abnahmetest verwendet einen einseitig eingespannten Euler-Bernoulli-Balken mit 10 × 3 mm Querschnitt, 1 N Endkraft, 3,4 GPa Modul und 1090 kg/m³ Dichte. Diese Werte sind Testkonstanten, keine zusaetzliche Materialfreigabe. Bei variabler Laenge L von 40 bis 100 mm gilt Masse proportional L, Steifigkeit proportional 1/L³ und erste Eigenfrequenz proportional 1/L². Damit dominiert L = 40 mm jedes laengere Design und ist das bekannte gemeinsame Optimum aller drei Ziele. Ohne vorgegebenen Gewinner findet der feste Seed in 30 Trials genau dieses Optimum. Gegen L = 100 mm betragen die erwarteten Verhaeltnisse 0,4 fuer Masse, 15,625 fuer Steifigkeit und 6,25 fuer Frequenz. Die Toleranz fuer die diskrete Optimumslaenge betraegt 10⁻¹² mm; die analytischen Kennwertverhaeltnisse verwenden die numerische pytest-Approximation.

Weitere Tests beweisen mit Aufrufzaehlern, dass zu duenne Waende, zu kleine Einspannquerschnitte und ungueltige Geometrie niemals den Evaluator erreichen. Sie pruefen relative Verletzungen, Solverfehler, Nullmoden, NaN, feste Baseline bei Wiederaufnahme, Schutz gegen geaenderte Studienvertraege, isolierte Parameterkopien, Exporte und den Vieweradapter. Der analytische Optimierungstest ersetzt weder den CalculiX-Balkentest noch den echten Frame-Probelauf.

Optuna-API-Quellen: [NSGAIISampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html), [Study: Persistenz, Enqueue und Pareto](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html), [Trial.set_constraint](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.set_constraint). Geprueft am 30.09.2026 gegen installierte Version 5.0.0.
