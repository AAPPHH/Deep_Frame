# I3: echter Vier-Trial-Probelauf

Der Lauf vom 30.09.2026 erfuellt die vereinbarte Mindestabnahme: **ein gegen die feste v0-Referenz verbessertes, geometrisch und nach allen fuenf relativen Constraints gueltiges Design**. Vier echte Auswertungen wurden versucht. **Ein Trial wurde erfolgreich berechnet; drei scheiterten an nichtpositiven Jacobi-Determinanten ihrer Gmsh-C3D10-Netze.** Diese drei Trials haben Status `FAIL`, keine Zielwerte und keinen Platz auf der Pareto-Front. Die Vernetzungspruefung wurde nicht abgeschaltet; weder Referenz noch Constraints wurden nachtraeglich angepasst.

Berechnungsstand: Commit `2b8bf1c`. Studie: `frame-v0-probe-v1`, Optuna 5.0.0, Seed 42. Gmsh 4.15.2 und CalculiX 2.22 berechneten fuer Referenz und Trial 0 jeweils drei statische Lastfaelle und sechs elastische Eigenfrequenzen. Die Material-, Last- und Einspannannahmen stehen in [integration.md](../integration.md), [fea.md](../fea.md) und [optimization.md](../optimization.md). Der Lauf ist ein Funktionsnachweis und keine konvergierte Optimierung.

## Referenz und gueltiger Kandidat

Unveraenderte Referenz: 135 mm diagonaler Radstand, Wide-X-Verhaeltnis 112/82, Arme 6,5 mm breit und 4,0 mm hoch. Trial 0 erhoeht ausschliesslich die Armhoehe auf **4,2 mm**; die Breite bleibt **6,5 mm**. Er ist der erste der beiden vor Beginn festgelegten Startkandidaten und durchlief dieselben Druck-, Geometrie-, FEA- und Constraint-Pruefungen wie alle anderen Trials. Die Herkunft der Grundabmessungen ist in [armattan_research.md](../armattan_research.md) festgehalten; die Armabmessungen sind eigene vorlaeufige PA6-CF-Entwurfsannahmen.

| Kennwert | v0 | Trial 0 | Aenderung | Fester Grenzwert |
| --- | ---: | ---: | ---: | ---: |
| FEA-Masse, g | 69,162903 | 69,594458 | +0,6240 % | maximal 72,621048 |
| Armsteifigkeit, N/mm | 5,598303 | 6,423631 | **+14,7425 %** | mindestens 5,318388 |
| Erste Eigenfrequenz, Hz | 204,5565 | 214,6914 | **+4,9546 %** | mindestens 194,328675 |
| Maximale Verschiebung, mm | 0,237311 | 0,206693 | -12,9023 % | maximal 0,249177 |
| Maximale Vergleichsspannung, MPa | 5,983980 | 5,927106 | -0,9504 % | maximal 6,582378 |

Alle Grenzwerte wurden einmal aus v0 mit den Faktoren Masse 1,05; Steifigkeit 0,95; Frequenz 0,95; Verschiebung 1,05; Spannung 1,10 gebildet. Die Massenzunahme ist innerhalb der vorab erlaubten 5 %. Der Kandidat verbessert Steifigkeit und Frequenz, minimiert aber nicht gleichzeitig die Masse.

Die ausgewiesene FEA-Masse umfasst ausschliesslich Frame und 37-g-Akku-Punktmasse. Die Frame-Masse steigt von **32,162903 g auf 32,594458 g**. Die Geometriechecks enthalten zusaetzlich Motoren, Props, Kamera und AIO; deren vollstaendige v0-Baugruppenmasse betraegt 105,062903 g. Diese weiteren Komponenten wurden nicht als FEA-Punktmassen eingerechnet.

## Alle Versuche und offene Vernetzung

| Trial | Armhoehe × Armbreite, mm | Auswahl | Ergebnis |
| --- | --- | --- | --- |
| 0 | 4,2 × 6,5 | Erster unveraenderter Startkandidat | `COMPLETE`, gueltig, einziger Pareto-Kandidat |
| 1 | 4,0 × 6,7 | Zweiter unveraenderter Startkandidat | `FAIL`, nichtpositive C3D10-Jacobi-Determinante |
| 2 | 4,0 × 6,9 | Optuna, Seed 42 | `FAIL`, nichtpositive C3D10-Jacobi-Determinante |
| 3 | 4,2 × 6,7 | Optuna, Seed 42 | `FAIL`, nichtpositive C3D10-Jacobi-Determinante |

Alle vier Designs bestanden die vorangestellten Druck- und Geometriepruefungen. Fuer die drei Netzfehler existiert **kein gueltiges FEA-Ergebnis**. Eine geometrisch gueltige Form ist noch keine Garantie fuer ein gueltiges quadratisches Tetraedernetz. Robusteres Vernetzen, etwa durch untersuchte Optimierung hoeherer Elemente mit weiterhin positiver Jacobi-Pruefung, bleibt offen. Eine veraenderte Netzstrategie benoetigt wegen des Auswertungsvertrags eine neue Studie; die vorliegende Studie bleibt unveraendert erhalten.

Die Referenz verwendete 45.491 C3D10-Elemente und 83.406 Knoten, Trial 0 47.582 Elemente und 86.594 Knoten. Minimale Jacobi-Determinanten: 0,0405774 bzw. 0,0296899 mm³, jeweils positiv. Das starre 4-mm-Akku-Querband koppelte in beiden Rechnungen 58 Knoten an den Akku-Schwerpunkt (0, 0, 34,5) mm. Referenz und Trial 0 benoetigten fuer ihre FEA-Auswertungen 52,39 bzw. 44,52 s. Keine Nullmoden wurden als erste Eigenfrequenz verwendet.

Die Suchraumgrenzen 6,3 × 3,8 und 6,9 × 4,4 mm wurden zusaetzlich geometrisch untersucht: CAD-Schnitte durch alle vier freien Armhaelse ergaben 23,94 bzw. 30,36 mm² und damit mehr als die geforderten 20 mm². Kritische feste Stege bleiben Motorwelle–Schraube 2,0 mm, AIO-Fenster 2,789 mm und Strap-Aussenrand 2,5 mm. Das ist eine Pruefung dieses konkreten Suchraums, kein universeller Wandstaerkennachweis.

## Nachweise und Wiederaufnahme

- [Vollstaendiger Ergebnis-Snapshot](optimization_probe.json): Referenz, alle vier Trials, Parameter, Vorpruefungen, feste Constraints, erfolgreiche FEA-Ergebnisse und die drei vollstaendigen Fehlerberichte einschliesslich Gmsh-Logs.
- [Gueltige Pareto-Front als JSON](optimization_pareto.json) und [CSV](optimization_pareto.csv).
- [Viewer-Screenshot des Kandidaten](optimization_trial0_viewer.png): OCP CAD Viewer zeigte `/Group/Trial 0`; Status `[1, 1]`, anschliessend Sichtpruefung des gespeicherten Screenshots.

Die Snapshot-Artefaktpfade sind relativ zum Repository; der installierte Solverpfad ist portabel als `.venv/calculix/calculix_2.22_4win/ccx_static.exe` dargestellt. Lokale persistente Datenbank: `exports/optimization/probe/study.sqlite3`. Vollstaendige lokale FEA-Dateien: `exports/fea/optimization_probe/`. Beim Wiederaufruf von `run_optimization.py` werden vier weitere Trials angehaengt, keine bisherigen Ergebnisse ersetzt. Der unveraenderte Code-/Material-/Last-/Werkzeugvertrag muss dazu weiterhin passen.

Der Orchestrator kopierte die abgeschlossene SQLite-Datenbank und rund 316 MB Rohartefakte in den main-Worktree. Die Datenbank-Pruefsumme stimmte vor der Kopie ueberein. Dort bestaetigte `run(RUN_CONFIG | {"n_trials": 0, "show_viewer": False})` mit der eigenen main-Umgebung den unveraenderten Vertrag `785b9b238b9370333c84fd3120cbfad01ae68743a2486f92c4e0a6b9275ce418`, dieselben vier Trials und die gespeicherte Referenz ohne Neuberechnung.

Vor dem Probelauf bestanden **69 Tests**, einschliesslich realer CalculiX-Balken- und Punktmassentests sowie sieben Integrationstests. Die Datenabnahme pruefte anschliessend alle festen Grenzen nochmals direkt gegen die gespeicherten Messwerte, beide unveraenderten Startkandidaten, vier echte Versuche, die positiven Netze der erfolgreichen Rechnungen und den Ausschluss der drei Fehler aus der Front. Ein unabhaengiger Read-only-Audit von Subagent B bestaetigte aus den rohen `.inp`- und `.dat`-Dateien die Kraftsummen -1 N, -3,6284605 N und +5 N, Dichte 1,09e-9 t/mm³, Akku-Masse 3,7e-5 t, E/nu, Richtungssteifigkeit und korrekte Frequenzspalte.

Offen bleiben besonders die Netzrobustheit fuer breitere Arme, Frame-Netzkonvergenz, Sensitivitaet gegen Einspann- und Akku-Patchannahmen sowie reale gedruckte Materialeigenschaften. Die Frequenzsteigerung ist unter diesen festen Pruefstandsbedingungen gemessen; freie Flugmoden, reale Crashfestigkeit und ein produktionsreifer Frame sind damit nicht nachgewiesen.
