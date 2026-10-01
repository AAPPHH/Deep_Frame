# Workstation: Rekonstruktion und separate FEA-Netzstudie

Die Referenz wurde auf der Workstation reproduziert und in getrennten Studien bei 3, 2.5 und 2 mm (native Voreinstellung/PaStiX, zwei Threads) sowie 2 und 1.5 mm (explizit SPOOLES, ein Thread) ausgewertet. Alle fuenf vollstaendigen Vergleichspaare mit insgesamt zehn FEA-Laeufen bestehen die historischen fuenf Vergleichsgrenzen. **Allgemeine Netzkonvergenz ist nicht bestaetigt:** Der alte freie Kandidat besteht den letzten 2->1.5-mm-Sensitivitaetstest, die v0-Spannungsspitze steigt dort aber erneut um 22.02 % und verletzt die unveraenderte 5-%-Grenze.

Der urspruengliche 1.5-mm-Versuch endete nach den drei statischen v0-Lastfaellen beim nativen Modal-Solver mit Windows-Heapfehler `0xC0000374`. Dieser Versuch bleibt vollstaendig als Fehler gespeichert; es gab keine automatische Wiederholung. Die spaetere erfolgreiche SPOOLES1-Untersuchung hat eine eigene Studien- und Solveridentitaet.

## Integritaet und Wiederherstellung

Bei Ankunft war `15e37bf` sauber; der Tag `phase1-workstation-2026-09-30` zeigt auf `938bbdf`. Seit dem Tag waren nur Uebergabedokumente hinzugekommen. Das grosse ZIP und der originale komplette Run wurden nicht uebertragen. Alle sieben versionierten Evidenzdateien bestehen ihre archivierten SHA256- und Bytezahlpruefungen; alle drei NPZ-Masken bestehen ihre eigenen Bytehashes. 24 von 26 Python-Dateien waren direkt bytegleich zum historischen Run, die beiden weiteren nach ausschliesslicher LF-Normalisierung (`components.py`, `model.py`).

Der 45-Iterations-Kandidat `default_t00` wurde aus den gespeicherten Feldern, exakten Preserve-/Verbotsregionen und seiner originalen Rekonstruktionskonfiguration automatisch neu erzeugt. Volumina: v0 29507.250563310336 mm3, Kandidat 53698.902835521665 mm3. Beide neu exportierten STL-Dateien sind sogar bytegleich zu den historischen STL-Hashes. STEP-Dateien besitzen eigene neue Hashes. Die komplette geometrische/Fertigungspruefung besteht erneut, einschliesslich 4182 Normalenstrahlen mit minimal 2 mm gemessener Wandstaerke.

Die frische 3-mm-FEA reproduziert fuer beide Koerper alle gespeicherten physikalischen Resultatfelder exakt: alle drei statischen Lastfaelle, alle sechs Eigenfrequenzen, Netzgroessen und Punktmassenkopplung. Neue Laufverzeichnisse, STEP-Header und Laufzeiten sind getrennte Provenienz. Die spaeter fuer die Workstation-Studien hinzugefuegte Optimierungs-Callback-Aenderung ist in der neuen Quellprovenienz explizit enthalten.

## Methode und vorab festgelegte Grenzen

Alle Stufen verwenden identische CAD-Geometrie, isotropes PA6-CF, die archivierten Last-/Klemmregionen und den 37-g-Akku mit demselben Schwerpunkt und derselben starren Patchkopplung. Innerhalb jeder Studie wird nur die maximale Gmsh-Netzvorgabe kleiner; minimale Netzgroesse 0.5 mm, zwoelf Kruemmungspunkte, C3D10 mit geraden Mittelknoten, Gmsh 4.15.2 und CalculiX 2.22 bleiben gleich. Die erste Studie nutzt einen Gmsh-Thread und zwei CalculiX-Threads; die getrennte SPOOLES-Folgestudie einen CalculiX-Thread.

Vor dem ersten FEA-Start festgelegter Sensitivitaetstest zwischen aufeinanderfolgenden Netzen: hoechstens 2 % fuer Armsteifigkeit, maximale Verschiebung und erste Eigenfrequenz sowie 5 % fuer maximale Vergleichsspannung. Der Bezugswert ist jeweils das groebere Netz. Dies ist ein inkrementeller Sensitivitaetstest, kein Beweis asymptotischer Konvergenz. Die physischen Selektorboxen bleiben gleich; deren Knotenzahlen und die gleiche Kraft pro ausgewaehltem Knoten aendern sich mit dem Netz.

## Vollstaendige Resultate mit nativer Solver-Voreinstellung

Die Frame-Massen bleiben 32.162903 g (v0) und 58.531804 g (freie 45-Iterations-Referenz). Der Akku ist darin nicht enthalten.

| Koerper | Netz mm | Elemente | Knoten | k N/mm | u_max mm | sigma_max MPa | f1 Hz | Zeit s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 3 | 45491 | 83406 | 5.163786 | 0.25542346 | 5.983499 | 204.5040 | 49.42 |
| v0 | 2.5 | 52949 | 97025 | 5.149509 | 0.25582465 | 6.819513 | 202.5610 | 62.75 |
| v0 | 2 | 64931 | 119383 | 5.147838 | 0.25624653 | 6.700761 | 200.2521 | 75.96 |
| Freie Referenz | 3 | 60933 | 110430 | 194.391068 | 0.00806285 | 0.517476 | 1075.7640 | 61.76 |
| Freie Referenz | 2.5 | 71302 | 128046 | 191.974296 | 0.00815862 | 0.632510 | 1068.7230 | 78.44 |
| Freie Referenz | 2 | 81024 | 143983 | 187.118130 | 0.00832870 | 0.650058 | 1068.3390 | 89.85 |

| Koerper | Verfeinerung mm | delta k | delta u_max | delta f1 | delta sigma_max | Screen bestanden |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v0 | 3 -> 2.5 | 0.276 % | 0.157 % | 0.950 % | 13.972 % | nein |
| v0 | 2.5 -> 2 | 0.032 % | 0.165 % | 1.140 % | 1.741 % | ja |
| Freie Referenz | 3 -> 2.5 | 1.243 % | 1.188 % | 0.655 % | 22.230 % | nein |
| Freie Referenz | 2.5 -> 2 | 2.530 % | 2.085 % | 0.036 % | 2.774 % | nein |

Bei 2 mm betragen die Kandidat/v0-Verhaeltnisse: Masse 1.81985, Steifigkeit 36.34888, erste Eigenfrequenz 5.33497, maximale Verschiebung 0.03250 und maximale Spannung 0.09701. Alle historischen Grenzen sind damit weiterhin erfuellt. Fuer die Referenz bleibt das 4-mm-Designraster unveraendert; diese Untersuchung verfeinert ausschliesslich das unabhaengige FEA-Netz.

Die Spannungsspitze stammt auf den urspruenglichen 3/2.5/2-mm-Stufen bei v0 aus `camera_side` am Bereich um [9, 19.75, 25.75] mm und beim freien Kandidaten aus `arm_tip` nahe der AIO-Unterseitenklemmung um [-14.25, 15.67, 0.26] mm. Die JSON-Evidenz speichert Element-/Integrationspunktnummern und Elementgrenzen. Elementmittelpunkte lokalisieren die Umgebung; sie sind nicht die exakten Integrationspunktkoordinaten und beweisen keine Singularitaet.

## Fehlgeschlagene Zusatzstufe 1.5 mm

Der vorgezogene reine Gmsh-Test erzeugte 97610 Elemente fuer v0 und 149614 fuer den Kandidaten in 4.11 bzw. 6.99 s, jeweils mit positiven Jacobians. Die RAM-/Zeitprognose aus dem 2-mm-Lauf lag innerhalb der vorhandenen Ressourcen. Der anschliessende v0-Versuch berechnete die drei statischen Faelle, brach aber nach insgesamt 87.18 s in `case_3` (Punktmassen-Modell, PaStiX General Matrix, N=509598, nnz=36407448) mit Exit 3221226356 ab. Vorher waren rund 20.85 GiB RAM frei; RAM-Erschoepfung ist nicht belegt. Die Fehlerklasse war bereits im ersten Workstation-Testlauf sporadisch aufgetreten.

Der unveraenderte Fehler liegt unter `exports/topology/workstation_20260930/mesh_study/results/baseline_1p5mm/`. Die nachfolgende Untersuchung alternativer nativer Solver kennzeichnet die Solveridentitaet als neue Studie. Teilresultate dieser Stufe gehen nicht in einen vollstaendigen v0/Kandidatenvergleich oder einen Konvergenznachweis ein.

## Getrennte SPOOLES1-Folgestudie mit 2-mm-Anker

Die Ursachenpruefung erhielt alle Originaldateien. Zwei getrennte Modaldiagnosen am byteidentischen v0-1.5-mm-Modell liefen mit einem Thread durch: native Voreinstellung/PaStiX und explizit SPOOLES liefern alle sechs Eigenfrequenzen auf Ausgabepraezision identisch. Danach wurde fuer die gesamte Folgeuntersuchung `linear_solver="SPOOLES"` und `threads=1` ausdruecklich gespeichert. Die Voreinstellung bestehender Pipelineaufrufe wurde nicht veraendert.

Zuerst wurden beide Koerper nochmals bei 2 mm gerechnet, um Backend-/Threadwechsel und Netzverfeinerung getrennt zu pruefen. Alle fuenf Hauptkennwerte, alle sechs Frequenzen und Netzmetadaten stimmen exakt mit dem alten 2-mm-Paar ueberein. Die einzige zusaetzlich gefundene v0-Vektorkomponentendifferenz liegt bei 1.26e-12 mm. Erst danach folgten die zwei 1.5-mm-FEA. Alle vier neuen Laeufe enden erfolgreich; Rohartefakte liegen getrennt unter `exports/topology/workstation_20260930/mesh_study_spooles1/`.

| Koerper | Netz mm | Elemente | Knoten | k N/mm | u_max mm | sigma_max MPa | f1 Hz | Zeit s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 2 | 64931 | 119383 | 5.147838 | 0.25624653 | 6.700761 | 200.2521 | 83.23 |
| v0 | 1.5 | 97610 | 177627 | 5.158165 | 0.25643801 | 8.176502 | 201.7339 | 141.09 |
| Freie Referenz | 2 | 81024 | 143983 | 187.118130 | 0.00832870 | 0.650058 | 1068.3390 | 113.64 |
| Freie Referenz | 1.5 | 149614 | 257102 | 187.120571 | 0.00840162 | 0.653915 | 1063.3510 | 276.33 |

| Koerper | Verfeinerung mm | delta k | delta u_max | delta f1 | delta sigma_max | Screen bestanden |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v0 | 2 -> 1.5 | 0.2006 % | 0.0747 % | 0.7400 % | 22.0235 % | nein |
| Freie Referenz | 2 -> 1.5 | 0.0013 % | 0.8755 % | 0.4669 % | 0.5934 % | ja |

Die letzte Verfeinerung stabilisiert die Kennwerte des freien Referenzkandidaten innerhalb des festgelegten Screens. Ein einzelner bestandener Schritt beweist keine asymptotische Konvergenz. Die erneut stark steigende v0-Spannung verhindert weiterhin einen gemeinsamen Netzkonvergenznachweis und eine als konvergiert bezeichnete Spannungsverhaeltniszahl. Die historischen Akzeptanzgrenzen bleiben dennoch auf jeder berechneten Stufe erfuellt; die Werte sind jeweils als Ergebnisse des deklarierten diskreten Modells zu lesen. Unter 1.5 mm wurde nicht weiter gerechnet.

## Reproduktion und Daten

Der [Runner](../../tools/run_workstation_mesh_study.py) bindet Evidenz- und Quellhashes, Rekonstruktion, physikalische Inputs, Solveridentitaet und eigene Datei an die Studienidentitaet. Neue Ausgabeverzeichnisse muessen leer sein; Wiederaufnahme prueft jede rohe Artefaktdatei, das gespeicherte FEA-Resultat und die Vollstaendigkeit aller Lastfaelle. Drei gezielte [Negativtests](../../tests/test_workstation_mesh_study.py) bestehen. Die Ausgangs-Abnahme und historische Dichte werden nicht ueberschrieben.

Die erfolgreiche Wiederaufnahme aller sechs Resultate wurde vor der spaeteren Erweiterung um eine explizite Solverauswahl geprueft: sechs Cachetreffer, kein neuer FEA-Aufruf. Die alten Runnerbytes liegen unter `exports/topology/workstation_20260930/mesh_study/provenance/run_workstation_mesh_study.py` (SHA256 `8f1fe74c5bc3e8ef912f928c38f5fadb49b603ea1fef496b628f3f925d625152`), die vorherigen FEA-Core-Dateien unter `exports/topology/workstation_validation/source_before_backend/`. Nach der Aenderung an Runner und FEA-Core weicht die Quellprovenienz ab: Der aktuelle Runner darf den alten Lauf daher nicht als identischen Cache fortsetzen. Historische Records und Hashes bleiben lesbar und unveraendert; neue Rechnungen brauchen ein eigenes Ausgabeverzeichnis.

Beide Studien enthalten inzwischen je alle 26 Original-Python-Module und ihren jeweiligen Runner unter `provenance/`, mit verifizierter `source_manifest.json`. Nach Abschluss der Rechnungen wurde der FEA-Core nochmals ausschliesslich fuer JSON-sichere Fehler-Metadaten und strengere Thread-Eingabepruefung angepasst. Auch diese Quellveraenderung verhindert eine faelschlich identische Wiederaufnahme der alten Runs. Fuer eine neue Berechnung mit dem aktuellen Stand ist ein frisches Verzeichnis zu verwenden; Solver und Threadzahl werden ausdruecklich gesetzt:

```powershell
.\.venv\Scripts\python.exe -B tools/run_workstation_mesh_study.py --output exports/topology/workstation_20260930/mesh_study_spooles1_recheck --mesh-sizes 2 1.5 --linear-solver SPOOLES --threads 1
```

Der fehlgeschlagene urspruengliche 1.5-mm-Versuch wird dabei nicht ersetzt oder automatisch wiederholt. Exakte Cache-Wiederaufnahme setzt die Wiederherstellung der zur jeweiligen Studie archivierten Quellbytes und ihrer uebrigen Provenienz voraus.

Die [kompakte JSON-Evidenz](workstation_mesh_study.json) enthaelt Audit, alle statischen Faelle, alle sechs Moden pro erfolgreichem Lauf, Netzkennwerte, Vergleichsgrenzen, Hashes und den Fehlerfall. Vollstaendige STEP/STL, Gmsh-Netze, CalculiX-Eingaben und -Ausgaben sowie Logs liegen unter `exports/topology/workstation_20260930/mesh_study/` und `mesh_study_spooles1/`. Die [urspruenglichen Modellgrenzen](../topology_phase1.md) bleiben bestehen: isotropes trockenes Ersatzmaterial, feste Klemmungen, starre Akku-Kopplung, keine Vollcopter-Flugmoden und keine physische Druck-/Crashvalidierung.
