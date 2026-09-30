# Punktmassen-Sensitivitaet des freien Phase-1-Kandidaten

Der frische SIMP-Lauf mit 45 Iterationen wurde ohne nachtraegliches Aendern des Dichtefelds bei Schwelle 0.20 rekonstruiert. Elf protokollierte lokale Manifold-Fuellzellen ergeben einen gueltigen Solid aus 819 Voxeln und 147 Quadern. Die Frame-Masse betraegt 58.531804 g; mit dem 37-g-Akku sind es 95.531804 g. Der Lauf ist ein nicht konvergierter Demonstrator.

Die zusaetzliche Modalanalyse ersetzt ausschliesslich den skalaren Akku-Massenwert im bereits erfolgreich gerechneten CalculiX-Eingabedeck. Alle Netzknoten, Elemente, Materialwerte, Randbedingungen und Kopplungen bleiben identisch. Das Eingabedeck enthaelt `*MASS` mit 3.7e-05 t beziehungsweise 7.4e-05 t vor `*STEP` und `*FREQUENCY`. Die Materialdichte ist 1.09e-09 t/mm3. Der Referenzknoten liegt im Akku-Schwerpunkt bei z=34.5 mm und ist ueber `*RIGID BODY` mit 64 Knoten am 4-mm-breiten Strapband des Akkudecks verbunden. Der Evaluator prueft, dass diese Knoten nicht zu einer Fixierung gehoeren.

| Modus | Akku 37 g, Hz | Akku 74 g, Hz |
| --- | ---: | ---: |
| 1 | 1075.764 | 824.3202 |
| 2 | 1192.776 | 902.7666 |
| 3 | 1446.945 | 1070.140 |
| 4 | 1548.478 | 1525.608 |
| 5 | 1972.901 | 1902.266 |
| 6 | 2095.669 | 2090.076 |

Alle sechs Eigenfrequenzen sinken; keine Starrkoerpermoden wurden verworfen. Dies belegt, dass die Akku-Punktmasse an der Modalanalyse teilnimmt. Es ist kein Beleg fuer die Gueltigkeit der isotropen Materialannahme oder fuer eine reale Flugfreigabe. Der starre Patch unterdrueckt seine lokale Verformung; eine Sensitivitaet gegen Patchbreite, verteilte Akkumasse, Netzfeinheit und Druckanisotropie bleibt offen.

Der urspruengliche Kandidat besteht alle vier unabhaengigen Hauptanalysen mit Steifigkeit 194.391068 N/mm, maximaler Verschiebung 0.008062851 mm und maximaler Vergleichsspannung 0.517476108 MPa. Seine Geometrie besteht die 25 Preserve- und 56 Freiraumpruefungen. 4182 CAD-Normalenstrahlen messen mindestens 2.000 mm; der endliche Screen ist kein globaler Mindestdickenbeweis. Es wurden keine geschlossenen CAD-Hohlraeume oder eingeschlossenen Voxel-Leerraeume gefunden. Support ist erforderlich und erlaubt.

Die [maschinenlesbare Zusammenfassung](topology_phase1_mass_sensitivity.json) nennt Quellcommit, SHA-256-Dateipruefsummen und lokale Rohartefakte. Diese Zusatzdiagnostik liegt getrennt vom offiziellen Pipeline-Abnahmelauf. Das Dichtefeld stammt aus `Deep_Frame_P1B/exports/topology/final_solver`; die CAD- und Solverartefakte bleiben unter `Deep_Frame_P1C/exports/topology/final45_threshold20` erhalten.
