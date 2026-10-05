# Freier 4/3-mm-Frame: Wiederaufnahme und DGX-Übergabe

Stand 05.10.2026. Der Nutzer hat während der Nachprüfung auf Neuberechnung auf der DGX umgestellt. Deshalb wurden die noch laufende Sigma-Auswertung und die Layout-Nachoptimierung des reparierten Körpers gestoppt. Die großen Felder, STLs, Netze und Bilder werden nicht übertragen. Dieser Bericht enthält vorhandene Messungen, keine Freigabe als druckfertiger Frame.

Der Optimierungslauf `free_layout4_opt/fine2` war bereits abgeschlossen: 155 Iterationen, etwa 15,56 g Dichtefeld, f1 etwa 300 Hz. Er wurde nicht wiederholt. Rohkörper und v3c wurden übernommen und der Kamerakontakt anschließend gezielt korrigiert.

| Messung | v3c vor Kontaktkorrektur | v3c mit Kontaktkorrektur |
|---|---:|---:|
| STL-Masse | 15,5321 g | 15,6706 g |
| Luftspalt am Kameraanschluss links / rechts | 0,474 / 0,300 mm | 0 / 0 mm |
| Kontaktquerschnitt 0,01 mm vor der Fläche, links / rechts | 0 / 0 mm² | 45,08 / 45,07 mm² |
| Armspitzensteifigkeit | 23,8066 N/mm | 23,6350 N/mm |
| f1, Körper-FEA | 170,201 Hz | 669,565 Hz |
| Frontalcrash, p99,9 von Mises / Drucknormalspannung | 71,239 / 22,243 MPa | 6,494 / 1,913 MPa |
| Sigma tr / größter Eigenwert | 0,45988 / 0,13486 N mm | abgebrochen, neu berechnen |
| Standreserve, Vorgabe mindestens 15 mm | −4,509 mm | −4,443 mm |
| Propellerunterkante über Boden, Vorgabe mindestens 17 mm | 19,743 mm | 19,743 mm |
| Wandregel: tiefer Anteil, höchstens 0,5 % | 1,421 % | 1,364 % |
| Größte tiefe Wandverletzung, höchstens 5 mm³ | 47,654 mm³ | 47,654 mm³ |

Quelle: `free_contact_metrics.json`, `camera_contact_diagnosis.json`. Kontaktquerschnitte werden mit 0,1-mm-Raster in Schnittebenen des echten STL innerhalb eines 4-mm-Kreises um die Schraubachse gemessen. Das ist keine Optimierer-Federfläche. Die Körper-FEA verwendet ihre deklarierten Vergleichslastfälle und Einspannungen; ihre Frequenz ist nicht direkt die erodierte Optimiererfrequenz. Die Ersatzoberfläche weicht nach der Korrektur maximal 0,282 mm vom STL ab; die bisher als Warnung behandelte 0,20-mm-Grenze bleibt überschritten.

## Ursache und Korrektur

Das konservativ gerasterte Kamera-Keep-out lässt bereits am Rohkörper etwa 0,918 mm Abstand zur exakten Kamera-Seitenfläche. Die Splinerekonstruktion hatte nur Akku-Kontaktflächen bündig gezogen und verkleinerte zusätzlich den vorhandenen Kamerafußabdruck. Dadurch blieben zwar seitliche Schraublaschen sichtbar, ihre reale Kontaktfläche zur Kamera fehlte.

Commits `19eb1a2` und `f1831d5` erhalten den vorhandenen Rohkontakt-Fußabdruck lokal im Kreis um die Schraubachse und führen ihn zur exakten Seitenfläche. Die Reichweite beträgt 2 mm beziehungsweise mindestens Gitterweite plus Boolean-Abstand. Keep-outs und Schraubbohrung werden anschließend wieder exakt ausgeschnitten. Ein leerer Kontaktbereich erzeugt keine neue Lasche. Es werden keine Kamerabügel vorgeschrieben. Nachweis: 16 Rekonstruktionstests bestanden über Ray; beide Anschlussflächen ohne Luftspalt; ein wasserdichter Körper. Gemessener Spitzen-RAM der Rekonstruktion: 1,954 GiB.

## Form, Stand, Stack und Akku

Die gesichtete v3c-Isometrie zeigt zwei offene seitliche Schutzstreben vor der Kamera. Oberhalb der Kamera entsteht kein geschlossener Querbügel. Die Kameraanschlüsse gehen seitlich in die vorderen Armwurzeln und den vorderen unteren Querverbund. Ein geschlossener Schutz vor und über der Kamera ist damit weiterhin nicht nachgewiesen.

Die vier Stackpfosten sind in den unteren Querverbund und die seitlichen Übergänge zu den Armwurzeln eingebunden. Der Akku liegt auf einem offenen rechteckigen Randverband mit Querstrebe auf z = 20 mm; die Tragstreben führen zu den Armwurzeln und dem zentralen Tragwerk. Diese Beschreibung bezieht sich auf die gesichteten Körperbilder, nicht auf eine zusätzliche Festigkeitsfreigabe.

Der tiefste Körperpunkt liegt auf z = −0,509 mm. Die Kontaktfläche ist nur die etwa 142,23 mm² große Hülle zweier zentraler tiefer Bereiche bei y ungefähr 5,4 bis 11,2 mm, nicht die vier Motorpads. Der Schwerpunkt liegt hinter dieser Hülle: Standreserve negativ. Die Kontaktkorrektur verändert dieses Standproblem nicht. Die größten Wandverletzungen liegen um die vier Stackpfosten. Beide Bedingungen bleiben verfehlt.

Die Neun-Kriterien-Auswertung des korrigierten Körpers ist wegen fehlendem Sigma **unvollständig**. Bereits fertig: 20/20 Bohrungen, Bauteilpassung und Werkzeugzugang bestanden, ein Körper, Luftstromfläche etwa 5,2 %, Überhänge etwa 21,2 %, Druckzeit mit Unterstützung etwa 221 Minuten. Armsteifigkeit, f1 und die drei ausgewerteten Crashfestigkeiten bestehen. Wandregel besteht nicht; Sigma zählt bis zur Neuberechnung als nicht erfüllt. Die gesonderte Standbedingung besteht ebenfalls nicht.

## Layout und Kameramaße

Die vorhandene Nachoptimierung `layout_optimization_free.json` gehört zum **alten** 15,5321-g-v3c-Körper. Sie erfüllt horizontal ±1 mm mit Schwerpunkt y = 0,9962 mm. Ihr gerundeter Vorschlag lautet Akku-y −2,7 mm, Akkuunterkante 20,0 mm, Kamera-y 40,1 mm, Kamerabodenabstand 11,2 mm, Stackabstand 3,0 mm. Gegenüber dem gebauten Layout: Akku +0,1 mm, Kamera −1,0 mm längs und +4,2 mm höher. Alpha Rollen/Nicken/Gieren: 2471,4 / 2244,4 / 113,84 rad/s². Das ist ein neuer Layoutvorschlag; seine Anschlüsse müssen in einer neuen Geometrie erzeugt werden.

Der reparierte Körper hat im unveränderten gebauten Layout Schwerpunkt y = 1,0454 mm und verletzt die ±1-mm-Bedingung um 0,0454 mm. Seine erneute Layoutoptimierung wurde für den DGX-Wechsel gestoppt. Der alte Layoutnachweis darf hierfür nicht übernommen werden.

`free_body_measure_raw.json` und `free_body_measure_v3c.json` enthalten die früheren Rückvoxelisierungen des STL in das 4/3-mm-Optimierergitter, positiv-x-Hälfte gespiegelt. Vor der Kontaktkorrektur ergaben sich dort Kamerawege 168,57 / 111,57 mm, Abschirmung 0,23826 / 0,02543 und Coverage 0,32131. Das Verfahren verliert lokale Anschlüsse durch die 50-%-Binärschwelle; diese Werte sind ausdrücklich **keine Messungen des korrigierten Körpers** und keine direkte Ganzkörper-FEA. Neue Kamerafeder-/Coverage-/Abschirmungswerte und Twist stehen aus. Der Rohkörper ließ sich in der vorhandenen Körper-FEA trotz konfigurierter Netzversuche nicht auswerten; die fehlende FEA wird nicht als bestanden gewertet.

## Auf der DGX fortsetzen

Alle folgenden Eingaben sind relativ zum Repository; Python muss in der eingerichteten Umgebung verfügbar sein. `tools/compute.py` muss auf den laufenden Ray-Head zeigen. Die Neuberechnung muss zuerst `exports/runs/free_layout4_opt/fine2/density_half.npz` mit dem portierten Optimierungslauf erzeugen. Die Rekonstruktionsanforderung enthält die vollständige damalige Bauteil-/Layoutkonfiguration und verwendet den integrierten aktuellen Quellcode, keinen historischen Worktree.

```sh
python tools/compute.py geometry -- python exports/layout/postprocess_free.py export
python tools/compute.py geometry -- python run.py stage exports/layout/free_contact_reconstruction_dgx.json
python exports/layout/postprocess_free.py stage
python tools/evaluate_frame.py run exports/layout/free_contact_evaluation_dgx.json
python tools/compute.py cpu -- python exports/layout/postprocess_free.py contacts
python tools/compute.py cpu -- python exports/layout/postprocess_free.py layout
```

Die Evaluation verteilt ihre Unteraufgaben selbst über Ray. Die 4-GiB-Klasse `geometry` reicht laut gemessenem Peak für diesen 4/3-mm-Körper; bei 0,75 mm muss die Ressourcenklasse anhand der dortigen Größe neu gewählt werden.

Danach fehlen noch: vollständiger Sigma-/Kamera-/Twist-Nachweis, Datenblatt des neuen Körpers mit Dynamikzeile, vier maßstabsgleiche Vergleichsansichten v3c | Schienen-v3b | ManaFly sowie der Vergleich 0,75 mm | 4/3 mm | ManaFly. Die Bilder und großen Körperdateien werden auf der DGX neu erzeugt. Das Kabelrouting übernimmt den korrigierten Körper; sein dortiger Neulauf und Nachweis sind eine eigene Aufgabe.
