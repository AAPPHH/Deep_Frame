# Frame v0, Komponenten und Checks

## Frame v0

`deep_frame.frame.build_geometry(parameters)` liefert einen geschlossenen `build123d.Solid`. `reference_parameters()` liefert eine unabhaengige JSON-faehige Kopie aus `CONFIG`; `validate_geometry()` prueft sie ohne FEA. Die urspruengliche Smoke-Konfiguration und `run.py smoke` bleiben erhalten. Die Defaults verweisen ueber `default_sources` auf jede Kennwertzeile in `docs/research/design_defaults.json` und deren Armattan-/Prinzip-IDs.

Der Radstand135 mm und das Verhaeltnis112/82 ergeben seitlich108.92648 mm und laengs79.74974 mm. Vier gerade6.5 x4 mm Arme verbinden den Zentralkoerper mit runden Motorpads. Das Motorlochbild folgt demselben Komponentenparameter wie der Motorplatzhalter. AIO-Bohrungen liegen bei x/y=+-12.75 mm. `minimum_wall_mm=2` bestimmt Seitenwaende und konstruktive Mindeststege. `structural_margins()` prueft unter anderem Bohrung-zu-Wellenloch, Bohrung-zu-Rand, AIO-Fenster, Wandfenster und Strap-Stege; es ist keine globale FEM- oder Slicer-Wanddickenanalyse.

Der Akku liegt mit30-mm-Breite auf einem40 x70 mm Deck bei Z28 (gemeinsame Knotenebene der Topologiegitter). Zwei seitliche Waende leiten die Decklast um das Board herum zum zentralen Boden und zu den Armwurzeln. Das Mittelfenster spart Volumen; es ist keine geschlossene Akkuwanne. Zwei Paare2 x12-mm-Schlitze bei x=+-16.5, y=+-12 fuehren quer ueber die breite Akkuoberflaeche. Der Boden hat ein20-mm-Fenster: das urspruengliche22-mm-Fenster liess am AIO-Loch weniger als2 mm Material. Das zentrale Motorwellenloch ist2.8 mm, um beim9-mm-Lochkreis einen2-mm-Netzsteg zu halten. Beide Werte sind eigene Druckableitungen, keine Armattan-Geometrie.

Die gedrehte Lux-Huelle steht bei y35; die seitliche Schraubachse folgt ihrem Schwerpunkt und damit dem Tilt. Vorne und hinten verbinden obere Stege die Kaefigwaende. Der Heckbereich enthaelt nach hinten offene XT30-/Balancer-Aufnahmen, Kabeldurchlass und vertikale3-mm-Antennenbohrung bei y=-56. Haltepassung, reale Kabelbiegeradien und Zugentlastung bleiben prototypisch.

Das Teil ist zusammenhaengend in einem Druck fertigbar. Z0 ist die ebene Unterseite. Akkudeck, Kaefigbruecken und Wandfenster erzeugen Bruecken/Ueberhaenge; lokale Stuetzstrukturen oder ein validiertes Brueckenprofil sind erforderlich. Einteiliger Solid bedeutet keine Zusicherung eines supportfreien Drucks. Schraubenmontage, AIO-Kabelzugang, reale Kamera-Gewinde, Propellerflex, Temperatur und Fluglasten muessen am physischen Prototyp geprueft werden. Die Integralarme sind nach strukturellem Bruch nicht einzeln austauschbar; der Frame wird nachgedruckt.

FEA-Selektoren: Motorzentren sind `(+-54.46324036,+-39.87487241)` mm. Motorpad-Unterseite liegtZ0; ein Selektor mit x/y um das jeweilige Zentrum+-9.6 mm und z+-0.1 mm erreicht das Pad. Akkudeck-OberflaecheZ28 kann ueber x+-20.1, y+-35.1, z27.9..28.1 selektiert werden. Ein zentraler kleiner Quader kann wegen des Deckfensters leer sein. Akku-Punktmasse37 g liegt bei `(0,0,33.5)` mm. Beim Armsteifigkeitsfall den Zentralkoerper fixieren und das freie Motorpad belasten; alle vier Motoren zu fixieren misst nicht die freie Armsteifigkeit.

## Parametrische Komponenten und Annahmen

Koordinaten: x quer, y nach vorne, z nach oben. Die ungedrehte Komponente liegt mit ihrer Unterseite auf z=0; Breite ist x, Laenge y. Kamera-Tilt dreht um x und hebt die gedrehte Huelle anschliessend auf z=0. Montagehoehe ist deshalb nicht unabhaengig vom Tilt festgeschrieben. Alle Masse und Massen stehen in `COMPONENT_DEFAULTS` von `deep_frame/config.py` und werden von der zentralen `CONFIG` uebernommen.

| Komponente | Huelle / Masse | Herkunft und Grenze |
|---|---|---|
| HDZero AIO15 |31.3 x 31.3 mm; 25.5 x 25.5 M2; 7.2 g|Nutzervorgabe; komplette Stackhoehe vorlaeufig 6 mm. Boardmodell hat echte vier Durchgangsbohrungen.|
| HDZero Lux |Laenge14, Breite16, Hoehe14 mm; 2.3 g|Nutzervorgabe; Tilt20 Grad als Designstartwert. Seitliche Kameraschraubenlage/-groesse ist noch nicht durch Lux-Zeichnung validiert.|
| GNB5502S120A |63 x 30 x 11 mm; 37 g|Nutzervorgabe; Masse einschliesslich Leitungen/Steckern am Akkuschwerpunkt konzentriert.|
| Motor |Durchmesser14.2, Hoehe14.6 mm; 5.9 g|Vorlaeufige [GEPRC GR1105 Herstellerreferenz](https://geprc.com/product/gep-gr1105-motor/), [Masszeichnung](https://geprc.com/wp-content/uploads/2019/05/22-6199766706.jpg), [Massenabbildung](https://geprc.com/wp-content/uploads/2019/05/22-8095453337.jpg). Die Hoehe schliesst die obere Welle ein; das Modell nutzt konservativ einen vollen Zylinder. Die Masse umfasst die abgebildeten Kabel.|
| Prop |HQProp T2.5X2X3V2S: 63.5-mm-Scheibe, 5 mm hoch (Nabe 9.8 x 5 mm, Welle 1.5 mm); 1.2 g; 3 Blaetter, Steigung 2, PC, 2 CW + 2 CCW|Nutzerentscheidung (COMPONENT_LIBRARY). Ohne Adapterringe sitzt die Nabe direkt auf der Glocke (`prop_motor_gap_mm` = 0, als geplanter Kontakt Motor/eigener Prop gewertet); die Scheibe reicht ueber die Nabenhoehe. Schub SCHAETZUNG ~206 g = 2.02 N aus der Zeile HQ T65R, 7.4 V, 100 % der GTS-V3-1203-8000KV-Tabelle (aehnlicher, nicht identischer Prop). Andere Propgroessen ohne Bibliothekseintrag nutzen weiter die vorlaeufige Scheibenregel.|
| XT30U-F |10.2 x 12.4 x 5.2 mm|[AMASS-Zeichnung Seite2](https://images.100y.com.tw/pdf_file/AMASS-XT30U.pdf#page=2); Anschluss-/Kabelraum separat zu pruefen.|
| Balancer |9.8 x 7.5 x 5.7 mm;3 Pins|Vorlaeufige [JST XHP-3-Huelle](https://www.jst-mfg.com/product/pdf/eng/eXH.pdf#page=4), tatsaechlichen GNB-Stecker pruefen.|

Die Formulierung 9 x 9 mm ist nicht eindeutig mit der gewaehlten typischen 11xx-Referenz vereinbar: deren Zeichnung zeigt vier M2 auf einem **9-mm-Lochkreis**, nicht ein Quadrat mit9-mm-Seiten. Vorlaeufig wird das belegte Lochkreisbild verwendet (`mount_layout="bolt_circle"`, Achsen bei +-9/sqrt(8) mm). `mount_layout="square"` erzeugt weiterhin das explizite9-x-9-Quadrat, benoetigt mit2.2-mm-Bohrung aber eine groessere Motorhuelle als14.2 mm. Die Wahl ist keine Bestaetigung eines bereits ausgewaehlten Motors. Vor Bestellung/Druck muss das reale Modell die Default-Annahme ersetzen.

XT30 und Balancer bekommen0 g Zusatzmasse, damit die37-g-Akkuangabe nicht doppelt gezaehlt wird. Ihr tatsaechlicher Massenversatz bleibt unbekannt. Schrauben, Akku-Straps, Kabel ausserhalb der angegebenen Komponentenmassen, Antenne und Druckhohlraeume sind nicht zusaetzlich modelliert. Die Baugruppenmasse ist eine Ersatzmodellbilanz.

PA6-CF ist noch kein festgelegtes Filament. Als generische Annahme wird die [Bambu PA6-CF TDS v2](https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf) mit1.09 g/cm3 verwendet. FEA nutzt separat4430 MPa Young-Modul und angenommene Querkontraktion0.30. Isotropie, voller Materialanteil und trockener Werkstoff sind Rechenannahmen; ein reales Druckprofil und Feuchte-/Richtungsabhaengigkeit sind damit nicht nachgewiesen.

## Geometrie-, Kollisions- und Massenchecks

`validate_geometry(parameters)` liefert `passed`, `violations` und `checks` als JSON-faehiges Dict. Pruefungen sind Topologie, Ist-Bohrungsachsen/-durchmesser im CAD, konstruktive Wandstege, alle paarweisen Abstaende/Volumenueberschneidungen sowie Akku-Prop-Draufsicht. Fehlerhafte Bauteil- oder Frameparameter werden als fehlgeschlagene Geometrie gemeldet. Der Optimierer kann das Dict vor jeder FEA auswerten.

Der minimale Abstand zwischen zwei Shapes wird durch OpenCascade bestimmt. Bei Kontakt werden echte Schnittvolumina berechnet. Kontakte Frame/AIO, Frame/Akku, Frame/Motoren und Frame/Stecker sind als geplante Auflage mit0-mm-Mindestabstand bezeichnet; ein positives Schnittvolumen ueber1e-6 mm3 scheitert trotzdem. Fuer Prop-Paare und alle propellernahen Teile gelten2 mm, fuer sonstige getrennte Komponenten0.5 mm. Die Toleranz1e-6 mm betrifft nur CAD-Numerik. Eine reale Passung benoetigt zusaetzlich Prozess- und Montagezugaben.

Die Projektion ueberlagert die Vereinigung aller Propellerschwenkscheiben mit dem Akku-Grundriss entlang z. Doppelt ueberdeckte Bereiche werden nur einmal gezaehlt. Prozent bedeutet Schnittflaeche geteilt durch Akku-Draufsicht; es ist keine Aussage ueber3D-Kollision. Der Default erlaubt0 Prozent. Ein100-mm-Radstand erzeugt echte Prop-Prop- und Prop-Frame-Durchdringungen und verletzt diesen Check.

Frame-Masse folgt CAD-Volumen mal1.09 g/cm3. Jede Komponente verwendet ihre konfigurierte Masse und die homogene CAD-Ersatzhuelle; OpenCascade liefert Volumenintegrale und lokales Traegheitsmoment um ihren Massenschwerpunkt. Mit dem Steiner-Satz werden alle Tensoren in den gemeinsamen Baugruppenschwerpunkt umgerechnet. Ausgabe ist g, mm und g mm2. Der Tensor gilt in globalen x/y/z-Achsen und hat auch die Nebendiagonalelemente; eine Bounding-Box-Naeherung wird nicht verwendet. Akku-Leitungen und Stecker sind massemaessig in37 g enthalten und werden nicht doppelt gezaehlt.

`run.py frame` schreibt STL, STEP und das Check-Dict unter `exports/frame_v0.*` und zeigt die Baugruppe im laufenden OCP CAD Viewer. Propellerscheiben sind transparent; reale Schnittkoerper werden deckend rot als `COLLISION ...` angezeigt. `assembly_scene()` bietet dieselben Shapes/Farben ohne Viewer fuer Tests. Ein korrekt dargestellter Frame ist keine Festigkeitsfreigabe; FEA, Druckversuch und reale Montage bleiben separate Nachweise.

Ausfuehren aus dem Repository mit seiner installierten Umgebung:

```powershell
.\.venv\Scripts\python.exe run.py frame
.\.venv\Scripts\python.exe -m pytest -q
```

Validierung am2026-09-30:41 pytest-Tests bestanden. DefaultFrame32.1629 g, Gesamtmodell105.0629 g, Schwerpunkt(-0.0121,0.7638,18.9069) mm; Akku/Prop-Projektion0 Prozent. Das vollstaendige Dict liegt unter `docs/validation/frame_v0.json`, die frische OCP-Viewer-Canvas unter `docs/validation/frame_v0_viewer.png`. STL wurde als wasserdicht, orientiert und ein zusammenhaengender Volumenkoerper mit trimesh geprueft. `collisions_100mm.png` zeigt den absichtlich ungueltigen100-mm-Radstand mit roten Schnittkoerpern.
