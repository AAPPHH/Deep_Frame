# Geometrie-, Kollisions- und Massenchecks

`validate_geometry(parameters)` liefert `passed`, `violations` und `checks` als JSON-faehiges Dict. Pruefungen sind Topologie, Ist-Bohrungsachsen/-durchmesser im CAD, konstruktive Wandstege, alle paarweisen Abstaende/Volumenueberschneidungen sowie Akku-Prop-Draufsicht. Fehlerhafte Bauteil- oder Frameparameter werden als fehlgeschlagene Geometrie gemeldet. Der Optimierer kann das Dict vor jeder FEA auswerten.

Der minimale Abstand zwischen zwei Shapes wird durch OpenCascade bestimmt. Bei Kontakt werden echte Schnittvolumina berechnet. Kontakte Frame/AIO, Frame/Akku, Frame/Motoren und Frame/Stecker sind als geplante Auflage mit0-mm-Mindestabstand bezeichnet; ein positives Schnittvolumen ueber1e-6 mm3 scheitert trotzdem. Fuer Prop-Paare und alle propellernahen Teile gelten2 mm, fuer sonstige getrennte Komponenten0.5 mm. Die Toleranz1e-6 mm betrifft nur CAD-Numerik. Eine reale Passung benoetigt zusaetzlich Prozess- und Montagezugaben.

Die Projektion ueberlagert die Vereinigung aller Propellerschwenkscheiben mit dem Akku-Grundriss entlang z. Doppelt ueberdeckte Bereiche werden nur einmal gezaehlt. Prozent bedeutet Schnittflaeche geteilt durch Akku-Draufsicht; es ist keine Aussage ueber3D-Kollision. Der Default erlaubt0 Prozent. Ein100-mm-Radstand erzeugt echte Prop-Prop- und Prop-Frame-Durchdringungen und verletzt diesen Check.

Frame-Masse folgt CAD-Volumen mal1.09 g/cm3. Jede Komponente verwendet ihre konfigurierte Masse und die homogene CAD-Ersatzhuelle; OpenCascade liefert Volumenintegrale und lokales Traegheitsmoment um ihren Massenschwerpunkt. Mit dem Steiner-Satz werden alle Tensoren in den gemeinsamen Baugruppenschwerpunkt umgerechnet. Ausgabe ist g, mm und g mm2. Der Tensor gilt in globalen x/y/z-Achsen und hat auch die Nebendiagonalelemente; eine Bounding-Box-Naeherung wird nicht verwendet. Akku-Leitungen und Stecker sind massemaessig in37 g enthalten und werden nicht doppelt gezaehlt.

`run_frame.py` schreibt STL, STEP und das Check-Dict unter `exports/frame_v0.*` und zeigt die Baugruppe im laufenden OCP CAD Viewer. Propellerscheiben sind transparent; reale Schnittkoerper werden deckend rot als `COLLISION ...` angezeigt. `assembly_scene()` bietet dieselben Shapes/Farben ohne Viewer fuer Tests. Ein korrekt dargestellter Frame ist keine Festigkeitsfreigabe; FEA, Druckversuch und reale Montage bleiben separate Nachweise.

Ausfuehren aus dem Repository mit seiner installierten Umgebung:

```powershell
.\.venv\Scripts\python.exe run_frame.py
.\.venv\Scripts\python.exe -m pytest -q
```

Validierung am2026-09-30:41 pytest-Tests bestanden. DefaultFrame32.1629 g, Gesamtmodell105.0629 g, Schwerpunkt(-0.0121,0.7638,18.9069) mm; Akku/Prop-Projektion0 Prozent. Das vollstaendige Dict liegt unter `docs/validation/frame_v0.json`, die frische OCP-Viewer-Canvas unter `docs/validation/frame_v0_viewer.png`. STL wurde als wasserdicht, orientiert und ein zusammenhaengender Volumenkoerper mit trimesh geprueft. `collisions_100mm.png` zeigt den absichtlich ungueltigen100-mm-Radstand mit roten Schnittkoerpern.
