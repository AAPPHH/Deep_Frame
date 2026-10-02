# Optimierungsproblem: Zielfunktion, Nebenbedingungen und feste Bewertungsstufe

Stand 2026-10-02, Proof of Concept. Grundlage: neun Bewertungskriterien, für Referenzen und eigene Kandidaten identisch gemessen mit `tools/evaluate_frame.py` (Teile `geometry`, `walls`, `fea`, `slicer` als Ray-Jobs über `compute.py`). Die Ziel- und Warnbereiche stehen als Dicts in `deep_frame/config.py` (`EVALUATION_CONFIG["targets"]`, `EVALUATION_CONFIG["warnings"]`) und werden bei jedem `report` mitgeprüft.

Regel: Ein Kandidat ist nur gut, wenn alle Bedingungen gelten (absolute Bedingungen und Zielbereiche), unabhängig von der Mechanik. Nicht auswertbare Bedingungen gelten als verfehlt. Warnbereiche werden nur gemeldet und entscheiden nicht. Werte, die auf Materialannahmen beruhen, sind mit (A) markiert.

## Zielfunktion

Masse des gedruckten Frames minimieren (`geometry.mass.frame_mass_g`, Volumen der STL × 1,09 g/cm³, massiver Körper). Die Masse ist bewusst keine Nebenbedingung. Kontext aus den Referenzen: ManaFly 3 29,3 g (76-mm-Props, Achsabstand 160 mm), Aether4 86,2 g als massiver Körper (102-mm-Props, 191 mm). Gedruckt ist Aether4 hohl mit 2 Wänden, etwa 40 g. Aus Aether4 wird deshalb kein Massenband abgeleitet.

## Materialmodell

- Filament: Bambu Lab PA6-CF. Quelle: Technical Data Sheet V3.0, https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf (ISO 527 / ISO 1183; Proben 100 % Füllung, getempert und 12 h bei 80 °C getrocknet).
- Aus dem Datenblatt: E_xy 4430 MPa, E_z 2170 MPa, Zugfestigkeit XY 102 MPa, Zugfestigkeit Z 48 MPa, Dichte 1,09 g/cm³. Z liegt also gemessen bei 49 % von XY; die 50–60-%-Ersatzregel wird nicht gebraucht.
- Annahmen (A), nicht im Datenblatt: ν_xy = ν_xz = 0,30; G_xy = E_xy/(2(1+ν)) = 1704 MPa; G_xz = G_yz = E_z/(2(1+ν)) = 835 MPa. Zustand trocken und getempert, keine Feuchte, Poren, Plastizität, Ermüdung oder Dehnratenabhängigkeit.
- CalculiX: `*ELASTIC, TYPE=ENGINEERING CONSTANTS` mit `*ORIENTATION`, Materialachse 3 = Druckrichtung des Frames. Eigene Frames und ManaFly werden flach gedruckt (Modell-z). Aether4 wird stehend gedruckt; seine Druckachse stammt aus der Original-STL-Lage: (0; −0,640; 0,768).
- Crash-Festigkeit: 99,9-%-Wert der Integrationspunkte, v. Mises gegen σ_xy = 102 MPa und |σ| längs der Druckachse gegen σ_z = 48 MPa, jeweils mit Sicherheitsfaktor 2,0 (linear-statisch mit Ersatzlasten, keine Stoßdynamik).
- Lasten (`EVALUATION_CONFIG["loads"]`): Armspitze 3,6 N (2 × Schub 1,80 N) nach oben auf dem vorderen linken Motorsitz, Zentralbefestigung fest. Crash front 125 g × 25 g auf die Kamera, Crash Arm 125 g × 12,5 g schräg (2:2:1) auf den Motorsitz, Crash hinten 125 g × 12,5 g auf das Akkudeck mit festen Motorsitzen (Annahme). Modal: Unterseiten der vier Motorsitze fest, Akku als starr gekoppelte Punktmasse.

## Referenzen und Belege

| Referenz | evaluation.json | Anmerkung |
|---|---|---|
| ManaFly 3 BETA V4 | C:/clones/Deep_Frame-eval/exports/evaluation/ref_manafly3/evaluation.json | alle Bedingungen erfüllt; Steckbrief C:/clones/Deep_Frame/Examples/Frames/ManaFly3/datasheet.md |
| BM Aether 4 | C:/clones/Deep_Frame-eval/exports/evaluation/ref_aether4/evaluation.json | verfehlt nur `tools_reachable` (innere hintere Motorschrauben nicht geradlinig von unten erreichbar); walls/fea/slicer aus dem identischen Lauf exports/evaluation/aether4 übernommen (Ray-Ausfall); Steckbrief C:/clones/Deep_Frame/Examples/Frames/BM+Aether4/datasheet.md |

Evidenz der Einzelläufe: docs/validation/frame_evaluation_manafly3.md, docs/validation/frame_evaluation_aether4.md. Gesamttabelle aller Kandidaten: C:/clones/Deep_Frame-neural/exports/datasheets/EVALUATION.md.

## Skalierung

Die Referenzen haben andere Größen (Achsabstand 160 und 191 mm) als unsere Domänen (131–142 mm, meist 135 mm). Größenabhängige Kriterien werden deshalb dimensionslos verglichen (`scaled` in evaluation.json, berechnet aus den Motorpositionen):

- Armlänge L = mittlerer horizontaler Abstand der Motoren vom Motormittelpunkt, Achsabstand D = 2L.
- Armneigung = (F/k)/L: Durchbiegung der Armspitze unter 3,6 N bezogen auf die Armlänge (Neigungswinkel in rad).
- Stütze/Vol. = Stützmaterial des Slicers / Frame-Volumen.
- Symmetrie, Schwerpunktversatz, Höhe jeweils / D. Trägheit Izz / (Gesamtmasse × L²).
- Luftstrom ist bereits auf die Propfläche normiert (Materialanteil in der Projektion der vier Propscheiben, je Frame mit eigener Propgröße).
- Nicht skaliert: f1 (Anregung durch Motordrehzahl, nicht durch Rahmengröße; beide Referenzen liegen bei 346 Hz), Mindeststrebenbreite (Düse 0,4 mm), Überhanganteil, Öffnungsanteil, Crash-Ausnutzung.

## Nebenbedingungen und Zielbereiche

Die Bereiche umschließen beide Referenzen mit gerundetem Abstand; sie sind einseitig, wo die Physik eine Richtung vorgibt. Spalte „Für uns“ rechnet auf D = 135 mm, L = 67,5 mm um.

| Nr. | Kriterium | Größe (Pfad in evaluation.json) | Zielbereich | ManaFly | Aether4 | Für uns | Art | Zuordnung | Grund |
|---|---|---|---|---|---|---|---|---|---|
| 2 | Schwerpunkt | scaled.cog_offset_per_wheelbase | ≤ 0,010 | 0,0013 | 0,0031 | ≤ 1,35 mm | Warnung | äußere Schleife | Hängt an der Komponentenplatzierung der Domäne, nicht an der Topologie |
| 2 | Trägheit | scaled.izz_per_mass_arm2 | ≤ 0,45 (K) | 0,42 | 0,38 | – | Warnung | äußere Schleife | Folgt aus Komponenten und Achsabstand |
| 3 | Luftstrom | geometry.airflow.prop_ring_share | ≤ 0,15 | 0,110 | 0,140 | ≤ 15 % | hart | Optimierer | Material in der Propprojektion ist eine lineare Bedingung auf das Dichtefeld (passive Leerzone oder Strafterm) |
| 4 | Bohrbilder | geometry.assembly.bolt_patterns_passed | alle Löcher | 24/24 | 24/24 | 20/20 | hart, absolut | Rekonstruktion | Bohrungen als exakte Polyeder-Booleans nach der Extraktion (D1); ein Dichtegitter von 1,3–2 mm löst M2 nicht auf |
| 4 | Passung (Keep-outs frei) | geometry.assembly.fit_passed | ≤ 1 mm³ je Keep-out | ok | ok | ok | hart, absolut | Rekonstruktion (+ Optimierer) | Optimierer hält Keep-outs als passive Leerzonen; nur die exakte Subtraktion garantiert ≤ 1 mm³ |
| 4 | Werkzeug und Stecker | geometry.assembly.tools_passed | alle erreichbar | ok | nein | ok | hart, absolut | Optimierer (Domäne) | Werkzeugkorridore als verbotene Regionen in die Domäne, dann frei von Material |
| 6 | Ein Körper | geometry.form.mesh_bodies | = 1 | 1 | 1 | 1 | hart, absolut | Rekonstruktion | Konnektivität nach Extraktion; Inseln entfernen oder verwerfen |
| 5 | Überhang | geometry.printability.overhang_share | ≤ 0,25 | 0,161 | 0,238 | ≤ 25 % | hart | äußere Schleife | Heute nur geprüft; ein AM-Überhangfilter im Optimierer ist Folgearbeit |
| 5 | Mindestwandstärke (Öffnung r = 1 mm) | walls.deep_fraction | ≤ 0,06 | 0,0506 | 0,0426 | ≤ 6 % | hart | Rekonstruktion | Öffnung/Offset im Feld vor der Extraktion |
| 5 | Stützbedarf | scaled.support_per_volume | ≤ 1,5 | 1,12 | 1,45 | ≤ 1,5 × Vol. | hart | äußere Schleife | Nur der Slicer misst ihn |
| 5 | Druckzeit | scaled.print_min_per_g | ≤ 16 min/g | 15,7 | 12,6 | – | Warnung | äußere Schleife | Generische Slicer-Geschwindigkeiten, nur relativ vergleichbar |
| 6 | Strebe p10 | geometry.form.strut_width_mm.p10 | ≥ 1,2 mm (Grenze 1,19 wegen Voxelstufe 0,4 mm) | 1,2 | 1,4 | ≥ 1,2 mm | hart | Optimierer + Rekonstruktion | Filterradius bzw. Mindestlängenmaß im Optimierer, Öffnung in der Rekonstruktion |
| 6 | Symmetrie | scaled.symmetry_per_wheelbase | ≤ 0,0025 | 0,00024 | 0,00079 | ≤ 0,34 mm rms | hart | Optimierer | Spiegelsymmetrie über Halbdomäne erzwingen |
| 6 | Strebe p90 | geometry.form.strut_width_mm.p90 | ≤ 6,5 mm | 3,6 | 6,4 | ≤ 6,5 mm | Warnung | äußere Schleife | Stilmaß |
| 6 | Querschnitt H/B p50 | geometry.form.section_ratio.p50 | 1,0–1,4 | 1,12 | 1,25 | – | Warnung | äußere Schleife | Stilmaß |
| 6 | Draufsicht-Material | geometry.airflow.bbox_share | ≤ 0,45 (offen ≥ 55 %) | 0,29 | 0,41 | – | Warnung | äußere Schleife | Stilmaß |
| 6 | Schlaufen | geometry.form.loops.loops | ≥ 20 | 26 | 28 | – | Warnung | äußere Schleife | Stilmaß, abhängig vom Filterradius |
| 6 | Rauheit | geometry.form.roughness.curvature_neighbour_rms_per_mm | ≤ 0,18 /mm | 0,168 | 0,129 | – | Warnung | Rekonstruktion | Glättung des Feldes |
| 6 | Bauhöhe | scaled.height_per_wheelbase | ≤ 0,40 | 0,20 | 0,40 | ≤ 54 mm | Warnung | äußere Schleife | Domänenhöhe |
| 7 | Steifigkeit Armspitze (A) | scaled.arm_tip_slope | ≤ 0,007 rad | 0,00683 (6,6 N/mm) | 0,00298 (12,7 N/mm) | k ≥ 7,6 N/mm | hart | Optimierer | Nachgiebigkeit unter dem Lastfall arm_tip ist die Standard-Nebenbedingung der TO |
| 8 | Erste Eigenfrequenz (A) | fea.eigenfrequencies_hz.0 | ≥ 330 Hz | 346 | 347 | ≥ 330 Hz | hart | äußere Schleife | Eigenwert-Sensitivitäten mit Modenwechsel sind teuer; modale FEA läuft nur außen |
| 9 | Crash front/Arm/hinten (A) | assessment.crash.*.utilisation_xy, utilisation_z | ≤ 1 bei SF 2,0 | max. 0,24 / 0,19 | max. 0,19 / 0,11 | ≤ 1 | hart, absolut | äußere Schleife | Spannungsnebenbedingungen sind im Optimierer nicht umgesetzt; heute weit unter der Grenze |
| – | FEA gelöst, Slicer ok | fea.status, slicer.passed | ok | ok | ok | ok | hart, absolut | äußere Schleife | Voraussetzung für 7–9 und den Stützbedarf |

Hart = Teil der Gut-Regel (`target:<name>` bzw. absolute Bedingung in `assessment.missed`). Warnung = `assessment.warnings`, steht in der Bewertungszeile hinter „Warnung:“.

## Widersprüche und Ausnahmen

- Aether4 verfehlt `tools_reachable`. Die Bedingung bleibt für unsere Kandidaten absolut. Bei Aether4 ist das eine dokumentierte Ausnahme: Die inneren hinteren Motorschrauben erreicht man in der Praxis schräg oder mit kurzem Bit.
- Die vorläufige Wandregel in `topology_implicit_validation.py` (Tiefenanteil ≤ 0,5 %) ist mit ihrer eigenen Kalibriergeometrie unvereinbar: ManaFly misst 5,06 % und Aether4 4,26 % (`walls.calibrated_rule_passed = false` bei beiden). Als Zielbereich gilt deshalb ≤ 6 %. Die 0,5-%-Regel ist hier keine Bedingung.
- Masse, Steifigkeit, f1 und Crash von Aether4 gelten für den massiven Körper; der echte Druck ist hohl. Deshalb kein Massenband und nur einseitige Mechanikgrenzen.
- Symmetrie: Alle klassischen Kandidaten liegen bei 1,7–2,4 mm rms (≈ 1,3–1,8 % von D), die Neural-Kandidaten auf derselben alten Domäne bei 0,2–0,5 mm. Die Grenze von 0,34 mm wurde nicht aufgeweitet; die klassische GPU-SIMP-Route braucht eine erzwungene Spiegelsymmetrie.
- Strebe p90 ≤ 6,5 mm und Schlaufen ≥ 20 verfehlen fast alle eigenen Kandidaten. Das sind Warnungen: Unsere Frames sind gröber und weniger verzweigt als die Referenzen.
- Schwerpunktversatz: Alle eigenen Kandidaten liegen bei 1,2–3,0 % von D (Kamera vorn, Akku mittig). Das liegt an der Komponentenplatzierung der Domäne, nicht an der Topologie, deshalb nur eine Warnung.
- f1 der Referenzen liegt bei 344–347 Hz je nach Neuvernetzung. Die Grenze von 330 Hz lässt etwa 4 % Abstand.

## Feste Bewertungsstufe nach jedem neuen Kandidaten

Jeder Treiber (implicit_study, Neural- und fast-Treiber) schreibt nach jedem neuen Kandidaten eine `frame.json` und ruft eine Zeile auf:

```
cd /c/clones/Deep_Frame-eval && /c/clones/Deep_Frame/.venv/Scripts/python.exe tools/evaluate_frame.py run <frame.json>
```

Minimale `frame.json`:

```
{"name": "<Kandidat, Route, Domäne>", "stl": "<.../geometry.stl>", "output": "C:/clones/Deep_Frame-eval/exports/evaluation/<kandidat>",
 "domain": "<JSON der Design-Domäne, mit der der Kandidat optimiert wurde, z. B. .../density_source/inputs.json>",
 "datasheet": "C:/clones/Deep_Frame-neural/exports/datasheets/<kandidat>.md", "print_axis": [0, 0, 1]}
```

- `domain` ist ein Dateipfad. Die Datei enthält entweder einen Schlüssel `domain` (wie `inputs.json`) oder auf oberster Ebene direkt die Domäne. Ein Dict direkt in der frame.json wird nicht angenommen. Geprüft: combo_heavy mit `domain` ergibt dieselben Motoren, Bohrbilder, Komponenten, Keep-outs, Stecker und Selektoren wie die explizite frame.json. Daraus kommen Motoren, Bohrbilder, Komponenten, Keep-outs, Stecker und FEA-Selektoren, genau wie bei `"ours": true`, aber aus der tatsächlich verwendeten Domäne statt aus `build_design_domain` des Eval-Worktrees. Ohne `domain` gilt `"ours": true` mit `domain_grid` oder eine explizite Beschreibung (Referenzen).
- `run` reicht vier Ray-Jobs über `compute.py` ein (geometry = cpu, walls = wall_check, fea = fea_modal, slicer = cpu), wartet auf alle und ruft `report` auf. `report` schreibt `evaluation.json` und `bewertung.md` und hängt bei gesetztem `datasheet` den Abschnitt „## Bewertungszeile (neun Kriterien)“ an den Steckbrief an. Gibt es den Steckbrief noch nicht, wird er angelegt. Steht dieselbe Zeile schon darin, wird nichts angehängt.
- `run` blockiert bis zum Ende der Ray-Jobs (Warteschlange oft 10–30 min) und läuft deshalb am besten im Hintergrund. Fehlt danach ein Teil, zählen `report <frame.json>` (leicht, direkt) die fehlenden Kriterien als verfehlt.
- Bekannte Falle: Liegt die STL-Unterseite über z = 0 (fast_simp: z ≈ 0,03–0,09 mm), wählen die Fixture-Boxen keine Knoten aus. Dann die Fixture-Boxen in der frame.json bis z = 0,5 erweitern.
- Neue Zeilen enthalten die Ziel- und Warnbereiche. Ältere Steckbriefzeilen vom 2026-10-02 vor 15:40 enthalten nur die absoluten Bedingungen; maßgeblich ist EVALUATION.md.

## Stand der Kandidaten (2026-10-02)

Die vollständige Tabelle steht in C:/clones/Deep_Frame-neural/exports/datasheets/EVALUATION.md. Kurzfassung:

- ManaFly erfüllt alle Bedingungen. Aether4 verfehlt nur `tools_reachable`.
- neuralproto_fmax00625_f12 (51,6 g) ist der einzige eigene Kandidat, der alle Bedingungen erfüllt. Er hat nur Warnungen (Strebe p90, Schlaufen, Schwerpunkt) und ist 1,8 × so schwer wie ManaFly.
- Klassische Route, alte Domäne (45–57 g): Alle verfehlen Symmetrie, die meisten auch den Luftstrom (≈ 15–16 %). Bei wb142, impact2 (Standardnetz), f18 und f25 scheitert zusätzlich die Tet-Vernetzung (SICN).
- fast-Routen, neue Domäne (17–42 g): Alle verfehlen Bohrbilder und Keep-outs, weil das Gitter keine M2-Bohrungen auflöst und die exakte Rekonstruktion fehlt. SIMP-Kandidaten verfehlen zusätzlich die Symmetrie, fast_simp_v07 die Öffnung r = 1 mm (6,6 %), fast_simp_v05 die FEA (Vernetzung) und fast_neural_v05/v07 f1 (167/317 Hz). fast_neural_v07_mw zerfällt in 8 Körper.
- Leichteste Richtung mit erfüllter Mechanik: fast_neural_v10 (42,3 g) und fast_simp_v10 (42,1 g), die nur an Montage (und SIMP an Symmetrie) scheitern. Das ist der Hebel für die Rekonstruktion.
