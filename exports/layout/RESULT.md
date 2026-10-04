# Phase 2: Akkulagerung ohne vorgegebene Geometrie (Phase-1-Layout)

Stand 04.10.2026, 14:40. Branch `feature/layout-battery` (`/c/clones/Deep_Frame-layout`). Bild: `exports/layout/battery_vs_rails.png`. Zahlen: `exports/layout/battery_vs_rails.json`.

## Kurzantwort

- **Welche Lagerung entsteht?** Ein geschlossener **Rechteckring („Bilderrahmen“) unter dem Akkurand**. Er trägt eine durchgehende Oberkante genau auf der Akkuunterseite (z 25,8 mm), dazu einen **Querriegel im vorderen Drittel**. Der Ring steht über vier Eckstützen auf der Unterplatte und dem Stack.
  - Seitliche Lippen im Seitenband entstehen nicht. Die Seitenbänder bleiben leer.
  - Die Auflage liegt dort, wo die Lasten in die Stützen gehen: an den Ecken und Längskanten des Akkus statt mittig.
  - Gegenüber den Schienen (zwei Längsrippen an den Akkukanten mit Querstegen) wird die Auflage zum Ring, also zusätzlich zu Querkanten vorn und hinten.
- **Masse:** Rohkörper 17,0 g gegenüber 15,7 g mit Schienen (+8 %). Recon v3b 18,6 g gegenüber 17,6 g (+6 %).
- **Schwerpunkt und Trägheit:**
  - Der Rahmenschwerpunkt wandert 4,0 mm nach vorn (y 9,3 statt 5,3 mm) und 1,8 mm nach oben (z 14,7 statt 12,9 mm).
  - Rahmenträgheit Ixx +7 %, Iyy −12 %, Izz −2 %.
  - Ganzes Fluggerät am Phase-1-Layout: Schwerpunkt y 1,73 mm. Damit ist **die Phase-1-Bedingung CoG horizontal ±1 mm um 0,73 mm verletzt**. Das Layout war mit dem Schienenrahmen als Rahmenanteil optimiert worden. Das wird berichtet, nicht gelockert. Ein neuer Phase-1-Durchlauf mit diesem Rahmenanteil wäre der nächste Schritt.
- **α je Achse** (Phase-1-Layout, Komponenten gleich):
  - frei roh: 2433 / 2039 / 111,2 rad/s² (Rollen / Nicken / Gieren);
  - Schienen v3b: 2375 / 2072 / 111,5 rad/s²;
  - Änderung: Rollen +2,5 %, Nicken −1,6 %, **Gieren (Minimum) −0,3 %**;
  - frei v3b: 2409 / 1994 / 109,6 rad/s² (Gieren −1,7 %).

## Vergleich

Alle Werte stammen aus `battery_vs_rails.json`. Das Fluggerät ist jeweils komplett: Rahmen-STL × PA6-CF 1,09 g/cm³ plus Komponenten über `LayoutModel`, mit τ wie in Phase 1.

| Körper | Layout | Rahmen g | Rahmen-SP x/y/z mm | Rahmen Ixx/Iyy/Izz g mm² | Gerät g | Gerät-SP y / z über Rotorebene mm | Gerät Ixx/Iyy/Izz g mm² | α Rollen/Nicken/Gieren rad/s² | verletzt (Layout) |
|---|---|---|---|---|---|---|---|---|---|
| frei roh (`battery_free_raw_2`) | Phase 1 | 17,01 | −0,04 / 9,26 / 14,71 | 14728 / 12385 / 25581 | 86,3 | 1,73 / +0,46 | 77530 / 88746 / 152308 | 2433 / 2039 / 111,2 | cg_y |
| frei recon v3b (`battery_free_v3_2`) | Phase 1 | 18,61 | −0,13 / 9,83 / 14,60 | 16135 / 13168 / 27602 | 87,9 | 1,98 / +0,30 | 79306 / 89648 / 154580 | 2409 / 1994 / 109,6 | cg_y |
| Schienen recon v3b (`simp_mma_cov3_v3_2`) | Phase 1 | 17,62 | 0,14 / 5,26 / 12,90 | 13744 / 14023 / 25971 | 86,9 | 0,97 / +0,04 | 76305 / 90938 / 151903 | 2375 / 2072 / 111,5 | – |
| Schienen recon v3b | eigenes (v3b-Regel) | 17,62 | wie oben | wie oben | 86,9 | 1,99 / +0,36 | 75564 / 93659 / 148441 | 2306 / 2092 / 114,1 | cg_y, camera_sees_no_prop |

Mechanik nach dem Neun-Kriterien-Evaluator (gleiche Lastfälle; Grenzen tr ≤ 0,687 und λmax ≤ 0,229 N mm):

| Körper | Masse g | Armspitze N/mm | f1 Hz | Σ tr / λmax N mm | verfehlt |
|---|---|---|---|---|---|
| frei roh | 17,01 | 13,4 | 394 | 0,722 (105 %) / 0,230 (100,6 %) | bolt_patterns, keep_outs_free, Wandregel (wie jeder Rohkörper), **target:sigma_mean, target:sigma_worst** |
| Schienen roh (`simp_mma_cov3_raw_1`) | 15,72 | 10,9 | 407 | 0,687 / 0,220 | bolt_patterns, keep_outs_free, Wandregel, Symmetrie (vor 1bca977) |
| frei recon v3b | 18,61 | – | – | – | **FEA nicht auswertbar** (siehe unten) |
| Schienen recon v3b | 17,62 | 10,5 | 473 | 0,602 / 0,207 | keine |

Optimierer, Endstand fein 4/3 mm (`exports/runs/battery_free_opt/fine/result.json`):

- konvergiert nach 222 + 117 Iterationen, 1,8 h GPU;
- Feld 19,7 g (Schienen 18,2 g), Rohkörper 17,0 g;
- aktiv: tr und λmax des Lastmodells, crash_arm_front_right, crash_arm_rear_left, f1 = 303 Hz;
- nicht aktiv:
  - Akkuverschiebung, höchstens 0,106 mm (crash_arm_rear_left) gegen 0,5 mm; Flug 3σ 0,066 mm;
  - Auflagefläche 610 mm² gegen mindestens 350 mm².

Der Ring entsteht also nicht aus den Akku-Nebenbedingungen. Er entsteht aus dem Lastmodell (Akkublock von Σ über die Federn) und aus den Crashfällen, in denen die Akkuträgheit über die Federn auf den Rahmen geht.

## Was gemacht wurde

1. **Formulierung** (`deep_frame/topology_problem.py` `BatterySupport`, `docs/load_covariance.md` Abschnitt „Freie Akkulagerung“). Commits 86478db, a37f1bb, 7672644, 5ddc802, a9c434a.
   - Schalter `TOPOLOGY_CONFIG["battery_support"]`: Standard bleibt `"rails"`, `"free"` gilt nur für diesen Lauf.
   - Akku als Starrkörper: 37 g, Quaderträgheit, Keep-out `battery_insertion`.
   - Federn an allen Knoten zwischen erlaubten und Akku-Keep-out-Zellen:
     - Unterseite: Normalenfeder z und Schub x/y;
     - Seitenband: Normalenfeder x bzw. y;
     - Steifigkeit k = Fläche × Pad × (10⁻⁴ + ρ³).
   - Pad-ANNAHME: 1,0 N/mm³ normal und 0,3 N/mm³ Schub (Silikon-Haftpad, G ≈ 0,3 MPa).
   - Designabhängige Last: f_j = K_j T_j D⁻¹ w. Sie gilt für die sechs Akkuspalten von Σ (linearisiert, Seitenbänder mit Sekante 0,5) und für jeden Inertia-Relief-Fall. Die Akkuträgheit und die crash_back-Last greifen am Körper an.
   - z nur Druck: Active-Set auf die Starrkörperbewegung, Zug → 10⁻³ k. Den Kontaktzustand bestimmt eine ANNAHME von 10 N Gummibandvorspannung. Das Band selbst ist nicht modelliert.
   - Nebenbedingungen:
     - Federweg x/y am Akkuschwerpunkt ≤ 0,5 mm, im Flug als 3σ aus dem Σ-Akkublock, im Crash je Fall;
     - Auflagefläche ≥ 350 mm² (Schienenvariante 2 × 3,5 × 50 mm);
     - der Akkuanteil der Modalmasse wird nach k_z verteilt.
   - Sensitivitäten mit Lastterm: d(uᵀf)/dk = [T s]([u] − [T r]).
2. **FD-Prüfung:**
   - Minigitter-Test `test_free_battery_support_equilibrium_and_gradients`: Gleichgewicht und alle Zeilen, rel. 1e-4;
   - Rahmengitter 68 × 64 × 24 (`docs/validation/battery_support_fd.json`): alle 28 Zeilen, größter relativer Fehler 1e-7;
   - Suite: 730 bestanden, 8 übersprungen.
3. **Lauf:** SIMP+MMA mit Lastmodell, gleiche Grenzen und Kalibrierung wie `simp_mma_cov3`, 2 mm → 4/3 mm, gleicher Startentwurf (26,6-g-Feld).
   - Dazu ein neutraler Start: 0,5 in den Zellen bis zwei Zellen um den Akku. Das Startfeld trug den Akku nur auf der alten Höhe 28 mm.
   - Ohne diesen Start lief der erste MMA-Schritt mit Crash-Nachgiebigkeit ×300 in NaN.
4. **Körper und Steckbrief:**
   - roh: `exports/runs/battery_free_raw_2`;
   - recon v3b: `exports/runs/battery_free_v3_2`. Rekonstruktionsmodul und Konfiguration stammen aus b74dcf2 und liegen per git archive über dem Layout-Branch, weil der alte Stand das Phase-1-Layout (negatives battery_y) nicht abbildet.
   - STLs: `C:/clones/Deep_Frame-neural/exports/battery_free/geometry.stl` und `.../battery_free_v3b/geometry.stl`.
   - Der Steckbrief `datasheet.md` jedes Laufs enthält Zeile 12 Dynamik (α, Schwerpunkt, Trägheit).

## Offene Punkte (berichtet, nicht gelockert)

- **Evaluator-Selektor:** Der Akku-Selektor des Evaluators war ein 4-mm-Band in Akkumitte. Die freie Lagerung berührt dort nicht. Der erste Evaluationslauf (`*_1`) brach deshalb mit „Empty node selector“ ab.
  - Bei freier Lagerung gilt der Selektor jetzt für die ganze Akkufläche auf der Unterseitenebene (Commit „evaluator battery selector“).
  - Lasten und Massen sind unverändert. Die Schienenvariante ist davon nicht betroffen.
  - Das Σ-Maß des Evaluators ändert sich dadurch für die freie Variante. Die Kalibrierung 1,384 / 1,374 stammt vom Schienenrahmen.
  - Die 105 % / 100,6 % bei tr / λmax sind daher eine Kalibrierfrage. Erst ein Kalibrierlauf mit freier Lagerung würde zeigen, ob die Grenzen eingehalten werden.
- **recon v3b frei:** v3b legt die Oberkante unter dem Akku auf z 25,5 statt 25,8 mm. Schienen glättet v3b als feste Bereiche exakt, eine freie Auflage nicht. Damit bleibt **0,3 mm Spalt zum Akku**, und der FEA-Selektor findet keine Knoten.
  - Mechanik des v3b-Körpers: nicht ausgewertet (zählt als verfehlt).
  - Nötig wäre in v3b eine Ebenenglättung für `battery_contact`, so wie für `battery_rail_` im Rohexport.
- **Schwerpunkt:** CoG y +0,73 mm über der ±1-mm-Grenze. Grund ist der neue Rahmenanteil (Ring vorn schwerer). Phase 1 müsste mit diesem Rahmen erneut laufen.
- **Akku-Nebenbedingungen:** Sie sind bei den angenommenen Padsteifigkeiten weit inaktiv. Bei weicherem Pad oder kleinerer Mindestfläche könnten Seitenlippen entstehen. Nicht untersucht.
