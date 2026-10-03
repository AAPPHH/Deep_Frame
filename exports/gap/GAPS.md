# Lückenfinder Schnittstellensteifigkeit: SIMP+MMA gegen ManaFly 3 und Aether 4

Reine Messung. Es wurde keine Bedingung eingebaut; alle Grenzen unten sind Vorschläge zur Freigabe.

## Methode

- Bewertungs-FEA (CalculiX auf dem Tet-Netz der jeweiligen Bewertung, gleiches Material, gleiche Lagerung). Werkzeug `tools/interface_stiffness.py`.
- Je Schnittstelle und globaler Richtung eine Einheitslast: 1 N (Fx, Fy, Fz) bzw. 1 N·mm (Mx, My, Mz) um die Achse durch den Knotenschwerpunkt der Schnittstelle.
- Steifigkeit arbeitskonjugiert: k = P² / Σ fᵢ·uᵢ, also Last durch mittlere Verschiebung bzw. mittlere Verdrehung. F in N/mm, M in N·mm/rad.
- Lagerung: Motorsitze, Akkuschienen und Kamera gegen die festgehaltenen Stack-Befestigungen; der Stack gegen die vier festgehaltenen Motorsitze.
- Achsen: x quer (rechts +), y längs (vorne +), z hoch.
- Ray-Läufe: ManaFly 3 raysubmit_ZySKb3BcpRqiRmsh, Aether 4 raysubmit_4TVzfwpbqWHLzjY6, SIMP+MMA roh raysubmit_J2FUEKPp8WHipna3, SIMP+MMA 1:1-Rekonstruktion raysubmit_ARBKGa5cq2FBNGRm. Alle vier wiederholen die früheren Läufe in `exports/interface_stiffness/` (Abweichung ≤ 1e-9 relativ, Aether 4 identisch).

## Vergleichbarkeit

| Rahmen | Armlänge (Mitte bis Motor) | Achsabstand | Rahmenmasse |
|---|---|---|---|
| ManaFly 3 BETA V4 | 80.0 mm | 160.1 mm | 29.3 g |
| BM Aether 4 | 95.3 mm | 190.7 mm | 86.2 g |
| SIMP+MMA roh | 66.2 mm | 132.5 mm | 14.6 g |
| SIMP+MMA Rekonstruktion | 66.2 mm | 132.5 mm | 16.4 g |

- Aether 4 ist mit rund 191 mm Achsabstand der größere Rahmen und mit 86 g fast dreimal so schwer wie ManaFly (29 g). Der Kandidat hat 133 mm und 16 g. Aether 4 ist eine massive Plattenkonstruktion und liegt fast überall höher; die schwächere Referenz ist deshalb fast immer ManaFly.
- Skalierung wie bei der bestehenden Armspitzen-Grenze (`scaled.arm_tip_slope`, gleiche Neigung (F/k)/L): Kräfte k_skaliert = k_ref · L_ref / L_Kandidat, also ManaFly × 1,208 und Aether 4 × 1,439. Für den kleineren Kandidaten wird die Grenze damit strenger.
- Momente bleiben unskaliert, denn die Verdrehung unter gleichem Moment ist schon ein Winkel wie die Armneigung. Bei geometrischer Ähnlichkeit mit gleichen Querschnitten würde k_M ~ EI/L ebenfalls mit L_ref/L wachsen; das ist hier bewusst nicht angesetzt.
- Die Tabellen zeigen Verhältnisse unskaliert und in Klammern skaliert (nur bei Kräften verschieden). Entschieden wird auf der Rekonstruktion, unskaliert: Lücke = unter 70 % beider Referenzen. Fälle, die erst skaliert unter 70 % fallen, sind eigens markiert.
- Kein Massenbezug: der Kandidat ist gut halb so schwer wie ManaFly. Für die Bedingung zählt die absolute Steifigkeit.

## Tabellen je Schnittstelle

Rekon./Referenz unskaliert, in Klammern skaliert. roh/beide = Rohfeld durch die schwächere Referenz (unskaliert). Lücke: R = Rekonstruktion unter 70 % beider Referenzen; (s) = nur skaliert; r = nur das Rohfeld.

### motor_front_left

| Richtung | ManaFly | Aether 4 | SIMP+MMA roh | SIMP+MMA Rekon. | Rekon./ManaFly | Rekon./Aether 4 | roh/beide | Lücke |
|---|---|---|---|---|---|---|---|---|
| Fx | 64.89 | 88.82 | 44.48 | 46.07 | 71 % (59 %) | 52 % (36 %) | 69 % | (s) |
| Fy | 54.81 | 62.9 | 3.589 | 3.03 | 6 % (5 %) | 5 % (3 %) | 7 % | R |
| Fz | 6.564 | 12.43 | 8.466 | 8.094 | 123 % (102 %) | 65 % (45 %) | 129 % |  |
| Mx | 2917 | 2.004e+04 | 1503 | 1414 | 48 % | 7 % | 52 % | R |
| My | 4710 | 2.105e+04 | 2953 | 3198 | 68 % | 15 % | 63 % | R |
| Mz | 2.147e+04 | 1.267e+05 | 2307 | 1980 | 9 % | 2 % | 11 % | R |

### motor_front_right

| Richtung | ManaFly | Aether 4 | SIMP+MMA roh | SIMP+MMA Rekon. | Rekon./ManaFly | Rekon./Aether 4 | roh/beide | Lücke |
|---|---|---|---|---|---|---|---|---|
| Fx | 65.05 | 88.73 | 41.88 | 43.71 | 67 % (56 %) | 49 % (34 %) | 64 % | R |
| Fy | 55.9 | 62.99 | 3.346 | 2.854 | 5 % (4 %) | 5 % (3 %) | 6 % | R |
| Fz | 6.762 | 12.52 | 7.936 | 7.826 | 116 % (96 %) | 63 % (43 %) | 117 % |  |
| Mx | 2928 | 2.021e+04 | 1448 | 1390 | 47 % | 7 % | 49 % | R |
| My | 4733 | 2.102e+04 | 2829 | 3306 | 70 % | 16 % | 60 % | R |
| Mz | 2.156e+04 | 1.234e+05 | 2215 | 1941 | 9 % | 2 % | 10 % | R |

### motor_rear_left

| Richtung | ManaFly | Aether 4 | SIMP+MMA roh | SIMP+MMA Rekon. | Rekon./ManaFly | Rekon./Aether 4 | roh/beide | Lücke |
|---|---|---|---|---|---|---|---|---|
| Fx | 31.8 | 125.6 | 32.2 | 32.56 | 102 % (85 %) | 26 % (18 %) | 101 % |  |
| Fy | 14.72 | 135 | 10.33 | 9.412 | 64 % (53 %) | 7 % (5 %) | 70 % | R |
| Fz | 6.295 | 21.79 | 7.431 | 7.527 | 120 % (99 %) | 35 % (24 %) | 118 % |  |
| Mx | 5673 | 3.513e+04 | 2310 | 2305 | 41 % | 7 % | 41 % | R |
| My | 8700 | 2.915e+04 | 2920 | 2755 | 32 % | 9 % | 34 % | R |
| Mz | 1.387e+04 | 6.96e+04 | 5087 | 4454 | 32 % | 6 % | 37 % | R |

### motor_rear_right

| Richtung | ManaFly | Aether 4 | SIMP+MMA roh | SIMP+MMA Rekon. | Rekon./ManaFly | Rekon./Aether 4 | roh/beide | Lücke |
|---|---|---|---|---|---|---|---|---|
| Fx | 31.56 | 125 | 32.62 | 34.45 | 109 % (90 %) | 28 % (19 %) | 103 % |  |
| Fy | 14.5 | 135.4 | 10.41 | 9.846 | 68 % (56 %) | 7 % (5 %) | 72 % | R |
| Fz | 6.094 | 21.55 | 7.874 | 7.324 | 120 % (99 %) | 34 % (24 %) | 129 % |  |
| Mx | 5612 | 3.5e+04 | 2403 | 2321 | 41 % | 7 % | 43 % | R |
| My | 8593 | 2.918e+04 | 2996 | 2770 | 32 % | 9 % | 35 % | R |
| Mz | 1.382e+04 | 6.987e+04 | 5249 | 4508 | 33 % | 6 % | 38 % | R |

### stack

| Richtung | ManaFly | Aether 4 | SIMP+MMA roh | SIMP+MMA Rekon. | Rekon./ManaFly | Rekon./Aether 4 | roh/beide | Lücke |
|---|---|---|---|---|---|---|---|---|
| Fx | 433.7 | 2606 | 308.7 | 351.6 | 81 % (67 %) | 13 % (9 %) | 71 % | (s) |
| Fy | 683.5 | 1874 | 208.4 | 218.8 | 32 % (26 %) | 12 % (8 %) | 30 % | R |
| Fz | 94.32 | 310.4 | 113.3 | 133.7 | 142 % (117 %) | 43 % (30 %) | 120 % |  |
| Mx | 3.434e+04 | 9.273e+04 | 2.137e+04 | 3.834e+04 | 112 % | 41 % | 62 % | r |
| My | 1.898e+04 | 1.131e+05 | 2.439e+04 | 3.385e+04 | 178 % | 30 % | 129 % |  |
| Mz | 4.051e+05 | 1.943e+06 | 1.041e+05 | 1.716e+05 | 42 % | 9 % | 26 % | R |

### battery_rails

| Richtung | ManaFly | Aether 4 | SIMP+MMA roh | SIMP+MMA Rekon. | Rekon./ManaFly | Rekon./Aether 4 | roh/beide | Lücke |
|---|---|---|---|---|---|---|---|---|
| Fx | 49.73 | 123.7 | 50.14 | 61.65 | 124 % (103 %) | 50 % (35 %) | 101 % |  |
| Fy | 88.2 | 132.4 | 89.83 | 97.82 | 111 % (92 %) | 74 % (51 %) | 102 % |  |
| Fz | 406.9 | 246.1 | 229.7 | 213.6 | 52 % (43 %) | 87 % (60 %) | 93 % | (s) |
| Mx | 2.192e+04 | 8273 | 2.149e+04 | 1.481e+04 | 68 % | 179 % | 260 % |  |
| My | 4.391e+04 | 7.455e+04 | 4.949e+04 | 4.366e+04 | 99 % | 59 % | 113 % |  |
| Mz | 1.409e+05 | 1.307e+05 | 4.048e+04 | 4.334e+04 | 31 % | 33 % | 31 % | R |

### camera

| Richtung | ManaFly | Aether 4 | SIMP+MMA roh | SIMP+MMA Rekon. | Rekon./ManaFly | Rekon./Aether 4 | roh/beide | Lücke |
|---|---|---|---|---|---|---|---|---|
| Fx | 27.62 | 46.88 | 88.83 | 101.5 | 368 % (304 %) | 217 % (150 %) | 322 % |  |
| Fy | 60.16 | 42.03 | 113.4 | 123.3 | 205 % (170 %) | 293 % (204 %) | 270 % |  |
| Fz | 42.01 | 41.06 | 92.69 | 112.4 | 268 % (221 %) | 274 % (190 %) | 226 % |  |
| Mx | 2.529e+04 | 3.057e+04 | 2.931e+04 | 4.695e+04 | 186 % | 154 % | 116 % |  |
| My | 1.913e+04 | 5.811e+04 | 2.73e+04 | 3.144e+04 | 164 % | 54 % | 143 % |  |
| Mz | 5.587e+04 | 8.693e+04 | 4.232e+04 | 4.671e+04 | 84 % | 54 % | 76 % |  |

## Vorgeschlagene Bedingungen (nicht eingebaut)

Grenze = schwächere Referenz nach Skalierung (Kräfte × L_ref/L_Kandidat, Momente unskaliert). Abdeckung: ob bestehende oder gerade geplante Bedingungen (Armspitze senkrecht, Armspitze seitlich, Verwindung, Crash-Nachgiebigkeiten, f1) die Richtung schon erfassen.

| # | Schnittstelle | Richtung | Lücke | Rekon. | roh | Grenze | Quelle unskaliert | Rekon./Grenze | Abdeckung |
|---|---|---|---|---|---|---|---|---|---|
| 1 | motor_front_left | Fx | nur skaliert | 46.07 | 44.48 | ≥ 78.4 | ManaFly 64.89 | 59 % | nur Crash Arm (schräg 2:2:1, Grenze 1,5 × Referenz-Nachgiebigkeit, locker); Armspitze seitlich (geplant) wirkt quer zum Arm, in x nur mit Anteil 0,59 |
| 2 | motor_front_left | Fy | R, roh | 3.03 | 3.589 | ≥ 66.23 | ManaFly 54.81 | 5 % | teilweise: Armspitze seitlich (geplant; quer zum Arm = (0,59; 0,81), also überwiegend Fy) + Crash Arm |
| 3 | motor_front_left | Mx | R, roh | 1414 | 1503 | ≥ 2917 | ManaFly 2917 | 48 % | indirekt: Armspitze senkrecht und Verwindung (geplant) begrenzen die Armbiegung, nicht die Kippsteifigkeit des Sitzes |
| 4 | motor_front_left | My | R, roh | 3198 | 2953 | ≥ 4710 | ManaFly 4710 | 68 % | indirekt: Armspitze senkrecht und Verwindung (geplant), wie Mx |
| 5 | motor_front_left | Mz | R, roh | 1980 | 2307 | ≥ 2.147e+04 | ManaFly 2.147e+04 | 9 % | indirekt: Armspitze seitlich (geplant) begrenzt die Biegung in der Ebene, nicht die Drehung des Sitzes |
| 6 | motor_front_right | Fx | R, roh | 43.71 | 41.88 | ≥ 78.59 | ManaFly 65.05 | 56 % | nur Crash Arm (schräg 2:2:1, Grenze 1,5 × Referenz-Nachgiebigkeit, locker); Armspitze seitlich (geplant) wirkt quer zum Arm, in x nur mit Anteil 0,59 |
| 7 | motor_front_right | Fy | R, roh | 2.854 | 3.346 | ≥ 67.54 | ManaFly 55.9 | 4 % | teilweise: Armspitze seitlich (geplant; quer zum Arm = (0,59; 0,81), also überwiegend Fy) + Crash Arm |
| 8 | motor_front_right | Mx | R, roh | 1390 | 1448 | ≥ 2928 | ManaFly 2928 | 47 % | indirekt: Armspitze senkrecht und Verwindung (geplant) begrenzen die Armbiegung, nicht die Kippsteifigkeit des Sitzes |
| 9 | motor_front_right | My | R, roh | 3306 | 2829 | ≥ 4733 | ManaFly 4733 | 70 % | indirekt: Armspitze senkrecht und Verwindung (geplant), wie Mx |
| 10 | motor_front_right | Mz | R, roh | 1941 | 2215 | ≥ 2.156e+04 | ManaFly 2.156e+04 | 9 % | indirekt: Armspitze seitlich (geplant) begrenzt die Biegung in der Ebene, nicht die Drehung des Sitzes |
| 11 | motor_rear_left | Fy | R | 9.412 | 10.33 | ≥ 17.78 | ManaFly 14.72 | 53 % | teilweise: Armspitze seitlich (geplant; quer zum Arm = (0,59; 0,81), also überwiegend Fy) + Crash Arm |
| 12 | motor_rear_left | Mx | R, roh | 2305 | 2310 | ≥ 5673 | ManaFly 5673 | 41 % | indirekt: Armspitze senkrecht und Verwindung (geplant) begrenzen die Armbiegung, nicht die Kippsteifigkeit des Sitzes |
| 13 | motor_rear_left | My | R, roh | 2755 | 2920 | ≥ 8700 | ManaFly 8700 | 32 % | indirekt: Armspitze senkrecht und Verwindung (geplant), wie Mx |
| 14 | motor_rear_left | Mz | R, roh | 4454 | 5087 | ≥ 1.387e+04 | ManaFly 1.387e+04 | 32 % | indirekt: Armspitze seitlich (geplant) begrenzt die Biegung in der Ebene, nicht die Drehung des Sitzes |
| 15 | motor_rear_right | Fy | R | 9.846 | 10.41 | ≥ 17.51 | ManaFly 14.5 | 56 % | teilweise: Armspitze seitlich (geplant; quer zum Arm = (0,59; 0,81), also überwiegend Fy) + Crash Arm |
| 16 | motor_rear_right | Mx | R, roh | 2321 | 2403 | ≥ 5612 | ManaFly 5612 | 41 % | indirekt: Armspitze senkrecht und Verwindung (geplant) begrenzen die Armbiegung, nicht die Kippsteifigkeit des Sitzes |
| 17 | motor_rear_right | My | R, roh | 2770 | 2996 | ≥ 8593 | ManaFly 8593 | 32 % | indirekt: Armspitze senkrecht und Verwindung (geplant), wie Mx |
| 18 | motor_rear_right | Mz | R, roh | 4508 | 5249 | ≥ 1.382e+04 | ManaFly 1.382e+04 | 33 % | indirekt: Armspitze seitlich (geplant) begrenzt die Biegung in der Ebene, nicht die Drehung des Sitzes |
| 19 | stack | Fx | nur skaliert | 351.6 | 308.7 | ≥ 524 | ManaFly 433.7 | 67 % | indirekt: Crash seitlich (anderer Lastangriff) |
| 20 | stack | Fy | R, roh | 218.8 | 208.4 | ≥ 825.8 | ManaFly 683.5 | 26 % | indirekt: Crash front/hinten greifen an Kamera bzw. Akkudeck an, nicht am Stack; f1 hat keine Masse am Stack |
| 21 | stack | Mx | nur roh | 3.834e+04 | 2.137e+04 | ≥ 3.434e+04 | ManaFly 3.434e+04 | 112 % | teilweise: Verwindung (geplant), über die Motorsitze gekoppelt |
| 22 | stack | Mz | R, roh | 1.716e+05 | 1.041e+05 | ≥ 4.051e+05 | ManaFly 4.051e+05 | 42 % | nein: Gier-Torsion (torsion_yaw) wird nur überwacht, nicht begrenzt |
| 23 | battery_rails | Fz | nur skaliert | 213.6 | 229.7 | ≥ 354.1 | Aether 4 246.1 | 60 % | Crash- und f1-Fälle |
| 24 | battery_rails | Mz | R, roh | 4.334e+04 | 4.048e+04 | ≥ 1.307e+05 | Aether 4 1.307e+05 | 33 % | nein: torsion_yaw nur Monitor; Crash hinten belastet das Akkudeck längs |

Zählung: 20 Richtungen sind auf der Rekonstruktion unskaliert unter 70 % beider Referenzen (R). Dazu kommen 3 nur skaliert (motor_front_left Fx 59 %, stack Fx 67 %, battery_rails Fz 60 %) und 1 nur im Rohfeld (stack Mx, Rekonstruktion 112 % von ManaFly); diese 4 sind Hinweise, keine Vorschläge. Ausnahme: motor_front_left Fx sollte wegen der Spiegelsymmetrie zusammen mit motor_front_right Fx (R, 67 %) behandelt werden.

## Einordnung

- Größte Lücken: Motorsitze vorne Fy (5 % von ManaFly) und Mz (9 %); Motorsitze hinten Mx, My, Mz (32–41 %) und Fy (64–68 %); Stack Fy (32 %) und Mz (42 %); Akkuschienen Mz (31 % von Aether 4). Kamera, Motorsitze Fz und Akkuschienen Fx, Fy, Mx, My liegen auf oder über mindestens einer Referenz.
- Motor Fy ist die Längsrichtung, beim Armwinkel 36° gegen x also überwiegend quer zum Arm. Die geplante Bedingung Armspitze seitlich greift genau dort und dürfte die Fy-Lücke vorne großteils schließen, wenn ihre Grenze aus denselben Referenzen kommt. Motor Fx ist nur über den Crash-Fall Arm erfasst.
- Die Verdrehung der Motorsitze (Mx, My, Mz an allen vier Sitzen, 12 der 20 Lücken) erfasst keine Bedingung direkt. Armspitze senkrecht, seitlich und Verwindung begrenzen Verschiebungen, nicht die Verdrehung des Sitzes. Ob die neuen Bedingungen sie mitziehen, zeigt erst die Nachmessung.
- Gier-Torsion (Stack Mz, Akkuschienen Mz) ist nicht begrenzt; `torsion_yaw` läuft nur als Monitor. Naheliegender Kandidat für eine harte Bedingung: eine Bedingung auf die Gier-Torsion deckt beide Zeilen ab.
- Stack Fy: kein Lastfall greift am Stack in Längsrichtung an.
- Vorschlag zum Vorgehen: zuerst den laufenden SIMP+MMA-Lauf mit Verwindung und Armspitze seitlich neu messen. Nur was danach noch unter 70 % beider Referenzen liegt, als Bedingung mit der Grenze aus dieser Tabelle einbauen. Vorrang hätten dann Gier-Torsion (eine Bedingung für Stack Mz und Akkuschienen Mz) und die Kippsteifigkeit der Motorsitze.

## Dateien

- Messwerte: `exports/gap/{manafly3,aether4,simp_mma_raw,simp_mma_recon}/interface_stiffness.json` und `.md`, CalculiX-Dateien in `ccx/`.
- Markierte Richtungen maschinenlesbar: `exports/gap/gaps_flags.json`.
- Frühere Vergleichstabelle: `exports/interface_stiffness/comparison/interface_gaps.md`.
