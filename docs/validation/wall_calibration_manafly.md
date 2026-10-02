# Kalibrierung der Wandprüfung an ManaFly3

Ziel: Unsere 2-mm-Wandprüfung an einem realen Frame kalibrieren, der fliegt und Crashs übersteht (ManaFly 3" BETA V4). Gemessen wurde mit zwei Methoden, auf drei Netzen und mit identischen Einstellungen.

**Status:** Umgesetzt als blockierende Prüfung (Commit 9d464e6). **VORLÄUFIG:** kalibriert an der Herstellergeometrie von ManaFly (BETA V4, kein eigener Test); Aether4 als zweiter Datenpunkt (ergänzt, Abschnitt unten); endgültig nach eigenen Falltests in M2.

## Eingaben

| Netz | Datei | sha256 (12) | Dreiecke | Volumen mm³ |
|---|---|---|---|---|
| ManaFly3 | `Deep_Frame-neural/exports/manafly_ref/manafly3_repaired.stl` (Reparatur: 0,2-mm-Voxel, MC, dezimiert, um 180° um z gedreht, z ab 0; Stand 10:46) | 3d39cdc16bc8 | 184 156 | 26 900 |
| gpu708 t025 c00 | `exports/implicit/poc/gpu708_t025/candidates/c00/geometry.stl` | 59b1fcb1a07c | 146 102 | 51 904 |
| fine c01 c00 | `exports/implicit/poc/fine_c01/candidates/c00/geometry.stl` | e7d9ce76f346 | 136 966 | 41 409 |

Grenzen der Referenz:
- Die Wandstärke ist durch die Reparatur auf etwa ±0,1–0,2 mm quantisiert. Die 2,0-mm-Grundplatte erscheint repariert mit 1,98 mm; Strahlprobe bei (−12,4; −45,3): Quelle 2,00 mm, repariert 1,98 mm.
- Abstand Quelle → Reparatur: p99 0,97 mm, max 2,7 mm. Strukturen unter etwa 0,5 mm sind also nicht aufgelöst.
- Die Netzprüfung der Pipeline meldete Selbstschnitte im reparierten Netz (Stand 10:38). Das betrifft nur einzelne Strahlproben (< 0,01 mm).
- Eine frühere Reparatur (sha 19957a00c381, 400k Dreiecke) ergab: A 8,16 % dünn (< 1,8 mm: 4,65 %), B tief 5,6 %, größte tiefe Komponente 397 mm³. Das Bild ist dasselbe, die Regel unten gilt für beide Stände.

## Methoden (Einstellungen)

- **A – Strahlprüfung:** `wall_screen` aus `deep_frame/topology_implicit_validation.py` (Stand 5c65f21).
  - Pipeline-Einstellungen: `_validation_settings` mit Probenabstand 1,0 mm, Toleranz 1e-5 und Knickwinkel `remesh_feature_deg` = 60°. Minimum 2,0 mm.
  - **Keine Bohrungszugabe** (`bores=()`), für alle drei Netze gleich. ManaFly hat keine vorgeschriebenen Bohrungen im Sinne unserer Pipeline; seine Montagelöcher fallen unter „Motoraufnahme“ bzw. „Zentralteil“.
  - Das Probenbudget wurde auf 20 M angehoben (nur ein Budget, kein Kriterium).
  - Die Verteilung stammt aus einem zweiten, identischen Strahllauf mit Obergrenze 8 mm. Seine Zahl dünner Proben stimmt in allen drei Fällen exakt mit `wall_screen` überein.
- **B – Morphologische Öffnung:** Kugel mit r = 1,0 mm.
  - Voxelgitter h = 0,1 mm, Belegung per z-Windungszahl. Öffnung exakt über EDT (Erosion: EDT ≥ r, Dilatation: EDT ≤ r), gekachelt in x mit 22 Voxeln Halo.
  - Entfernt = Teil minus Öffnung. Komponenten mit 26er-Nachbarschaft.
  - Effektive Schwelle ≈ 1,9–2,0 mm. Analytisch geprüft: Platte 1,85 mm wird vollständig entfernt, Platten von 1,95/2,0/2,1 mm bleiben bis auf Kantenrundung erhalten.
  - Die Öffnung rundet auch jede konvexe scharfe Kante ab, die Entfernungstiefe ist dort ≤ 0,41 mm, an Ecken ≤ 0,73 mm.
  - Deshalb gibt es zusätzlich **„tief entfernt“**: Komponenten, deren maximaler Abstand zur Oberfläche ≥ 0,45 mm ist. Das sind echte Wände < 2 mm, kein Kantenabtrag.
  - Analytisch geprüft: Platte 1,7 mm → 1 tiefe Komponente; Platte 2,0 mm → 51 flache Kantenkomponenten, 0 tiefe.
- Ortsklassen sind grob.
  - Unsere Frames: Preserve-Boxen ±2 mm aus der Domäne.
  - ManaFly: Motoraufnahme = innerhalb 13 mm um (±57,5; ±57) in xy; Zentralteil = |x| < 22, |y| < 36, z < 18; Oberrahmen = z ≥ 18; sonst Strebe/Arm.
  - „Grundebene“ = z < 0,05 mm.
- Laufzeit: A 9–13 s, B 2–3 min pro Netz (CPU-Slot, 8 Threads).

## Ergebnisse

| Kennzahl | ManaFly3 | gpu708 t025 | fine c01 |
|---|---|---|---|
| **A** Strahlproben | 184 342 | 289 152 | 275 346 |
| **A** dünne Proben < 2,0 mm | **20 542** | 762 | 729 |
| **A** Anteil dünn | **11,1 %** | 0,26 % | 0,26 % |
| **A** Anteil < 1,8 / < 1,5 / < 1,0 mm | 5,27 / 3,74 / 1,67 % | 0,009 / 0,006 / 0,003 % | 0,004 / 0,003 / 0,001 % |
| **A** min gemessen | 0,004 mm (Reparatur-Artefakt); echt ab ≈ 0,25 mm | 2,4e-5 mm (Grundebene z = 0, Artefakt) | 0,19 mm (Kante am XT30-Keep-out) |
| **A** p1 / p5 aller Proben | 0,84 / 1,78 mm | 2,16 / 2,50 mm | 2,16 / 2,26 mm |
| **A** p1 / p5 / p50 der dünnen Proben | 0,72 / 0,80 / 1,80 mm | 0,42 / 1,83 / 1,99 mm | 1,50 / 1,92 / 1,99 mm |
| **A** Ort der dünnen Proben | Zentralteil 8 746, Strebe/Arm 5 101, Motoraufnahme 3 981, Oberrahmen 1 469, Grundebene 1 245 | Motoraufnahme 710 (alle 1,73–2,0 mm, Steg zwischen Schraub- und Wellenbohrung), Akku/Strap 26, Kamera 15, Strebe 7, sonstige 4 | Motoraufnahme 711 (1,66–2,0 mm, Bohrungsstege), Akku/Strap 16, Stecker 2 |
| **B** entfernt gesamt | **1 558 mm³ = 5,79 %** | 256 mm³ = 0,49 % | 284 mm³ = 0,69 % |
| **B** davon Kantenrundung (flach) | 195 mm³ (0,72 %) | 256 mm³ (0,49 %) | 280 mm³ (0,68 %) |
| **B** tief entfernt (echte Wand < 2 mm) | **1 363 mm³ = 5,06 %**, 44 Komp. | **0** | 4,2 mm³ = 0,010 %, 2 Komp. |
| **B** größte tiefe Komponente | **346 mm³** (Zentralteil: Gitter der Grundplatte, ≈ 70 × 42 × 3 mm) | – | 3,1 mm³ (Akku/Strap-Auflage) |
| **B** Komponenten gesamt / > 1 mm³ | 26 472 / 76 | 16 853 / 78 | 17 975 / 86 |
| **B** tief entfernt nach Ort | Zentralteil 574, Strebe/Arm 340, Motoraufnahme 261, Oberrahmen 188 mm³ | – | Akku/Strap 4,2 mm³ |

Zum Vergleich protokolliert die Pipeline mit Bohrungszugabe 338 (gpu708) bzw. 305 (fine c01) dünne Proben. Die Differenz von 424 Proben liegt an den Motor-Bohrungsstegen.

## Wie ein Frame aussieht, der fliegt und Crashs übersteht

- **ManaFly verletzt eine strikte 2-mm-Regel massiv.**
  - Methode A: 11 % der Proben liegen unter 2 mm, 3,7 % unter 1,5 mm, 1,7 % unter 1,0 mm.
  - Methode B: 5,1 % des Volumens sind echte Wände unter 2 mm.
  - Rund die Hälfte der dünnen A-Proben liegt bei 1,8–2,0 mm, also in der Quantisierungszone der Reparatur. Belastbar sind die Anteile unter 1,8 mm (5,3 %) und B.
  - Die Grundplatte ist nominell genau 2,0 mm dick; ihre Gitterstege sind in der Ebene schmaler.
  - Die Motorsitze haben Lippen von 1,0–1,4 mm (z 8,0 → 9,0/9,2).
  - Die Rails und Bügel oben haben Rippen bis herab auf ≈ 0,25 mm.
  - Steifigkeit und Crashfestigkeit kommen aus der Höhe der Struktur: Bügel, Rails und ein 32 mm hoher Rahmen. Lokale Wandstärke allein erklärt sie nicht.
- **Unsere Frames liegen um mehr als eine Größenordnung darunter.**
  - Methode A: 0,26 % dünne Proben, fast alle zwischen 1,66 und 2,0 mm an den Bohrungsstegen der Motoraufnahme.
  - Methode B: 0 bzw. 0,01 % tief entfernt.
  - Einzelne Proben unter 0,5 mm (Grundebene z = 0, Schnittkanten an Keep-out-Ebenen x = ±15,5 mm) sind laut B keine Wände, sondern Keile bzw. tangentiale Artefakte.
  - Methode A allein kann diese Fälle nicht von echten dünnen Wänden trennen; Methode B kann es.
- **Kantenrundung (flacher Abtrag von B) liegt bei allen drei bei 0,5–0,9 %.** Das ist kein Wandmaß und darf nicht in eine Regel eingehen.

## Kalibrierte Regel (umgesetzt, vorläufig)

Die 2-mm-Mindestwand bleibt als Auslegungsvorgabe unverändert, und die Feld-Öffnung (r = 1,35 mm) bleibt Pflicht. Die Kalibrierung zeigt: Ein flugfähiger, crashfester Frame liegt weit über allen Werten unserer Frames. Die Regel ist deshalb von ManaFly abgeleitet, mit Sicherheitsfaktor ≈ 10 (mindestens 5, falls ein Teil von ManaFlys tiefem Volumen nur Quantisierung um 2,0 mm ist). Sie bewertet das Endnetz, nicht einzelne Strahlproben:

1. **Primär, blockierend (Methode B, h ≤ 0,1 mm, r = 1,0 mm):**
   - tief entferntes Volumen (Tiefe ≥ 0,45 mm) ≤ **0,5 %** des Teilvolumens (ManaFly 5,1 %);
   - keine einzelne tiefe Komponente > **5 mm³** (ManaFly 346 mm³; das ist etwa ein 2,5 × 2 × 1 mm großer Steg);
   - in Motor-Preserve-Zonen gar keine tiefe Komponente.
2. **Sekundär, Warnung (Methode A, ohne Bohrungszugabe):**
   - Anteil dünner Proben < 2,0 mm ≤ **1 %** (ManaFly 11 %, unter 1,8 mm 5,3 %);
   - Anteil < 1,5 mm ≤ **0,05 %** (ManaFly 3,7 %);
   - Proben unter 0,1 mm und Proben in der Grundebene zählen als Artefakt, wenn B dort keine tiefe Komponente findet.
   - `wall_screen_blocking` bleibt `False`.

Umsetzung (Commit 9d464e6): `wall_rule` in `deep_frame/topology_implicit_validation.py`, aufgerufen von `MeshAcceptance.features`. Bericht unter `checks.features.wall_opening` (blockierend) und `checks.features.wall_warning` (Warnung). Werte in `IMPLICIT_CONFIG`:

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `wall_voxel_mm` | 0,1 | Voxelgröße h (Gate, darf nur kleiner werden) |
| `wall_opening_radius_mm` | 1,0 | Öffnungsradius r (Gate, darf nur größer werden) |
| `wall_deep_mm` | 0,45 | Tiefe, ab der eine Komponente „tief“ ist (Gate, nur kleiner) |
| `wall_deep_max_fraction` | 0,005 | tiefes Volumen / Teilvolumen (Gate, nur kleiner) |
| `wall_deep_component_max_mm3` | 5,0 | größte tiefe Komponente (Gate, nur kleiner) |
| `wall_motor_zone_margin_mm` | 2,0 | Motorzone = Preserve-Regionen mit „motor“ im Namen, AABB ± Rand (Gate, nur größer) |
| `wall_tile_voxels` | 120 | Kachelbreite in x (nur Speicher) |
| `wall_thin_max_fraction` | 0,01 | Warnung: Anteil A < Minimum |
| `wall_very_thin_mm` / `wall_very_thin_max_fraction` | 1,5 / 0,0005 | Warnung: Anteil A < 1,5 mm |

Die Warnanteile werden vor jeder Bohrungszugabe gezählt. Die Artefaktausnahme (Proben < 0,1 mm, Grundebene) ist nicht umgesetzt; sie betrifft nur die Warnung. Spalten ohne Abschluss in z (Windungszahl ≠ 0) und ein leeres Teil lassen die Prüfung scheitern. Bericht: tiefes Volumen und Anteil, tiefe Komponenten, größte, Motortreffer, flacher Abtrag (Kantenrundung), Laufzeit.

Nachprüfung aller PoC-Frames mit der eingebauten Prüfung (`exports/implicit/poc/*/candidates/c00/geometry.stl`, Regionen aus `density_source/inputs.json`, je ≈ 2 min CPU): alle bestehen den blockierenden Teil, keine Motortreffer. wb142 verfehlt nur die Warnung (A < 1,5 mm 0,069 % > 0,05 %).

| Frame | B tief | größte tiefe Komponente | Motortreffer | flach mm³ | A < 2,0 mm | A < 1,5 mm | blockierend |
|---|---|---|---|---|---|---|---|
| bat_large | 0 % | – | 0 | 259 | 0,26 % | 0,003 % | bestanden |
| bat_small | 0 % | – | 0 | 254 | 0,26 % | 0,002 % | bestanden |
| combo_compact | 0 % | – | 0 | 249 | 0,27 % | 0,003 % | bestanden |
| combo_heavy | 0 % | – | 0 | 256 | 0,28 % | 0,031 % | bestanden |
| fine_c01 | 0,010 % (2 Komp.) | 3,1 mm³ | 0 | 280 | 0,27 % | 0,003 % | bestanden |
| gpu708_t025 | 0 % | – | 0 | 256 | 0,26 % | 0,006 % | bestanden |
| load_arm2 | 0 % | – | 0 | 255 | 0,27 % | 0,006 % | bestanden |
| load_impact2 | 0 % | – | 0 | 254 | 0,26 % | 0,002 % | bestanden |
| wb131 | 0 % | – | 0 | 251 | 0,27 % | 0,003 % | bestanden |
| wb142 | 0 % | – | 0 | 255 | 0,33 % | **0,069 %** | bestanden (Warnung) |

Ursprüngliche Kalibrierung (Skript im Scratchpad):

| Frame | B tief entfernt | größte tiefe Komponente | A < 2,0 mm | A < 1,5 mm |
|---|---|---|---|---|
| gpu708 | 0 % | – | 0,26 % | 0,006 % |
| fine c01 | 0,010 % | 3,1 mm³ | 0,26 % | 0,003 % |

ManaFly verfehlt sie bewusst, denn die Regel enthält den Sicherheitsfaktor. B als Gate kostet ≈ 2 min CPU pro Kandidat bei h = 0,1 mm.

## Zweiter Datenpunkt: BM Aether4

Zweiter realer Frame, der fliegt (BM Aether4, Quelle `Examples/Frames/BM+Aether4`). Anders als ManaFly ist er ein direkter STL-Export ohne Voxel-Reparatur; der Quantisierungsvorbehalt entfällt.

Eingaben:

| Netz | Datei | sha256 (12) | Dreiecke | Volumen mm³ |
|---|---|---|---|---|
| Aether4 body12 (12-mm-Motorlochbild) | `Deep_Frame-neural/exports/aether4_ref/aether4_body12.stl` (orientiert, z ab 0, 164 × 165 × 76 mm) | d5ced8a6130f | 455 622 | 79 054 |
| Aether4 body9 (9-mm-Motorlochbild) | `Deep_Frame-neural/exports/aether4_ref/aether4_body9.stl` | b4f8bf39e2d5 | 458 724 | 80 847 |

- Beide Netze sind geschlossen, orientiert und ein Körper; die Netzprüfung der Pipeline (Topologie, Selbstschnitte) besteht.
- Der Export enthält Splitterdreiecke (minimaler Dreieckswinkel 0,4°).
- Methoden und Einstellungen wie oben (A ohne Bohrungszugabe; B mit h = 0,1 mm, r = 1,0 mm, tief ≥ 0,45 mm). B ist hier in x und y gekachelt (100 Voxel, Halo 24 Voxel), sonst identisch.
- Ortsklassen:
  - Motoraufnahme = innerhalb 13 mm um (±69,6; 64,78) und (±70,3; −64,78);
  - Zentralteil = |x| < 22, |y + 16,35| < 26, z < 15;
  - Oberrahmen/Kamerawiege = y > 30, z ≥ 45;
  - Grundebene = z < 0,05 mm; sonst Strebe/Arm.

| Kennzahl | Aether4 body12 | Aether4 body9 | ManaFly3 (oben) |
|---|---|---|---|
| **A** Strahlproben | 1 043 840 | 1 075 564 | 184 342 |
| **A** dünne Proben < 2,0 mm | **201 716 = 19,3 %** | 184 376 = 17,1 % | 11,1 % |
| **A** Anteil < 1,8 / < 1,5 / < 1,0 mm | 16,0 / **12,2** / 5,86 % | 13,5 / 9,73 / 4,46 % | 5,27 / 3,74 / 1,67 % |
| **A** Anteil < 0,1 mm | 499 Proben = 0,05 % (Splitter/Berührflächen des Exports) | 0,05 % | – |
| **A** min gemessen | 0,0011 mm bei (30,4; 21,2; 51,0), Strebe/Arm | 0,0011 mm, gleiche Stelle | 0,004 mm |
| **A** p1 / p5 aller Proben | 0,36 / 0,86 mm | 0,46 / 1,11 mm | 0,84 / 1,78 mm |
| **A** p1 / p5 / p50 der dünnen Proben | 0,25 / 0,35 / 1,37 mm | 0,29 / 0,43 / 1,44 mm | 0,72 / 0,80 / 1,80 mm |
| **A** Ort der dünnen Proben | Zentralteil 87 425, Oberrahmen 42 546, Strebe/Arm 41 677, Motoraufnahme 30 068, Grundebene 0 | Zentralteil 87 427, Oberrahmen 42 708, Strebe/Arm 40 891, Motoraufnahme 13 350, Grundebene 0 | siehe oben |
| **B** entfernt gesamt | 3 487 mm³ = 4,41 % | 3 398 mm³ = 4,20 % | 1 558 mm³ = 5,79 % |
| **B** davon Kantenrundung (flach) | 120 mm³ (0,15 %) | 168 mm³ (0,21 %) | 195 mm³ (0,72 %) |
| **B** tief entfernt (echte Wand < 2 mm) | **3 367 mm³ = 4,26 %**, 28 Komp. | **3 229 mm³ = 3,99 %**, 27 Komp. | 1 363 mm³ = 5,06 %, 44 Komp. |
| **B** größte tiefe Komponente | **1 333 mm³** (Zentralteil: Bodenplatte unter dem Stack, 40 × 54 × 7,5 mm, max. Tiefe 0,91 mm) | **1 333 mm³** (dieselbe Bodenplatte) | 346 mm³ |
| **B** zweitgrößte | 1 140 mm³ (Mittelsteg vorn bei x = 0, y ≈ 33, z ≈ 31; 33 × 42 × 25 mm, Tiefe 0,99 mm) | 1 009 mm³ (derselbe Mittelsteg vorn) | – |
| **B** Komponenten gesamt / > 1 mm³ | 19 602 / 52 | 20 389 / 84 | 26 472 / 76 |
| **B** tief entfernt nach Ort | Zentralteil 1 362, Strebe/Arm 1 324, Motoraufnahme 361, Oberrahmen 320 mm³ | Zentralteil 1 362, Strebe/Arm 1 188, Oberrahmen 444, Motoraufnahme 235 mm³ | Zentralteil 574, Strebe/Arm 340, Motoraufnahme 261, Oberrahmen 188 mm³ |
| **B** tiefe Komponenten in Motorzonen | alle vier Motoraufnahmen, je 46–69 mm³ (Tiefe 0,64–0,92 mm) | ja: vorn je 63 mm³ (Tiefe 0,91) und 18 mm³ (Tiefe 0,8); Motorzonen gesamt 235 mm³ | ja |

Eine einfachere Öffnung (Erosion EDT > r, Kachel 150 Voxel) ergab für body12 konsistent 3 720 mm³ = 4,71 % entfernt, größte Komponente 1 411 mm³. Laufzeit B: 8,7 min bei 455k Dreiecken (CPU-Slot).

Was Aether4 über die vorläufige Regel sagt:
- **Aether4 verfehlt alle drei blockierenden Kriterien deutlich, wie ManaFly:**
  - tief entfernt 4,26 % gegen Grenze 0,5 % (Faktor 8,5);
  - größte tiefe Komponente 1 333 mm³ gegen 5 mm³;
  - tiefe Komponenten in allen vier Motorzonen.
  - Warnschwellen: A < 2,0 mm 19,3 % gegen 1 %, A < 1,5 mm 12,2 % gegen 0,05 %.
- **ManaFly ist kein Ausreißer.** Zwei unabhängig konstruierte, fliegende Frames haben 4–5 % ihres Volumens in echten Wänden unter 2 mm (B tief: Aether4 4,26 % bzw. 3,99 % für body12/body9, ManaFly 5,06 %). Der Sicherheitsfaktor der Regel liegt gegenüber Aether4 bei ≈ 8,5, gegenüber ManaFly bei ≈ 10. Die Regel bleibt damit eine Regel mit Sicherheitsfaktor; sie bildet nicht die Grenze dessen ab, was fliegt.
- **Bei Aether4 stammt das tiefe Volumen überwiegend aus Platten knapp unter 2 mm.**
  - Die maximale Entfernungstiefe der großen Komponenten liegt bei 0,91–0,99 mm, die Plattendicke also bei ≈ 1,8–2,0 mm.
  - Die Motorsitze haben eine Tiefe von 0,64 mm, also Wände von ≈ 1,3 mm.
  - Das tiefe Kriterium zählt eine solche Platte vollständig. Das Kriterium „größte tiefe Komponente ≤ 5 mm³“ greift deshalb schon bei einer flächigen 1,9-mm-Platte. Für unsere Frames ist das gewollt, denn die 2 mm bleiben Auslegungsvorgabe; als Maß für die Schwere einer Wandverletzung taugt der Wert aber nicht.
- **A ist bei Aether4 noch weniger aussagekräftig als bei ManaFly.** 19,3 % dünne Proben gehen bis auf 0,001 mm herab; 0,05 % liegen unter 0,1 mm und stammen von Splitterdreiecken und Berührflächen des Exports. Das bestätigt, dass A Warnung bleibt und B entscheidet.
- **Folgerung:** Die Schwellen bleiben unverändert. Unsere Frames (B tief 0 bzw. 0,010 %) liegen mehr als zwei Größenordnungen unter beiden realen Frames.

Reproduktion Aether4: `Deep_Frame-neural/exports/aether4_ref/wall.py` (A: `rays`, einfache Öffnung: `opening`), `wall_cal.py` + `cal_b.py` (A-Verteilung und B mit tiefer Klassifikation). Rohwerte stehen in `wall_rays_body12.json`, `wall_rays_body9.json`, `wall_opening_body12.json` `wall_cal_body12.json` und `wall_cal_body9.json` im selben Ordner (nicht eingecheckt).

## Reproduktion

Gerechnet wurde mit dem Skript `wall_calibration.py` im Scratchpad dieser Sitzung (nicht eingecheckt). Es verwendet `wall_screen`, `corner_normals`, `segment_hits` und `_triangle_samples` unverändert. Die Einstellungen stehen oben; die Rohwerte wurden als `wall_calibration.json` (alle drei Netze, ManaFly-Stand 19957a00c381) und `v2/wall_calibration.json` (ManaFly-Stand 3d39cdc16bc8) im selben Scratchpad abgelegt.
