# Kalibrierung der Wandprüfung an ManaFly3

Ziel: Unsere 2-mm-Wandprüfung an einem realen Frame kalibrieren, der fliegt und Crashs übersteht (ManaFly 3" BETA V4). Gemessen wurde mit zwei Methoden, auf drei Netzen und mit identischen Einstellungen. Status: Proof of Concept, 2026-10-02.

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

## Empfohlene kalibrierte Regel (Vorschlag, nicht umgesetzt)

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

Beide aktuellen Frames erfüllen diese Regel:

| Frame | B tief entfernt | größte tiefe Komponente | A < 2,0 mm | A < 1,5 mm |
|---|---|---|---|---|
| gpu708 | 0 % | – | 0,26 % | 0,006 % |
| fine c01 | 0,010 % | 3,1 mm³ | 0,26 % | 0,003 % |

ManaFly verfehlt sie bewusst, denn die Regel enthält den Sicherheitsfaktor. Wenn die Regel eingeführt wird, ist B als Gate günstig: 2–3 min CPU pro Kandidat bei h = 0,1 mm.

## Reproduktion

Gerechnet wurde mit dem Skript `wall_calibration.py` im Scratchpad dieser Sitzung (nicht eingecheckt). Es verwendet `wall_screen`, `corner_normals`, `segment_hits` und `_triangle_samples` unverändert. Die Einstellungen stehen oben; die Rohwerte wurden als `wall_calibration.json` (alle drei Netze, ManaFly-Stand 19957a00c381) und `v2/wall_calibration.json` (ManaFly-Stand 3d39cdc16bc8) im selben Scratchpad abgelegt.
