# Implizite Route vs. CAD-Route vs. Raster-Referenz (gpu708 t025)

Stand: 2026-10-01, Code c407b01 (feature/implicit-integration: Reopening nach Ripple-Glättung cb09893, FEA-Vernetzungskette f8a45ff, Retry bei CalculiX-Crash 701efff, protected_opening als Option, Standard aus). Alle Werte aus den Records auf der Platte (`exports/implicit/**`, `C:/clones/Deep_Frame/exports/topology/**`), zusammengefasst in `implicit_vs_cad_gpu708_t025.json`. Läufe mit Treiber-Standardeinstellungen (`IMPLICIT_CONFIG`, keine FEA-Overrides), nur Schwelle/Extension und für fein subdivisions 5 / Schwelle 0.5 gesetzt (Plan Abschnitt 11).

Spalten:
- **(a)** implizit, gespeicherte GPU-Dichte gpu708 (8/3 mm, 708 Updates), t 0.25, extension preserve: `exports/implicit/gpu708_t025/candidates/c00`.
- **(b)** implizit, robuste Feinrechnung 4/3 mm (R 3.4 mm, beta bis 16), t 0.5: c00 preserve, c01 preserve_forbidden: `exports/implicit/fine_robust`.
- **(c)** CAD-Route (Dichte -> Netz -> B-rep -> OCCT), nur Status, nicht repariert oder neu gerechnet.
- **(d)** FEA-geprüfter Rasterrahmen grid4_iter300 density_t01 (50.95 g).

## Vergleich

| | (a) implizit gpu708 t025 | (b) implizit fein c00 | (b) implizit fein c01 | (c) CAD-Route | (d) Raster t01 |
|---|---|---|---|---|---|
| Bestanden | **nein** (geometry_invalid) | **nein** (field_disconnected) | **nein** (geometry_invalid) | **nein** | **ja** |
| Fehlschlagende Checks | nur features (2-mm-Wand): 52 dünne Samples, min 0.00046 mm | Feldstufe: abgelöster Körper 38.4 mm³ ohne Mount im final_witness⁷; nicht extrahiert | nur features: 40, min 0.031 mm | Hauptlauf ef_buffer050_preunion: reconstruction/decimation (8 degenerierte Dreiecke). Weitester Lauf envelope_bounds: validation/topology `closed_manifold_single_solid` (Tessellation nicht wasserdicht); Rest nicht ausgewertet, keine FEA | keine (Wand min 2.000 mm, 0 dünn) |
| Übrige 10 Checks + Massen-Screen | bestanden⁵ | nicht ausgewertet | bestanden⁵ | – | bestanden |
| deviation gegated (verarbeitetes Feld)⁵ | 0.070 mm | – | 0.094 mm | – | – |
| Bohrungsstege mit Polygon-Toleranz⁶ | 424 Samples, min 1.9839 mm (Toleranz 0.0191) | – | 424, min 1.9839 | – | – |
| Opening entfernt (Dichte / komponiert / Reopening) | 1202 / 3581 / 754 mm³ | 2515 / 5438 / 868 mm³ | 1208 / 4796 / 1073 mm³ | – | – |
| Dichte (geteilt, einmalig) | 576 s GPU | 2771 s GPU (116 It., 24.7 s/It.) | (gleich) | 576 s GPU (gpu708) | 820 s CPU |
| Feld + Extraktion + Booleans | 116.0 s | 79.9 s bis Abbruch | 105.0 s | Extraktion 77.7, Nähen 37.9, Booleans 28.5, STEP 4.3 s | Rekonstruktion 1.5 s |
| Validierung | 34.4 s | – | 31.1 s | 22.2 s (bis Abbruch) | – |
| Metriken + Renders | 3.2 s | – | 2.7 s | – | – |
| Kandidat gesamt ohne FEA | 154.8 s | 79.9 s | 140.1 s | 270.5 s (Hauptlauf 17.5 s) | – |
| v0-Baseline-FEA (einmal je Studie) | 54.1 s | – | 52.6 s | – | – |
| FEA (Netz / Lösen)¹ | 304 s (23 / 282) | – | 259 s (42 / 217) | – | 55 s |
| Kandidat gesamt (Record runtime_s) | 513.3 s | 79.9 s | 451.8 s | – | – |
| Peak RSS | 2.67 GB | 2.54 GB | 2.84 GB | – | – |
| Rahmenmasse | 57.04 g (1.77x v0) | – | 45.67 g (1.42x) | 61.31 g (nur Screening) | 50.95 g (1.58x) |
| f1 | 1015.7 Hz¹ | – | 748.2 Hz¹ | – | 964.0 Hz |
| arm_tip Steifigkeit / Nachgiebigkeit | 244.5 N/mm / 0.00409 mm/N¹ | – | 279.0 / 0.00358¹ | – | 142.1 / 0.00704² |
| max. Verschiebung / von Mises | 0.0064 mm / 0.60 MPa¹ | – | 0.0052 / 0.68¹ | – | 0.0118 / 0.58 |
| Vergleich v0 (Steif./f1-Verhältnis) | würde bestehen (47.3 / 4.97)¹ | – | würde bestehen (54.0 / 3.66)¹ | – | bestanden (27.5 / 4.71) |
| Tet-Netz | 129 481 C3D10, SICN min 0.067 | – | 115 730, SICN min 0.078 | – | 58 196 |
| freie Scharfkanten / Fläche | 0.039 /mm | – | 0.051 | – | 0.237 |
| freier achsnormaler Anteil | 0.158 | – | 0.204 | – | 1.000 |
| freier Rasterebenen-Anteil (gpu708-Raster / eigenes Raster) | 0.009 / 0.009 | – | 0.008 / 0.010 | – | 0.464 / 1.000 |
| \|H\| r=1 mm p50 / p95 / p99 / max | 0.102 / 0.466 / 0.553 / 0.874 | – | 0.118 / 0.467 / 0.557 / 0.827 | – | 0.000 / 0.500 / 0.832 / 1.260 |
| Krümmung Nachbar-RMS | 0.152 /mm | – | 0.158 | – | 0.268 |

v0-Baseline (identisch in beiden Studien, SPOOLES 1 Thread): arm_tip 5.164 N/mm, f1 204.5 Hz, Rahmenmasse 32.16 g.

¹ Diagnostische FEA (Kandidat ist geometry_invalid, einziger Fehlschlag features, Massen-Screen bestanden): Treiber-Subklasse `int_e2e/run.py` ruft `evaluate` mit den Standard-`MESH_KEYS` auf `geometry.ply` auf; Lastfälle, Material, Punktmassen, SPOOLES 1 Thread, SICN- und Randabweichungs-Gates unverändert. Ergebnis in `record["diagnostic_fea"]` und `diagnostic_fea_result.json`; Status bleibt geometry_invalid, success false, Ledger unverändert. Vernetzung: remesh_hxt 2.0 mm scheitert bei beiden (2 gefaltete Kanten der vorbereiteten Oberfläche), remesh_delaunay 2.0 mm übersprungen (a) bzw. gescheitert (c01), remesh_hxt 1.5 mm erfolgreich. Randknoten-Abweichung 1e-10 mm, Flächenabtastung zu `geometry.ply` max 0.123 (a) / 0.151 mm (c01). Alle FEA-Werte und "würde bestehen" sind Diagnose; es gab keine Abnahme-FEA im Sinne von C13.
² (d) gelöst mit PaStiX, 2 Threads (keine SPOOLES-Nachrechnung vorhanden).
⁵ **Nutzerentscheidung 1:** deviation (C9) wird gegen das verarbeitete Feld gegated, wie `density_isosurface` der CAD-Route. Gemessen wird also der Extraktions-, Remesh- und Boolean-Fehler; ein echter 0,3-mm-Fehler scheitert weiterhin. Der Plan (Abschnitt 5, C9 ii) verlangte ≤ 0,20 mm gegen die rohe Dichte-Isofläche. Diese Distanz enthält die gewollte Formänderung durch sigma_d, Opening und Ripple-Glättung und wird nur als Diagnose gespeichert. Nach Plandefinition scheiterte deviation im Stand 567f2cc bei (a), (b) c00 und (b) c01 (0,257 / 0,344 / 0,345 mm > 0,20); dann bestünden nur 9 von 11 Checks. Für c407b01 nicht neu gemessen.
⁶ **Nutzerentscheidung 2:** Der analytische Motorbohrungs-Steg ist genau 2,0 mm. Umschriebene Polygone (Bohrung bis +0,01 mm) können ihn nie erreichen. Nur Sehnen mit beiden Enden auf vorgeschriebenen Bohrungspolygonen dürfen deshalb um die summierte Polygon-Übergröße kürzer sein. Alle anderen Stellen behalten die strikte Grenze von 2,0 mm; ein freier 1,98-mm-Steg scheitert.
⁷ Fein c00: Alle Witnesses bis einschließlich opening_witness zeigen 1 Komponente mit allen Mounts. Nach Ripple-Glättung und Reopening (entfernt 868 mm³) hat final_witness 2 Komponenten; alle Mounts liegen in Komponente 1. Der zweite Körper: 2026 Samples, 38.4 mm³ (0.09 % von 40 585 mm³), Box x [-1.2, 2.0], y [-41.7, -37.7], z [0.4, 5.2], max phi 1.64 mm, also ein echter Klumpen unter dem AIO/XT30-Bereich, dessen Hals das Reopening durchtrennt hat. Diagnose read-only mit `int_e2e/detached.py`.

Formmetriken für (a), (b) c01, (d) mit derselben Implementierung (`surface_metrics` mit der gpu708-Domain, Regionen in allen drei Domains identisch; `ball_curvature` r 1 mm, 5000 Samples, seed 0). (d) ist die OCCT-Tessellation (0.03 mm) des t01-STEP, also dasselbe Netz wie die Surface-Maturity-Referenz.

![Vergleich](implicit_vs_cad_gpu708_t025.png)

`implicit_vs_cad_gpu708_t025.png`: (a), (b) c01, (d) aus denselben vier Ansichten (Isometrie vorn links, Isometrie hinten rechts, Draufsicht, flach von vorn) im gleichen Maßstab; Anzeige auf 30k Dreiecke dezimiert. (b) c00 (field_disconnected) und (c) haben keine Geometrie zum Rendern.

Dünne Samples (C6), Cluster: Messerkanten-Lippen < 0.4 mm, wo eine Preserve-Schale auf eine Keep-out-Kappe trifft (Batterieband z ≈ 29 bei x ≈ ±6..15.5, y ≈ ±16..24; AIO/XT30-Unterseite z ≈ 0.1-2.7, y ≈ -38..-53; bei (a) 0.0005 mm bei (2.7, -58.6, 10)), dazu 1.3-1.98 mm bei (13, -35, 8.5) und z ≈ 25, x ≈ ±19.8. Bei c01 ca. 10 Samples mit 1.960-1.999 mm an der Motor-Unterseitenebene z = 4.2 (|x| 47-62): Hundertstel-Effekt, protokolliert, allein nicht maßgeblich.

## Sweep gpu708 (Code c407b01)

Schwellen 0.20/0.25/0.30 x Extensions none/preserve/preserve_forbidden, t025/preserve = Spalte (a). Drei Studien `exports/implicit/sweep/{t020,t025,t030}`, je ein CPU-Slot, Zusammenfassung `sweep/summary.json`.

| Kandidat | Status | Fehlschlag | Dünn / min Wand | Masse | f1 / arm_tip³ | runtime_s (Record) |
|---|---|---|---|---|---|---|
| t020 c00 none | geometry_invalid | features + Massen-Screen 2.008 | 100 / 0.036 mm | 64.58 g | – (Masse scheitert ebenfalls) | 156.2 s |
| t020 c01 preserve | geometry_invalid | features | 130 / 0.013 mm | 63.41 g | FEA-Vernetzung scheitert (alle 12 Versuche)⁴ | 156.9 s |
| t020 c02 preserve_forbidden | extension_changed_topology | field (Euler -43 -> -44) | – | – | – | 3.3 s |
| t025 c00 none | geometry_invalid | features | 36 / 0.021 mm | 58.61 g | 1041.7 Hz / 245.7 N/mm | 153.8 s |
| t025 c01 preserve_forbidden | extension_changed_topology | field (-44 -> -43) | – | – | – | 3.3 s |
| t030 c00 none | geometry_invalid | features | 32 / 0.092 mm | 53.29 g | 560.8 / 232.1 | 150.7 s |
| t030 c01 preserve | extension_changed_topology | field (-44 -> -48) | – | – | – | 3.2 s |
| t030 c02 preserve_forbidden | extension_changed_topology | field (-44 -> -47) | – | – | – | 3.3 s |

Erfolgsrate 0/8, Geometrie-Erfolgsrate 0/8; 4/8 bis zum Endnetz gebaut. Laufzeit pro Kandidat: Mittel 78.8 s, Median 77.0 s; gebaute Kandidaten 154.4 s (Geometrie + Validierung). Wanduhr ca. 5.5 min auf 3 parallelen CPU-Slots.
³ Diagnostische FEA (Skript `sweep/diag_fea.py`, Treiber-Standard), nicht im Ledger; remesh_hxt 2.0 mm im ersten Versuch, 99 324 bzw. 92 616 C3D10, 165-194 s; Vergleich zu v0 bestanden.
⁴ `fea._prepare_surface`: `meshing_merge_close_vertices` mit `fea_merge_distance_mm` 0.05 verschweißt die Gegenflächen eines dünnen Stegs bei (22.1, 37.3, 10.4) (echtes 0.9-1.3-mm-Glied zwischen motor_leads und camera_tool_access, nur t020).

## Shell-Wettbewerb (bedb04e -> c407b01)

Gemessen: dünne C6-Samples auf gpu708 t025 preserve und fein, summiert. Logs im Scratchpad `compete/`.

| Variante | gpu708 | fein | Summe |
|---|---|---|---|
| **Basis (Preserve-Schale, Standard)** | 52 (min 0.00046 mm) | 40 (min 0.031 mm) | **92** |
| protected_opening | 370 (min 0.20) | 467 (min 0.43) | 837 |
| fillet | 3687 (27 unaufgelöst) | 5128 | 8815 |
| ohne Schale | 4338 | 5398 | 9736 |

Alle drei Alternativen sind schlechter als die Basis. protected_opening bleibt als Option mit Tests, Standard aus (c407b01).

## Gegenüber 567f2cc

Das Reopening nach der Ripple-Glättung (cb09893) senkt die dünnen Samples (gpu708 86 -> 52, fein c01 85 -> 40) und die Masse (57.77 -> 57.04 g, 46.39 -> 45.67 g), trennt aber bei fein c00 den 38-mm³-Klumpen ab. Die FEA läuft jetzt mit den Standard-Vernetzungsschlüsseln (Zielkette 2.0 -> 1.5 mm, Elementbudget) statt der Diagnose-Overrides 1.0-1.2 mm, daher 115-129k statt 180k Elemente und 259-304 s statt 351-524 s. Die vorherigen Zahlen stehen in der Git-Historie dieser Datei (f0a1a32, 9ddf704).

Ledger `exports/run_log.jsonl`: 11 Zeilen, alle git c407b01, sauberer Baum (3 Einzelkandidaten, 8 Sweep), alle success false.

## Empfehlung

Standardroute: **implizite Route**. Jeder gebaute Kandidat liefert in ca. 2.5 min ein geschlossenes, orientiertes, selbstschnittfreies Einzelkörper-Netz mit exakten Bohrungen, Keep-out- und Hüllkurven-Freiheit; 10 von 11 Checks und der Massen-Screen bestehen nach den aktuellen Definitionen (Nutzerentscheidungen ⁵ ⁶). Die diagnostische FEA mit Standardeinstellungen läuft durch (4-5 min) und läge mechanisch klar über v0 und über (d). Die CAD-Route erreichte auf gpu708 in sechs Läufen nie die FEA. Formqualität: Scharfkanten 0.04-0.05 /mm statt 0.24, achsnormaler Anteil 0.16-0.20 statt 1.0, keine Voxelstufen. Noch kein implizites Teil ist freigegeben; (d) bleibt bis dahin die einzige geprüfte Referenz. Nächster Hebel ist allein die 2-mm-Wand an den Preserve/Keep-out-Kanten.

## Offene Probleme

1. **2-mm-Wand scheitert bei jedem gebauten Kandidaten** (32-130 dünne Samples). Rest: Messerkanten-Lippen < 0.4 mm, wo die in F8 wiederhergestellte δ-Schale (0.3 mm) von der exakten Keep-out-Kappe in X1 angeschnitten wird, ohne dass der Feldkörper sie stützt (Hypothese, nicht verifiziert). Vorschlag: F8-Restore der aufgeblähten Preserves auf Feldkörper ∪ exakte Preserves begrenzen. Dazu t020: echter 0.9-1.3-mm-Steg.
2. **Fein c00 field_disconnected** (Fußnote ⁷). Vorgeschlagener Patch, **Nutzerentscheidung offen**: in `_connected` nur für final_witness, wenn alle Mounts in einer Komponente liegen, die übrigen Komponenten entfernen (Feld auf -h, lokal reinitialisieren), als `detached_removed` [{volume_mm3, bbox}] protokollieren und nur oberhalb eines kleinen Volumenanteils (z. B. 0.005) scheitern. Entfernen ohne Mount überbrückt nie. Alternative: Fehlschlag beibehalten und als Problem des Dichtelaufs werten. Der Code sagt bewusst "neither bridged nor removed".
3. **FEA-Oberflächenvorbereitung**: 2.0-mm-Ziel faltet Kanten bei (a) und c01, 1.5 mm rettet (9-25 s). t020 c01: Merge 0.05 mm verschweißt einen dünnen Steg, alle 12 Versuche scheitern. Vorschlag: Merge nur bei fehlschlagenden mesh_checks oder Schwelle 1e-4 mm. Nadeldreiecke aus den exakten Booleans (min. Winkel 3e-5-2e-4°) schließen direct_hxt immer aus; Kollaps von Kanten < 1e-3 mm vor dem Export vorgeschlagen.
4. **Extension-Guard** weist auf gpu708 4 von 5 Extension-Varianten außerhalb von (a) ab (Euler-Zahl ändert sich um 1-4); auf den Varianten alle 8 (siehe `implicit_variants_pilot.md`).
5. **Feinrechnung**: Die Öffnung entfernt noch 4.8-5.4 cm³, die Mindestbreite stammt also nicht allein vom Optimierer. Der Lauf ist **nicht konvergiert** im Sinne des Plans (Abschnitt 11): objective_stall bei β 16 nach 14 Iterationen, change_tolerance nie erreicht, 24.7 s/It. statt 11-12 s.
6. (d) wurde mit PaStiX/2 Threads gelöst, nicht mit dem SPOOLES-Profil der impliziten Route.
7. Kosmetisch: `forbidden` und `preserve` sind in `validation.checks` je Region geschlüsselt, ihr oberstes `passed` ist None.
