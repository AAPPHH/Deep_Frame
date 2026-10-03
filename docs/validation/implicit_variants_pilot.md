# Datensatz-Pilot: implizite Route auf 8 GPU-Varianten

Stand: 2026-10-01, Code c407b01 (feature/implicit-integration), sauberer Baum, kein Code geändert. Daten: `exports/implicit/variants/<lauf>/`, Ledger `exports/implicit/variants/run_ledger.jsonl` (20 Zeilen, eine je Laufverzeichnis), zusammengefasst in `implicit_variants_pilot.json`. Jeder Lauf im CPU-Jobslot, Peak RSS ≤ 2.7 GB.

## Ergebnis

**0 von 8 Varianten freigegeben.** Mit der angeforderten Einstellung (t 0.25, preserve) stoppen alle 8 nach ca. 3 s in der Feldstufe mit `extension_changed_topology`. Der unveränderte Treiber nimmt keine Variante an: `prepare()` scheitert in `comparison_inputs` (wb131: `ValueError: Physical comparison input changed: comparison_load_cases`, 0.08 s, keine Ledger-Zeile). Für die Tabellen B und C liefen Scratchpad-Harnesses (`variant_harness.py` mit domänenbezogenem `comparison_inputs`, `guard_harness.py` mit einseitigem Guard); sie sind **nicht** die angeforderte Konfiguration.

| Variante | Änderung gegenüber gpu708 | A: t025 preserve (Standard + D1) | Guard Komp./Euler orig -> erw. | B: extension none (Harness) | C: einseitiger Guard, preserve (Harness) |
|---|---|---|---|---|---|
| wb131 | Radstand 131.25 mm | extension_changed_topology, 3.3 s | 1/-43 -> 1/-45 | Booleans scheitern (D5), 104 s | nur features: 39 dünn / 0.0006 mm, 55.40 g; FEA scheitert (D7), 2039 s |
| wb142 | Radstand 142 mm | extension_changed_topology, 3.1 s | 1/-44 -> 1/-47 | nur features: 234 / 0.0013 mm, 59.19 g; FEA scheitert (D7), 277 s | – |
| bat_small | Akku 57x26x10 mm, 29 g | extension_changed_topology, 3.4 s | 1/-43 -> 1/-45 | Extraktion scheitert (D6), 76 s | – |
| bat_large | Akku 68x34x14 mm, 55 g, Deck 44 mm | extension_changed_topology, 3.1 s | 1/-46 -> 3/-46 | nur features: 25 / 0.028 mm, 58.91 g; f1 805.8 Hz, 250.8 N/mm, Vergleich v0 bestanden, 366 s | nur features: 31 / 0.0005 mm, 56.93 g; f1 772.7 Hz, 243.9 N/mm, Vergleich bestanden, 359 s |
| load_arm2 | arm_tip-Kraft 2 N, Gewicht 2 | extension_changed_topology, 3.0 s | 1/-44 -> 1/-47 | nur features: 24 / 0.084 mm, 58.50 g; f1 1063.6 Hz, 313.4 N/mm, bestanden, 377 s | – |
| load_impact2 | Akku-Stoß 20 g, Gewicht 2 | extension_changed_topology, 3.1 s | 1/-42 -> 1/-43 | nur features: 35 / 0.013 mm, 58.88 g; f1 1155.1 Hz, 246.9 N/mm, bestanden, 369 s | – |
| combo_compact | wb131 + bat_small + load_arm2 | extension_changed_topology, 3.1 s | 1/-40 -> 1/-43 | features + supports (7 eingeschlossene Hohlräume): 26 / 0.027 mm, 55.93 g; keine FEA, 133 s | – |
| combo_heavy | wb142 + bat_large + Kamera 30° + load_impact2 | extension_changed_topology, 3.0 s | 1/-46 -> 3/-47 | Booleans scheitern (D5), 103 s | – |

FEA-Werte sind Diagnose (Kandidat bleibt geometry_invalid). v0 wird je Variante aus den eigenen Parametern gebaut (gpu708: 5.16 N/mm, 204.5 Hz; bat_large: 5.60 N/mm, 193.1 Hz); Steifigkeitsverhältnisse von 40-60x spiegeln die dünne v0-Platte. Die Läufe `bat_large_none` und `wb142_none` (FEA "invalid") sind Artefakte des ersten Harness (STL mit process=False geladen); gültig sind die `_r2`-Wiederholungen. In Tabelle C hält `extension_guard.original` topology(a ∧ b), nicht den Wert des Standardcodes.

## Raten und Laufzeit

- Erfolgsrate: 0/8 angefordert, 0/8 unter none, 0/2 unter einseitigem Guard. Diagnostische FEA ok: 3/8 (none), 1/2 (einseitiger Guard).
- Laufzeit pro Kandidat: ca. 3 s bei Abbruch in der Feldstufe. Bis zur FEA: ca. 100 s Geometrie + 32 s Validierung + 49 s v0-Baseline (einmal je Variante) + 172-192 s FEA, zusammen 360-380 s. Ausreißer 2039 s (wb131 C, davon 1853 s Tet-Vernetzung).
- Dünne Samples 24-39 je gebautem Kandidaten, außer wb142 none (234, Raster-Blobs).

## Was die Datensatzproduktion blockiert (nach Schwere)

1. **D1 `comparison_inputs` an topology_phase1 festgenagelt** (`tools/mature_pipeline.py:342-352`): verlangt Material, Punktmassen, comparison_load_cases und alle Nicht-Topologie-Parameter gleich `docs/validation/topology_phase1/inputs.json`. Jede Variante weicht ab (Lastbox wandert mit dem Radstand; Akku-Massen; Radstand, Deckbreite, Akku, Kamerawinkel, Lastfaktoren). Vorschlag: Selbstkonsistenz prüfen, `domain[name] == build_design_domain(inputs["parameters"])[name]` für die drei Blöcke; nur fea_settings und relative_constraints aus der historischen Datei. Besteht offline 8/8.
2. **D4 zweiseitiger Extension-Guard** (`deep_frame/topology_implicit.py:355-358`): verlangt Gleichheit von Komponenten und Euler-Zahl. Die Extension entfernt 2.2-2.7 cm³ Raster-Preserve-Blob (fügt ca. 0.2 cm³ hinzu) und öffnet dabei Henkel an den battery_coupling-Preserves; bei bat_large/combo_heavy verwaisen zwei Splitter von 3.6-4.1 mm³. Entfernen kann nicht überbrücken; gpu708 bestand nur zufällig. Vorschlag: einseitig `_topology(occupied ∧ extended)` gegen `_topology(extended)` vergleichen, also nur hinzugefügtes Material prüfen. Offline 8/8; abgelöste Splitter fängt der Final-Witness weiterhin.
3. **2-mm-Wand** scheitert bei jedem gebauten Kandidaten (Messerkanten an Preserve/Keep-out-Kanten, siehe `implicit_vs_cad_gpu708_t025.md`). Ohne Behebung ist kein Kandidat abnahmefähig.
4. **D7 Tet-Vernetzung** (`deep_frame/fea.py`): `_prepare_surface` erzeugt gefaltete Kanten bei jedem Ziel (wb131 C: 2 Falten; wb142 none: 12 Falten + Selbstschnitte, HXT-PLC-Fehler bei (-15.5, -6, 24.69)); danach laufen classify_hxt und classify_delaunay je in den 900-s-Timeout (30 min je gescheitertem Kandidaten). Vorschlag: Faltenreparatur; Classify-Fallbacks mit ca. 120 s Timeout oder überspringen, wenn alle Remesh-Ziele an der Topologie scheitern.
5. **D5 `_round_float32`** (`topology_implicit.py:528-534`) verwirft den Kandidaten bei einem einzigen verschmolzenen Vertex (wb131/combo_heavy none: 3 Kanten < 1e-4 mm auf z = 0 bei (-53.7, 48.5)). Vorschlag: `Manifold.simplify(~1e-5)` vor `to_mesh64` oder Merge mit erneuter Manifold-Prüfung und Protokoll.
6. **D6 Marching Cubes** erzeugt eine 8-Flächen-Hohlschale von -0.001 mm³ (bat_small none bei (-41, 43.4, 10.9)), Extraktion scheitert mit 2 Körpern. Vorschlag: negative, vom Außenraum getrennte Feldinseln vor MC füllen oder Schalen mit |V| < h³ verwerfen und protokollieren.
7. **D2 keine Ledger-Zeile bei Abbruch in `prepare()`**: `log_run` wird nur aus `candidate()` gerufen. Vorschlag: im except-Zweig von `ImplicitStudy.run()` `log_run` mit candidate None, failure_stage "prepare" und Fehler. Sonst sind Datensatz-Fehler im Ledger unsichtbar.
8. **D3 kein Diagnose-FEA-Pfad** im Treiber, wenn nur features scheitert (`if not passed or geometry_only: return`); die Harnesses ergänzen ihn mit Kennzeichnung `fea_diagnostic_only`.

Alle Patches sind vorgeschlagen, keiner angewendet. Reihenfolge für die Datensatzproduktion: D1 und D4 (Varianten erreichen die Geometrie), dann 2-mm-Wand (Abnahme), dann D7/D5/D6 (Robustheit und Laufzeit), D2/D3 (Nachvollziehbarkeit).
