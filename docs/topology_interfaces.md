# Freie Topologieoptimierung: Schnittstellen v1

Dieser additive Vertrag erweitert `docs/interfaces.md`; die funktionierende v0-Geometrie und die bestehende FEA bleiben unveraendert als Referenz und unabhaengiger Pruefer. Python bleibt ohne Kommentare, Docstrings, argparse und unnoetige Prints. Alle Konfigurationen sind Dictionaries.

## Methode und Verantwortlichkeiten

Phase 1 verwendet klassische dreidimensionale, gefilterte SIMP-Dichteoptimierung auf regulaeren Hex8-Zellen. Die Startdichte ist im freien Bereich gleichfoermig. Mehrere statische Lastfaelle bestimmen die Materialverteilung; Eigenfrequenzen, Spannungen, Verschiebungen, Masse und Steifigkeit werden bei der unabhaengigen Kandidatenbewertung mit der vorhandenen gmsh/CalculiX-Pipeline gemeinsam geprueft. Die Akzeptanz darf nicht allein auf SIMP-Ersatzkennwerten beruhen. Eine diskrete Kandidatenauswahl kann die gemeinsam formulierten Ziele und Constraints verwenden; Gradienten aller Ziele sind keine Voraussetzung dieses Vertrags.

Es gibt keine v0-Armmaske, keine X-Frame-Saat und keine vorgeschriebenen Verbindungen zwischen Anschluessen. Komponentenpositionen, Anschlussflaechen, aeusserer Bauraum und Fertigungsgrenzen sind die ausdruecklich verbleibenden menschlichen Formannahmen. `build_frame` und `build_geometry` duerfen bei Designraumerzeugung und freier Optimierung nicht aufgerufen werden.

## Designraum

`deep_frame.topology_domain.build_design_domain(parameters: dict) -> dict`

`parameters` enthaelt die vorhandenen Komponenten-, Material- und Lastfallparameter sowie optional `topology`. Die Funktion kopiert Eingaben und liefert:

- `schema_version`: `deep-frame-topology-domain-v1`.
- `grid`: `origin_mm` ist die untere aeussere Ecke, `spacing_mm` ist immer eine Liste `[hx,hy,hz]`, `shape` ist `[nx,ny,nz]`, `axis_order` ist `xyz`, `order` ist `C`. Zellmittelpunkt `(i,j,k)` ist `origin + spacing * ([i,j,k]+0.5)`. C-Flatten: z laeuft am schnellsten. Gitterknoten besitzen entsprechend `(nx+1,ny+1,nz+1)`.
- `allowed`, `preserve`, `forbidden`: bool-NumPy-Arrays mit genau `grid.shape`. `allowed` umfasst freie und feste Materialzellen; `preserve` ist Teilmenge von `allowed`; `forbidden` ist das Komplement von `allowed`. Freie Zellen sind `allowed & ~preserve`. Vorgeschriebene Dichte ist 1 in Preserve und 0 in Forbidden. Subvoxel-Bohrungen werden zusaetzlich geometrisch exakt verarbeitet.
- `regions`: Liste benannter generischer raeumlicher Primitive mit `name`, `role` (`preserve`, `forbidden`, `allowed`) und `purpose`. Box: `kind: box`, `min_mm`, `max_mm`; achsparalleler Zylinder: `kind: cylinder`, `center_mm`, `radius_mm`, `height_mm`, `axis` (`x`, `y`, `z`). `center_mm` liegt im geometrischen Mittelpunkt. Primitive sind JSON-faehig. `rasterize: false` kennzeichnet subvoxelgenaue Ausschnitte, die nur bei Rekonstruktion ausgeschnitten werden; keine still verschwundenen Bohrungen.
- `material`, `point_masses`, `load_cases`, `fea_settings`: bestehender FEA-Vertrag, unveraendertes Einheitensystem. Die vorhandenen drei mechanischen Lastfaelle plus Modaltest bleiben enthalten. Zusaetzliche benannte Anschlusslastfaelle sind erlaubt, wenn geringe Lasten fuer sonst lastfreie notwendige Anbauteile deklariert werden.
- `manufacturing`: mindestens `nozzle_width_mm`, `minimum_wall_nozzles`, `minimum_feature_mm`, `minimum_attachment_area_mm2`, `supports_allowed`, `build_direction`. Supportfreiheit wird nicht behauptet. Diskrete Connectivity, Featuregroesse und lokale Anschlussquerschnitte sind vor FEA zu pruefen.
- `metadata`: JSON-faehige Entstehungsdaten mit freiem/festem/verbotenem Zellvolumen und Preserve-Anteil, Komponentenpositionen, Annahmen und Referenzbezug. Kein Shape in diesem Dict.

Die Zellen beschreiben die Optimierungsdiskretisierung; `regions` bleiben die genaue geometrische Wahrheit fuer Bauraumgrenzen, Bauteilfreiheit, Anschluesse und Montagezugang. Konservative Rasterabweichungen werden dokumentiert. Thin-Box-FEA-Selektoren gelten in physischen mm; der interne Hex8-Adapter darf diese fuer Knotenselektion um hoechstens eine halbe Gitterweite erweitern und muss die Erweiterung speichern. Eine leere Auswahl ist ein Fehler. Finale FEA nutzt die exakten Selektoren und die rekonstruierte Geometrie.

## Optimierer und Felddaten

`deep_frame.topology_optimization.optimize_topology(domain: dict, settings: dict) -> dict`

Das Ergebnis hat `status` (`ok`, `invalid`, `failed`), `density` (NumPy-Array), JSON-faehige `summary`, `history`, `diagnostics`. Optional weitere Felder sind explizit zu benennen. Status `ok` beschreibt einen abgeschlossenen Dichteoptimierungslauf, noch keine mechanisch akzeptierte Geometrie. Der Optimierer darf weder v0-Geometrie noch Rekonstruktion oder CalculiX importieren. Austauschbare Generatoren koennen denselben Feld-/Domain-Vertrag verwenden.

Settings umfassen `volume_fraction`, `filter_radius_mm`, `penalization`, `min_stiffness_ratio`, `max_iterations`, `change_tolerance`, `move_limit`, `projection_beta`, optional `case_weights`. Referenzlasten werden einzeln normiert, damit N/N-mm-Skalen keinen Lastfall versehentlich ausschalten. Iterationshistorie speichert insbesondere Volumen, Zielfunktion, Einzelcompliances, Aenderung und Konvergenzstatus. Fehlgeschlagene Loesungen werden nie als gueltige Nullwerte gespeichert.

## Rekonstruktion, Bewertung und Speicherung

`deep_frame.topology_geometry.reconstruct_topology(domain: dict, density, settings: dict) -> build123d.Solid`

`deep_frame.topology_geometry.validate_topology(solid, domain: dict, settings: dict) -> dict`

Der Rekonstruktionsadapter kann einen anderen belegten klassischen Algorithmus verwenden, muss aber automatisch ohne manuelles Nachmodellieren genau einen gueltigen geschlossenen Solid erzeugen. Er vereinigt erforderliche exakte Preserve-Primitiven und schneidet exakte Forbidden-Primitiven einschliesslich Bohrungen aus. Es darf keine von Hand eingezeichneten Ersatzarme oder Verbindungsrippen geben. Getrennte notwendige Preserve-Anschluesse duerfen nicht still entfernt werden. Fehler werden als nachvollziehbare Invalid-/Failed-Kandidaten gespeichert.

Die Validierung liefert `passed`, `violations`, `checks`. Pflichtchecks: ein Solid, Preserve-Abdeckung, Forbidden-Ueberdeckung, Komponenten-/Propellerfreiraum, Mindestfeature-/Duesenverhaeltnis und Anschlussquerschnitte. Die genaue Fertigungspruefmethode sowie moegliche Rastergrenzen werden dokumentiert. Nur gueltige Kandidaten erreichen `deep_frame.fea.evaluate`.

Persistenz enthaelt mindestens JSON-Manifest mit Domain ohne Arrays, Material, BCs, Lasten, Massen, Parametern, Seeds, Methode, Versionen/Hashes, Iterationshistorie und Kandidatenstatus; NPZ mit `density`, `allowed`, `preserve`, `forbidden`; resultierende STEP/STL; FEA-Ergebnis-Dicts und Solverartefakte; Vergleich zur unveraenderten v0. NPZ-Feldlayout steht im Manifest. Nicht erfolgreiche Designs bleiben als gekennzeichnete Datenpunkte erhalten. Ein reproduzierbarer Run benoetigt keinen manuellen CAD-Eingriff.

## Arbeitspakete und Akzeptanz

- A: dieser Vertrag, `topology_domain.py`, `topology_config.py`, Domain-Tests/-Dokumentation.
- B: `topology_optimization.py`, klassische Hex8/SIMP-Implementierung, analytische und Gradient-/Maskentests, Methodendokumentation.
- C: `topology_geometry.py`, Rekonstruktion, Fertigungs-/Geometrievalidierung, generischer Evaluator und Persistenz/Run-Adapter, unabhaengiger FEA-Test und Vergleich.
- Root orchestriert und integriert; jede Umsetzung findet in eigenen Branches/Worktrees statt.

Mindestens eine automatisch erzeugte, geometrisch und mechanisch gueltige Struktur muss sichtbare freie Lastpfade haben. Ein Vergleich zu v0 berichtet Masse, Steifigkeit, Maximalspannung, Maximalverschiebung und Eigenfrequenzen unter denselben physikalischen Last-/Materialannahmen. V0 bleibt Referenz; ein anderer Randbedingungsadapter muss auf beide Geometrien angewendet und begruendet werden. Ein besserer Einzelwert allein erfuellt Phase 1 nicht.
