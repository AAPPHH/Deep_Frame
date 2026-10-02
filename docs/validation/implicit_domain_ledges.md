# Preserve-Geometrie gegen Keep-outs: Messung auf gespeicherten Dichten

Stand: 2026-10-02, Branch feature/domain-ledges auf b42adde, Messung mit dem Arbeitsbaum dieses Commits (Skripte im Scratchpad `domain_check/`: `build.py`, `rays.py`, `run.sh`). Beide Dichten wurden auf der alten Domaene optimiert. Fuer diese Messung ersetzt `build.py` nur die Regionen durch `build_design_domain(inputs["parameters"])`; die gespeicherten Masken bleiben (Preserve-Masken weichen um 48 Zellen auf gpu708 und 122 Zellen fein ab, allowed/forbidden um 0). Die Preserves werden in der impliziten Route exakt neu aufgepraegt, deshalb ist das fuer diesen Schritt zulaessig. Neue Dichten auf der neuen Domaene stehen aus.

## Aenderungen an der vorgegebenen Geometrie

| Region | alt | neu | Grund |
|---|---|---|---|
| Motorkontakte | r 9.5, Oberkante z 4.0 | r 9.7, Oberkante 4.5 (Schnitt bei 4.0) | Rand um Motorhuelle 1.9 -> 2.1 mm; Messung 1.96-1.999 mm an z 4.2 |
| Akkuauflagen, Koppelflaechen, Strap-Oesen | Oberkante 29.0 | 29.5 (Schnitt bei 29.0, ausserhalb der Akkuhuelle 0.5 mm hoeher) | buendiger Kontakt |
| Strap-Oesen | Rand 2.0 | Rand 2.1 (`strap_rim_mm`) | Rand um Strap-Schlitz |
| XT30-/Balancer-Auflage | Oberkante 2.5 | 3.0 (Schnitt bei 2.5, Enden 3.0 hoch) | buendiger Kontakt |
| Kamera-Laschen | r 4.0 | r 4.1 (`camera_mount_radius_mm`) | Rand um 2-mm-Werkzeugkorridor 2.0 -> 2.1 |
| Antennenoese | r 3.5 | r 4.3 (`antenna_eyelet_radius_mm`) | Rand 2.0 -> 2.8; schliesst die Balancer-Ecke ein (Abstand vorher 0.66 mm) |
| AIO-Bosse | buendig 5.5 | unveraendert buendig | Abstand zur AIO-Huellwand 0.2 mm < Aufblaehung + Reserve; eine Verlaengerung erzeugte 230 Splitterproben (Lauf 1) |

Bohrungen, Schraubpositionen, Komponentenhuellen und Freiraeume bleiben unveraendert. Neue Selbstpruefung: `metadata.prescribed_clearance` (Test `test_prescribed_preserves_keep_two_millimetre_walls_against_keepouts`).

## Duenne Proben (C6, ohne Bohrungsstege)

| Lauf | vorher (b42adde) | Lauf 1: alle buendigen Kontakte verlaengert | Lauf 2: dieser Commit |
|---|---|---|---|
| gpu708 t025 preserve | geometry_built, 52 duenn, min 0.0005 mm | geometry_invalid in `booleans` (float32 verschmilzt 1 Vertex) | geometry_invalid in `booleans` (gleicher Vertex) |
| fein c01 preserve_forbidden | geometry_built, 40 duenn, min 0.031 mm | geometry_built, 308 duenn (230 an AIO-Huellwaenden) | geometry_built, 34 duenn, min 0.0024 mm; einzige verletzte Pruefung `features` |

Cluster fein, vorher -> Lauf 2: Motorkappen z 4.2: 11 -> 0. Antennenoese/Balancer-Ecke z 0: 4 -> 0. XT30/Balancer-Kanten z 0-3.2: 12 -> 11. Akku-/Strap-Kanten z 29-29.7: 8 -> 16. Strap-Schlitz und -Unterseite z 24.7-25: 5 -> 7. Die Bohrungsstege (424 Proben, min 1.9839 mm) bleiben innerhalb der C6-Toleranz.

## Offene Punkte

1. gpu708: Nach den Booleans liegen bei (-56.664, 49.629, 0) zwei Vertices exakt aufeinander (Nullkante, degenerierte Dreiecke am z=0-Schnitt des vorderen linken Motorkontakts, r 9.7 + 0.3). `_round_float32` meldet `merged_vertices = 1`, die Pruefung scheitert. Dieser Fehler gehoert zur Robustheit der Boolean-Ausgabe, nicht zur Domaene. Vorschlag: degenerierte Kanten unter der float32-Marge vor dem Runden kollabieren (`Manifold.simplify` oder Kantenkollaps) und danach erneut pruefen.
2. Buendige Kontakte innerhalb der Keep-out-Grundflaeche schneiden weiterhin Kanten an z = 29 (0.11-0.8 mm). Das Feld wird in einem Band `constraint_offset_mm` = 0.3 vor jedem Keep-out abgeschnitten, und die Preserve-Waende liegen nur 0.5 mm innerhalb der Akkuhuelle (< Aufblaehung + Band). An der Kehle zwischen Schale und abgeschnittenem Feld kippt das Netz nach aussen. Die Verlaengerung allein reicht dafuer nicht.
3. XT30-/Balancer-Enden: Die Steckerhuellen schneiden die Kappenrundung der 3.0 mm hohen Enden quer (0.22-1.9 mm, 11 Proben).
4. gpu708 vorher, nicht domaenenbedingt: (13.4, -34.9, 8.5) 1.84-1.93 mm und (18.9, 37.3, 15.9) 1.04 mm (naechste Region 1.3-3 mm entfernt); Kamera-Schraubbohrung tritt an der freien Laschen-Innenseite x = 10 aus (0.032 mm). Auf gpu708 nach dieser Aenderung nicht messbar, weil die Booleans scheitern.
