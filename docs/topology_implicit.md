# Implizite Geometrie-Route

Die implizite Route ersetzt auf dem kritischen Pfad die Kette Dichte -> Dreiecksnetz -> B-Rep -> OCCT-Booleans. Alle Formoperationen laufen in einem Distanzfeld. Danach folgen genau eine Extraktion und ein exakter Boolean-Batch mit manifold3d. Das Ergebnis ist ein geschlossenes Dreiecksnetz (`geometry.stl`, `geometry.ply`). Genau dieses Netz wird geprueft, an die FEA gegeben und gedruckt. STEP ist nur noch optionale Ansicht.

Code: Feld, Extraktion und Booleans in [deep_frame/topology_implicit.py](../deep_frame/topology_implicit.py), Abnahme und Kruemmungsmass in [deep_frame/topology_implicit_validation.py](../deep_frame/topology_implicit_validation.py), Treiber in [tools/implicit_study.py](../tools/implicit_study.py). Alle Einstellungen stehen im einfachen Dict `IMPLICIT_CONFIG` in [deep_frame/config.py](../deep_frame/config.py). Wand-, Anhaftungs-, Volumen- und Reifegrenzen werden aus `SURFACE_VALIDATION_SETTINGS` referenziert und nicht kopiert.

## Feld-Pipeline

Feinraster: `subdivisions = 10` auf dem 8/3-mm-Dichteraster ergibt h = 0,267 mm. Werte sind float32 und innen positiv.

1. **Preserve-Erweiterung:** Preserve-Zellen erhalten die Dichte der naechsten freien Zelle. Die Variante `preserve_forbidden` erweitert zusaetzlich nur teilweise verbotene Zellen. Ein Topologie-Guard vergleicht die Komponentenzahl und die Euler-Zahl der harten Komposition mit und ohne Erweiterung. Bei einer Abweichung endet der Kandidat mit `extension_changed_topology`.
2. **Upsampling** (PCHIP, gemeinsam mit der CAD-Route) und Pegelfeld rho - t.
3. **Reinitialisierung** mit bandkorrigierter EDT, danach Gauss-Glaettung sigma_d = 0,4 mm und erneute Reinitialisierung.
4. **Anschluesse** werden als exakte Primitiv-SDFs per kubischem Smooth-Max eingefuegt. Die Uebergangsweite ist k = 2,0 mm, die Inflation delta = 0,3 mm.
5. **Keep-outs und Bauraum** werden danach per hartem Minimum mit dem Abstand c = 0,3 mm abgezogen. Verrundungen koennen deshalb nie in Keep-outs wachsen.
6. **Opening** mit r = 1,35 mm. Erosion {min(phi, d) > r}, d ist die exakte Euklid-Distanz zu den linearen Nullstellen von phi auf den Gitterkanten mit Vorzeichenwechsel; die Dilatation ist r - h^2/(2r) minus derselben Nullstellen-Distanz zur erodierten Menge. Der Abzug h^2/(2r) deckt die Ueberschaetzung der Punktwolken-Distanz, sodass keine Kugel aus phi herausragt und keine Keilschneiden entstehen. Gradientenprojizierte Bandpunkte werden dafuer nicht verwendet, weil sie an Mittelachsen 0,25-0,3 mm zu weit reichten und Restschalen nahe null hinterliessen. Jedes verbleibende Merkmal enthaelt eine Kugel mit mindestens 1 mm Radius. Das ergibt die 2-mm-Mindestwand plus Raster- und Glaettungsreserve.
7. **Ripple-Glaettung** (sigma_r = 0,25 mm), danach Reinitialisierung und **erneutes Opening** mit demselben Radius, dann harte Wiederherstellung der inflationierten Anschluesse. Die Glaettung hob nahe null liegende Opening-Reste um etwa 0,1 mm an und erzeugte so 1,1-1,5 mm duenne Glieder (gpu708: 59 Strahlen bei y 37, z 10-11; fine c01: 52 Strahlen bei z 16-18). Das Mindestwandkriterium ist deshalb die letzte Operation auf dem freien Feld; Kosten etwa 22 s je Kandidat. Ein Beschnitt dieser Wiederherstellung mit dem Keep-out-Abstand c wurde gemessen und verworfen: er erzeugt unter schmalen Anschlusswaenden und auf Motorauflagen Lippen von 1,85-1,99 mm (2355 statt 180 duenne Strahlen am gpu708-Rahmen). Die Option `protected_opening` (Opening der mit der Anschluss-Schale vereinigten, an Huelle und Keep-outs ausser vorgeschriebenen Bohrungen beschnittenen Form, danach nur exakte Preserve-Wiederherstellung) bleibt verfuegbar, ist aber standardmaessig aus. Im Schalenvergleich (noshell, fillet, protectopen) war sie die beste der drei Varianten, lag aber deutlich hinter der Basis: 837 statt 92 duenne Strahlen auf gpu708 und fine zusammen (370 statt 52 bzw. 467 statt 40).

Zeugen fuer die Verbindung werden im Feld geprueft, nie durch Ergaenzen erzeugt. Alle Pflichtanschluesse muessen in der unveraenderten Dichtekomposition und in der geglaetteten, geoeffneten Dichte vor dem Smooth-Union in einer 6-zusammenhaengenden Komponente liegen. Nach dem Opening und im Endfeld wird das erneut geprueft. Ein Verstoss ergibt `mount_disconnected`. Der Kandidat wird abgelehnt und nicht ueberbrueckt.

## Extraktion, Booleans und Bohrungen

Marching Cubes (Lewiner) laeuft auf dem mit -h umrandeten Feld. Danach folgt isotropes Remeshing mit pymeshlab (Ziel 0,6 mm). Die Gates sind dieselben wie bei der Dezimierung in der CAD-Route: hoechstens 0,20 mm Abweichung und hoechstens 1 % Volumenaenderung. Danach berechnet manifold3d in einem Batch ((Netz + Preserves) - alle Forbidden-Koerper) geschnitten mit dem Bauraum.

Zwei bewusste Entscheidungen gehen ueber "nur Bohrungen exakt" hinaus:

- **Exakte Booleans auch fuer Preserves und Keep-outs.** Ein abgetastetes Feld erreicht die unveraenderten Grenzen nicht: 1e-5 mm3 Fehlvolumen und Durchdringung sowie exakte Ebenen bei z = 0, 4 und 29 mm fuer die FEA-Selektoren. Die Feld-Offsets delta und c sorgen dafuer, dass diese Booleans nur quer schneiden oder nichts aendern. Sie schaben nie tangential.
- **Zylinder als umschriebene Polygone.** Die Segmentzahl folgt aus einer Toleranz von 0,01 mm. Bohrungen sind dadurch nie zu klein und hoechstens 0,01 mm zu gross; das Uebermass wird je Bohrung gespeichert. Die Voreinstellung von manifold3d (8 Segmente, 0,107 mm zu klein bei r = 1,4) wird nie verwendet.

## Abnahme auf dem Endnetz

`validate_implicit` prueft das Endnetz und dessen erneut geladenes Binaer-STL. Eine Pruefung, die sich nicht auswerten laesst, gilt als nicht bestanden.

| Pruefung | Inhalt |
|---|---|
| gates | eingefrorene Grenzwerte; nur Stichprobenbudgets duerfen ueberschrieben werden |
| topology, self_intersections | geschlossen, orientiert, ein Koerper, keine Hohlraumschalen, STL-Rundreise; MeshLab plus exakte Paarzertifikate |
| envelope, forbidden | Volumen ausserhalb bzw. im Keep-out <= 1e-5 mm3, zusaetzlich analytische Durchdringung <= 1e-4 mm |
| preserve | Fehlvolumen <= 1e-5 mm3, Bohrungsraender, Anhaftungsflaeche |
| features | exakte float64-Strahlen entlang der Innennormalen; Mindestwand 2 mm. Nur Sehnen, deren beide Enden auf vorgeschriebenen Bohrungen liegen, duerfen um die Summe der Polygon-Uebermasse (aus Segmentzahl und Radius) kuerzer sein (Nutzerentscheidung 2). |
| supports | Ueberhaenge und Zugaenglichkeit (eingeschlossene Hohlraeume) |
| deviation | Abstand zum verarbeiteten Feld (nach Glaettung und Opening, wie `density_isosurface` der CAD-Route) in der freien Zone <= 0,20 mm; Abstand zur rohen Dichte-Isoflaeche nur als Diagnose (Nutzerentscheidung 1; `density_diagnostic` ohne, `density_diagnostic_first_opening` mit den Aenderungen des erneuten Openings). Diese Diagnose liegt derzeit auf beiden Rahmen ueber 0,20 mm (gpu708 t025: 0,257 mm vor, 0,313 mm nach dem erneuten Opening; fine c01: 0,345 bzw. 0,377 mm); der Gate sieht das erneute Opening konstruktionsbedingt nicht |
| connectivity | Feldzeugen, ein Koerper, alle Preserves vorhanden |
| surface_maturity | scharfe Kantenlaenge je freier Flaeche und achsparalleler Flaechenanteil hoechstens 0,5 x Referenz (Voxelroute grid4_iter300 t01) |

Zwei Abnahmeregeln weichen vom Architekturplan ab. Beide sind Nutzerentscheidungen, keine Lockerung durch die Implementierung:

1. **deviation gegen das verarbeitete Feld.** Der Plan (Abschnitt 5, C9 ii) verglich in der freien Zone mit der rohen Dichte-Isoflaeche pchip(rho) - t. Diese Distanz enthaelt die gewollte Formaenderung durch sigma_d, Opening und Ripple-Glaettung. Gegated wird deshalb wie bei `density_isosurface` der CAD-Route der Abstand des Endnetzes zum Marching-Cubes-Netz des verarbeiteten Feldes. Das ist der Extraktions-, Remesh- und Boolean-Fehler. Der Abstand zur rohen Dichte wird als `free_zone.density_diagnostic` mit `gated: false` gespeichert. Auf gpu708 t025 betraegt er 0,257 mm, auf der Feinrechnung 0,344 und 0,345 mm, also jeweils ueber 0,20 mm. Nach Plandefinition wuerde deviation dort scheitern. Ein echter Extraktionsfehler von 0,3 mm scheitert weiterhin (Test `test_extraction_error_against_processed_field_fails_the_gate`).
2. **Bohrungsstege.** `_primitive_wall_checks` gibt fuer die Motorbohrungen einen analytischen Steg von genau 2,0 mm vor. Umschriebene Polygone (D2) machen jede Bohrung bis zu 0,01 mm groesser, deshalb kann dieser Steg 2,0 mm nie erreichen. Sehnen, deren Anfang und Ende beide auf dem realisierten Polygon einer vorgeschriebenen Bohrung (rasterize=False) liegen, duerfen deshalb genau um die Summe der beiden Polygon-Uebermasse kuerzer sein. Auf gpu708 betrifft das 424 Samples, min 1,9839 mm bei einer Toleranz von 0,0191 mm. Alle anderen Sehnen behalten die strikte Grenze 2,0 mm - 1e-5. Ein freier 1,98-mm-Steg und ein Bohrungssteg mit 4,47 mm Achsabstand scheitern weiterhin (Tests in `tests/test_topology_implicit_validation.py`).

Danach folgen der Massen-Screen (hoechstens 2,0 x v0) und die FEA, siehe [fea.md](fea.md). Die FEA erhaelt das Endnetz ueber den STL-Zweig. Die Tetraeder-Versuche laufen nacheinander in jeweils eigenen Prozessen (`tet_attempts`). Die `remesh_*`-Versuche laufen zuerst ueber alle Oberflaechen-Zielkantenlaengen `fea_remesh_targets_mm` (2,0 -> 1,5 -> 1,2 -> 1,0 mm), danach folgen `refine_hxt` (Unterteilung auf `fea_refine_edge_mm`, dann Remesh auf das groebste Ziel), `classify_*` und `direct_hxt`. Ein Ziel, dessen vorbereitete Oberflaeche Faltkanten oder Selbstschnitte hat, wird fuer die uebrigen `remesh_*`-Versuche desselben Ziels uebersprungen. Jeder Versuch hat ein Elementbudget aus dem Speicher: `min(fea_memory_budget_mb - privater Speicher des Prozesses, freier Speicher) / fea_memory_per_element_kb` (9728 MB, 38 kB je C3D10 fuer SPOOLES gemessen bei 113k-369k Elementen). Liegt die lineare Tetraederzahl darueber, scheitert der Versuch vor der Ordnungserhoehung mit `over_budget`; feinere `remesh_*`-Ziele werden dann mit Begruendung uebersprungen. Jeder Versuch steht mit Name, Ziel, Status, Diagnose, Elementzahl und Laufzeit in `result["mesh"]["attempts"]`. Gates sind minSICN >= 0,01 und die Randabweichung, die derzeit nur an den Knoten geprueft wird. Lastfaelle, Material, Punktmasse und Vergleichsgrenzen sind identisch zu [topology_pipeline.md](topology_pipeline.md).

## Formqualitaet und Bilder

- `surface_metrics` liefert die vorhandenen Masse fuer scharfe Kanten je freier Flaeche und den achsparallelen Flaechenanteil. Fuer Netze gilt dasselbe wie fuer CAD.
- `ball_curvature` integriert das Cohen-Steiner/Morvan-Mittelkruemmungsmass ueber 1-mm-Kugeln um 5000 gleichmaessig verteilte, geseedete Oberflaechenpunkte. Normiert wird mit pi r^2: Kugel 1/R, Zylinder 1/(2R), Ebene 0. Gespeichert werden p50/p95/p99/max von |H| und die RMS der |H|-Spruenge zwischen Nachbarpunkten unter 1,5 mm. Das Mass haengt nicht von der Triangulierung ab. Es wird nur protokolliert und ist kein Gate.
- `render_views` erzeugt headless mit matplotlib Agg vier Ansichten (Isometrie, oben, vorne, seitlich) einer nur fuer die Darstellung dezimierten Kopie. Dazu kommen exakte Schnitte des vollstaendigen Netzes bei z = 2 und 27 mm mit Preserve- und Forbidden-Umrissen. Alle PNGs werden mit SHA-256 im Record gespeichert. Die Sichtpruefung bleibt Pflicht.

## Treiber, Records und Lauf-Ledger

```powershell
.\.venv\Scripts\python.exe tools/implicit_study.py run study.json
.\.venv\Scripts\python.exe tools/implicit_study.py render render.json
.\.venv\Scripts\python.exe tools/implicit_study.py summarize summary.json
```

`run` benoetigt `source` (hashgepruefte Dichtequelle) und `output` (neues, leeres Verzeichnis). Alle Schluessel aus `IMPLICIT_CONFIG` sowie `geometry_only`, `reference_step`, `study_timeout_s`, `run_log` und `section_heights_mm` lassen sich ueberschreiben; unbekannte Schluessel werden abgelehnt. Kandidaten sind das Produkt `thresholds x extensions` mit den IDs cNN.

Das Ausgabelayout entspricht der CAD-Studie: `manifest.json` (Schema `deep-frame-implicit-geometry-study-v1`), `status.json`, `density_source/`, `provenance/`, `baseline/` und `reference_metrics.json`. Je Kandidat gibt es `candidates/cNN/` mit `record.json`, `progress.jsonl`, Endnetz, `renders/`, `raw_fea_result.json` und bei Fehlern `failure/`. Der Record enthaelt Parameter, Status, `failure_stage`, Laufzeiten je Stufe (`timings_s`: Feldschritte, extraction, remesh, booleans, export, validation je Pruefung, metrics, renders, fea, fea_mesh, fea_solve), Spitzen-RSS, Pruefergebnisse, Formmasse, Kruemmung, Masse, FEA und Vergleich. Das Manifest fasst `success_rate`, `geometry_success_rate` und `runtime_per_candidate_s` zusammen.

Jeder Kandidat jedes Laufs haengt eine JSON-Zeile an `exports/run_log.jsonl` an (`topology_pipeline.log_run`). Das gilt fuer implizite Studien, die CAD-Route (`tools/mature_pipeline.py geometry`), die Rasterroute (`tools/workstation_study.py candidates`, Art `raster`, je in diesem Aufruf bearbeitetem Kandidaten eine Zeile; Status `ok` wird als `accepted` gezaehlt, der Originalstatus steht in `raster_status`) und Dichtelaeufe (`tools/topology_study.py run`). Der End-to-End-Treiber `tools/mature_pipeline.py run` nimmt `run_log` entgegen, loest den Pfad einmal absolut auf und reicht ihn an Dichte- und Geometriestufe weiter; endet die Dichtestufe ohne eigene Zeile (Zeitlimit, Absturz), schreibt der Treiber eine Fehlzeile `timed_out` oder `failed`. Jede Zeile enthaelt Zeit, Git-Commit, Laufverzeichnis, Art (`implicit`, `cad`, `raster`, `density`), Kandidat, Quell-Hash, Erfolg, Status, Fehlerstufe, Laufzeit und Stufenzeiten. `summarize` fasst daraus je Lauf Erfolgsrate, Statusverteilung und Laufzeit je Kandidat zusammen.

Die Belege des Vergleichs auf der gespeicherten gpu708-Dichte und der Feinrechnung stehen in [validation/implicit_vs_cad_gpu708_t025.md](validation/implicit_vs_cad_gpu708_t025.md), die Feinrechnung selbst in [validation/robust_fine_run.md](validation/robust_fine_run.md). Die Feinrechnung stoppte durch Stagnation der Zielfunktion, nicht durch Konvergenz der Designaenderung (siehe dort).
