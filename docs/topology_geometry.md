# Designraum, Rekonstruktion und Fertigungspruefung

## Freier Designraum und verbleibende Formannahmen

`deep_frame.topology_geometry.build_design_domain(parameters)` erzeugt aus einem einfachen Parameter-Dict die freie Ausgangsdomaene. `TOPOLOGY_CONFIG` in `deep_frame/config.py` ergaenzt die bestehenden Komponenten-, Material- und Integrationsparameter. Der Adapter ruft weder `build_frame` noch `build_geometry` auf. Aenderungen an v0-Armbreite, Armwurzel, Zentralplatte oder Deckfenstern beeinflussen seine Masken nicht; ein Test setzt diese v0-Parameter absichtlich auf geometrisch unbrauchbare Werte.

Der Startbauraum ist ein einziger Quader von x = -68 bis 68 mm, y = -64 bis 64 mm und z = 0 bis 32 mm. Er umfasst auch Material zwischen Motoren, hinter der Elektronik und oberhalb der ueblichen Arme. Es gibt keine vorgeschriebenen Arme, keine zentrale Platte, keine Deckwaende, keine Kamerakaefigwaende und keinen Hecksteg. Der Optimierer kann innerhalb dieses Raums durchgehende Flaechen, schraege Pfade, Verzweigungen, Rippen und unterschiedliche Querschnitte erzeugen. Motor- und Komponentenpositionen sowie die aeusseren Bauraumgrenzen bleiben in diesem Lauf fest.

### Quantifizierte Freiheit

Die Standarddiskretisierung besitzt 34 x 32 x 8 Zellen mit 4 mm Kantenlaenge. Zellen sind in der Reihenfolge xyz und C-order gespeichert; z laeuft beim Flattening am schnellsten.

| Menge | Zellen | Rastervolumen |
| --- | ---: | ---: |
| Gesamter Quader | 8704 | 557056 mm3 |
| Zulaessiger Materialraum | 5842 | 373888 mm3 |
| Feste lokale Anschluesse | 162 | 10368 mm3 |
| Frei optimierbare Zellen | 5680 | 363520 mm3 |
| Gesperrt | 2862 | 183168 mm3 |

Die Preserve-Zellen belegen **2.7730 Prozent** des zulaessigen Raums. **97.2270 Prozent** bleiben freie Optimierungsvariablen. Diese Angaben betreffen das Raster; exakte Kontaktgrenzen und Bohrungen aendern das physische Endvolumen. Die Preserve-Maske enthaelt mehrere getrennte Komponenten und kann allein keinen tragenden Frame bilden.

`volume_fraction = 0.10` begrenzt das SIMP-Dichtevolumen auf 37388.8 mm3, entsprechend 40.754 g bei 1.09 g/cm3. Dies ist kein vorweggenommenes Ergebnisgewicht: Dichteschwelle, exakte lokale Anschluesse, Ausschnitte und Qualitaetspruefungen bestimmen das tatsaechliche Gewicht. Die zugehoerigen Werte und die tatsaechlichen Masken werden je Lauf gespeichert.

### Feste funktionale Anschluesse

- Vier lokale Motorkontakte im ManaFly-Stil: 7.7 mm Radius (2.1 mm Rand um die M2-Bohrungen auf dem 9-mm-Lochkreis des GTS V3 1203), vollstaendig unter der 8.38-mm-Motorhuelle; Oberkante 0.5 mm in die Motorhuelle hinein, der exakte Schnitt legt die Auflage auf die Motorunterseite z = 4 mm. M2-Bohrungen und Wellenfreigang werden exakt ausgeschnitten. Es gibt keine vorgeschriebenen Verbindungen zu einem Zentrum.
- Vier AIO15-Bosse: 3.2 mm Radius, Oberkante buendig bei 5.5 mm (nur 0.2 mm neben der AIO-Huellwand, deshalb nicht verlaengert), Bohrungen gemaess dem bestehenden 25.5-mm-M2-Lochbild. Der umgebende Boden bleibt frei.
- Zwei Laengsschienen fuer den Akku-Strap im ManaFly-Stil (Breite 3.5 mm, Laenge 50 mm, 3 mm hoch unter z = 28 mm). Ihre Aussenkante liegt unter der Akkukante (`battery_rail_edge_inset_mm` = 0, x = 11.5..15 mm); ihre Oberseite liegt 0.5 mm in der Akkuhuelle, die exakte Auflage bleibt bei z = 28 mm. Der Strap umschlingt Akku und Schiene.
- Die Akkuauflage `deck_top_mm` = 28 liegt auf einer gemeinsamen Knotenebene der 4-, 2-, 4/3- und 1-mm-Gitter (Vielfache von 4 mm ab z = 0). So treffen Schienenzellen, Akku-Punktmasse und der Lastselektor `battery_impact` (z = 28 +- 0.01 mm) auf jedem dieser Gitter vorhandene Preserve-Knoten, ohne Selektor-Erweiterung oder lokale Verschiebung in den Studienwerkzeugen. Die Schienen-x-Lage enthaelt auf jedem dieser Gitter mindestens einen Zellmittelpunkt.
- Ein ELRS-Antennenfreiraum von 3 mm ueber dem AIO-Board (Keep-out `elrs_antenna_clearance`).
- Zwei kleine Kamera-Schraublaschen (Radius 4.1 mm, 2.1-mm-Rand um den 2-mm-Werkzeugkorridor) und zwei obere Schutz-/Lastkontaktflaechen. Ihre lokale Breite von 6 mm erreicht die Rastergrenze x = +/-16 mm; dies verhindert subvoxelduenne Reste zwischen Schraubbohrung und freiem Material. Die Breite ist eine eigene Diskretisierungs-/Montageannahme, keine kopierte v0-Geometrie. Dazwischen ist weder eine Kaefigwand noch ein Verbindungsbogen vorgeschrieben.
- Keine Sitze, Oesen oder Zugangskorridore fuer XT30, Balancer (JST-XH) und VTX-Antenne (Nutzerentscheidung): Die Stecker haengen an den Akkuleitungen, Antenne und Stecker werden mit Gummiband dort befestigt, wo es passt. XT30 und Balancer bleiben Bibliotheksteile (Masse im Akku enthalten), sind aber weder Keep-out noch Preserve; das Heck hinter dem Stack ist bis auf Motor- und Prop-Regionen freier Bauraum.

Diese lokalen Primitiven sind menschliche Formannahmen fuer die notwendigen Schnittstellen. Sie duerfen nicht als frei optimierte Geometrie ausgegeben werden. Die tragenden Verbindungen zwischen ihnen entstehen aus dem freien Materialfeld.

### Preserve-Geometrie gegen Keep-outs

Die implizite Route blaeht jede Preserve-Primitive um `preserve_inflation_mm` = 0.18 mm auf und schneidet Keep-outs danach exakt. Liegt eine Preserve-Flaeche buendig auf einer Keep-out-Flaeche, trifft dieser Schnitt die Kappe der aufgeblaehten Schale; uebrig bleiben Stufen in Hoehe der Aufblaehung und Schneiden mit Wandmessungen von 0.0005 bis 0.7 mm (Akkuauflagen z = 28, Motorkappen z = 4). Die Wandpruefung bleibt streng; korrigiert wird die vorgegebene Geometrie:

- Buendige horizontale Kontakte (Preserve-Oberseite = Keep-out-Unterseite bei gemeinsamer Grundflaeche) werden um `flush_overlap_mm` = 0.5 mm in das Keep-out verlaengert und dort als Preserve-Ausschnitt deklariert, ausser die Preserve-Primitive liegt naeher als Aufblaehung plus Reserve (0.28 mm) an einer Seitenwand des Keep-outs; dann wuerde die Schale als Splitter austreten, der Kontakt bleibt buendig und ist eine Verletzung der Selbstpruefung (keine Ausnahme). Die AIO-Bosse haben deshalb r = 3.1 mm (`aio_boss_radius_mm`, 0.3 mm Abstand zur AIO-Huellwand, Rand um die Schraubbohrung 2.0 mm); die Klemmselektoren behalten `aio_contact_radius_mm` = 3.2 mm. Der exakte Schnitt laeuft dann durch die senkrechte Schalenwand; die wirksame Auflageebene bleibt unveraendert. Keep-outs innerhalb einer koaxialen vorgeschriebenen Bohrung (Antennen-Einfuehrkanal) zaehlen nicht. Die Liste steht in `metadata.flush_contact_extensions`.
- Jede Keep-out-Wand, die eine Preserve-Primitive schneidet, und jedes koaxiale Keep-out-Zylinderpaar muss einen Rand von mindestens 2.0 mm plus `prescribed_wall_margin_mm` = 0.1 mm lassen (Polygon- und Float-Reserve).
- Zwei Preserve-Primitiven ueberlappen oder liegen weiter als dieser Rand plus beide Aufblaehungen auseinander.

`prescribed_clearance` prueft diese Regeln nach der Verlaengerung und speichert das Ergebnis in `metadata.prescribed_clearance`; die Standarddomaene muss sie bestehen. Vorgeschriebene Bohrungen sind ausgenommen, ihre Stege regelt die C6-Bohrungstoleranz. Die Ebene z = 0 ist Bauraumgrenze, kein Keep-out, und bleibt unveraendert. Messung auf den gespeicherten Dichten: `docs/validation/implicit_domain_ledges.md`.

### Verbotene Volumina und Montage

Die vorhandenen parametrisierbaren Komponenten-Platzhalter liefern die Bauraumhuellen fuer AIO15, Lux-Kamera mit Tilt, GNB5502S120A, vier Motoren und Props. Hardware erhaelt 0.5 mm Zusatzfreiraum; geplante untere Auflageflaechen behalten Kontaktabstand null ohne positives Durchdringungsvolumen. Props erhalten radial und axial 2 mm Freiraum. Die Prop-Huelle ist die HQProp-T2.5X2X3V2S-Scheibe (63.5 mm) ueber die volle Nabenhoehe (5 mm ab Motoroberkante, Propebene aus der Nabengeometrie); mit Freiraum ergibt das einen Zylinder r = 33.75 mm, z = 11.9..20.9 mm.

Zusaetzlich gesperrt werden Batterieentnahme nach oben, AIO-Einschub von rechts bei abgesteckten Kabeln, Kameraeinbau/Linsenkorridor nach vorne, vorlaeufige Motorkabelkorridore und Schraubbohrungen. Diese Korridore sind nachvollziehbare Montageannahmen, keine vollstaendige Simulation aller Werkzeuge, biegsamen Kabel oder des Kamera-Sichtfeldes.

Die Kamera-Schraubbohrung endet exakt an den lokalen Laschen. Ausserhalb davon ist ein 4-mm-Werkzeugkorridor konservativ rasterisiert. Der Antennenkanal reicht durch den gesamten Bauraum nach oben; oberhalb der festen Oese wird auch er konservativ rasterisiert. Damit kann weder ein mathematisch vorhandener Schraubenkanal eine duenne freie Resthaut erzeugen noch eine kurze Antennenbohrung unter einer spaeter hinzukommenden Rasterzelle blind enden.

Eine freie Rasterzelle wird konservativ entfernt, sobald ihr Quader eine Hardware-/Prop-/Zugangshuellenflaeche mit positivem Volumen schneidet. Reine Beruehrung einer Auflageflaeche entfernt die darunterliegende Zelle nicht. Damit schneidet die exakte Rekonstruktion keine zufaellig duennen Resthaeute aus freien Hardware-Randzellen. Subvoxel-Bohrungen und Strap-Schlitze haben `rasterize: false` und werden zwingend geometrisch ausgeschnitten.

Preserve-Ausschnitte sind fuer Bohrungen und Oesenschlitze ausdruecklich mit `allow_preserve_subtraction` gekennzeichnet. Potenzielle Paare werden konservativ anhand positiver AABB-Ueberdeckung protokolliert; die exakte CSG-Differenz ist das geometrische Soll. Eine neue, nicht deklarierte Preserve-/Forbidden-Ueberdeckung wird vor der Optimierung abgelehnt, statt die Kontaktflaeche still zu verlieren.

### Vergleichbare Lasten und Anschlussnachweis

Der Adapter uebernimmt die bisherigen Armspitzen-, 10-g-Akku-Aufprall-, seitlichen Kameralast- und Modalannahmen. Fuer den Armspitzenfall werden ausschliesslich die Unterseiten der vier AIO-Montagekontakte geklemmt: vier Boxselektoren um x/y = +/-12.75 mm mit je 3.2 mm Halbausdehnung und z = +/-0.01 mm. Der Optimierer kann dadurch keine willkuerlich grosse freie Bodenflaeche auf eine breite numerische Klemme aufbauen.

Diese vier Faelle stehen in `comparison_load_cases` und muessen unveraendert auf **v0 und freie Struktur** angewandt werden. Alte v0-Ergebniswerte mit breiter Zentralklemme sind keine gueltigen Vergleichswerte fuer diesen Lauf. Alle anderen Faelle fixieren weiterhin die vier Motor-Unterseiten. Der Akku wird identisch an die Querflaeche bei z = 28 mm gekoppelt; die Punktmasse liegt bei [0, 0, 33.5] mm.

Im internen Hex8-Adapter verwendet dieser Frame-Designraum ausdruecklich `interface_node_policy: preserve_adjacent`: Lasten und Fixierungen treffen nur Knoten an festen Kontaktzellen. Eine um bis zu eine halbe Zellenweite erweiterte Auswahl darf dadurch keine beliebigen freien Materialknoten im breiten geometrischen Selektor belasten. Selektorerweiterung und Policy werden gespeichert. Die abschliessende FEA verwendet weiterhin die exakten Kontaktflaechen am rekonstruierten Solid.

17 weitere Faelle belasten sonst nicht direkt belastete Pflichtanschluesse einzeln mit 0.05 N nach unten. Sie verhindern, dass lastfreie Montageschnittstellen als unverbundene Inseln verbleiben. In der normierten Compliance-Zielfunktion besitzen die drei Hauptfaelle zusammen 90 Prozent Gewicht, alle 17 Anschlussfaelle zusammen 10 Prozent. Eine kleine Kraft alleine wuerde nach Compliance-Normierung das Gewicht nicht verringern. Der separate geometrische Nachweis der Verbindung aller Pflichtanschluesse bleibt notwendig.

### Fertigung und Quellen

Die vorlaeufige Fertigungsgrenze betraegt 2 mm, abgeleitet aus 0.4-mm-Duese und mindestens fuenf Bahnen. Exakte Bohrungsumrandungen und Anschlussquerschnitte werden zusaetzlich geprueft. Stuetzmaterial ist erlaubt; Supportfreiheit, bestimmte Druckfestigkeit oder nacharbeitsfreie Montage werden nicht behauptet. Die Fertigungspruefung ist ein geometrisches Screening und ersetzt keinen Druckversuch.

Die feste Hardwareanordnung und ihre Quellen bleiben in `COMPONENT_DEFAULTS`, `FRAME_DEFAULTS` und `FRAME_DEFAULT_SOURCES` von `deep_frame/config.py` sowie in [armattan_research.md](armattan_research.md) nachvollziehbar. Die Motorhuelle stammt vorlaeufig vom [GEPRC GR1105](https://geprc.com/product/gep-gr1105-motor/); sein 9-mm-Lochkreis ist kein 9x9-mm-Quadrat. Dichte und isotrope Materialannahme bleiben der dokumentierte [Bambu-PA6-CF-Datensatz](https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf). Rasteraufloesung, Kontaktgroessen, Bauraum, Korridore und relative Anschlusslasten sind eigene offengelegte Modellierungsentscheidungen.

## Boden des Bauraums

### Klaerung: die zwei kantigen Bloecke am Boden

Grundlage: Lauf `battery_free` (Phase-1-Layout, freie Akkulagerung, SIMP+MMA 4/3 mm, Dichte `C:/clones/Deep_Frame-layout/exports/runs/battery_free_opt/fine/density_half.npz`). Die Domaene ist mit demselben Aufbau rekonstruiert wie im Lauf (`docs/validation/bottom_domain/run_cfg.json`, `tools/formulation_study.patched_builder`, also `R2Domain` mit getrimmten Motorpads). Befehl: `python -m tools.topology_study floor docs/validation/bottom_domain/floor_before.json` (Ray `cpu`). Bild: `docs/validation/bottom_domain/before/floor.png`, Zahlen: `docs/validation/bottom_domain/before/floor.json`.

Jeder der beiden Bloecke (x = -12.75 und x = +12.75) besteht aus drei Teilen:

| Teil | Quelle | Ausdehnung | Vorgabe oder Optimierer | am Boden z = 0 |
|---|---|---|---|---|
| zwei Saeulen | Preserve `aio_contact_0/1` (links) bzw. `aio_contact_2/3` (rechts) aus `topology_geometry._component_regions` | Zylinder r = 3.1 mm, z = 0..11.8 mm (`base_thickness_mm` 3.0 + `aio_standoff_mm` 8.8), Achsen bei y = -12.75 / +12.75 | **Vorgabe**: 356 mm3 exakt, 398 mm3 im Raster je Saeule, also 4 x 398 = 1593 mm3 = 29 % aller Preserve-Zellen (5547 mm3) | flache Unterseite ist die eigene Grundflaeche der Vorgabe, kein Zuschnitt |
| Untergurt | Optimiererdichte zwischen den Saeulen | x = 10..14 mm, y = -8.7..10 mm, 38 Zellen je Seite in der Bodenschicht (90 mm3); Hoehe ueber z = 0 im Median 1.33 mm, hoechstens 2.67 mm | **Optimierer** | **abgeschnitten**: liegt mit voller Breite auf der Bauraumgrenze z = 0; ein voller Querschnitt (Mindeststegbreite 2.5 mm robust, Filterradius 4 mm) passt darunter nicht |
| Stackfixierung | `aio_fixtures` in `build_design_domain` (Evaluator `center_fixtures`, Lastfaelle arm_tip/crash_*) | Quader r = 3.2 mm um jede Schraubachse, z = -0.01..0.01 | Vorgabe der Lagerung | fixiert die Saeulenunterseiten auf z = 0 |

Von der Seite (Schnitt x = 12.67) sieht man je Seite die beiden Saeulen (blau) und dazwischen den eine bis zwei Zellen dicken Untergurt (orange) auf z = 0; von vorn decken sich die Saeulen eines Paars zu je einem Rechteck 6.2 x 11.8 mm. Das sind die beiden Bloecke. Die Saeulen sind seit dem Phase-1-Layout so hoch, weil `aio_contact_*` den ganzen Abstand Boden bis AIO-Unterseite vorgibt (bei Standoff 3 mm waren es 5.5 mm). Radial gibt es nichts zu verkleinern: 3.1 - 1.1 = 2.0 mm ist genau die Mindestwand.

Weitere Zellen in der Bodenschicht (z = 0..1.33): 112 Preserve-Zellen, davon 84 in den AIO-Saeulen und 28 im Ende des Buegelrohrs (Pfad z = 1.5 mm, Radius 1.6 mm, y = 47..56); 76 Optimiererzellen, alle im Untergurt. Die Motorpads (z = 6.67..9.33) und ihre Schraubkorridore (graue Kreuze) beruehren den Boden nicht; dort wirkt bereits die Armwurzel-Korrektur.

### Vorgaben auf das Lochbild verkleinert: AIO-Sitzaugen

- `aio_contact_*` ist jetzt ein **Sitzauge** um jede Stack-Schraube statt einer Saeule ab Boden: r = 3.1 mm (2.0 mm Wand um die 2.2-mm-Bohrung), Oberkante unveraendert auf der Grommet-Sitzebene `base_thickness_mm + aio_standoff_mm` (plus 0.5 mm Buendigverlaengerung in die AIO-Huelle), Hoehe mindestens `aio_eye_height_mm` = 8/3 mm; die Unterkante liegt auf der naechsten Gitterknotenebene darunter. Auf dem 4/3-mm-Gitter des Laufs: z = 8.0..11.8 mm statt 0..11.8 mm. Auf dem groben 4-mm-Testgitter mit Standoff 3 mm reicht das Auge bis z = 0 (keine Ebene dazwischen).
- Darunter liegt `aio_tool_access_*`: ein rasterisierter Keep-out-Zylinder r = `aio_tool_radius_mm` = 2.0 mm (Schraubenkopf r 1.9 mm aus der Evaluator-Pruefung `tool_reachable` plus 0.1 mm) vom Boden bis zur Augenunterseite. Die Augenunterseite ist die Kopfauflage; Buendig- und Randregeln gelten fuer diesen Korridor nicht, er wird deshalb erst nach `prescribed_clearance` angehaengt. Die durchgehende Bohrung `aio_screw_*` reicht bis zum Boden.
- Wie das Auge getragen wird (von der Seite, als Rohr um den Korridor, ueber Streben), entscheidet der Optimierer. Die Rekonstruktion v3b vereinigt jede Preserve-Primitive generisch mit festem Uebergang (`smooth_union` mit `preserve_blend_mm`, `aio_contact` steht in `load_path_mounts`); es ist keine Aenderung an v3b noetig.
- **Lagerung verschoben:** Die Stackfixierung (`arm_tip`, `thrust_all`, `crash_*` im Domaenenaufbau, Evaluator `center_fixtures`) fasst jetzt die vier Augen ueber ihre Hoehe (Quader r 3.2 mm, z = Augenunterseite - 0.01 bis Oberkante + 0.01; im Lauf 7.99..11.81 statt -0.01..0.01). Die Lagerung wandert damit um rund 9 mm nach oben. Die Evaluator-Lochbildpruefung `stack_25.5` schneidet 1.5 mm ueber der Augenunterseite (bisher fest z = 1.5, fuer Saeulen ab Boden identisch).
- Der Optimierer selbst ist davon nur ueber die Anbindungslasten und die Punktmassen betroffen (`R2Domain` rechnet `arm_tip`, `thrust_all` und `crash_*` mit Inertia Relief ohne Fixierung). Die Sigma-Kalibrierung 1.384 / 1.374 ist mit der alten Lagerung gemessen und muss auf dem neuen Rahmen neu bestimmt werden.

Domaenendiff auf dem Laufgitter (`docs/validation/bottom_domain/eyes/floor.json`, Feld wie oben): Preserve -1195 mm3 (5547 -> 4352 mm3; je Auge 398 -> 100 mm3 Raster, 356 -> 115 mm3 exakt), erlaubter Raum -626 mm3 (die vier Werkzeugkorridore). In der Bodenschicht bleiben 28 Preserve-Zellen (Buegelrohrende) und die alten Saeulenfusszellen werden frei.

## Dichtefeld zu pruefbarer freier Geometrie

`deep_frame.topology_geometry.reconstruct_topology(domain, density, settings)` verwendet ausschliesslich das raeumliche Dichtefeld, die Masken und deklarierte generische Anschluss-/Freiraum-Primitiven. Es importiert keinen v0-Frameaufbau. `validate_topology` prueft den erzeugten Koerper vor der unabhaengigen gmsh/CalculiX-Auswertung.

Die voreingestellte Dichteschwelle0.4 wird auf zellzentrierte xyz/C-Daten angewendet. Preserve-Zellen muessen Dichte1 besitzen, verbotene Zellen0. Zusammenhang bedeutet sechs flaechige Nachbarn. Optionale Entfernung lastfreier Inseln ist standardmaessig aus; aktiviert entfernt sie nur Komponenten ohne irgendeinen Preserve-Anschluss und protokolliert Anzahl und Zellzahl. Getrennte notwendige Anschluesse sind immer Fehler. Es werden keine Bruecken, Arme oder Rippen ergaenzt.

`repair_manifold_voxels=True` aktiviert optional eine lokale Aufloesung diagonaler Kanten-/Punktkontakte in bereits sechsfach verbundenen Feldern. Bei einer mehrdeutigen2x2-Belegung wird die erlaubte angrenzende Leerezelle mit groesster urspruenglicher Dichte gefuellt. Dreidimensionale Punktkontakte werden ebenso lokal behandelt. Die Reparatur darf kein Forbidden-Voxel beruehren und stoppt bei `maximum_repair_voxels` (Default5 Prozent der belegten Zellen). Jede hinzugefuegte Zelle speichert Index, Anlass, urspruengliche Dichte, binaren Vorher-/Nachherzustand und hinzugefuegtes Volumen. Getrennte Pflichtanschluesse werden vor diesem Schritt abgelehnt; die Operation zeichnet keine Verbindungswege zwischen ihnen. Masse und saemtliche geometrischen/mechanischen Gates gelten danach unveraendert.

Ein deterministisches Greedy-Verfahren zerlegt belegte Zellen in disjunkte achsparallele Quader. Sechs Achsenreihenfolgen werden durchgerechnet; die mit den wenigsten Quadern wird verwendet. OpenCascade vereinigt diese Quader. Exakte Preserve-Primitiven werden hinzugefuegt und exakte Forbidden-Primitiven inklusive Subvoxel-Bohrungen ausgeschnitten. Ein abschliessender Schnitt begrenzt alles auf den Gitterbauraum. Der Report speichert Dichteschwelle, Zell-/Quaderzahl, Raster- und Endvolumen, Volumenaenderung durch die exakten Operationen, Laufzeit und jede allgemeine Reparatur.

Das bewusst stufige Ergebnis vermeidet eine heimliche Glaettung der optimierten Materialverteilung. Marching Cubes wurde als Alternative geprueft: die [offizielle scikit-image-Dokumentation](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) beschreibt Isosurface-Triangulierung und deren topologische Eigenschaften. Eine weitere Rueckfuehrung vieler Dreiecke in robuste CAD-Flaechen und exakte Schraubanschluesse waere erforderlich. Fuer4-mm-Demonstratoren wird deshalb die volumenexakte Quadervereinigung mit [build123d/OpenCascade-Booleans](https://build123d.readthedocs.io/en/latest/direct_api_reference.html) verwendet; das ist keine glatt optimierte Endproduktion.

### Geometrie und Fertigung

Pflichtpruefungen sind genau ein gueltiger geschlossener CAD-Solid sowie ein orientiertes, wasserdichtes, zusammenhaengendes trianguliertes Volumen. Jeder exakte Freiraum muss bis auf1e-5 mm3 leer sein. Jeder Preserve-Koerper abzueglich ausdruecklicher Bohrungen/Freiraeume muss vollstaendig vorhanden bleiben.

Freie Materialzellen sind mindestens eine Gitterweite breit. Der diskrete Feature-Screen verlangt, dass die kleinste Gitterweite mindestens die deklarierte Feature- und Duesenbahngrenze erreicht. Ein feineres Gitter wird ohne zusaetzlichen Feature-Nachweis abgelehnt. Rasterisierte Hardware-Freiraeume duerfen belegte Voxels nicht nachtraeglich anschneiden; eine nichtkonservative Maske wird vor der Rekonstruktion abgelehnt. Subvoxel-Bohrungen und Strap-Schlitze werden separat an exakten Anschlussprimitiven ueber Bohrung-Rand-, Bohrung-Bohrung- und Schlitz-Rand-Netzstege geprueft.

Nach allen Booleans folgt ein weiterer Wandstaerkenscreen direkt am CAD. Jede Flaeche wird an neun UV-Punkten (0.2/0.5/0.8 je Richtung) geprueft; Punkte auf Trimkanten werden ausgelassen, weil ein entlang einer Kante verlaufender Strahl keine eindeutige Materialdicke misst. Liegen alle UV-Punkte ausserhalb der beschnittenen Flaeche, wird der Schwerpunkt ihres groessten Dreiecks exakt auf die CAD-Flaeche projiziert. Ein wiederverwendeter OpenCascade-Intersektor liefert den ersten positiven Schnitt des nach innen gerichteten Normalenstrahls. Nicht aufloesbare Flaechen sind Fehler. Die Ausgabe nennt Strahlzahl, kleinste Dicke, duenne Samples und ungepruefte Flaechen. Dies ist eine endliche geometrische Stichprobe, kein globaler Mindestdickenbeweis zwischen den Abtastpunkten. Ein adversarialer exakter Boxschnitt mit0.5-mm-Reststeg scheitert ausdruecklich an dieser Pruefung.

Die lokale Anschlusspruefung misst mittlere Materialquerschnitte in0.02-mm-Scheiben unmittelbar ausserhalb der sechs AABB-Grenzflaechen jedes Preserve-Koerpers und summiert die vorhandenen Verbindungsflaechen. Regionspezifische Mindestflaechen und Wandstaerken gehen vor pauschalen Annahmen. Diese lokale Pruefung ist kein globaler minimaler Lastpfadquerschnitt und keine Slicer-Simulation. Sie verhindert eine alleinige nominelle Flaechenbehauptung ohne Kontakt zur rekonstruierten Struktur. Einteilige CAD-Topologie ersetzt keine Festigkeitspruefung.

Abwaerts gerichtete Flaechen loesen konservativ Supportbedarf aus. Mehrere geschlossene CAD-Shells weisen eingeschlossene Hohlraeume nach; ein sechsfach verbundener Flood-Fill des gepolsterten Voxel-Aussenraums prueft zusaetzlich die grobe Zugaenglichkeit der Leerzellen. Eingeschlossene Hohlraeume sind ungueltig. Enge exakte Durchgaenge, Werkzeugreichweite und tatsaechliche Supportentfernung bleiben Grenzen dieses endlichen Screens. Der Demonstrator erlaubt Support, verwendet Aufbauachse+z und behauptet keinen supportfreien Druck. Reale Extrusionsbreite, Rastertreppen, anisotrope Festigkeit, Feuchte, lokale Kerben und Montagepassung bleiben experimentell zu pruefen.

### Unabhaengiger Volumenmesher

Bestehende FEA-Defaults bleiben erhalten. Zwei ausdrueckliche Optionen stehen zusaetzlich zur Verfuegung: `mesh_second_order_linear=True` erzeugt geradlinige Mittelknoten von C3D10-Tetraedern; gekruemmte CAD-Raender sind damit durch die linearen Randdreiecke angenaehert. `mesh_high_order_optimize`0..4 aktiviert explizit die von [Gmsh dokumentierten Hochordnungsoptimierer](https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options). Beide Parameter werden im Netzmetadatum gespeichert. Der Positivitaetscheck aller `minDetJac`-Werte, Volumenkonnektivitaet, Last-/Fixture-Selektion und CalculiX-Fehlerchecks bleiben aktiv. V0 und freier Kandidat verwenden bei Vergleichen immer dieselben Einstellungen.

Beide Optionen bestehen den echten analytischen Kragbalkenvergleich fuer Verschiebung und erste Eigenfrequenz mit3-Prozent-Toleranz. Ein unabhaengiges synthetisches3D-Schleifenfeld mit306 Voxeln wurde automatisch zu sechs Quadern und einem geschlossenen Solid rekonstruiert, als STEP/STL exportiert und erfolgreich mit gmsh/CalculiX statisch und modal bewertet. Dieser Methodentest ist noch kein Erfolg einer freien SIMP-Frameoptimierung.
