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

- Vier lokale Motorkontakte: 9.7 mm Radius (2.1 mm Rand um die 7.6-mm-Motorhuelle), Oberkante 0.5 mm in die Motorhuelle hinein; der exakte Schnitt legt die Auflage auf die Motorunterseite z = 4 mm. M2-Bohrungen und Wellenfreigang werden exakt ausgeschnitten. Es gibt keine vorgeschriebenen Verbindungen zu einem Zentrum.
- Vier AIO15-Bosse: 3.2 mm Radius, Oberkante buendig bei 5.5 mm (nur 0.2 mm neben der AIO-Huellwand, deshalb nicht verlaengert), Bohrungen gemaess dem bestehenden 25.5-mm-M2-Lochbild. Der umgebende Boden bleibt frei.
- Vier einzelne Akku-Auflageflaechen sowie zwei kurze Koppelflaechen an der Querlinie y = 0. Ihre Oberseite liegt 0.5 mm in der Akkuhuelle; die exakte Auflage bleibt bei z = 29 mm. Ausserhalb der Akkuhuelle (|x| > 15.5 mm) stehen die Koppelflaechen dadurch 0.5 mm hoeher. Die Koppelflaechen dienen derselben 37-g-Akku-Punktmasse wie im Vergleichsmodell.
- Vier lokale Strap-Oesen mit 2.1-mm-Rand (`strap_rim_mm`), Oberkante wie die Akkuauflagen 0.5 mm ueber z = 29 mm, deren Schlitze seitlich neben der 30-mm-Akkubreite liegen. Durch sie laeuft der Strap ueber die breite Akkuflaeche. Die tragende Verbindung der Oesen wird optimiert.
- Zwei kleine Kamera-Schraublaschen (Radius 4.1 mm, 2.1-mm-Rand um den 2-mm-Werkzeugkorridor) und zwei obere Schutz-/Lastkontaktflaechen. Ihre lokale Breite von 6 mm erreicht die Rastergrenze x = +/-16 mm; dies verhindert subvoxelduenne Reste zwischen Schraubbohrung und freiem Material. Die Breite ist eine eigene Diskretisierungs-/Montageannahme, keine kopierte v0-Geometrie. Dazwischen ist weder eine Kaefigwand noch ein Verbindungsbogen vorgeschrieben.
- Lokale XT30- und Balancer-Auflagen mit zugaenglichen Enden fuer die Befestigung durch Band oder Kleber (Oberkante 0.5 mm in der Steckerhuelle, Enden ausserhalb dadurch 3.0 statt 2.5 mm hoch) sowie eine VTX-Antennenaufnahme mit 3-mm-Bohrung und 4.3 mm Aussenradius (`antenna_eyelet_radius_mm`), die die Balancer-Auflagenecke einschliesst. Befestigungsmethode, Steckerbelegung und Kabelradien sind vorlaeufige Montageannahmen.

Diese lokalen Primitiven sind menschliche Formannahmen fuer die notwendigen Schnittstellen. Sie duerfen nicht als frei optimierte Geometrie ausgegeben werden. Die tragenden Verbindungen zwischen ihnen entstehen aus dem freien Materialfeld.

### Preserve-Geometrie gegen Keep-outs

Die implizite Route blaeht jede Preserve-Primitive um `preserve_inflation_mm` = 0.3 mm auf und schneidet Keep-outs danach exakt. Liegt eine Preserve-Flaeche buendig auf einer Keep-out-Flaeche, trifft dieser Schnitt die Kappe der aufgeblaehten Schale; uebrig bleiben 0.3 mm hohe Stufen und Schneiden mit Wandmessungen von 0.0005 bis 0.7 mm (Akkuauflagen z = 29, XT30/Balancer z = 2.5, Motorkappen z = 4). Die Wandpruefung bleibt streng; korrigiert wird die vorgegebene Geometrie:

- Buendige horizontale Kontakte (Preserve-Oberseite = Keep-out-Unterseite bei gemeinsamer Grundflaeche) werden um `flush_overlap_mm` = 0.5 mm in das Keep-out verlaengert und dort als Preserve-Ausschnitt deklariert, ausser die Preserve-Primitive liegt naeher als Aufblaehung plus Reserve (0.4 mm) an einer Seitenwand des Keep-outs; dann wuerde die Schale als Splitter austreten, der Kontakt bleibt buendig und steht in `prescribed_clearance.flush_kept_near_wall` (AIO-Bosse). Der exakte Schnitt laeuft dann durch die senkrechte Schalenwand; die wirksame Auflageebene bleibt unveraendert. Keep-outs innerhalb einer koaxialen vorgeschriebenen Bohrung (Antennen-Einfuehrkanal) zaehlen nicht. Die Liste steht in `metadata.flush_contact_extensions`.
- Jede Keep-out-Wand, die eine Preserve-Primitive schneidet, und jedes koaxiale Keep-out-Zylinderpaar muss einen Rand von mindestens 2.0 mm plus `prescribed_wall_margin_mm` = 0.1 mm lassen (Polygon- und Float-Reserve).
- Zwei Preserve-Primitiven ueberlappen oder liegen weiter als dieser Rand plus beide Aufblaehungen auseinander.

`prescribed_clearance` prueft diese Regeln nach der Verlaengerung und speichert das Ergebnis in `metadata.prescribed_clearance`; die Standarddomaene muss sie bestehen. Vorgeschriebene Bohrungen sind ausgenommen, ihre Stege regelt die C6-Bohrungstoleranz. Die Ebene z = 0 ist Bauraumgrenze, kein Keep-out, und bleibt unveraendert. Messung auf den gespeicherten Dichten: `docs/validation/implicit_domain_ledges.md`.

### Verbotene Volumina und Montage

Die vorhandenen parametrisierbaren Komponenten-Platzhalter liefern die Bauraumhuellen fuer AIO15, Lux-Kamera mit Tilt, GNB5502S120A, vier Motoren, Props, XT30 und Balancer. Hardware erhaelt 0.5 mm Zusatzfreiraum; geplante untere Auflageflaechen behalten Kontaktabstand null ohne positives Durchdringungsvolumen. Props erhalten radial und axial 2 mm Freiraum. Die 65-mm-Scheibe bleibt die bewusst konservative Nutzervorgabe.

Zusaetzlich gesperrt werden Batterieentnahme nach oben, AIO-Einschub von rechts bei abgesteckten Kabeln, Kameraeinbau/Linsenkorridor nach vorne, Steckerentnahme nach oben, vorlaeufige Motor-/Balancer-Kabelkorridore und Schraub-/Antennenbohrungen. Diese Korridore sind nachvollziehbare Montageannahmen, keine vollstaendige Simulation aller Werkzeuge, biegsamen Kabel oder des Kamera-Sichtfeldes.

Die Kamera-Schraubbohrung endet exakt an den lokalen Laschen. Ausserhalb davon ist ein 4-mm-Werkzeugkorridor konservativ rasterisiert. Der Antennenkanal reicht durch den gesamten Bauraum nach oben; oberhalb der festen Oese wird auch er konservativ rasterisiert. Damit kann weder ein mathematisch vorhandener Schraubenkanal eine duenne freie Resthaut erzeugen noch eine kurze Antennenbohrung unter einer spaeter hinzukommenden Rasterzelle blind enden.

Eine freie Rasterzelle wird konservativ entfernt, sobald ihr Quader eine Hardware-/Prop-/Zugangshuellenflaeche mit positivem Volumen schneidet. Reine Beruehrung einer Auflageflaeche entfernt die darunterliegende Zelle nicht. Damit schneidet die exakte Rekonstruktion keine zufaellig duennen Resthaeute aus freien Hardware-Randzellen. Subvoxel-Bohrungen und Strap-Schlitze haben `rasterize: false` und werden zwingend geometrisch ausgeschnitten.

Preserve-Ausschnitte sind fuer Bohrungen und Oesenschlitze ausdruecklich mit `allow_preserve_subtraction` gekennzeichnet. Potenzielle Paare werden konservativ anhand positiver AABB-Ueberdeckung protokolliert; die exakte CSG-Differenz ist das geometrische Soll. Eine neue, nicht deklarierte Preserve-/Forbidden-Ueberdeckung wird vor der Optimierung abgelehnt, statt die Kontaktflaeche still zu verlieren.

### Vergleichbare Lasten und Anschlussnachweis

Der Adapter uebernimmt die bisherigen Armspitzen-, 10-g-Akku-Aufprall-, seitlichen Kameralast- und Modalannahmen. Fuer den Armspitzenfall werden ausschliesslich die Unterseiten der vier AIO-Montagekontakte geklemmt: vier Boxselektoren um x/y = +/-12.75 mm mit je 3.2 mm Halbausdehnung und z = +/-0.01 mm. Der Optimierer kann dadurch keine willkuerlich grosse freie Bodenflaeche auf eine breite numerische Klemme aufbauen.

Diese vier Faelle stehen in `comparison_load_cases` und muessen unveraendert auf **v0 und freie Struktur** angewandt werden. Alte v0-Ergebniswerte mit breiter Zentralklemme sind keine gueltigen Vergleichswerte fuer diesen Lauf. Alle anderen Faelle fixieren weiterhin die vier Motor-Unterseiten. Der Akku wird identisch an die Querflaeche bei z = 29 mm gekoppelt; die Punktmasse liegt bei [0, 0, 34.5] mm.

Im internen Hex8-Adapter verwendet dieser Frame-Designraum ausdruecklich `interface_node_policy: preserve_adjacent`: Lasten und Fixierungen treffen nur Knoten an festen Kontaktzellen. Eine um bis zu eine halbe Zellenweite erweiterte Auswahl darf dadurch keine beliebigen freien Materialknoten im breiten geometrischen Selektor belasten. Selektorerweiterung und Policy werden gespeichert. Die abschliessende FEA verwendet weiterhin die exakten Kontaktflaechen am rekonstruierten Solid.

17 weitere Faelle belasten sonst nicht direkt belastete Pflichtanschluesse einzeln mit 0.05 N nach unten. Sie verhindern, dass lastfreie Montageschnittstellen als unverbundene Inseln verbleiben. In der normierten Compliance-Zielfunktion besitzen die drei Hauptfaelle zusammen 90 Prozent Gewicht, alle 17 Anschlussfaelle zusammen 10 Prozent. Eine kleine Kraft alleine wuerde nach Compliance-Normierung das Gewicht nicht verringern. Der separate geometrische Nachweis der Verbindung aller Pflichtanschluesse bleibt notwendig.

### Fertigung und Quellen

Die vorlaeufige Fertigungsgrenze betraegt 2 mm, abgeleitet aus 0.4-mm-Duese und mindestens fuenf Bahnen. Exakte Bohrungsumrandungen und Anschlussquerschnitte werden zusaetzlich geprueft. Stuetzmaterial ist erlaubt; Supportfreiheit, bestimmte Druckfestigkeit oder nacharbeitsfreie Montage werden nicht behauptet. Die Fertigungspruefung ist ein geometrisches Screening und ersetzt keinen Druckversuch.

Die feste Hardwareanordnung und ihre Quellen bleiben in `COMPONENT_DEFAULTS`, `FRAME_DEFAULTS` und `FRAME_DEFAULT_SOURCES` von `deep_frame/config.py` sowie in [armattan_research.md](armattan_research.md) nachvollziehbar. Die Motorhuelle stammt vorlaeufig vom [GEPRC GR1105](https://geprc.com/product/gep-gr1105-motor/); sein 9-mm-Lochkreis ist kein 9x9-mm-Quadrat. Dichte und isotrope Materialannahme bleiben der dokumentierte [Bambu-PA6-CF-Datensatz](https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf). Rasteraufloesung, Kontaktgroessen, Bauraum, Korridore und relative Anschlusslasten sind eigene offengelegte Modellierungsentscheidungen.

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
