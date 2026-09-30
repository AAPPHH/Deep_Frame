# Freier Designraum und verbleibende Formannahmen

`deep_frame.topology_domain.build_design_domain(parameters)` erzeugt aus einem einfachen Parameter-Dict die freie Ausgangsdomaene. `topology_config.py` ergaenzt die bestehenden Komponenten-, Material- und Integrationsparameter. Der Adapter ruft weder `build_frame` noch `build_geometry` auf. Aenderungen an v0-Armbreite, Armwurzel, Zentralplatte oder Deckfenstern beeinflussen seine Masken nicht; ein Test setzt diese v0-Parameter absichtlich auf geometrisch unbrauchbare Werte.

Der Startbauraum ist ein einziger Quader von x = -68 bis 68 mm, y = -64 bis 64 mm und z = 0 bis 32 mm. Er umfasst auch Material zwischen Motoren, hinter der Elektronik und oberhalb der ueblichen Arme. Es gibt keine vorgeschriebenen Arme, keine zentrale Platte, keine Deckwaende, keine Kamerakaefigwaende und keinen Hecksteg. Der Optimierer kann innerhalb dieses Raums durchgehende Flaechen, schraege Pfade, Verzweigungen, Rippen und unterschiedliche Querschnitte erzeugen. Motor- und Komponentenpositionen sowie die aeusseren Bauraumgrenzen bleiben in diesem Lauf fest.

## Quantifizierte Freiheit

Die Standarddiskretisierung besitzt 34 x 32 x 8 Zellen mit 4 mm Kantenlaenge. Zellen sind in der Reihenfolge xyz und C-order gespeichert; z laeuft beim Flattening am schnellsten.

| Menge | Zellen | Rastervolumen |
| --- | ---: | ---: |
| Gesamter Quader | 8704 | 557056 mm3 |
| Zulaessiger Materialraum | 5918 | 378752 mm3 |
| Feste lokale Anschluesse | 162 | 10368 mm3 |
| Frei optimierbare Zellen | 5756 | 368384 mm3 |
| Gesperrt | 2786 | 178304 mm3 |

Die Preserve-Zellen belegen **2.7374 Prozent** des zulaessigen Raums. **97.2626 Prozent** bleiben freie Optimierungsvariablen. Diese Angaben betreffen das Raster; exakte Kontaktgrenzen und Bohrungen aendern das physische Endvolumen. Die Preserve-Maske enthaelt mehrere getrennte Komponenten und kann allein keinen tragenden Frame bilden.

`volume_fraction = 0.10` begrenzt das SIMP-Dichtevolumen auf 37875.2 mm3, entsprechend 41.284 g bei 1.09 g/cm3. Dies ist kein vorweggenommenes Ergebnisgewicht: Dichteschwelle, exakte lokale Anschluesse, Ausschnitte und Qualitaetspruefungen bestimmen das tatsaechliche Gewicht. Die zugehoerigen Werte und die tatsaechlichen Masken werden je Lauf gespeichert.

## Feste funktionale Anschluesse

- Vier lokale Motorkontakte: 9.5 mm Radius, Oberkante an der Motorunterseite. M2-Bohrungen und Wellenfreigang werden exakt ausgeschnitten. Es gibt keine vorgeschriebenen Verbindungen zu einem Zentrum.
- Vier AIO15-Bosse: 3.2 mm Radius, Oberkante 5.5 mm, Bohrungen gemaess dem bestehenden 25.5-mm-M2-Lochbild. Der umgebende Boden bleibt frei.
- Vier einzelne Akku-Auflageflaechen sowie zwei kurze Koppelflaechen an der Querlinie y = 0. Ihre Oberseite liegt bei z = 29 mm. Die Koppelflaechen dienen derselben 37-g-Akku-Punktmasse wie im Vergleichsmodell.
- Vier lokale Strap-Oesen mit 2-mm-Rand, deren Schlitze seitlich neben der 30-mm-Akkubreite liegen. Durch sie laeuft der Strap ueber die breite Akkuflaeche. Die tragende Verbindung der Oesen wird optimiert.
- Zwei kleine Kamera-Schraublaschen und zwei obere Schutz-/Lastkontaktflaechen. Dazwischen ist weder eine Kaefigwand noch ein Verbindungsbogen vorgeschrieben.
- Lokale XT30- und Balancer-Auflagen mit zugaenglichen Enden fuer die Befestigung durch Band oder Kleber sowie eine VTX-Antennenaufnahme mit 3-mm-Bohrung. Befestigungsmethode, Steckerbelegung und Kabelradien sind vorlaeufige Montageannahmen.

Diese lokalen Primitiven sind menschliche Formannahmen fuer die notwendigen Schnittstellen. Sie duerfen nicht als frei optimierte Geometrie ausgegeben werden. Die tragenden Verbindungen zwischen ihnen entstehen aus dem freien Materialfeld.

## Verbotene Volumina und Montage

Die vorhandenen parametrisierbaren Komponenten-Platzhalter liefern die Bauraumhuellen fuer AIO15, Lux-Kamera mit Tilt, GNB5502S120A, vier Motoren, Props, XT30 und Balancer. Hardware erhaelt 0.5 mm Zusatzfreiraum; geplante untere Auflageflaechen behalten Kontaktabstand null ohne positives Durchdringungsvolumen. Props erhalten radial und axial 2 mm Freiraum. Die 65-mm-Scheibe bleibt die bewusst konservative Nutzervorgabe.

Zusaetzlich gesperrt werden Batterieentnahme nach oben, AIO-Einschub von rechts bei abgesteckten Kabeln, Kameraeinbau/Linsenkorridor nach vorne, Steckerentnahme nach oben, vorlaeufige Motor-/Balancer-Kabelkorridore und Schraub-/Antennenbohrungen. Diese Korridore sind nachvollziehbare Montageannahmen, keine vollstaendige Simulation aller Werkzeuge, biegsamen Kabel oder des Kamera-Sichtfeldes.

Eine freie Rasterzelle wird konservativ entfernt, sobald ihr Quader eine Hardware-/Prop-/Zugangshuellenflaeche mit positivem Volumen schneidet. Reine Beruehrung einer Auflageflaeche entfernt die darunterliegende Zelle nicht. Damit schneidet die exakte Rekonstruktion keine zufaellig duennen Resthaeute aus freien Hardware-Randzellen. Subvoxel-Bohrungen und Strap-Schlitze haben `rasterize: false` und werden zwingend geometrisch ausgeschnitten.

Preserve-Ausschnitte sind fuer Bohrungen und Oesenschlitze ausdruecklich mit `allow_preserve_subtraction` gekennzeichnet. Potenzielle Paare werden konservativ anhand positiver AABB-Ueberdeckung protokolliert; die exakte CSG-Differenz ist das geometrische Soll. Eine neue, nicht deklarierte Preserve-/Forbidden-Ueberdeckung wird vor der Optimierung abgelehnt, statt die Kontaktflaeche still zu verlieren.

## Vergleichbare Lasten und Anschlussnachweis

Der Adapter uebernimmt die bisherigen Armspitzen-, 10-g-Akku-Aufprall-, seitlichen Kameralast- und Modalannahmen. Fuer den Armspitzenfall werden ausschliesslich die Unterseiten der vier AIO-Montagekontakte geklemmt: vier Boxselektoren um x/y = +/-12.75 mm mit je 3.2 mm Halbausdehnung und z = +/-0.01 mm. Der Optimierer kann dadurch keine willkuerlich grosse freie Bodenflaeche auf eine breite numerische Klemme aufbauen.

Diese vier Faelle stehen in `comparison_load_cases` und muessen unveraendert auf **v0 und freie Struktur** angewandt werden. Alte v0-Ergebniswerte mit breiter Zentralklemme sind keine gueltigen Vergleichswerte fuer diesen Lauf. Alle anderen Faelle fixieren weiterhin die vier Motor-Unterseiten. Der Akku wird identisch an die Querflaeche bei z = 29 mm gekoppelt; die Punktmasse liegt bei [0, 0, 34.5] mm.

Im internen Hex8-Adapter verwendet dieser Frame-Designraum ausdruecklich `interface_node_policy: preserve_adjacent`: Lasten und Fixierungen treffen nur Knoten an festen Kontaktzellen. Eine um bis zu eine halbe Zellenweite erweiterte Auswahl darf dadurch keine beliebigen freien Materialknoten im breiten geometrischen Selektor belasten. Selektorerweiterung und Policy werden gespeichert. Die abschliessende FEA verwendet weiterhin die exakten Kontaktflaechen am rekonstruierten Solid.

17 weitere Faelle belasten sonst nicht direkt belastete Pflichtanschluesse einzeln mit 0.05 N nach unten. Sie verhindern, dass lastfreie Montageschnittstellen als unverbundene Inseln verbleiben. In der normierten Compliance-Zielfunktion besitzen die drei Hauptfaelle zusammen 90 Prozent Gewicht, alle 17 Anschlussfaelle zusammen 10 Prozent. Eine kleine Kraft alleine wuerde nach Compliance-Normierung das Gewicht nicht verringern. Der separate geometrische Nachweis der Verbindung aller Pflichtanschluesse bleibt notwendig.

## Fertigung und Quellen

Die vorlaeufige Fertigungsgrenze betraegt 2 mm, abgeleitet aus 0.4-mm-Duese und mindestens fuenf Bahnen. Exakte Bohrungsumrandungen und Anschlussquerschnitte werden zusaetzlich geprueft. Stuetzmaterial ist erlaubt; Supportfreiheit, bestimmte Druckfestigkeit oder nacharbeitsfreie Montage werden nicht behauptet. Die Fertigungspruefung ist ein geometrisches Screening und ersetzt keinen Druckversuch.

Die feste Hardwareanordnung und ihre Quellen bleiben in `component_defaults.py`, `frame_defaults.py` und `docs/armattan_research.md` nachvollziehbar. Die Motorhuelle stammt vorlaeufig vom [GEPRC GR1105](https://geprc.com/product/gep-gr1105-motor/); sein 9-mm-Lochkreis ist kein 9x9-mm-Quadrat. Dichte und isotrope Materialannahme bleiben der dokumentierte [Bambu-PA6-CF-Datensatz](https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf). Rasteraufloesung, Kontaktgroessen, Bauraum, Korridore und relative Anschlusslasten sind eigene offengelegte Modellierungsentscheidungen.
