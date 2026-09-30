# Phase 1: freie klassische 3D-Topologieoptimierung

Die automatische Pipeline hat eine eigenstaendige tragende Struktur erzeugt, geometrisch geprueft und mit der vorhandenen gmsh/CalculiX-FEA gegen v0 verifiziert. Der Generator verwendet keine v0-Geometrie als Maske, Saat oder Verbindungsmodell. v0 bleibt unveraendert als Vergleichskoerper erhalten.

Abnahmelauf: `2cd7bcbc15b67fc7`, Kandidat `default_t00`, Status `ok`. Im abschliessenden Integrationsstand bestehen **124 Tests**.

![Feste Anschluesse, v0 und automatisch erzeugte Struktur im gleichen Massstab](validation/topology_phase1/geometry_comparison.png)

Links stehen nur die vorgeschriebenen lokalen Anschluesse. Rechts sind zusaetzliche Verzweigungen, rueckwaertige Verbindungen, raeumliche Deckabstuetzungen und ein zusaetzlicher Pfad am vorderen linken Motor sichtbar. Diese Verbindungen sind im linken Bild nicht vorgegeben. Die stufigen Oberflaechen entsprechen der gewaehlten 4-mm-Diskretisierung; sie wurden nicht manuell geglaettet oder nachmodelliert.

## Methode, Freiheit und Formannahmen

Die innere Optimierung verwendet dreidimensionale Hex8-Elastizitaet, SIMP mit Exponent 3, einen 6-mm-Dichtefilter, glatte Projektion und ein Optimality-Criteria-Update bei 10 Prozent Volumenbudget. Die freie Anfangsdichte ist ueberall gleich. Dieses etablierte, nachvollziehbare Verfahren unterstuetzt Mehrlastfaelle und aktive/passive Zellen; Quellen und Ableitungen stehen in [der Methodendokumentation](topology_method.md).

**5684 Zell-Dichten** bestimmen Material, Querschnitte, Verzweigungen und Verbindungen im gesamten zulaessigen 3D-Raum. 158 feste Zellen belegen nur **2.7046 Prozent** der 5842 erlaubten Zellen. Das volle Start-Cuboid misst 136 x 128 x 32 mm. Komponentenhuellen, Propeller-, Kabel-, Schraub- und Montagezugangsraeume schneiden verbotene Volumina daraus aus. [Domain und Quellen](topology_domain.md) legen alle Angaben offen.

Menschlich fest bleiben der Bauraum, 135-mm-Motorabstand und Komponentenpositionen, lokale Motor-/AIO-/Kamera-/Akku-/Strap-/Stecker-/Antennenkontakte, Lasten, Klemmungen, 2-mm-Mindestfeature und Druckrichtung. Es gibt keine vorgeschriebenen Arme oder tragenden Verbindungen. Die lokale Kameralaschenbreite von 6 mm passt zur Rasteraufloesung; reale Kamera- und Steckerbefestigung bleiben vorlaeufig.

Drei Hauptlastfaelle erhalten gemeinsam 90 Prozent des normierten Compliance-Ziels, 17 kleine Anschlusslastfaelle zusammen 10 Prozent. Die unabhaengige aeussere Bewertung beruecksichtigt Masse, Steifigkeit, maximale Verschiebung, Spannung und erste Eigenfrequenz gemeinsam durch feste Constraints und eine fuenfdimensionale Pareto-Auswahl. Modal- und Spannungswerte sind keine SIMP-Ersatzwerte: Sie stammen aus der erneut vernetzten exakten Geometrie.

## Verifizierter Vergleich

Beide Koerper verwenden dasselbe isotrope PA6-CF-Ersatzmaterial (1.09 g/cm3, E = 4430 MPa, nu = 0.30), dieselben Lasten, denselben 37-g-Akku am Schwerpunkt [0, 0, 34.5] mm und dieselben Netzeinstellungen. Der Armtest klemmt bei beiden die vier AIO-Unterseiten; die anderen Faelle klemmen die vier Motorunterseiten. Deshalb wurde v0 neu ausgewertet. Alte Zahlen mit breiter Zentralklemmung werden nicht vermischt.

| Kennwert | v0 | Freie Struktur | Vorab festgelegte Grenze |
| --- | ---: | ---: | --- |
| Frame-Masse | 32.1629 g | 58.5318 g | maximal 2 x v0 |
| Armsteifigkeit | 5.1638 N/mm | 194.3911 N/mm | mindestens 0.5 x v0 |
| Maximale Verschiebung | 0.255423 mm | 0.008063 mm | maximal 2 x v0 |
| Maximale Vergleichsspannung | 5.98350 MPa | 0.51748 MPa | maximal 2 x v0 |
| Erste Eigenfrequenz | 204.504 Hz | 1075.764 Hz | mindestens 0.7 x v0 |

Alle fuenf Grenzen sind erfuellt. Die freie Struktur ist **81.99 Prozent schwerer**, bei diesen Randbedingungen aber erheblich steifer. Eine leichtere Loesung oder globale Optimalitaet wird nicht behauptet. Das FEA-Modell umfasst Frame plus Akku; die tabellierte Frame-Masse enthaelt den Akku nicht. Kamera, AIO, Motoren und Props sind geometrische Randbedingungen, ihre Massen werden in dieser mechanischen Vergleichsauswertung nicht zusaetzlich angesetzt.

Der Lauf pruefte sechs automatische Dichteschwellen. Bei 0.20 entsteht der einzige gueltige und damit einzige Pareto-Kandidat `default_t00`. Die Schwellen 0.25 bis 0.50 trennen notwendige Anschluesse ab und werden vor der FEA mit gespeichertem Grund verworfen. Der akzeptierte Koerper besitzt genau einen geschlossenen Solid, ein wasserdichtes Netz, freie Hardwarebereiche und vollstaendige Pflichtanschluesse. 11 protokollierte lokale, dichtegeleitete Zellfuellungen beseitigen diagonale nichtmannigfaltige Kontakte; sie zeichnen keine Ersatzarme ein und verbinden keine zuvor getrennten Pflichtinseln.

Die finale FEA besitzt 60933 C3D10-Elemente und 110430 Knoten bei 3 mm Netzvorgabe; v0 besitzt 45491 Elemente und 83406 Knoten. Ein zusaetzlicher Test verdoppelt ausschliesslich die Akkumasse von 37 auf 74 g: Die erste Frequenz sinkt von 1075.764 auf **824.3202 Hz**, alle sechs Moden sinken. Damit ist die Punktmasse im Modalmodell wirksam. 64 Knoten des 4-mm-Querbands sind starr mit dem Akku-COM gekoppelt; dieses Modell unterdrueckt lokale Patchverformung und bildet keine intrinsische Akku-Rotationstraegheit um dessen Schwerpunkt ab.

## Grenzen und gespeicherte Daten

Der Lauf endet nach 45 Updates am Iterationslimit; formale Dichtekonvergenz wird nicht behauptet. Die normierte Compliance sinkt von 1 auf 0.0133265. 17.28 Prozent der freien Zellen bleiben zwischen Dichte 0.1 und 0.9. Schwellenbildung und Rekonstruktion erhoehen die Dichte-Modellmasse von 40.754 g auf die gemessenen 58.532 g. Feinere Raster, Projektionsfortsetzung und Netzkonvergenz bleiben weitere klassische Verbesserungen.

![Gespeicherte Dichtequerschnitte mit Pflichtanschluessen](validation/topology_phase1/density_slices.png)

Die Fertigungspruefung untersucht Konnektivitaet, Anschlussquerschnitte, exakte Bohrungs-/Schlitzstege und 4182 CAD-Normalenstrahlen: gemessene Mindestdicke 2.000 mm, keine fehlenden Flaechen oder geschlossenen Hohlraeume. Die endliche Flaechenabtastung ist kein Beweis der globalen Minimaldicke zwischen den Proben. Stuetzmaterial ist erforderlich und erlaubt; Zugangs-/Hohlraumchecks ersetzen keine Slicerplanung. Druckanisotropie, Feuchte, reale Hardwarepassung, transiente Einschlaege und reale Flugrandbedingungen sind noch nicht verifiziert. Der Vergleich ist linear-elastisch und nutzt feste Motor-/AIO-Klemmungen; die Frequenzen sind keine freifliegenden Gesamtcopter-Moden.

Die unveraenderte Armspitzenlast greift nur am vorderen linken Motor an, der Kameraschlag in einer seitlichen Richtung. Ohne Symmetrieauflage ist eine asymmetrische Struktur daher zu erwarten. Der bestandene Vergleich bestaetigt diese deklarierten Lastfaelle und keine allseitige Crashbestaendigkeit. Eine spaetere Lastfallfamilie muss alle relevanten Richtungen und Kontaktbedingungen explizit abbilden.

Der [maschinenlesbare Abnahmebeleg](validation/topology_phase1/acceptance.json) verlinkt den vollstaendigen lokalen Run und seine Hashes. Gespeichert werden Parameter und regionale Randbedingungen, xyz-Raster und Masken, Design-/physische Dichte als NPZ, Material, Punktmassen, Lastfaelle, Optimierungssettings, Seed, alle Iterationen, Rekonstruktionsaenderungen, gueltige und ungueltige Kandidaten, exakte STEP/STL-Geometrie, FEA-Eingaben/Netze/Logs/Ergebnisse, Pareto-Front, Paket-/Solverversionen und Quellcode-/Artefakthashes. Damit existieren positive und negative Beispiele fuer einen spaeteren Trainingsdatensatz. Deep Learning, Diffusion und PINNs werden nicht eingesetzt.

Der [Run-Manifest-Snapshot](validation/topology_phase1/run_manifest.json), [Felder](validation/topology_phase1/fields.npz), [Optimierungshistorie](validation/topology_phase1/optimization.json) und [Modal-Punktmassentest](validation/topology_phase1_mass_sensitivity.md) sind als kompakte Belege versioniert. Die umfangreichen Solverdateien und STEP/STL-Exporte verbleiben unter `exports/topology/phase1/`; die vorhandene `.gitignore` wird eingehalten. Erneuter automatischer Lauf: `python run_topology.py`. Die [Pipeline-Dokumentation](topology_pipeline.md) beschreibt Wiederaufnahme, Artefaktpruefung und austauschbare Adapter.
