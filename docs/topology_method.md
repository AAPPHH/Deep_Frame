# Freie 3D-Strukturverteilung mit SIMP

Der Generator optimiert eine skalare Materialdichte fuer jede freie dreidimensionale Zelle. Er beginnt mit einer raeumlich konstanten freien Dichte. Weder die v0-Geometrie noch Armpfade, Kreuzkonturen, Skelettlinien oder von Hand entworfene Verbindungsrippen werden eingelesen. Lastpfade und Verzweigungen entstehen durch das Elastizitaetsproblem im erlaubten Volumen. Die Dichte ist richtungsunabhaengiges isotropes Material, kein orthotrop orientierbares Mikrostrukturmodell.

Die Methode folgt dem etablierten SIMP-Ansatz mit raeumlichem Dichtefilter, Optimality-Criteria-Update und Volumenbeschraenkung. Dreidimensionale Mehrlastprobleme, aktive/passive Zellen und die prinzipielle Methode beschreibt [Liu und Tovar (2014), Originalartikel im Hochschulrepositorium](https://scholarworks.indianapolis.iu.edu/items/a2165e92-8941-46ca-9cc7-c59cefa4fe96). Die Herleitung effizienter Dichtefilter und des OC-Updates ist in [Andreassen et al. (2011), DTU-Publikationsnachweis](https://orbit.dtu.dk/en/publications/efficient-topology-optimization-in-matlab-using-88-lines-of-code/) dokumentiert. Der Python-Code ist eine eigene Implementierung der Formeln mit numerischer Gaussintegration und uebernimmt keinen fremden Programmtext.

## Diskretisierung und Randbedingungen

Jede regulaere Hex8-Zelle besitzt acht Knoten mit je drei translatorischen Freiheitsgraden. Die Steifigkeitsmatrix folgt aus `integral(B.T D B dV)` mit trilinearen Ansatzfunktionen und acht Gauss-Punkten. Die volle dreidimensionale isotrope Elastizitaetsmatrix umfasst Normal- und Schubverformung. Die Elementmatrix ist nicht aus einem zweidimensionalen Balkenmodell extrudiert. Der Solver benutzt mm, N und MPa; die separate analytische Modalverifikation verwendet eine konsistente Massenmatrix und Tonnen (`g/cm3 * 1e-9`).

Alle statischen Lastfaelle des Domain-Dicts werden beruecksichtigt. Eine angegebene Last ist die Gesamtkraft und wird gleichmaessig auf die gewaehlten Knoten verteilt. Faelle mit derselben Einspannung teilen eine sparse Faktorisierung; ihre rechten Seiten werden zusammen geloest. Der normalisierte lineare Loesungsrest muss kleiner als `1e-4` bleiben. Das ist eine numerische Fehlerschranke, keine Aussage ueber die Genauigkeit des groben Netzes.

Knoten, die ausschliesslich an verbotene Zellen grenzen, werden weder belastet noch als freie Solver-Freiheitsgrade benutzt. Ihre Entfernung aus Selektoren steht in `summary.system.selector_filtering`. Verbotene Zellen tragen keine Ersatzsteifigkeit zur Systemmatrix bei. Ein an einem duennen physikalischen Selektor vorbeilaufendes Gitter wird nicht durch Null-Lasten toleriert: Nur wenn kein Knoten mit erlaubter Materialnachbarschaft enthalten ist, darf der interne Adapter den Selektor auf jeder Seite um genau eine halbe Zellweite erweitern. Jede Erweiterung und ihre Knotenzahl steht in `summary.system.selector_expansions`. Die unabhaengige geometrische FEA benutzt die urspruenglichen exakten Selektoren. Diese Diskretisierungsdifferenz ist eine bekannte Ersatzmodellabweichung.

`interface_node_policy: preserve_adjacent` beschraenkt alle Last- und Einspannselektoren zusaetzlich auf Knoten mit mindestens einer angrenzenden Preserve-Zelle. Damit bleiben bei breiten Selektorboxen nur die funktionalen Anschluesse als Kraftuebergangsflaechen uebrig. Entfernte freie Nachbarknoten werden separat protokolliert. Die Frame-Domain waehlt diese Policy; der generische Solver erlaubt mit `allowed_adjacent` auch Probleme ohne festes Interface-Maskenschema. Beide benutzen die gesamte vorgegebene Kraft auf den verbleibenden Knoten.

## Dichte, Filter und Optimierung

Die SIMP-Interpolation ist `E(rho) = E0 * (epsilon + (1-epsilon) * rho**p)` mit standardmaessig `p=3` und `epsilon=1e-6`. Die schwache Ersatzsteifigkeit verhindert singularitaetsbedingte Abbrueche in beinahe leeren erlaubten Materialzellen. Sie ist kein erlaubter physischer Werkstoff: Rekonstruktion, Connectivity und finale FEA entfernen diese Ersatzstruktur. Verbotene Zellen werden vollstaendig aus der Steifigkeitsassembly entfernt.

Der Filter verwendet lineare Abstandgewichte `max(0, r-distance)` zwischen erlaubten Zellmittelpunkten. Verbotene Zellen werden nicht zum Nachbarschaftsnenner addiert. Auf die gefilterte Dichte folgt eine optionale glatte tanh-Projektion. Preserve-Dichten sind danach exakt 1 und verbotene Dichten exakt 0. Bei beiden sind Ableitungen null; Filter und Projektion werden mit der Kettenregel rueckwaerts differenziert. Der Filter ist eine Regularisierung; er garantiert allein noch keine druckbare Mindestwand.

Das innere Ziel ist die gewichtete Summe der Compliance `F.T U`, jeweils normiert auf den Wert bei der gleichen uniformen Anfangsverteilung. Standardmaessig erhalten alle statischen Faelle dasselbe Gewicht. Auch deklarierte kleine Anschlusskraefte bleiben damit in der Topologiesynthese wirksam. Ihre physikalische Lastgroesse ist im abschliessenden Solver unveraendert; die Gewichtung ist eine ausdrueckliche technische Priorisierung, keine zusaetzliche physische Kraft.

Die Volumenbeschraenkung bezieht sich auf die physische gefilterte Dichte einschliesslich Preserve-Zellen, geteilt durch die Anzahl **erlaubter** Zellen. Verbotene Zellen zaehlen nicht zum Bezugsvolumen. Das reale Budget ist `volume_fraction * allowed_cells * voxel_volume_mm3`. Zu geringe Budgets werden vor der ersten Optimierungsiteration abgelehnt; auch die unvermeidliche Filtertransition um Preserves wird geprueft. Eine konstante freie Startdichte wird durch skalare Bisektion so gewaehlt, dass sie das Budget einhaelt.

Das OC-Update nutzt die exakten Compliance- und Volumenableitungen, eine maximale Dichteaenderung und eine bisektierte Volumenmultipliziererin. `minimum_design_density` verhindert mathematisch irreversible exakt leere freie Designvariablen. Filterdichten und vorgeschriebene Nullzellen sind davon getrennt. Das Verfahren darf an jeder freien Zelle Material entfernen oder hinzufuegen; es ist nicht auf das Abtragen einer v0-Struktur beschraenkt.

Ein Lauf endet an der Aenderungstoleranz, am Iterationslimit oder an einem optionalen Laufzeitlimit. `status: ok` bedeutet nur, dass die Dichteauswertung beendet wurde. `summary.converged` und `stop_reason` unterscheiden Konvergenz und begrenzte Probenlaeufe. Alle finalen Kennwerte werden an der tatsaechlich zurueckgegebenen Dichte neu berechnet. Fuer ungueltige Eingaben oder Numerikfehler entstehen `invalid` beziehungsweise `failed`, eine Diagnose und kein erfundenes Kennwertergebnis.

## Mehrzielbewertung und Fertigung

Masse ist im inneren Problem ueber das Volumenbudget beschraenkt, Steifigkeit ueber die Mehrlast-Compliance optimiert. Eigenfrequenz, Maximalspannung und Verschiebung werden nicht als analytisch unzutreffende Gradienten angehaengt: Die automatische aeussere Kandidatenauswahl rekonstruiert Solid-Kandidaten und bewertet alle Ziele und Constraints mit der vorhandenen gmsh/CalculiX-Pipeline. Der 37-g-Akku wird dort an der vorgegebenen Deckposition gekoppelt. Das ist eine hierarchische klassische Optimierung mit mechanischem Screening, kein monolithischer MMA-Mehrzielsolver.

Die im Dichteergebnis gespeicherten Hex8-Spannungen sind SIMP-Gauss-Punktwerte des Ersatzmodells. Sie sind Diagnosewerte und gelten nicht als erfuellte Spannungsconstraints des gedruckten Solids. Voxel-Moden mit Punktmassen werden ausdruecklich abgelehnt, weil nur der vorhandene CalculiX-Adapter die vereinbarte raeumliche Starrkopplung am Akku-COM implementiert. Die modal verwendete volle Hex8-Massenmatrix dient dem analytischen Balkentest ohne Punktmassen.

Vor der unabhaengigen FEA muessen exakte Preserve-/Forbidden-Geometrie, Komponentenfreiraum, Einteiligkeit, Mindestfeatures, Duesenverhaeltnis und Anschlussquerschnitte bestehen. Das Volumengitter begrenzt die kleinsten frei erzeugbaren Strukturen. Die konfigurierten Fertigungschecks und zulaessigen Supports sind nicht mit einer garantierten Festigkeit realer anisotroper PA6-CF-Drucke gleichzusetzen.

## Numerische Pruefungen

`tests/test_topology_optimization.py` prueft:

- Genau sechs Starrkoerpermoden der Hex8-Steifigkeitsmatrix, konstante dreidimensionale Dehnungsenergie und konsistente translatorische Masse.
- Den vollmaterialigen 48 x 6 x 6-mm-Kragbalken unter 1 N: 0,0730389 mm Durchbiegung gegen 0,0770504 mm nach Euler-Bernoulli (Abweichung -5,21 %); erste Eigenfrequenz 867,609 Hz gegen 848,082 Hz (+2,30 %). Das Netz hat 24 x 4 x 4 Hex8-Zellen. Die Tests erlauben 8 % beziehungsweise 5 % wegen endlichem Seitenverhaeltnis und vollintegrierter grober Hex8-Biegediskretisierung.
- Zentrale finite Differenzen fuer die volle SIMP-/Filter-/Projektions-Kettenregel und den physischen Volumengradienten.
- Complianceverbesserung eines echten dreidimensionalen Mehrlastproblems mit freier raeumlicher Dichteverteilung; Preserve-/Forbiddenmasken und Volumenbudget.
- Reproduzierbarkeit, Eingabeunveraenderlichkeit, unmoegliche Budgets, fehlerhafte Selektoren, begrenzte protokollierte Selektorerweiterung und saubere Fehlerstatus.

## Gespeicherte Solverdaten und verbleibende Grenzen

`density` ist das physische Feld, `design_density` das ungefilterte Optimierungsfeld. Der externe Persistenzadapter speichert beide nach Bedarf als NPZ; fuer Reproduktion aus Domain und Settings reicht die deterministische uniforme Initialisierung. `summary` und `history` enthalten Settings, Methodik, Materialeinheiten, Gitter-/Freiheitsgradzahlen, Auswahlabweichungen, normalisierte Fallgewichte, Anfangscompliances, Einzelcompliances pro Iteration, Ziel, Dichtesumme, Aenderung, Restfehler, Zeit und Konvergenzstatus. Finalwerte enthalten Dichtevolumen/-masse sowie Verschiebungs-/Spannungs-/Steifigkeitsdiagnostik pro statischem Fall. Die unabhaengigen CAD-/FEA-Ergebnisse werden getrennt davon gespeichert.

Noch nicht enthalten sind garantierte globale Optimalitaet, automatische Netzkonvergenz, adaptive Gitter, robuste erosion/dilation-Optimierung, orientierte Druckanisotropie, Supportvolumen-Minimierung, Ermuedung, nichtlineare Aufprallphysik und Spannungs-/Eigenfrequenzgradienten im inneren SIMP-Loop. Die Methode kann lokale Optima und graue Zwischenmaterialien erzeugen; unterschiedliche Volumenbudgets, Lastgewichte und Schwellen liefern unterschiedliche Kandidaten. Fehlerhafte oder nicht verifizierbare Kandidaten bleiben als solche erhalten.
