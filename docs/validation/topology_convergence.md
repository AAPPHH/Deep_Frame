# Iterations- und Konvergenzstudie

Zusaetzlich zum automatisch CAD-/FEA-verifizierten 45-Iterations-Standardlauf wurde derselbe freie Designraum nochmals aus uniformer Anfangsdichte mit bis zu 150 Iterationen gerechnet. Die zweite Variante aendert ausschliesslich das Iterationslimit, die strengere Dichteaenderungstoleranz `0.005` statt `0.015` und den Laufzeitdeckel von 900 s. Domain, Material, 20 statische Lastfaelle, Gewichte, 0.10-Volumenbudget, Filter und Preserve-Knotenauswahl sind identisch.

| Kennwert | 45 Iterationen | 150 Iterationen |
|---|---:|---:|
| Normiertes Mehrlast-Complianceziel zu Beginn | 1.0 | 1.0 |
| Ziel am Ende | 0.013326498 | 0.013207095 |
| Max. Designvariablenaenderung am Ende | 0.025576 | 0.009717 |
| Geforderte Aenderungstoleranz | 0.015 | 0.005 |
| Konvergiert nach eigener Stoppregel | nein | nein |
| Ersatzdichte-Frame-Masse | 40.753792 g | 40.753792 g |
| Freie Zellen mit Dichte zwischen 0.1 und 0.9 | 17.277 % | 17.153 % |
| Laufzeit des isolierten Solvers | 144.4 s | 472.4 s |

Die weitere Zielverbesserung nach Iteration 45 betraegt 0.896 %. Die RMS-Aenderung des physischen Dichtefelds ist 0.01373; einzelne lokale Zellen aendern sich jedoch noch um bis zu 0.3170. Ein nahezu konstantes Gesamtziel beweist somit keine stationaere Geometrie. Das urspruengliche Aenderungskriterium `0.015` wird im laengeren Lauf erstmals bei Iteration 83 unterschritten. Die strengere Toleranz `0.005` wird innerhalb von 150 Iterationen nicht erreicht; beide Datensaetze werden wahrheitsgemaess als begrenzte, nicht nach ihrer jeweiligen Stoppregel konvergierte Laeufe gefuehrt.

Bei Dichteschwelle 0.20 sind beide Felder bereits vor Rekonstruktion zusammenhaengend: 808 belegte Zellen nach 45 Iterationen, 797 nach 150. Bei 0.25 sind es zwei Komponenten mit 714 Zellen beziehungsweise eine Komponente mit 703 Zellen. Das laengere Feld kann daher weitere leichtere Kandidaten liefern. Diese Moeglichkeit ist **kein** neuer mechanisch akzeptierter Entwurf: Fuer das 150er-Feld wurden in dieser Zusatzstudie weder CAD-Rekonstruktion noch Fertigungschecks oder CalculiX ausgefuehrt. Der mechanische Phase-1-Abnahmenachweis bezieht sich auf den vollstaendigen automatischen 45er-Run.

Die numerische Vorgeschichte mit beiden Settings-Dicts, allen Iterationszielen, Dichtesummen, Aenderungen, Restfehlern, Quellcode- und Felddigests sowie Schwellen-Connectivity steht in [topology_convergence.json](topology_convergence.json). Vollstaendige Dichte-/Designfelder, Masken, Domains und Einzelcompliancehistorien bleiben lokal unter `exports/topology/final_solver` und `exports/topology/converged_solver` des Optimierer-Worktrees erhalten. Die beiden Datensaetze verwenden dieselbe generische Pipeline-Schnittstelle und koennen mit den gespeicherten Dict-Settings erneut aus uniformer Initialisierung erzeugt werden.

Fuer einen spaeteren Teacher-Datensatz sollten Konvergenzstatus, Iterationsbudget und Extraktionsschwelle deshalb als eigene Qualitaetsmerkmale erhalten bleiben. Mechanische Gueltigkeit nach unabhaengiger FEA und numerische Konvergenz des Dichteoptimierers sind unterschiedliche Nachweise. Eine garantierte globale Optimalitaet wird fuer keinen Lauf behauptet.
