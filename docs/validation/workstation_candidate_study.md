# Workstation: neue Kandidaten nach 300 SIMP-Updates

Die unveraenderte 4-mm-Domain wurde mit 300 Updates neu optimiert. Der beste nach den bestehenden fuenf Grenzen und der bestehenden Auswahlformel akzeptierte Kandidat ist **density_t01** (Dichteschwelle 0.25) mit **50.95298 g**. Gegenueber dem eingefrorenen 58.53180-g-Kandidaten sind das **12.95 % weniger Rahmenmasse**. Die vier weiteren Schwellen scheitern explizit an getrennter Geometrie.

Die Dichteiteration endete weiterhin am Iterationslimit; formale Dichtekonvergenz wurde nicht erreicht. Die neuen Geometrien wurden mit 3-mm-C3D10-FEA, dem damaligen nativen Standardbackend **PaStiX** und **settings.threads=2** geprueft. Diese Werte wurden nicht nachtraeglich auf das spaeter eingefuehrte SPOOLES-Profil umetikettiert. Eine eigene Netzkonvergenzstudie dieser geaenderten Geometrien fehlt; Ergebnisse der historischen Geometrie sind dafuer kein Ersatz.

| Schwelle | Status | Masse g | Steifigkeit N/mm | f1 Hz | max. u mm | max. Spannung MPa |
|---|---|---:|---:|---:|---:|---:|
| 0.20 | gueltig | 57.12165 | 164.7982 | 1061.9350 | 0.0100268 | 0.647982 |
| 0.25 | gueltig | 50.95298 | 142.1080 | 964.0308 | 0.0118054 | 0.575509 |
| 0.30 | Density threshold produced 2 face-connected components | - | - | - | - | - |
| 0.35 | Density threshold produced 4 face-connected components | - | - | - | - | - |
| 0.40 | Density threshold produced 7 face-connected components | - | - | - | - | - |
| 0.50 | Density threshold produced 13 face-connected components | - | - | - | - | - |

Die sechs Schwellen, 2-mm-Mindestmerkmale, maximal 40 automatische Manifold-Reparaturvoxel und das Verbot entfernter Inseln blieben unveraendert. Es wurden keine Verbindungen von Hand ergaenzt. Beide gueltigen Geometrien bestanden die vollstaendigen Geometrie-/Fertigungschecks und alle fuenf mechanischen Grenzen gegen v0.

Die neue 3-mm-v0-Baseline der parallelen Netzstudie wurde erst nach Gleichheit von Material, Akku, Lasten, Selektoren, Parametern, CAD-Masse, FEA-Einstellungen, allen Quellhashes, Python/Paketen und Solveridentitaet uebernommen. Alle 35 rohen Baseline-Artefakte wurden gehasht geprueft. Der bestehende Frame, seine Baseline und die alte Abnahme wurden nicht ueberschrieben.

Ein anschliessender kompletter Resume pruefte Inputs, alle Kandidatenrecords/-Artefakte, Rohresultate, Lastfaelle, Baselinebelege und mechanische Vergleiche erneut; es startete kein weiterer Solver.

Maschinenlesbarer Beleg: [workstation_candidate_study.json](workstation_candidate_study.json). Rohdaten: `exports/topology/workstation_20260930/candidate_study/grid4_iter300/`. Die folgenden Befehle dokumentieren die **historische Ausfuehrung mit den archivierten damaligen Quellbytes**. Der aktuelle Core/Runner weist ein Resume in diesen alten Ordnern wegen geaenderter Provenienz bewusst ab:

```powershell
.\.venv\Scripts\python.exe -B tools/run_workstation_candidate_study.py --source exports/topology/workstation_20260930/density_study/grid4_iter300 --output exports/topology/workstation_20260930/candidate_study/grid4_iter300 --geometry-only
.\.venv\Scripts\python.exe -B tools/run_workstation_candidate_study.py --source exports/topology/workstation_20260930/density_study/grid4_iter300 --output exports/topology/workstation_20260930/candidate_study/grid4_iter300 --baseline-study exports/topology/workstation_20260930/mesh_study
```

Eine neue, getrennte Verifikation derselben 300-Update-Dichte unter dem aktuellen Workstation-Profil nutzt einen frischen Ausgabepfad und berechnet eine eigene passende v0-Baseline. Sie ist ein neuer numerischer Datensatz und ersetzt keine der obigen Messungen:

```powershell
.\.venv\Scripts\python.exe -B tools/run_workstation_candidate_study.py --source exports/topology/workstation_20260930/density_study/grid4_iter300 --output exports/topology/workstation_20260930/candidate_study/grid4_iter300_spooles1_next --geometry-only --linear-solver SPOOLES --threads 1
.\.venv\Scripts\python.exe -B tools/run_workstation_candidate_study.py --source exports/topology/workstation_20260930/density_study/grid4_iter300 --output exports/topology/workstation_20260930/candidate_study/grid4_iter300_spooles1_next --linear-solver SPOOLES --threads 1
```

## Feines 8/3-mm-Raster: negative Geometriepruefung

Der feinere Lauf mit 51 x 48 x 12 Zellen und 8/3 mm Zellweite endete nach **79 Updates am Zeitbudget** (angefragt waren hoechstens 150). Es gab weiterhin keine formale Dichtekonvergenz; die letzte maximale Dichteaenderung war 0.059873. Alle sechs urspruenglichen Schwellen wurden mit unveraenderter Rekonstruktion, unveraenderten Keepouts und maximal 40 automatischen Reparaturvoxeln geprueft. **Keiner der sechs Kandidaten erreichte eine zulaessige Geometrie.**

| Schwelle | Negativer Befund |
|---|---|
| 0.20 | Manifold raster repair would enter forbidden cells |
| 0.25 | Manifold raster repair would enter forbidden cells |
| 0.30 | Manifold raster repair would enter forbidden cells |
| 0.35 | Density threshold produced 3 face-connected components |
| 0.40 | Density threshold produced 3 face-connected components |
| 0.50 | Density threshold produced 6 face-connected components |

Bei 0.20/0.25/0.30 ist das rohe Schwellenfeld zwar zusammenhaengend, die benoetigte automatische Manifold-Reparatur wuerde jedoch verbotene Zellen belegen. Der vorhandene harte Check bricht deshalb vor einer zulaessigen CAD-Rekonstruktion ab. Bei den drei hoeheren Schwellen liegen getrennte Komponenten vor. Nachgelagerte Fertigungs-, Masse- und Mechanikfreigaben werden fuer diese verworfenen Felder nicht behauptet.

Es wurde keine Geometrie von Hand repariert, kein Keepout entfernt und keine Grenze aufgeweicht. Deshalb starteten **null Kandidaten-FEA und null Baseline-FEA** fuer diesen Datensatz. Der abschliessende Resume pruefte erneut alle sechs Recordhashes und beendete die Studie mit `status="invalid"`. Das konfigurierte Profil SPOOLES/1 Thread wurde fuer diese Studie folglich nicht in einer FEA ausgefuehrt.

Die bisherige 50.95298-g-Auswahl stammt weiterhin allein aus der getrennten 4-mm/300-Update-Studie. Das feinere Raster hat in diesem Lauf keinen neuen mechanisch bewertbaren Kandidaten geliefert. Rohbelege: `exports/topology/workstation_20260930/candidate_study/grid8over3_iter150_spooles1/`.

Beide Kandidatenstudien besitzen unter `provenance/` jeweils alle 26 verwendeten Python-Module und den exakten Runner mit Hashmanifest. Alle 27 Dateien pro Studie stimmen bytegenau mit den jeweiligen Eingabehashes ueberein.
