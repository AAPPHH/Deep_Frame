# 0,75 mm: lokale Diagnose und DGX-Uebergabe, 05.10.2026

Der lokale 0,75-mm-Lauf ist **nicht abgeschlossen**. Keine vollstaendige MMA-Iteration, kein optimiertes Feld, kein Rohkoerper und keine v3c-Auswertung wurden erzeugt. Nach dem Wechsel auf DGX wurden weitere lokale GPU-Versuche eingestellt.

## Nachgewiesener Stand

- Ausgangspunkt: konvergiertes `free_layout4_opt/fine2`, freie Akku- und Kameralagerung, Standbedingung im Optimierer ausgeschaltet wie im Referenzlauf.
- Alte Probe: Abbruch mit relativem PCG-Residuum `4.621282672804353e-8` bei Soll `1e-8`.
- Aeltere gekuerzte Diagnose: ausschliesslich Twist-RHS problematisch; nach expliziter Residual-Ersetzung stieg dessen Fehler wieder bis `1.3447`.
- Korrektur: nach expliziter Ersetzung des Residuals wird die Krylov-Suchrichtung neu gestartet; bei frei schwebenden Systemen wird auch der Operator auf das Komplement des Starrkoerperkerns projiziert. Toleranz bleibt `1e-8`.
- Kleine bestehende Vergleichstests einschliesslich neuem Test zur Starrkoerperprojektion: **15 bestanden in 11,23 s**, Ray `raysubmit_jjxvifyszi5wTUgy`.
- Volle 0,75-mm-Probe danach: alle anderen rechten Seiten bis `9.7221e-9`; Twist bleibt bei **`4.199504523534821e-8` nach 3000 PCG-Schritten** (1000 je Versuch, zwei Wiederholungen). Abbruch nach etwa 554 s, weiterhin null MMA-Iterationen.
- Speicherspitzen: etwa **10,00 GiB dedizierte GPU / 0,131 GiB shared / 8,52 GiB Host-RSS**. Shared stieg beim CUDA-Aufbau von 0,072 auf 0,131 GiB und blieb anschliessend konstant. Kein beobachtetes GPU-Spilling.

Das belegt eine stabilere Behandlung der Projektion und des Residual-Neustarts, **keine Loesung des 0,75-mm-Konvergenzproblems**. Eine belastbare Sekunden-pro-MMA-Iteration-Zahl oder Vollrun-Dauer liegt nicht vor. Rund 235–257 Schritte anderer RHS sind nur eine lineare Teilmessung.

Die geplante isolierte Pruefung mit echter FP64-Hierarchie und gepinnter Formulierung wurde **nicht ausgefuehrt**: Job `raysubmit_h2aYiuruEBF3KTWJ` wurde in PENDING gestoppt, danach Ray-Status STOPPED und kein zugehoeriger Pythonprozess. Gepinnte Randbedingungen sind kein freigegebener Ersatz fuer die Starrkoerperprojektion.

## Aenderungen getrennt bewerten

- `f91e41ad402be87422ee138b52f802d7484e6974`: portabler Werkzeugfix. Bei `mma.start` wird nur das gespeicherte Design auf das neue Raster prolongiert; ein unpassendes Referenz-Dichtefeld wird nicht vorher eingebettet. `mma.multigrid` wird weitergereicht. Fertige Optimierungsergebnisse erhalten kompakte MG-Loesediagnosen.
- `5b73802f7e2964196ac81c84c1b2ffe2340738e5`: getestete Projektions-/Restart-Korrektur und kurze Fehlerdiagnose. Auf diesem Stand bleibt der volle Twist-Fall offen.
- Der nachfolgende optionale `multigrid.trace`-Patch ist **Diagnoseinstrumentierung ohne ausgefuehrten 0,75-mm-Nachweis**. Er protokolliert alle 100 Schritte rekursive, projizierte und unprojizierte wahre Residuen, Starrkoerperanteil, RHS-Norm, Verschiebungsmaximum und Datentypen. Der eigentliche Operator verwendet FP64; Mischpraezision betrifft den Praekonditionierer. Live-Werte zur RHS-Projektion und zum Rundungsboden muessen auf DGX noch erhoben werden.

## Relative DGX-Eingaben

`free_075_dgx_probe_cfg.json` und `free_075_dgx_cfg.json` enthalten keine alten Worktree-Pfade. Vorbedingungen aus dem neuen 4/3-mm-Lauf:

1. Der explizite Layoutvertrag der DGX-Konfiguration: `request` ist `null`, kein alter Optimierungsrequest erforderlich. Den frischen 4/3-mm-Lauf mit `dgx/free_43_coldstart.json` erzeugen; vollständige Startbefehle stehen in [dgx_handoff.md](../dgx_handoff.md).
2. `exports/runs/free_layout4_opt/fine2/design.npz`, `result.json` und `density_half.npz`: fertig berechnete 4/3-mm-Feinstufe.
3. Projektabhaengigkeiten und Ray auf DGX; GPU-Python und Scheduler passend zu Linux konfigurieren. `tools/compute.py` muss die tatsaechliche DGX-RAM-/GPU-Anforderung deklarieren. Der bisherige lokale Typ reserviert nur 14 GPU-GiB und darf nicht unveraendert als Deklaration fuer einen bis zu 80-GiB-cuDSS-Lauf verwendet werden.

Die vorbereiteten Konfigurationen waehlen **explizit `cuda_cudss`**. `auto` wuerde bei 0,75 mm erneut Multigrid waehlen. Ob cuDSS mit der verfuegbaren 80-GiB-GPU passt, ist erst durch die Probe nachzuweisen; hier wird kein passender Speicherbedarf behauptet.

Sobald ein DGX-Ray-Jobtyp mit ausreichender Host-/GPU-Reservierung registriert ist, werden diese Payloads ueber diesen Scheduler gestartet, jeweils aus dem Repository-Hauptverzeichnis:

```sh
python tools/formulation_study.py frame_mma docs/validation/free_075_dgx_probe_cfg.json
python tools/formulation_study.py frame_mma docs/validation/free_075_dgx_cfg.json
```

Dies sind die Payload-Kommandos **innerhalb des Ray-Jobs**, keine Anweisung zum lokalen Direktstart. Die Probe umfasst drei Iterationen bei Beta 64. Der Vollrun startet die gleiche Fein-Fortsetzung wie `fine2` bei Beta 8 und durchlaeuft 8/16/32/64; die letzte Stufe hat maximal 300 Iterationen und die bestehenden Konvergenz-/Best-feasible-Regeln. Der Vollrun ist erst nach bestandener Speicher-/Loeseprobe sinnvoll.

Fuer weitere MG-Diagnose den gesicherten Branch `codex/finish-075` verwenden; dessen Trace-Patch ist nicht in `main` integriert. `mma.linear_solver` auf `multigrid` setzen und `mma.multigrid` zum Beispiel mit `{"trace": true, "max_iterations": 400, "retries": 0}` belegen. Dabei die Fehlerschwelle `1e-8` beibehalten. Diese Konfiguration ist ein begrenzter Diagnoseversuch, kein nachgewiesen funktionierender Vollrun.

Nach Optimierung fehlen weiterhin Rohkoerper, v3c-Rekonstruktion, Neun-Kriterien-Auswertung, Kalibrierpruefung und gegebenenfalls ein Nachkalibrierlauf, STLs, Bilder und Vergleich mit 4/3 mm/ManaFly. Die kleinen Messbelege liegen in `run075_local_20261005.json`; keine grossen Felder, STLs oder alten Worktree-Verzeichnisse wurden eingecheckt.
