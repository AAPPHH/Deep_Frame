# Workstation: native CalculiX-Diagnose

Das unveraenderte Windows-Binary CalculiX 2.22 brach zweimal mit **0xC0000374** ab: zuerst beim Balken mit Punktmasse, spaeter bei der Modalanalyse der v0-Geometrie auf dem 1.5-mm-Netz. Die Abbrueche traten im Standard-PaStiX-Pfad auf. Die Ursache bleibt offen; weder Speichermangel noch ein behobener Threadfehler sind bewiesen. Die urspruenglichen Fehlerrecords und Rohdateien bleiben unveraendert.

Nach dem ersten Fehler bestanden 32 unveraenderte Einzelwiederholungen sowie die gezielte und die volle Testsuite. Trotzdem trat der Fehler spaeter beim groesseren Modell erneut auf. Ein bestandener Testlauf beseitigt diese bekannte native Instabilitaet daher nicht.

Genau zwei neue Diagnosejobs verwendeten eine separate Kopie des fehlgeschlagenen v0-1.5-mm-Modalinputs. Alle OMP-, CCX- und BLAS-Threadvariablen standen explizit auf 1. Beim zweiten Job wurde nur die Karte `*FREQUENCY` um `SOLVER=SPOOLES` ergaenzt; Geometrie, Material, Randbedingungen, 37-g-Akku und Modellmatrizen-Eingaben blieben identisch. Die Syntax ist in den [offiziellen CalculiX-2.22-Quellen](https://www.dhondt.de/ccx_2.22.src.tar.bz2) (`frequencys.f`) und der [offiziellen Dokumentation](https://github.com/Dhondtguido/CalculiX/blob/master/doc/CalculiX.tex) beschrieben. Das mitgelieferte `README_Install` bestaetigt SPOOLES und PaStiX im statischen Windows-Binary.

| Diagnose | Ergebnis | Laufzeit s | Peak-RAM GiB |
|---|---|---:|---:|
| PaStiX, 1 Thread | erfolgreich | 40.72 | 5.54 |
| SPOOLES, 1 Thread | erfolgreich | 47.54 | 2.40 |

Alle sechs Frequenzen stimmen auf der ausgegebenen `.dat`-Praezision exakt ueberein: **201.7339, 236.7104, 238.5035, 416.6727, 457.0306 und 591.803 Hz**. Absolute und relative Differenzen sind fuer jede ausgegebene Mode null.

Die beiden erfolgreichen Einzelversuche sind kein Nachweis dauerhafter Solverstabilitaet und identifizieren keine Ursache. Ein explizit gewaehlt anderes Backend muss als neue numerische Provenienz gefuehrt werden. Es darf kein automatischer, unsichtbarer Ersatz fuer einen fehlgeschlagenen Job sein.

Vollstaendiger Maschinenbeleg mit Input-/Binary-/Diagnoseskript-/Artefakthashes: [workstation_solver_diagnosis.json](workstation_solver_diagnosis.json). Rohjobs: `exports/topology/workstation_validation/solver_diagnosis/`.

Die FEA-Schnittstelle besitzt jetzt den optionalen Schalter `settings["linear_solver"]`: `None` oder ein fehlender Eintrag behaelt das native Standardbackend bei; explizit sind `SPOOLES` und `PASTIX` erlaubt. Ungueltige Werte werden vor Vernetzung/Solverstart verworfen. Beide Analysekarten (`*STATIC`, `*FREQUENCY`) tragen den gewaehlten Parameter; Ergebnisse nennen Backend und Threadzahl. Es gibt keinen automatischen Backendwechsel und keine automatischen Wiederholungen fehlgeschlagener Jobs.

Die Threadsteuerung wurde ebenfalls konkret korrigiert: `settings["threads"]` setzt jetzt neben `OMP_NUM_THREADS`, `CCX_NPROC_RESULTS` und `NUMBER_OF_CPUS` auch `CCX_NPROC_STIFFNESS` und `CCX_NPROC_EQUATION_SOLVER`. Damit koennen geerbte, hoeher priorisierte CCX-Umgebungsvariablen die angeforderte Threadzahl nicht mehr uebersteuern. Diese Aenderung ist Teil der neuen numerischen Provenienz.

Die aktuelle volle Suite besteht aus **151 bestandenen Tests in 36.86 s**. Dazu gehoeren echte Balkenrechnungen mit Punktmasse fuer beide expliziten Backends, fruehe Ablehnung ungueltiger Backends und ein direkter Subprozessnachweis fuer das Ueberschreiben geerbter Threadzahlen.

Neue Workstation-Verifikationen verwenden ausdruecklich `linear_solver="SPOOLES"` und `threads=1`. Ein gewoehnlicher `python run_topology.py`-Aufruf verwendet weiterhin das native Standardbackend, da keine Default-Konfiguration geaendert wurde. Die folgenden Beispiele starten neue Rechnungen in separaten Ausgabeverzeichnissen.

Die vollstaendige bestehende Topologiepipeline laesst sich unter PowerShell ohne Bearbeitung ihrer Defaults so starten:

```powershell
@'
from copy import deepcopy
from deep_frame.geometry import reference_parameters
from deep_frame.topology_pipeline import PIPELINE_CONFIG, run_topology
settings = deepcopy(PIPELINE_CONFIG)
settings["output_dir"] = "exports/topology/workstation_phase1_spooles1"
settings["fea_settings"].update(linear_solver="SPOOLES", threads=1)
result = run_topology(reference_parameters(), settings)
print(result["status"], result["selected_id"], result["run_dir"])
'@ | .\.venv\Scripts\python.exe -B -
```

Die getrennte Netzstudie und die Geometriepruefung eines abgeschlossenen feinen Dichtelaufs haben explizite CLI-Optionen:

```powershell
.\.venv\Scripts\python.exe -B tools/run_workstation_mesh_study.py --output exports/topology/workstation_20260930/mesh_study_spooles1_next --mesh-sizes 2.0 1.5 --linear-solver SPOOLES --threads 1
.\.venv\Scripts\python.exe -B tools/run_workstation_candidate_study.py --source exports/topology/workstation_20260930/density_study/grid8over3_iter150 --output exports/topology/workstation_20260930/candidate_study/grid8over3_iter150_spooles1 --geometry-only --linear-solver SPOOLES --threads 1
```

Erst nach der Geometrieauswertung kann derselbe Kandidatenbefehl ohne `--geometry-only` die unabhaengige FEA starten. Ohne eine ausdruecklich angegebene, vollstaendig passende Baseline-Studie rechnet er eine eigene v0-Baseline mit denselben Einstellungen. Veraenderte Quell-/Runnerhashes verhindern bewusst ein Resume alter Studien unter neuer Provenienz. Alte Bytes sind unter `exports/topology/workstation_validation/source_before_backend/` sowie im Quellenarchiv der Dichtestudie gesichert; bestehende Ergebnisrecords wurden nicht migriert oder ueberschrieben.

Der abschliessende Validierungsfix weist ungueltige Backend-/Threadwerte vor dem Uebernehmen in die Ergebnis-Metadaten ab. `threads` muss eine positive ganze Zahl sein; boolesche Werte, NaN und Strings werden abgelehnt. Auch ungueltige Resultate bleiben strikt JSON-faehig, ohne Mesh/Solver zu starten. Dies ist eine Vertragskorrektur fuer ungueltige Eingaben und aendert keine Rechnung mit den verwendeten gueltigen Einstellungen. Die letzte getestete Quellversion ist unter `exports/topology/workstation_validation/source_final/` mit SHA256-Manifest gesichert.
