# Kabelstufe: Uebergabe an DGX, 5. Oktober 2026

Code: `feature/cable-channels`, Kamera-Router-Fix `13d69b6`; 14 Kabeltests auf dem CPU-Interpreter ueber Ray bestanden. Die freie Kamera hatte keine `camera_mount_`-Preserves. Der Router nimmt nun vorhandene Graphknoten innerhalb des bestehenden `crash_front`-Lastkontakts als moegliche Starts. Er erzeugt keine neuen Tragstreben und spart den Kontaktbereich beim Ausschneiden aus. Fehlende reale Graphverbindungen bleiben ein Fehler.

Die neue portable Stufenkonfiguration ist `docs/validation/dgx/cables_free_layout4_v3c.json`. Vom Repository-Verzeichnis aus nach den Vorstufen starten:

```sh
python tools/compute.py reconstruction -- python tools/reconstruction_study.py cables docs/validation/dgx/cables_free_layout4_v3c.json
```

Voraussetzungen: `exports/runs/free_layout4_opt/free_layout4/density_fine.npz` aus dem DGX-Optimierungslauf, der daraus abgeleitete finale Kontaktkoerper `exports/runs/free_layout4_v3_contact/frame.stl` und dessen identische Layout-/Lastdomaene `domain.json`. Alle Pfade sind relativ zum Repository. Der DGX soll diese grossen Artefakte neu erzeugen. Das Graphcache wird dort neu berechnet; kein Windows-Cache uebertragen.

Der reparierte Kontaktkoerper war lokal mit 15.67059 g, einem wasserdichten Koerper und Kamera-Seitenkontaktflaechen 45.08/45.07 mm2 vorhanden. Die Kabelanwendung auf diesen Koerper wurde vor dem Start wegen des DGX-Wechsels gestoppt. Es gibt dafuer noch keinen Kanalnachweis und keinen FEA-Nachweis.

## Gesicherte Diagnose am vorherigen v3c-Koerper

Baseline: `free_layout4_v3_1/frame.stl`, SHA256 `e2b66d76e272f9fb6968f06822df0f38ad6363d7179521c6f919c6e06e248b90`. Diese Werte gelten ausschliesslich fuer den alten Koerper vor Kontakt- und Router-Fix.

| Messung | Ohne Kanaele | Mit Kanaelen |
|---|---:|---:|
| Masse | 15.5321 g | 14.0047 g |
| Wasserdicht / Koerper / gefaltete Kanten | wasserdicht / 1 / nicht neu gemessen | wasserdicht / 1 / 0 |
| Wandregel: tiefe entfernte Fraktion bei Oeffnung r=1 mm | 1.42 % | 15.10 % |
| Groesste tiefe Wandkomponente | 47.7 mm3 | 520.5 mm3 |
| Betroffene Motorzonen | 0 | 4 |

Alle vier Motorpfade waren kontinuierlich. Kabellaengen hinten links/rechts: 55.2/56.2 mm; vorne links/rechts: 59.1/59.4 mm. Die beiden hinteren Pfade bestanden die Propseitenpruefung; vorne blieben 6/266 bzw. 4/266 offene horizontale Aussenstrahlen. Direkte Strahlen zu den relevanten Propellertellern waren blockiert. Die gesamte strengere Pruefung blieb vorne negativ. Der Kamerapfad war vor dem Router-Fix nicht geroutet.

Zwoelf benutzte Glieder lagen mit 2.50 bis 4.09 mm Mindestbreite unter der erforderlichen 4.40-mm-Kanalschalenbreite. Die lokale Schale ergaenzt Material, der Schlitz entfernt jedoch tragendes Material. Vier Ansichten (Iso, oben, unten, Seite) und der vordere Querschnitt wurden visuell geprueft: durchgehende untere Schlitze, verbreiterte Ausgaenge und die noch kanalose Kamera waren sichtbar.

Die **2-mm-Wandregel ist verfehlt**. `wall_mm=1.0` und die geforderten Lippen ab 0.8 mm erfuellen diese Regel nicht. Die Motorprofile hatten geometrisch 1.23 mm Lippen. Keine Grenzwerte wurden gelockert. Ein bestandener STL-Export ist kein bestandener Wand-/Funktions-/FEA-Nachweis.

## Auf DGX noch auszufuehren

1. Neue Dichte und reparierte v3c-Kontakte erzeugen, dann die portable Kabelstufe ausfuehren.
2. Alle fuenf Pfade pruefen: Kontinuitaet, strenge Propseitenpruefung, Kabellaengen, gemeinsame Buendel, zu schmale Streben; vier Ansichten und Querschnitte am neuen Koerper pruefen.
3. Wandregel mit unveraenderten Grenzwerten ausfuehren. Der aktuelle Schalenentwurf bleibt konstruktiv unter 2 mm; Ergebnis ehrlich als Fehler ausweisen.
4. FEA beider finalen Koerper mit identischem Material, Lasten und Vernetzungsweg ausfuehren; Armsteifigkeit, f1, Crash und Oberflaechenabweichung vergleichen. Keine Zahlen aelterer gefalteter Koerper als Nachweis verwenden. Aeltere foldfreie Koerper scheiterten am Oberflaechenwinkel-/Selbstschnitt-Gate; diese technische Huerde ist nicht behoben.

Es werden keine STL-, NPZ-, PKL-, Netze oder Solver-Rohdaten mit dieser Uebergabe eingecheckt.
