# Layout-Bedingungen gegen ManaFly und Tadpole geprüft

Prüft, bevor das Phase-1-Layout (`df55790`, `LAYOUT_DEFAULT`: Akku y −3,3 mm, Akku-Unterseite 25,8 mm, Kamera y 49,4 mm, Kamera-Bodenabstand 7,0 mm, Stack-Abstand 8,8 mm; α Rollen/Nicken/Gieren 2375/2072/111,5 rad/s²) endgültig Standard bleibt, welche Bedingungen ManaFly und Tadpole verletzen und ob die Grenzen für Schwerpunktband und Sichtfeld realistisch sind.

Alle Rechnungen laufen mit demselben `LayoutModel` auf dem sauberen Stand `a37f1bb` (per `git archive`, also ohne die offenen Phase-2-Änderungen im Worktree) über Ray, Typ `cpu`. Die Config bleibt unverändert. Die Varianten sind als Override-Dicts in `exports/layout/constraint_check.py` hinterlegt (`VARIANTS`).

Dateien:
- Tabelle und alle Zahlen: `exports/layout/layout_constraint_check.json`
- Optimierungsläufe je Variante: `exports/layout/constraint_check_runs/*.json`
- Der Lauf `baseline` reproduziert `LAYOUT_DEFAULT` exakt.

## 1 Verletzte Bedingungen je Referenz

Komponenten (K) sind unsere Hardware auf den Aufnahmen der Referenz: GNB 2S 550 mit 37 g, AIO15, Lux. Bei ManaFly kommen GR1105-Hüllmaße und eine 3-Zoll-Prop-Regel dazu, bei Tadpole GTS 1203 und HQ T2.5. Das Sichtfeld wird mit dem im Modell konfigurierten 145/94° gerechnet. In Klammern stehen die Werte für den realen 4:3-Modus der Lux (126/94°) und der Anteil der Bildfläche, den die überstrichenen Propscheiben verdecken.

| Frame | Bedingung | Wert | Grenze | Marge | Robustheit gegenüber den Annahmen |
|---|---|---|---|---|---|
| ManaFly | `cg_above_rotor_plane` | −3,75 mm | ≥ 0 mm | −3,75 | **Nicht robust.** Die Unterschreitung entsteht durch unseren leichten 2S-Akku (37 g) auf einem 29-g-Rahmen mit Rotorebene z 25,0. Folgende Annahmen ändern den Wert: Motorsitz z 8 statt 10 (Datenblatt Feld 3: 8–9,2) ergibt −2,3 mm, ein 60-g-Akku mit 22 mm Höhe +1,1 mm, ein 4S-650-Akku mit 75 g +3,5 mm. Das reale Abfluggewicht von 142,7 g, angenommen mit 4 × 8,5 g Motoren, 4 × 2,5 g Props und 59-g-Akku, ergibt **+0,3 mm**. Bei unseren Motoren kippt das Vorzeichen ab 83 g Akkumasse, bei 8,5-g-Motoren mit 22 mm hohem Akku ab 56 g. |
| ManaFly | `camera_sees_no_prop` | −26,7° (4:3: −17,2°, 3,9 % des Bildes) | ≥ 0° | −26,7 | **Robust.** Keine Kameraposition innerhalb der Grenzen erfüllt die Bedingung. Bestwerte im 4:3-Modus: −15,1° bei 20° Neigung und −9,4° bei 40°. Grund ist ein 3-Zoll-True-X: Die Props reichen bis y ≈ 95 mm, der Rumpf endet bei 65 mm. Der reale Frame fliegt trotzdem mit Props im Bild. |
| Tadpole | `cg_above_rotor_plane` | −0,14 mm | ≥ 0 mm | −0,14 | **Unbestimmt.** Der Wert liegt im Rauschen des Handmodells (kein STL vorhanden). Mit der Seitenteil-Box auf z 13 statt 10 ergibt sich 0,0 mm, mit dem Akku 1 mm höher +0,3 mm. |
| Tadpole | `battery_over_stack` | 2,5 mm Spalt | ≥ 3 mm | −0,5 | **Nicht robust.** Ohne den 3-mm-Zuschlag für den ELRS-Draht ist die Bedingung erfüllt, mit dem Akku 1 mm höher ebenfalls. Die Boxhöhe von 15–18 mm stammt aus Fotos (±10 %). |
| Tadpole | `battery_over_camera` | −3,4 mm Spalt | ≥ 3 mm | −6,4 | **Annahme.** Kamera-y 28 mm und Bodenabstand 0 mm sind geschätzt. Real sitzt die Kamera zwischen den Alu-Seitenteilen unter der Top-Platte. |
| Tadpole | `camera_sees_no_prop` | −27,6° (4:3: −18,1°, 3,0 % des Bildes) | ≥ 0° | −27,6 | **Gilt nur für die angenommene Kameralage.** Eine weiter vorn und höher sitzende Kamera erreicht im 4:3-Modus +13°. |
| Tadpole | `camera_under_hoop` | −1,9 mm | ≥ 0 mm | −1,9 | **Modellartefakt.** Die um 20° geneigte 14-mm-Hüllbox ist 17,9 mm hoch und passt in keiner Lage unter den geschätzten 18-mm-Bügel. Mit einem um 10 % höheren Bügel (19,8 mm) bleiben −0,14 mm. Eine reale Nano-Kamera mit Drehachse passt. |
| Unser Regel-Layout | `cg_y` | 1,99 mm | ≤ 1 mm | −0,99 | **Robust.** Mit mittigem Akku und Kamera vorn liegt der Schwerpunkt 2 mm vor der Motormitte. Das ist ein echtes Trimmproblem. |
| Unser Regel-Layout | `camera_sees_no_prop` | −17,9° (4:3: −8,4°, 1,8 % des Bildes) | ≥ 0° | −17,9 | **Hängt vom Bildmodus ab.** Im 16:9-Modus sind es −17,9° und 5,5 % des Bildes. |

Robustheitsläufe im JSON (`robustness`):
- ManaFly: Akku 50/60/75 g, Motoren 8,5 g, Props 2,5 g, Motorsitz z 8, Akku 3 mm tiefer, Kameraneigung 30°/40°.
- Tadpole: Seitenteile 2/6 g und z 7/13, Akku ±1 mm, Bügel +10 %, Kamera +2 mm hoch bzw. y 33, Neigung 30°, ohne ELRS-Zuschlag, Props 2,5 g.

**Fazit zu 1:** Robust gegen die Annahmen ist bei beiden Referenzen nur die Sichtfeldverletzung. Dabei verdecken die Props im realen 4:3-Modus 3–4 % des Bildes, und die echten Frames fliegen so. Die Schwerpunktverletzung von ManaFly folgt aus unserer leichten Hardware und kippt mit realistischer Hardware. Die übrigen Tadpole-Verletzungen sind Schätzfehler des Handmodells.

## 2 Schwerpunktband

Die Rotorebene liegt überall auf Motoroberkante plus halber Propnabe, wie in `LayoutModel` und `LOAD_COVARIANCE`. Die Komponenten sind unsere Hardware (K) auf den Aufnahmen der Referenz, wie im Bewertungswerkzeug. Der Rahmen geht mit Masse und Schwerpunkt aus den STL-Auswertungen ein.

| Frame | Rotorebene z | Schwerpunkt über Rotorebene | Bemerkung |
|---|---|---|---|
| ManaFly 3" (K) | 25,0 | −3,75 mm | Das Bewertungswerkzeug setzt die Props auf z 26,6 (alter Prop-Spalt 2 mm), damit ergäben sich −5,3 mm. Schon die Definition verschiebt den Wert um 1,6 mm. |
| ManaFly 3", realistische Hardware (ANNAHME 142,7 g) | 25,0 | +0,3 mm | 4 × 8,5 g Motoren, 4 × 2,5 g Props, 59-g-Akku (4S 450–550) |
| Tadpole 2,5" (K, Handmodell) | 14,4 | −0,14 mm | unbestimmt (siehe 1) |
| 2"-Frame v2 (K) | 16,4 | +1,0 mm | Akku sitzt auf der Leiter |
| BM Aether 4 (K) | 22,6–22,8 vorn, 40,7 hinten | +1,0 mm gegen die mittlere Ebene | Die Motoren sitzen in zwei Ebenen, 18 mm versetzt. Eine einzelne „Rotorebene“ ist hier nicht definiert. |
| TBS Source One V5 (K) | 19,9 | −0,25 mm | 5"-Frame mit unserem 37-g-Akku, nur bedingt aussagekräftig |
| TBS Source One V5, typische 5"-Hardware (ANNAHME) | 30,0 | −1,1 mm | 2207-Motoren mit 33 g, 6S-1100-Akku mit 185 g auf dem Deck, Rahmen 118 g |
| Unser Regel-Layout | 21,7 | +0,36 mm | |
| Unser `LAYOUT_DEFAULT` | 21,7 | +0,04 mm | Die untere Grenze ist aktiv. |

**Ergebnis:** Alle Frames, die nachweislich fliegen, liegen innerhalb von ±4 mm um die Rotorebene. Mit realistischer Hardware liegen sie innerhalb von ±1,1 mm. Etwa die Hälfte liegt unter der Ebene. „Schwerpunkt über der Rotorebene“ erfüllen die Referenzen also nicht als Regel. Sie liegen *nahe* an der Ebene, weil der Akku oben sitzt.

**Physik (Mechanismus belegt, Größenordnung bei unseren Frames nicht quantifiziert):**
- Der Schub ist körperfest und wirkt parallel zur Körper-z-Achse. Ein senkrechter Abstand h zwischen Schwerpunkt und Rotorebene erzeugt mit Schub und Gewicht kein Moment. Ein Quadrocopter hat deshalb keine Pendelstabilität aus tief liegender Masse; das Gegenteil ist der bekannte Trugschluss der „Pendelrakete“. Statisch braucht kein Fall eine Grenze für h.
- h wirkt über Kräfte in der Rotorebene, die mit der Geschwindigkeit wachsen: Rotor-H-Kraft bzw. Blattschlag in Vorwärtsflug und Wind. Diese Kraft greift im Abstand h vom Schwerpunkt an.
  - Schwerpunkt unter der Ebene: Das Moment richtet die Nase gegen die Fahrtrichtung auf und dämpft die Translation passiv.
  - Schwerpunkt über der Ebene: Das Moment hat das umgekehrte Vorzeichen und wirkt leicht anfachend.
  - Das Vorzeichen ist hier hergeleitet; die Größenordnung ist für 2,5-Zoll-Props nicht bestimmt.
  - Dazu kommt der Rumpfwiderstand des oben liegenden Akkus mit eigenem Hebelarm.
- Ein Raten-Regler regelt mit dem Gyro-Kreis hoher Bandbreite alle diese Momente als Störung aus. Sie verändern das Flugverhalten im Vorwärtsflug, schränken aber weder Steuerautorität noch Stabilität ein.
- Messbar und schon im Ziel enthalten ist die Trägheit: Liegen die Massen in der Höhe näher am Gesamtschwerpunkt, sinken Ixx und Iyy, und α steigt.
- **Konvention, keine Tatsache:** In der FPV-Szene heißt es, ein Akku oben bzw. ein Schwerpunkt nahe der Propebene fühle sich „agiler/flippiger“ an. Ein Akku unten liege „ruhiger“. Beides wird geflogen.

**Folgerung:** Das Band [0, 5] mm ist eine Vorliebe (`cg_band_source`: Nutzervorgabe), keine physikalische Anforderung. Vorschlag: **Band [−5, +5] mm**, also „nahe der Rotorebene“. Alle Referenzen liegen darin, auch ManaFly mit unseren Komponenten (−3,75). Ein Band, das ManaFly und Tadpole ausschließt, lässt sich durch die Referenzen nicht begründen.

## 3 Kamera-Sichtfeld

**Lux-Datenblatt** ([docs.hd-zero.com/camera-lux](https://docs.hd-zero.com/camera-lux), abgerufen 2026-10-04; im Ordner `Hardware` liegt kein Lux-Datenblatt):
- 1/2"-Sensor, „optimized for 4:3“
- Max FOV im **4:3-Modus: D 155°, H 126°, V 94°**
- Max FOV im **16:9-Modus: D 170°, H 145°, V 82°**

**Linsenmodell:**
- Äquidistant (θ ∝ r): Für 4:3 ergibt sich die Diagonale aus √(63² + 47²) = 78,6° → 157° gegen 155° laut Datenblatt; für 16:9 sind es 167° gegen 170°.
- Rektilinear käme 4:3 nur auf 132° Diagonale.
- Das Modell ist also ein Fischauge und als äquidistantes Rechteck im Winkelraum richtig modelliert.

**Fehler in der Konfiguration:** `fov_deg = [145, 94]` nimmt je Richtung den größeren Wert. Diesen Modus hat die Kamera nicht. Die Bedingung wird dadurch strenger als jeder reale Modus:
- Bindend ist bei uns die Horizontale, denn die Props liegen seitlich.
- Im 16:9-Modus (145/82) ist das Ergebnis identisch mit der Union.
- Im 4:3-Modus (126/94) bleibt bei `LAYOUT_DEFAULT` 9,6° Reserve.

**Props am Bildrand, Praxis (Konvention, keine Tatsache):**
- Im Racing gelten Props im Bild als störend.
- Bei Freestyle-Toothpicks und Cinewhoops werden Props in den Bildecken verbreitet hingenommen.
- Der BM-Aether-Hersteller wirbt ausdrücklich mit „True X ohne Props im Bild“. Das zeigt, dass es ein Designziel ist, das viele Frames nicht erreichen.
- ManaFly verdeckt im 4:3-Modus 3,9 % und im 16:9-Modus 7,6 % des Bildes, Tadpole (Annahme) 3,0 %.

**Kameraposition je Grenze** (nur unser Frame neu optimiert, sonst alles wie konfiguriert):

| Sichtfeld | Kamera y | Kamera-Bodenabstand | α R/N/G | Props im Bild 4:3 |
|---|---|---|---|---|
| Union 145/94 (heute) | 49,4 | 7,0 | 2375/2072/111,5 | 0 % |
| 16:9 145/82 | 49,4 | 7,0 | identisch | 0 % |
| **4:3 126/94** | **41,1** (−8,3 mm) | 7,0 | 2375/2123/112,9 | 0 % (16:9: 0,9 %) |
| 4:3 mit 5° Randzugabe | 37,0 | 5,7 | 2266/2026/113,5 | 0,36 % |
| 4:3 mit 10° Randzugabe | 32,3 | 5,7 | 2266/2047/114,1 | 0,97 % |

Ergebnisse:
- Der reale 4:3-Modus erlaubt die Kamera 8,3 mm weiter hinten.
- Randzugaben bringen nur noch wenige Millimeter. Wegen des lexikografischen Ziels kosten sie 5–8 % bei Rollen und Nicken für +1 % beim Gieren. Akku und Stack wandern dabei nach oben auf 31,7 bzw. 14 mm, weil sonst `battery_over_camera` bricht.
- Mit Gierrang-Toleranz 3 % landet die Kamera bei 40,2 statt 42,1 mm, also nur 1,9 mm Gewinn.
- Eine Randzugabe lohnt sich deshalb nicht als Standard.

## 4 Empfehlung

| Grenze | Heute | Empfehlung | Begründung |
|---|---|---|---|
| `cg_horizontal_mm` | 1,0 | **behalten** | echte Trimmbedingung; das Regel-Layout verletzt sie robust |
| `cg_band_mm` | [0, 5] | **[−5, 5]** (Entscheidung des Nutzers, da Nutzervorgabe) | Physikalisch keine harte Bedingung. Die Referenzen liegen bei −3,75 … +1,0 mm. Die Untergrenze 0 erzwingt im Optimum ein Artefakt: Der Stack steht auf 8,8 mm hohen Abstandshaltern, um den Schwerpunkt anzuheben, damit der 37-g-Akku tiefer darf. Keine Referenz macht das. |
| `fov_deg` | [145, 94] (Union) | **[126, 94]** (realer 4:3-Modus) und 16:9 [145, 82] als Option für Piloten im 16:9-Modus | Die Union ist kein Kameramodus. Mit 16:9 bleibt das heutige Ergebnis. |
| Randzugabe Sichtfeld | 0 | **0 behalten**, höchstens als Option | Konvention statt Anforderung; bringt nur 2–4 mm Kameraweg |
| Kameraneigung im Check | 20° | behalten, Hinweis | 30–40° ändern die Verletzung der Referenzen nur um 0,5–3°; unser Optimum bleibt zulässig und gewinnt Reserve |

**Layout mit den empfohlenen Grenzen** (`cg_m5_fov_4x3`):

| Größe | Neues Layout | Änderung gegenüber `LAYOUT_DEFAULT` |
|---|---|---|
| Akku y | −2,8 mm | +0,5 |
| Akku-Unterseite | 20,0 mm | −5,8 |
| Kamera y | 41,1 mm | −8,3 |
| Kamera-Bodenabstand | 7,0 mm | 0 |
| Stack-Abstand | 3,0 mm | −5,8 |

- **α Rollen/Nicken/Gieren: 2448 / 2204 / 112,9 rad/s².** Das ist auf allen drei Achsen besser als heute: Rollen +3,1 %, Nicken +6,4 %, Gieren +1,2 %.
- Schwerpunkt −2,9 mm unter der Rotorebene, also im Bereich von ManaFly.
- Aktive Bedingungen: `battery_over_stack`, `camera_sees_no_prop` (4:3), `camera_under_hoop`, `cg_y`.
- Abstand Akku–Props im Grundriss: 5,0 mm Reserve über der 2-mm-Grenze. Der Akku liegt dabei neben den Propscheiben, seine Unterseite 1,7 mm unter der Rotorebene.
- Nur das Schwerpunktband ändern ([−5, 5], Sichtfeld wie heute): Akku 20,0 mm, Stack 3,0 mm, Kamera bleibt bei 49,4. α 2448/2149/111,5.
- Nur das Sichtfeld ändern (4:3, Band wie heute): Kamera 41,1, Rest wie heute. α 2375/2123/112,9.

**Einordnung:**
- Gieren begrenzt in allen Varianten mit 111–114 rad/s², und das Gieren hängt kaum vom Layout ab.
- Der Unterschied zwischen dem Regel-Layout und jedem Optimum liegt bei höchstens 6 % auf einer Achse. Das Regel-Layout erreicht bei α_min sogar 114 gegen 111,5, ist aber unzulässig (`cg_y`, Sichtfeld).
- Das Layout wird also von den Bedingungen bestimmt, nicht vom Agilitätsziel. Deshalb hängt alles an der Frage, ob die Grenzen stimmen.

**Soll das neue Layout Standard werden?**
- Das heutige `LAYOUT_DEFAULT` ist unter **allen** geprüften Grenzvarianten zulässig, auch unter den empfohlenen. Als Zwischenstandard schadet es nicht und ist besser als das Regel-Layout, das `cg_y` und das Sichtfeld verletzt.
- Es ist aber nur unter den beiden zu strengen Grenzen optimal.
- **Empfehlung:**
  1. Der Nutzer entscheidet über das Schwerpunktband.
  2. Danach wird `fov_deg` auf den realen Modus gesetzt.
  3. `run.py layout` wird neu gerechnet, und das Ergebnis `cg_m5_fov_4x3` (bzw. die Variante nur mit Sichtfeld, falls das Band bleibt) wird neuer Standard.
- Weil sich Rahmenanteil (Masse und Trägheit von v3b) und Akkuhöhe ändern, sollte der Frame danach einmal neu optimiert werden. Der Rahmenanteil im Layoutmodell stammt vom alten Layout (Abweichung zweiter Ordnung).

## Grenzen dieser Prüfung

- Reale Hardware von ManaFly und TBS ist geschätzt (ANNAHME, siehe `REAL` im Skript). Tadpole hat kein STL; das Handmodell stammt aus `LAYOUT_REFERENCES`.
- Die Bildanteile beziehen sich auf die überstrichenen Propscheiben. Reale Blätter erscheinen nur zeitweise oder verwischt; Bügel und Rahmen im Bild sind nicht abgezogen.
- Für die Physik in Abschnitt 2 sind Mechanismen und Vorzeichen angegeben, aber keine Größenordnung. Für Aussagen über das Flugverhalten fehlt ein Flugmodell mit Rotor-H-Kraft.
