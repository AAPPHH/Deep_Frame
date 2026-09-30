# Parametrische Komponenten und Annahmen

Koordinaten: x quer, y nach vorne, z nach oben. Die ungedrehte Komponente liegt mit ihrer Unterseite auf z=0; Breite ist x, Laenge y. Kamera-Tilt dreht um x und hebt die gedrehte Huelle anschliessend auf z=0. Montagehoehe ist deshalb nicht unabhaengig vom Tilt festgeschrieben. Alle Masse und Massen stehen in `deep_frame/component_defaults.py` und werden von der zentralen `CONFIG` uebernommen.

| Komponente | Huelle / Masse | Herkunft und Grenze |
|---|---|---|
| HDZero AIO15 |31.3 x 31.3 mm; 25.5 x 25.5 M2; 7.2 g|Nutzervorgabe; komplette Stackhoehe vorlaeufig 6 mm. Boardmodell hat echte vier Durchgangsbohrungen.|
| HDZero Lux |Laenge14, Breite16, Hoehe14 mm; 2.3 g|Nutzervorgabe; Tilt20 Grad als Designstartwert. Seitliche Kameraschraubenlage/-groesse ist noch nicht durch Lux-Zeichnung validiert.|
| GNB5502S120A |63 x 30 x 11 mm; 37 g|Nutzervorgabe; Masse einschliesslich Leitungen/Steckern am Akkuschwerpunkt konzentriert.|
| Motor |Durchmesser14.2, Hoehe14.6 mm; 5.9 g|Vorlaeufige [GEPRC GR1105 Herstellerreferenz](https://geprc.com/product/gep-gr1105-motor/), [Masszeichnung](https://geprc.com/wp-content/uploads/2019/05/22-6199766706.jpg), [Massenabbildung](https://geprc.com/wp-content/uploads/2019/05/22-8095453337.jpg). Die Hoehe schliesst die obere Welle ein; das Modell nutzt konservativ einen vollen Zylinder. Die Masse umfasst die abgebildeten Kabel.|
| Prop |65-mm-Scheibe,0.8 mm hoch; 0.7 g|Durchmesser vom Nutzer, bewusst groesser als exakte2.5 Zoll=63.5 mm; Hoehe und Masse vorlaeufig, gleichmaessige Ersatzscheibe.|
| XT30U-F |10.2 x 12.4 x 5.2 mm|[AMASS-Zeichnung Seite2](https://images.100y.com.tw/pdf_file/AMASS-XT30U.pdf#page=2); Anschluss-/Kabelraum separat zu pruefen.|
| Balancer |9.8 x 7.5 x 5.7 mm;3 Pins|Vorlaeufige [JST XHP-3-Huelle](https://www.jst-mfg.com/product/pdf/eng/eXH.pdf#page=4), tatsaechlichen GNB-Stecker pruefen.|

Die Formulierung 9 x 9 mm ist nicht eindeutig mit der gewaehlten typischen 11xx-Referenz vereinbar: deren Zeichnung zeigt vier M2 auf einem **9-mm-Lochkreis**, nicht ein Quadrat mit9-mm-Seiten. Vorlaeufig wird das belegte Lochkreisbild verwendet (`mount_layout="bolt_circle"`, Achsen bei +-9/sqrt(8) mm). `mount_layout="square"` erzeugt weiterhin das explizite9-x-9-Quadrat, benoetigt mit2.2-mm-Bohrung aber eine groessere Motorhuelle als14.2 mm. Die Wahl ist keine Bestaetigung eines bereits ausgewaehlten Motors. Vor Bestellung/Druck muss das reale Modell die Default-Annahme ersetzen.

XT30 und Balancer bekommen0 g Zusatzmasse, damit die37-g-Akkuangabe nicht doppelt gezaehlt wird. Ihr tatsaechlicher Massenversatz bleibt unbekannt. Schrauben, Akku-Straps, Kabel ausserhalb der angegebenen Komponentenmassen, Antenne und Druckhohlraeume sind nicht zusaetzlich modelliert. Die Baugruppenmasse ist eine Ersatzmodellbilanz.

PA6-CF ist noch kein festgelegtes Filament. Als generische Annahme wird die [Bambu PA6-CF TDS v2](https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf) mit1.09 g/cm3 verwendet. FEA nutzt separat4430 MPa Young-Modul und angenommene Querkontraktion0.30. Isotropie, voller Materialanteil und trockener Werkstoff sind Rechenannahmen; ein reales Druckprofil und Feuchte-/Richtungsabhaengigkeit sind damit nicht nachgewiesen.
