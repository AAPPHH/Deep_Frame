# Code-Stil

- Thematisch Zusammengehöriges in eine Datei; neue Datei erst ab ~1000 Zeilen. Keine Mini-Module.
- Kein argparse. Konfiguration als einfache Dicts.
- OOP, DRY. Keine Docstrings, keine Kommentare, keine unnötigen Leerzeilen.
- Prints auf das funktional Nötige beschränken.

# Arbeitsweise

- Als Orchestrator in Führungsrolle arbeiten; Ausführung delegieren.
- Ein Commit pro logischem Schritt.
- Am Ende jeder Aufgabe oder alle 15 Minuten eine kurze Bilanz im Chat.

# Projekt

- **Version 1, klassisch iterativ:** GPU-Topologieoptimierung → implizite Geometrie → FEA. Ziel: eine Pipeline, die ohne menschliche Formvorgaben automatisch druckfertige, geprüfte Frames im Stil von ManaFly erzeugt und robust und schnell genug ist, um Datensätze über variierte Anforderungen zu produzieren. Teacher für alles Weitere.
- **Version 2, teilweise ML:** ML beschleunigt oder ersetzt Teile der klassischen Schleife, z. B. FEA-Surrogat mit Active Learning oder Neural Reparameterization. Das Surrogat dient in Version 3 als Physik-Guidance.
- **Version 3, End-to-End-Diffusion:** Anforderungen (Motorpositionen, Komponenten, Keep-outs, Lasten, Zielwerte) rein, 3D-Form als Dichtefeld oder SDF raus. Training auf Daten aus Version 1, Guidance durch das Surrogat aus Version 2, jeder Kandidat mit echter FEA und Druck geprüft. Offenes Problem: harte Montage-Constraints einhalten.

# Regel für diese Datei

- Nichts weiter in diese CLAUDE.md schreiben.