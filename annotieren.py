#!/usr/bin/env python3
"""
Manuelle Zählung für die Validierung: ein Video abspielen und jedes Fahrzeug per Tastendruck zählen. Das
Ergebnis lässt sich mit den automatischen Zählungen vergleichen (vergleich.py).

    python annotieren.py clip.mp4 --spuren spuren.json --ausgabe manuell.csv

Eine Aufnahme (Video) erzeugst du zum Beispiel mit ffmpeg, ohne neu zu kodieren:
    ffmpeg -i "<stream-url>" -t 300 -c copy clip.mp4

Tasten:
    Leertaste   Pause / weiter
    1 ... 9     Fahrzeug an Zählstelle 1 ... 9 zählen (Zähllinie bzw. Bereich, wie im Video eingeblendet)
    , und .     ein Bild zurück / vor (in der Pause)
    - und +     langsamer / schneller abspielen
    p m b l     Fahrzeugtyp für die nächsten Zählungen: Pkw (Standard), Motorrad, Bus, Lkw
    u           letzte Zählung rückgängig machen
    s, Enter, q oder Esc   beenden (gespeichert wird bei jeder Änderung)

Zähle ein Fahrzeug in dem Moment, in dem es die Zähllinie überquert, bzw. im Interstate-Modus, wenn es im Bereich
auftaucht. Die Zeit stammt aus der Position im Video, also unabhängig davon, wie schnell du abspielst.
"""
import argparse
import csv
import json
import os

import cv2

TYPEN = {"p": "Pkw", "m": "Motorrad", "b": "Bus", "l": "Lkw"}
FARBEN = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (255, 0, 255), (0, 200, 255), (200, 200, 0)]


class Annotation:
    """Die manuell gezählten Fahrzeuge: (Videozeit in s, Zählstelle, Typ)."""

    def __init__(self):
        self.ereignisse = []

    def zaehlen(self, video_s, stelle_nr, typ="Pkw"):
        self.ereignisse.append((round(float(video_s), 2), int(stelle_nr), typ))

    def rueckgaengig(self):
        return self.ereignisse.pop() if self.ereignisse else None

    def anzahl(self, stelle_nr=None):
        return sum(1 for e in self.ereignisse if stelle_nr is None or e[1] == stelle_nr)

    def speichern(self, pfad):
        with open(pfad, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_s", "stelle_nr", "richtung", "typ"])
            for t, stelle, typ in sorted(self.ereignisse):
                w.writerow([t, stelle, "", typ])

    @classmethod
    def laden(cls, pfad):
        a = cls()
        with open(pfad, newline="", encoding="utf-8") as f:
            for zeile in csv.DictReader(f):
                a.zaehlen(float(zeile["video_s"]), int(zeile["stelle_nr"]), zeile.get("typ") or "Pkw")
        return a


def lade_zaehlstellen(pfad):
    """Zähllinien und Bereiche aus der Konfiguration (relative Koordinaten). Ohne Datei: weder noch."""
    if not pfad or not os.path.exists(pfad):
        return [], []
    with open(pfad, encoding="utf-8") as f:
        daten = json.load(f)
    if isinstance(daten, list):
        return daten, []
    return daten.get("linien", []), daten.get("rois") or ([daten["roi"]] if daten.get("roi") else [])


def zeichne(bild, linien, rois, annot, video_s, tempo, typ, pause):
    """Blendet Zählstellen, Zeit, Zählstand und Tastenhilfe ein."""
    h, w = bild.shape[:2]
    stellen = linien if linien else rois
    for i, st in enumerate(stellen):
        farbe = FARBEN[i % len(FARBEN)]
        if linien:
            p1 = (int(st["p1"][0] * w), int(st["p1"][1] * h))
            p2 = (int(st["p2"][0] * w), int(st["p2"][1] * h))
            cv2.line(bild, p1, p2, farbe, 2)
            ort, name = p1, st.get("name", "")
        else:
            p1, p2 = (int(st[0] * w), int(st[1] * h)), (int(st[2] * w), int(st[3] * h))
            cv2.rectangle(bild, p1, p2, farbe, 1)
            ort, name = p1, f"Bereich {i + 1}"
        cv2.putText(bild, f"{i + 1} {name}: {annot.anzahl(i + 1)}", (ort[0] + 4, max(14, ort[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, farbe, 2)
    if not stellen:
        cv2.putText(bild, f"Zaehlstelle 1-9 per Taste, bisher: {annot.anzahl()}", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    status = f"{int(video_s // 60)}:{video_s % 60:05.2f}  {'PAUSE' if pause else f'{tempo:g}x'}  Typ: {typ}  gesamt: {annot.anzahl()}"
    cv2.putText(bild, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(bild, "Leertaste=Pause  1-9=zaehlen  ,/.=Bild  -/+=Tempo  p/m/b/l=Typ  u=zurueck  s=Ende",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)


def annotiere(video, ausgabe, spuren=None, tempo=1.0, fortsetzen=False, fenster="Manuelle Zaehlung"):
    """Interaktive Schleife. Gibt die Annotation zurück."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Video {video} kann nicht geöffnet werden.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 240 else 25.0
    linien, rois = lade_zaehlstellen(spuren)
    anzahl_stellen = len(linien) or len(rois) or 9
    annot = Annotation.laden(ausgabe) if fortsetzen and os.path.exists(ausgabe) else Annotation()
    typ, pause = "Pkw", True

    def lies(index):
        """Frame mit diesem Index (0-basiert) lesen und anzeigen."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = cap.read()
        return (max(0, index), frame) if ok else (None, None)

    pos, bild = lies(0)
    if bild is None:
        raise SystemExit("Das Video enthält keine Bilder.")
    while True:
        anzeige = bild.copy()
        zeichne(anzeige, linien, rois, annot, pos / fps, tempo, typ, pause)
        cv2.imshow(fenster, anzeige)
        # In der Pause auf eine Taste warten, sonst im Bildtakt des Videos (angepasst an das Tempo)
        taste = cv2.waitKey(0 if pause else max(1, int(1000 / fps / tempo))) & 0xFF
        if taste == 255:                                   # keine Taste
            if not pause:
                ok, frame = cap.read()
                if ok:
                    pos, bild = pos + 1, frame
                else:
                    pause = True                           # Videoende: anhalten, auf Taste warten
            continue
        zeichen = chr(taste) if 32 <= taste < 127 else ""
        if taste in (13, 27) or zeichen in ("s", "q"):
            break
        elif zeichen == " ":
            pause = not pause
        elif zeichen.isdigit() and zeichen != "0":
            stelle = int(zeichen)
            if stelle <= anzahl_stellen:
                annot.zaehlen(pos / fps, stelle, typ)
                annot.speichern(ausgabe)                   # nichts verlieren, falls etwas abbricht
        elif zeichen == "u":
            if annot.rueckgaengig():
                annot.speichern(ausgabe)
        elif zeichen in TYPEN:
            typ = TYPEN[zeichen]
        elif zeichen == ".":
            neu_pos, neu_bild = lies(pos + 1)
            if neu_bild is not None:
                pos, bild, pause = neu_pos, neu_bild, True
        elif zeichen == ",":
            neu_pos, neu_bild = lies(pos - 1)
            if neu_bild is not None:
                pos, bild, pause = neu_pos, neu_bild, True
        elif zeichen in ("-", "_"):
            tempo = max(0.125, tempo / 2)
        elif zeichen in ("+", "="):
            tempo = min(8.0, tempo * 2)
    annot.speichern(ausgabe)
    cap.release()
    cv2.destroyAllWindows()
    return annot


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fahrzeuge in einem Video per Tastendruck zählen")
    ap.add_argument("video", help="Videodatei (z. B. mit ffmpeg aufgenommen)")
    ap.add_argument("--spuren", default="spuren.json", help="Konfiguration mit Zähllinien/Bereichen (zum Einblenden)")
    ap.add_argument("--ausgabe", default="manuell.csv", help="Ergebnisdatei")
    ap.add_argument("--tempo", type=float, default=1.0, help="Abspielgeschwindigkeit (1 = Echtzeit)")
    ap.add_argument("--fortsetzen", action="store_true", help="vorhandene Ausgabedatei laden und weiterzählen")
    args = ap.parse_args(argv)
    annot = annotiere(args.video, args.ausgabe, args.spuren, args.tempo, args.fortsetzen)
    print(f"{annot.anzahl()} Fahrzeuge gezählt, gespeichert in {args.ausgabe}")


if __name__ == "__main__":
    main()
