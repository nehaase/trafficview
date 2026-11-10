#!/usr/bin/env python3
"""
Nimmt einen Ausschnitt eines Streams als Videodatei auf, ohne ffmpeg (nur OpenCV, das TrafficView ohnehin braucht).

    python aufnehmen.py "<stream-url>" --sekunden 300 --ausgabe clip.mp4

Das Video wird neu kodiert (mp4v). Das reicht für die Validierung, ist aber nicht verlustfrei wie
`ffmpeg -i "<url>" -t 300 -c copy clip.mp4`. Die Bildrate stammt aus dem Stream. Meldet er keine brauchbare
Angabe, wird sie in den ersten Sekunden gemessen.
"""
import argparse
import os
import time

import cv2


def aufnehmen(url, sekunden, ausgabe, fps=None, messdauer=3.0, ausgabe_text=print):
    """Gibt (Anzahl Bilder, verwendete Bildrate) zurück."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"Die Quelle {url} kann nicht geöffnet werden.")
    ist_datei = os.path.isfile(url)
    if fps is None:
        deklariert = cap.get(cv2.CAP_PROP_FPS)
        fps = deklariert if 1 <= deklariert <= 120 else None
    writer, puffer, anzahl, fehler = None, [], 0, 0
    start = time.time()
    try:
        while sekunden is None or time.time() - start < sekunden:
            ok, frame = cap.read()
            if not ok:
                fehler += 1
                if ist_datei or fehler >= 5:
                    break
                time.sleep(0.5)
                continue
            fehler = 0
            if writer is None:
                puffer.append(frame)
                if fps is None and time.time() - start < messdauer:
                    continue                               # Bildrate erst messen
                if fps is None:
                    fps = len(puffer) / (time.time() - start)
                    ausgabe_text(f"Bildrate gemessen: {fps:.1f} Bilder/s")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(ausgabe, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"Die Ausgabedatei {ausgabe} kann nicht geschrieben werden.")
                for bild in puffer:
                    writer.write(bild)
                    anzahl += 1
                puffer.clear()
            else:
                writer.write(frame)
                anzahl += 1
    except KeyboardInterrupt:
        ausgabe_text("Aufnahme abgebrochen, das bisher Aufgenommene wird gespeichert.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if anzahl == 0:
        raise SystemExit("Es wurden keine Bilder empfangen. Ist die URL richtig und der Stream erreichbar?")
    return anzahl, fps


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stream als Videodatei aufnehmen (ohne ffmpeg)")
    ap.add_argument("url", help="Stream-URL oder Videodatei")
    ap.add_argument("--sekunden", type=float, default=300, help="Länge der Aufnahme (Standard 300 = 5 Minuten)")
    ap.add_argument("--ausgabe", default="clip.mp4", help="Ausgabedatei")
    ap.add_argument("--fps", type=float, default=None, help="Bildrate erzwingen (sonst aus dem Stream bzw. gemessen)")
    args = ap.parse_args(argv)
    print(f"Nehme {args.sekunden:g} s auf nach {args.ausgabe} ... (Strg+C beendet früher und speichert)")
    anzahl, fps = aufnehmen(args.url, args.sekunden, args.ausgabe, args.fps)
    print(f"Fertig: {anzahl} Bilder, {fps:.1f} Bilder/s, ungefähr {anzahl / fps:.0f} s Video -> {args.ausgabe}")


if __name__ == "__main__":
    main()
