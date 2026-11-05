#!/usr/bin/env python3
"""
Verkehrszähler für öffentliche Kamera-Streams (z. B. von opencctv.org verlinkt)

- Erkennt Fahrzeuge (Pkw, Motorrad, Bus, Lkw) mit YOLO und verfolgt sie (--tracker einfach | bytetrack)
- Beliebig viele Zähllinien (z. B. eine pro Spur), getrennt nach Richtung und Fahrzeugtyp
- Mehrere Auswertebereiche (ROIs), z. B. je einer pro Fahrtrichtung; jeder bekommt
  einen eigenen Tracker und wird einzeln vergrößert -> bessere Erkennung bei weniger Rechenlast
- Fahrzeugtyp per Mehrheitsentscheidung über alle Frames (--zusammenfassen: alles als "Fahrzeug")
- --skip auto: passt das Überspringen von Frames an die Rechenleistung an
- Nutzt automatisch GPU (CUDA/Apple MPS), falls vorhanden
- Minutenwerte pro Linie/Richtung in CSV

Installation:
    pip install ultralytics opencv-python requests numpy

Ablauf:
    1) Einmalig einrichten (Fenster, Mausklicks): Bereiche (Taste r) und Linien
         python trafficview.py "<url>" --setup
    2) Zählen:
         python trafficview.py "<url>" --skip auto --show
"""

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

FAHRZEUGE = {2: "Pkw", 3: "Motorrad", 5: "Bus", 7: "Lkw"}
FARBEN = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (255, 0, 255),
          (0, 200, 255), (200, 200, 0), (128, 0, 255), (0, 128, 255)]
ROI_FARBEN = [(0, 255, 255), (255, 255, 0), (0, 165, 255), (255, 0, 200)]

# FFmpeg soll bei abgebrochenen Verbindungen selbst neu verbinden
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;5",
)


# ----------------------------------------------------------------------------
# Bildquelle
# ----------------------------------------------------------------------------
class FrameQuelle:
    def __init__(self, url, snapshot, interval):
        self.url, self.snapshot, self.interval = url, snapshot, interval
        self.cap = None
        self._letzter_abruf = 0.0
        if not snapshot:
            self._verbinden()

    def _verbinden(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url)

    @property
    def fps(self):
        """Bildrate des Streams (Schätzwert 25, falls der Stream keine plausible Angabe liefert)."""
        if self.cap is None:
            return 1.0 / max(self.interval, 0.05)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if 1 <= fps <= 120 else 25.0

    def lesen(self, skip=1):
        """Liefert ein Frame; bei skip>1 werden skip-1 Frames vorher verworfen."""
        if self.snapshot:
            warte = self.interval - (time.time() - self._letzter_abruf)
            if warte > 0:
                time.sleep(warte)
            self._letzter_abruf = time.time()
            try:
                r = requests.get(self.url, timeout=10,
                                 headers={"User-Agent": "Mozilla/5.0 Verkehrszaehler"})
                r.raise_for_status()
                return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            except requests.RequestException as e:
                print(f"[Warnung] Snapshot-Abruf fehlgeschlagen: {e}")
                time.sleep(2)
                return None

        for _ in range(max(0, skip - 1)):
            if not self.cap.grab():
                break
        ok, frame = self.cap.read()
        if not ok:
            print("[Warnung] Stream unterbrochen, versuche Neuverbindung ...")
            time.sleep(2)
            self._verbinden()
            return None
        return frame

    def schliessen(self):
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------------
# Geometrie
# ----------------------------------------------------------------------------
def seite(punkt, p1, p2):
    """>0 / <0 = Seite der Linie p1->p2, auf der der Punkt liegt."""
    return (p2[0] - p1[0]) * (punkt[1] - p1[1]) - (p2[1] - p1[1]) * (punkt[0] - p1[0])


def segmente_schneiden(a, b, c, d):
    """Schneidet die Bewegung a->b das Linienstück c->d (nicht die unendliche Gerade)?"""
    def ori(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = ori(a, b, c), ori(a, b, d), ori(c, d, a), ori(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def roi_in_pixel(roi, w, h):
    """Relativer ROI [x1,y1,x2,y2] -> Pixelgrenzen (None = ganzes Bild)."""
    if not roi:
        return 0, 0, w, h
    x1, y1 = max(0, int(roi[0] * w)), max(0, int(roi[1] * h))
    x2, y2 = min(w, int(roi[2] * w)), min(h, int(roi[3] * h))
    if x2 - x1 < 32 or y2 - y1 < 32:
        print("[Warnung] ROI zu klein oder ungültig, nutze das ganze Bild.")
        return 0, 0, w, h
    return x1, y1, x2, y2


def auto_imgsz(breite, hoehe):
    """Kleine Ausschnitte werden vor der Erkennung ~1,5x hochgerechnet (min. 640, max. 1280)."""
    ziel = math.ceil(max(breite, hoehe) * 1.5 / 32) * 32
    return int(max(640, min(1280, ziel)))


def warne_bei_ueberlappung(rois):
    """Überlappende Bereiche können dasselbe Fahrzeug doppelt zählen."""
    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            a, b = rois[i], rois[j]
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                print(f"[Warnung] Bereich {i + 1} und {j + 1} überlappen sich. Fahrzeuge im "
                      f"Überlappungsbereich könnten doppelt gezählt werden. Besser nicht überlappen lassen.")


# ----------------------------------------------------------------------------
# Einfacher Tracker (Mittelpunkte)
# ----------------------------------------------------------------------------
class EinfacherTracker:
    """
    Schlanker Tracker, der Erkennungen anhand von Position und Bewegung zuordnet.
    - Keine Bestätigungsphase und keine IoU-Pflicht: Fahrzeuge werden auch nach größeren Sprüngen
      (z. B. bei --skip) wiedererkannt.
    - Hysterese: Neue Tracks starten nur bei sicherer Erkennung (Konfidenz >= neu_schwelle). Einen
      bestehenden Track dürfen aber auch schwache Erkennungen weiterführen. Das verhindert das
      Flackern, wenn die Konfidenz eines Fahrzeugs knapp um die Schwelle schwankt.
    """

    def __init__(self, max_luecke=8, faktor=2.0):
        self.max_luecke = max_luecke      # so viele Frames darf ein Fahrzeug fehlen
        self.faktor = faktor              # erlaubter Sprung in Fahrzeuggrößen
        self.tracks = {}
        self.naechste_id = 1

    def update(self, boxen, konfs=None, neu_schwelle=0.0):
        """
        boxen: Array (N,4) xyxy; konfs: Konfidenzen. Gibt die IDs in gleicher Reihenfolge zurück
        (None = schwache Erkennung ohne passenden Track, wird ignoriert).
        """
        n = len(boxen)
        konfs = [1.0] * n if konfs is None else list(konfs)
        ids = [None] * n
        mitten = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxen]
        masse = [(float(b[2] - b[0]), float(b[3] - b[1])) for b in boxen]
        groessen = [max(w, h, 1.0) for w, h in masse]

        paare = []
        for tid, t in self.tracks.items():
            schritte = 1 + t["fehlt"]
            erwartet = (t["pos"][0] + t["v"][0] * schritte, t["pos"][1] + t["v"][1] * schritte)
            for j in range(n):
                d = math.dist(mitten[j], erwartet)
                if t["fehlt"] == 0:   # aktive Tracks dürfen auch über die letzte Position zuordnen
                    d = min(d, math.dist(mitten[j], t["pos"]))
                grenze = self.faktor * max(groessen[j], t["groesse"]) * (1 + 0.5 * min(t["fehlt"], 4))
                if d <= grenze:
                    # Reihenfolge: sichere Erkennungen zuerst, dann aktive vor verlorenen Tracks
                    paare.append((konfs[j] < neu_schwelle, t["fehlt"] > 0, d / grenze, tid, j))
        paare.sort()

        benutzt_t, benutzt_d = set(), set()
        for _, _, _, tid, j in paare:
            if tid in benutzt_t or j in benutzt_d:
                continue
            t = self.tracks[tid]
            schritte = 1 + t["fehlt"]
            neu_v = ((mitten[j][0] - t["pos"][0]) / schritte, (mitten[j][1] - t["pos"][1]) / schritte)
            t["v"] = (0.6 * neu_v[0] + 0.4 * t["v"][0], 0.6 * neu_v[1] + 0.4 * t["v"][1])
            t["pos"], t["groesse"], t["fehlt"] = mitten[j], groessen[j], 0
            t["w"], t["h"] = masse[j]
            t["treffer"] += 1
            ids[j] = tid
            benutzt_t.add(tid)
            benutzt_d.add(j)

        for tid in list(self.tracks):
            if tid not in benutzt_t:
                self.tracks[tid]["fehlt"] += 1
                if self.tracks[tid]["fehlt"] > self.max_luecke:
                    del self.tracks[tid]

        for j in range(n):
            if ids[j] is None and konfs[j] >= neu_schwelle:
                self.tracks[self.naechste_id] = {"pos": mitten[j], "v": (0.0, 0.0), "groesse": groessen[j],
                                                 "fehlt": 0, "w": masse[j][0], "h": masse[j][1], "treffer": 1}
                ids[j] = self.naechste_id
                self.naechste_id += 1
        return ids


# ----------------------------------------------------------------------------
# Konfiguration (ROIs + Linien)
# ----------------------------------------------------------------------------
def lade_konfig(datei, standard_hoehe):
    """Gibt (linien, rois) zurück. Akzeptiert auch ältere Formate (Linienliste, einzelnes 'roi')."""
    if os.path.exists(datei):
        with open(datei, encoding="utf-8") as f:
            daten = json.load(f)
        if isinstance(daten, list):
            linien, rois = daten, []
        else:
            linien = daten.get("linien", [])
            rois = daten.get("rois") or ([daten["roi"]] if daten.get("roi") else [])
        print(f"{len(linien)} Zähllinie(n) und {len(rois)} Auswertebereich(e) aus {datei} geladen.")
        return linien, rois
    print(f"Keine {datei} gefunden -> eine waagerechte Linie bei {standard_hoehe:.0%} Bildhöhe, ganzes Bild.")
    return [{"name": "Linie1", "p1": [0.0, standard_hoehe], "p2": [1.0, standard_hoehe]}], []


def zeichne_linien(bild, spuren, zaehler=None):
    h, w = bild.shape[:2]
    for i, s in enumerate(spuren):
        farbe = FARBEN[i % len(FARBEN)]
        p1 = (int(s["p1"][0] * w), int(s["p1"][1] * h))
        p2 = (int(s["p2"][0] * w), int(s["p2"][1] * h))
        cv2.line(bild, p1, p2, farbe, 2)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        laenge = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx, ny = -dy / laenge, dx / laenge
        cv2.arrowedLine(bild, (mx, my), (int(mx + nx * 35), int(my + ny * 35)), farbe, 2, tipLength=0.4)
        text = s["name"]
        if zaehler is not None:
            mit = sum(v for (li, r, _), v in zaehler.items() if li == i and r == "mit_pfeil")
            gegen = sum(v for (li, r, _), v in zaehler.items() if li == i and r == "gegen_pfeil")
            text += f"  ->{mit} <-{gegen}"
        cv2.putText(bild, text, (p1[0] + 5, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, farbe, 2)


def zeichne_rois(bild, rois):
    h, w = bild.shape[:2]
    for i, roi in enumerate(rois):
        farbe = ROI_FARBEN[i % len(ROI_FARBEN)]
        p1, p2 = (int(roi[0] * w), int(roi[1] * h)), (int(roi[2] * w), int(roi[3] * h))
        cv2.rectangle(bild, p1, p2, farbe, 1)
        cv2.putText(bild, f"Bereich {i + 1}", (p1[0] + 4, p2[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, farbe, 1)


def setup_modus(quelle, datei):
    """Standbild öffnen: Bereiche (Taste r, 2 Klicks = gegenüberliegende Ecken) und Linien (2 Klicks)."""
    frame = None
    for _ in range(100):
        frame = quelle.lesen()
        if frame is not None:
            break
    if frame is None:
        raise SystemExit("Konnte kein Bild vom Stream holen.")

    h, w = frame.shape[:2]
    spuren, rois, offen, verlauf = [], [], [], []
    zustand = {"modus": "linie"}

    def maus(ereignis, x, y, *_):
        if ereignis != cv2.EVENT_LBUTTONDOWN:
            return
        offen.append((x / w, y / h))
        if len(offen) < 2:
            return
        (ax, ay), (bx, by) = offen
        if zustand["modus"] == "roi":
            rois.append([min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)])
            verlauf.append("roi")
            zustand["modus"] = "linie"
        else:
            spuren.append({"name": f"Spur{len(spuren) + 1}", "p1": [ax, ay], "p2": [bx, by]})
            verlauf.append("linie")
        offen.clear()

    fenster = "Setup"
    cv2.namedWindow(fenster)
    cv2.setMouseCallback(fenster, maus)
    while True:
        bild = frame.copy()
        zeichne_rois(bild, rois)
        zeichne_linien(bild, spuren)
        for p in offen:
            cv2.circle(bild, (int(p[0] * w), int(p[1] * h)), 5, (0, 255, 255), -1)
        hinweis = (f"Modus: BEREICH {len(rois) + 1} (2 Klicks = Ecken)" if zustand["modus"] == "roi"
                   else "Modus: LINIE (2 Klicks)")
        cv2.putText(bild, hinweis, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(bild, "r=neuer Bereich  c=alle Bereiche loeschen  u=zurueck  Enter=speichern  q=Abbruch",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.imshow(fenster, bild)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, ord("s")):
            break
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("Setup abgebrochen.")
        if k == ord("r"):
            zustand["modus"] = "roi"
            offen.clear()
        if k == ord("c"):
            rois.clear()
            verlauf[:] = [v for v in verlauf if v != "roi"]
        if k == ord("u"):
            if offen:
                offen.pop()
            elif verlauf:
                (rois if verlauf.pop() == "roi" else spuren).pop()
    cv2.destroyAllWindows()
    if not spuren:
        raise SystemExit("Keine Linie eingezeichnet.")
    warne_bei_ueberlappung(rois)
    with open(datei, "w", encoding="utf-8") as f:
        json.dump({"rois": rois, "linien": spuren}, f, indent=2, ensure_ascii=False)
    print(f"{len(spuren)} Linie(n) und {len(rois)} Bereich(e) in {datei} gespeichert. "
          f"Der Pfeil zeigt die Richtung 'mit_pfeil'. Namen kannst du in der Datei ändern.")
    return spuren, rois


# ----------------------------------------------------------------------------
# Hauptprogramm
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Verkehrszähler für öffentliche Kamera-Streams")
    ap.add_argument("url", help="Stream- oder Snapshot-URL der Kamera")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO-Modell (n=schnell, s/m=genauer)")
    ap.add_argument("--conf", type=float, default=0.2, help="Mindest-Konfidenz (0-1)")
    ap.add_argument("--conf-low", type=float, default=0.08,
                    help="Untere Schwelle: schwächere Erkennungen dürfen bestehende Tracks weiterführen "
                         "(nur --tracker einfach), neue Tracks brauchen --conf")
    ap.add_argument("--skip", default="1",
                    help="Nur jedes N-te Frame auswerten (1 = alle) oder 'auto' (passt sich an)")
    ap.add_argument("--max-skip", type=int, default=4, help="Obergrenze bei --skip auto")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="YOLO-Eingabegröße; 0 = automatisch passend zum Ausschnitt")
    ap.add_argument("--roi", action="append", default=None,
                    help="Auswertebereich relativ: x1,y1,x2,y2 (mehrfach nutzbar, z. B. "
                         "--roi 0,0.4,0.5,1 --roi 0.5,0.4,1,1); überschreibt die Konfigurationsdatei")
    ap.add_argument("--setup", action="store_true", help="Bereiche und Zähllinien per Maus festlegen")
    ap.add_argument("--spuren", default="spuren.json", help="Konfigurationsdatei")
    ap.add_argument("--line", type=float, default=0.55,
                    help="Standard-Linienhöhe (0-1), falls keine Konfiguration existiert")
    ap.add_argument("--snapshot", action="store_true", help="URL ist ein einzelnes JPEG-Bild")
    ap.add_argument("--interval", type=float, default=0.5, help="Abrufintervall (nur Snapshot)")
    ap.add_argument("--csv", default="verkehr.csv", help="CSV-Datei für Minutenwerte")
    ap.add_argument("--tracker", choices=["einfach", "bytetrack"], default="einfach",
                    help="'einfach' (Standard) erkennt Fahrzeuge auch bei Frame-Sprüngen zuverlässiger wieder; "
                         "'bytetrack' ist der Ultralytics-Tracker")
    ap.add_argument("--zusammenfassen", action="store_true",
                    help="Alle Fahrzeugtypen als 'Fahrzeug' zählen (robust, wenn Pkw/Lkw/Bus verwechselt werden)")
    ap.add_argument("--show", action="store_true", help="Live-Vorschau (Taste q = Ende)")
    ap.add_argument("--debug", action="store_true",
                    help="Zeigt rohe YOLO-Erkennungen (rot) und Statistik (kostet Rechenzeit)")
    args = ap.parse_args()

    auto_skip = args.skip.lower() == "auto"
    skip = 1 if auto_skip else max(1, int(args.skip))

    # Gerät wählen
    if torch.cuda.is_available():
        geraet, half = "cuda:0", True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        geraet, half = "mps", False
    else:
        geraet, half = "cpu", False
    print(f"Rechengerät: {geraet}{' (FP16)' if half else ''}")

    # FP16-Einstellung versionsunabhängig: neue Ultralytics nutzen 'quantize', alte 'half'
    praezision = {}
    if half:
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            praezision = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
        except ImportError:
            praezision = {"half": True}

    # Tracker-Profil (an --conf gekoppelt, tolerant gegenüber größeren Sprüngen)
    tracker_datei = "bytetrack_zaehler.yaml"
    with open(tracker_datei, "w", encoding="utf-8") as f:
        f.write("tracker_type: bytetrack\n"
                f"track_high_thresh: {args.conf}\n"
                f"track_low_thresh: {min(0.1, args.conf)}\n"
                f"new_track_thresh: {args.conf}\n"
                "track_buffer: 30\n"
                "match_thresh: 0.95\n"
                "fuse_score: True\n")

    quelle = FrameQuelle(args.url, args.snapshot, args.interval)
    if args.setup:
        spuren, rois_rel = setup_modus(quelle, args.spuren)
    else:
        spuren, rois_rel = lade_konfig(args.spuren, args.line)
    if args.roi:
        rois_rel = []
        for r in args.roi:
            werte = [float(v) for v in r.split(",")]
            if len(werte) != 4:
                raise SystemExit("--roi erwartet 4 Werte: x1,y1,x2,y2")
            rois_rel.append(werte)
        warne_bei_ueberlappung(rois_rel)

    # Ohne Bereich wird das ganze Bild ausgewertet (None). Jeder Bereich bekommt ein eigenes Modell,
    # damit jeder Tracker nur "seine" Bilder sieht und IDs sich nicht vermischen.
    bereiche = rois_rel if rois_rel else [None]
    modelle = [YOLO(args.model) for _ in bereiche]
    roh_modell = YOLO(args.model) if args.debug and args.tracker == "bytetrack" else None
    einfache_tracker = [EinfacherTracker() for _ in bereiche]

    namen = {k: "Fahrzeug" for k in FAHRZEUGE} if args.zusammenfassen else dict(FAHRZEUGE)
    typen = list(dict.fromkeys(namen.values()))
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        if f.tell() == 0:
            csv.writer(f).writerow(["zeit", "linie", "richtung", *typen, "gesamt"])

    letzte_position = {}                                  # (bereich, track_id) -> Mittelpunkt
    gezaehlt = set()                                      # (linie, (bereich, track_id))
    stimmen = defaultdict(lambda: defaultdict(float))     # (bereich, track_id) -> Typ -> Konfidenzsumme
    summe = defaultdict(int)
    minute = defaultdict(int)
    minute_start = datetime.now().replace(second=0, microsecond=0)
    verarbeitet = 0
    ema = None            # geglättete Rechenzeit pro Frame (Sekunden)
    imgsz = args.imgsz

    print(f"Starte Zählung ({len(bereiche)} Bereich(e), Skip: {'auto' if auto_skip else skip}) ... "
          f"Strg+C zum Beenden")
    try:
        while True:
            frame = quelle.lesen(skip)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            pix = [((s["p1"][0] * w, s["p1"][1] * h), (s["p2"][0] * w, s["p2"][1] * h)) for s in spuren]

            gesamt_dauer = 0.0
            debug_teile = []
            for bi, roi in enumerate(bereiche):
                rx1, ry1, rx2, ry2 = roi_in_pixel(roi, w, h)
                ausschnitt = frame[ry1:ry2, rx1:rx2]
                imgsz = args.imgsz or auto_imgsz(rx2 - rx1, ry2 - ry1)
                versatz = np.array([rx1, ry1, rx1, ry1], dtype=np.float32)

                t0 = time.time()
                if args.tracker == "einfach":
                    res = modelle[bi].predict(ausschnitt, imgsz=imgsz, classes=list(FAHRZEUGE),
                                              conf=min(args.conf, args.conf_low), verbose=False,
                                              device=geraet, **praezision, agnostic_nms=True)[0]
                    boxen = res.boxes.xyxy.cpu().numpy() + versatz
                    konfs = res.boxes.conf.cpu().tolist()
                    ids = einfache_tracker[bi].update(boxen, konfs, args.conf)
                    klassen = res.boxes.cls.int().cpu().tolist()
                    roh = res
                else:
                    res = modelle[bi].track(ausschnitt, persist=True, tracker=tracker_datei, imgsz=imgsz,
                                            classes=list(FAHRZEUGE), conf=args.conf, verbose=False,
                                            device=geraet, **praezision, agnostic_nms=True)[0]
                    if res.boxes is None or res.boxes.id is None:
                        boxen, ids, klassen, konfs = np.zeros((0, 4)), [], [], []
                    else:
                        boxen = res.boxes.xyxy.cpu().numpy() + versatz
                        ids = res.boxes.id.int().cpu().tolist()
                        klassen = res.boxes.cls.int().cpu().tolist()
                        konfs = res.boxes.conf.cpu().tolist()
                    roh = None
                gesamt_dauer += time.time() - t0

                if args.debug:
                    if roh is None:
                        roh = roh_modell.predict(ausschnitt, imgsz=imgsz, classes=list(FAHRZEUGE),
                                                 conf=args.conf, verbose=False, device=geraet,
                                                 **praezision, agnostic_nms=True)[0]
                    debug_teile.append(f"Bereich {bi + 1} ({rx2 - rx1}x{ry2 - ry1}, imgsz {imgsz}): "
                                       f"erkannt {len(roh.boxes)}, verfolgt {len([i for i in ids if i is not None])}")
                    if args.show and args.tracker == "bytetrack":
                        for b in roh.boxes.xyxy.cpu().numpy() + versatz:
                            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                                          (0, 0, 255), 1)

                for box, tid, kls, konf in zip(boxen, ids, klassen, konfs):
                    if tid is None:       # schwache Erkennung ohne passenden Track
                        continue
                    x1, y1, x2, y2 = box
                    mitte = (float((x1 + x2) / 2), float(y2 - (y2 - y1) * 0.2))
                    schluessel = (bi, tid)
                    # Typ = konfidenzgewichtete Mehrheit über alle bisherigen Frames dieses Fahrzeugs
                    stimmen[schluessel][namen[kls]] += konf
                    typ = max(stimmen[schluessel], key=stimmen[schluessel].get)

                    if schluessel in letzte_position:
                        alt = letzte_position[schluessel]
                        for i, (p1, p2) in enumerate(pix):
                            if (i, schluessel) in gezaehlt:
                                continue
                            if segmente_schneiden(alt, mitte, p1, p2):
                                richtung = "mit_pfeil" if seite(mitte, p1, p2) > 0 else "gegen_pfeil"
                                summe[(i, richtung, typ)] += 1
                                minute[(i, richtung, typ)] += 1
                                gezaehlt.add((i, schluessel))
                    letzte_position[schluessel] = mitte

                    if args.show:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{typ} #{tid}", (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            ema = gesamt_dauer if ema is None else 0.9 * ema + 0.1 * gesamt_dauer
            if auto_skip and not args.snapshot:
                skip = int(max(1, min(args.max_skip, math.ceil(ema * quelle.fps))))
            verarbeitet += 1
            if args.debug and verarbeitet % 10 == 0:
                print(f"[debug] Frame {verarbeitet}: " + " | ".join(debug_teile) +
                      f" | {ema * 1000:.0f} ms/Frame, skip {skip}")

            # Minutenwerte wegschreiben
            jetzt = datetime.now().replace(second=0, microsecond=0)
            if jetzt > minute_start:
                with open(args.csv, "a", newline="", encoding="utf-8") as f:
                    wr = csv.writer(f)
                    for i, s in enumerate(spuren):
                        for richtung in ("mit_pfeil", "gegen_pfeil"):
                            werte = [minute[(i, richtung, t)] for t in typen]
                            wr.writerow([minute_start.isoformat(), s["name"], richtung, *werte, sum(werte)])
                print(f"[{minute_start:%H:%M}] {sum(minute.values())} Fahrzeuge in dieser Minute "
                      f"({ema * 1000:.0f} ms/Frame, skip {skip})")
                minute.clear()
                minute_start = jetzt
                if len(letzte_position) > 5000:
                    letzte_position.clear()
                    gezaehlt.clear()
                    stimmen.clear()

            if args.show:
                if args.tracker == "einfach":
                    for tr in einfache_tracker:
                        for tid, t in tr.tracks.items():
                            if t["fehlt"] > 0 and t["treffer"] >= 3:
                                gx = t["pos"][0] + t["v"][0] * t["fehlt"]
                                gy = t["pos"][1] + t["v"][1] * t["fehlt"]
                                cv2.rectangle(frame, (int(gx - t["w"] / 2), int(gy - t["h"] / 2)),
                                              (int(gx + t["w"] / 2), int(gy + t["h"] / 2)), (0, 255, 255), 1)
                                cv2.putText(frame, f"#{tid} (nicht erkannt)", (int(gx - t["w"] / 2), int(gy - t["h"] / 2) - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                zeichne_rois(frame, rois_rel)
                zeichne_linien(frame, spuren, summe)
                cv2.putText(frame, f"{ema * 1000:.0f} ms/Frame  skip {skip}  imgsz {imgsz}",
                            (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Verkehrszaehler", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        quelle.schliessen()
        try:
            cv2.destroyAllWindows()
        except cv2.error:      # opencv-python-headless hat keine Fenster-Funktionen
            pass
        print("\nEndergebnis:")
        for (i, richtung, typ), n in sorted(summe.items()):
            print(f"  {spuren[i]['name']:10s} {richtung:12s} {typ:9s} {n}")


if __name__ == "__main__":
    main()
