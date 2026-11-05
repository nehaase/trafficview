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


def schnittparameter(a, b, c, d):
    """Wo schneidet die Bewegung a->b das Linienstück c->d? Gibt s in [0, 1] zurück (0 = a, 1 = b) oder None."""
    rx, ry, sx, sy = b[0] - a[0], b[1] - a[1], d[0] - c[0], d[1] - c[1]
    nenner = rx * sy - ry * sx
    if abs(nenner) < 1e-12:
        return None
    s = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / nenner
    return min(1.0, max(0.0, s))


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
      bestehenden Track dürfen aber auch schwache Erkennungen weiterführen (kein Flackern).
    - Plausibilitätsprüfung bei der Zuordnung: begrenzter Suchradius, ähnliche Boxgröße und ein
      Tempolimit. Dadurch "springt" ein Track nicht auf Fahrbahnmarkierungen oder andere Fahrzeuge.
    - Austritt: Verschwindet ein Fahrzeug am Rand des Auswertebereichs oder läuft seine erwartete
      Bahn aus dem Bereich heraus, wird der Track sofort beendet (keine Geisterbox außerhalb).
    - Bewegungsprüfung: Ein Track gilt erst als Fahrzeug ("bewegt"), wenn seine letzten Positionen
      gleichmäßig auf einer Linie liegen (lineare Regression) und der Weg merklich ist. Statische
      Fehlerkennungen (Fahrbahnmarkierungen, Schilder) und Sprünge zwischen ihnen bestehen den Test
      nicht und werden weder gezählt noch gezeigt.
    """

    def __init__(self, max_luecke=8, faktor=2.0, daempfung=0.8, min_bewegung=0.25, max_tempo=2.0):
        self.max_luecke = max_luecke        # so viele Frames darf ein Fahrzeug fehlen
        self.faktor = faktor                # erlaubter Sprung in Fahrzeuggrößen
        self.daempfung = daempfung          # Geschwindigkeit nimmt bei fehlender Erkennung pro Frame ab
        self.min_bewegung = min_bewegung    # nötiger Weg (in Boxgrößen) im Fenster, bevor ein Track als Fahrzeug gilt
        self.max_tempo = max_tempo          # max. Geschwindigkeit in Boxgrößen pro Frame
        self.gedaechtnis = 8                # so viele Frames gilt ein Track nach der letzten Bewegung als Fahrzeug
        self.tracks = {}
        self.naechste_id = 1
        self.frame = 0

    @staticmethod
    def _gleichmaessig(verlauf, min_weg, min_r2=0.85):
        """
        Bewegt sich der Track gleichmäßig? Lineare Regression über die letzten Positionen (Frame, x, y):
        Der Weg im Fenster muss >= min_weg sein und die Punkte müssen gut auf einer Geraden liegen.
        Ein einzelner Sprung (z. B. Wechsel zwischen zwei statischen Fehlerkennungen) fällt durch.
        """
        n = len(verlauf)
        if n < 3:
            return False
        ts = [v[0] for v in verlauf]
        tm = sum(ts) / n
        stt = sum((t - tm) ** 2 for t in ts)
        if stt <= 0:
            return False
        steigung, ssres, sstot = [], 0.0, 0.0
        for achse in (1, 2):
            werte = [v[achse] for v in verlauf]
            wm = sum(werte) / n
            m = sum((t - tm) * (w - wm) for t, w in zip(ts, werte)) / stt
            steigung.append(m)
            ssres += sum((w - (wm + m * (t - tm))) ** 2 for t, w in zip(ts, werte))
            sstot += sum((w - wm) ** 2 for w in werte)
        if sstot < 1e-9:
            return False
        weg = math.hypot(steigung[0], steigung[1]) * (ts[-1] - ts[0])
        return weg >= min_weg and 1.0 - ssres / sstot >= min_r2

    def _weg(self, t, schritte, gedaempft=False):
        """
        Erwartete Position nach `schritte` Frames. Für die Zuordnung fährt das Fahrzeug mit konstanter
        Geschwindigkeit weiter; für die Anzeige der Geisterbox nimmt die Geschwindigkeit ab, damit die
        Box nicht davonläuft.
        """
        d = self.daempfung
        faktor = schritte if (not gedaempft or d >= 1.0) else (1.0 - d ** schritte) / (1.0 - d)
        return (t["pos"][0] + t["v"][0] * faktor, t["pos"][1] + t["v"][1] * faktor)

    @staticmethod
    def _am_rand(box, grenzen, toleranz=2.0):
        """Berührt die Box den Rand des Auswertebereichs? (YOLO schneidet Boxen am Ausschnitt ab.)"""
        if grenzen is None:
            return False
        return (box[0] <= grenzen[0] + toleranz or box[1] <= grenzen[1] + toleranz or
                box[2] >= grenzen[2] - toleranz or box[3] >= grenzen[3] - toleranz)

    @staticmethod
    def _innerhalb(punkt, grenzen):
        return grenzen is None or (grenzen[0] <= punkt[0] <= grenzen[2] and grenzen[1] <= punkt[1] <= grenzen[3])

    def update(self, boxen, konfs=None, neu_schwelle=0.0, grenzen=None):
        """
        boxen: Array (N,4) xyxy; konfs: Konfidenzen; grenzen: (x1,y1,x2,y2) des Auswertebereichs.
        Gibt die IDs in gleicher Reihenfolge zurück (None = schwache Erkennung ohne passenden Track).
        """
        self.frame += 1
        n = len(boxen)
        konfs = [1.0] * n if konfs is None else list(konfs)
        ids = [None] * n
        mitten = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxen]
        masse = [(float(b[2] - b[0]), float(b[3] - b[1])) for b in boxen]
        groessen = [max(w, h, 1.0) for w, h in masse]
        flaechen = [max(w * h, 1.0) for w, h in masse]
        rand = [self._am_rand(b, grenzen) for b in boxen]

        paare = []
        for tid, t in self.tracks.items():
            erwartet = self._weg(t, 1 + t["fehlt"])
            tempo = math.hypot(t["v"][0], t["v"][1])
            for j in range(n):
                groesse = max(groessen[j], t["groesse"])
                dx, dy = mitten[j][0] - erwartet[0], mitten[j][1] - erwartet[1]
                if tempo >= 0.15 * groesse:
                    # Fahrendes Fahrzeug: Suchbereich ist eine Ellipse entlang der Fahrtrichtung. Quer dazu
                    # ist wenig Abweichung erlaubt, rückwärts gar keine (verhindert Sprünge auf Markierungen).
                    ux, uy = t["v"][0] / tempo, t["v"][1] / tempo
                    laengs, quer = dx * ux + dy * uy, -dx * uy + dy * ux
                    zurueck = (mitten[j][0] - t["pos"][0]) * ux + (mitten[j][1] - t["pos"][1]) * uy
                    if zurueck < -0.5 * groesse:
                        continue
                    abstand = math.hypot(laengs / (self.faktor * groesse * (1 + 0.3 * min(t["fehlt"], 4))),
                                         quer / (0.7 * groesse * (1 + 0.15 * min(t["fehlt"], 4))))
                elif t["treffer"] <= 3:
                    # Junger Track ohne bekannte Geschwindigkeit: darf soweit suchen, wie ein Fahrzeug
                    # mit Höchsttempo in der Zwischenzeit kommen kann
                    radius = groesse * max(self.faktor, self.max_tempo * (1 + t["fehlt"]))
                    abstand = math.hypot(dx, dy) / radius
                else:
                    # Lange stehender Track (z. B. Fahrbahnmarkierung): kleiner Radius, damit er keine
                    # vorbeifahrenden Fahrzeuge "übernimmt"
                    radius = 0.6 * groesse * (1 + 0.1 * min(t["fehlt"], 4))
                    abstand = math.hypot(dx, dy) / radius
                if abstand > 1.0:
                    continue
                # Boxgröße und Seitenverhältnis müssen zum Track passen (am Rand abgeschnittene Boxen
                # sind davon ausgenommen)
                if not (rand[j] or t["am_rand"]):
                    if not 0.2 <= flaechen[j] / t["flaeche"] <= 5.0:
                        continue
                    seiten_neu, seiten_alt = masse[j][0] / max(masse[j][1], 1.0), t["w"] / max(t["h"], 1.0)
                    if max(seiten_neu / seiten_alt, seiten_alt / seiten_neu) > 2.5:
                        continue
                # Reihenfolge: sichere Erkennungen zuerst, dann aktive vor verlorenen Tracks
                paare.append((konfs[j] < neu_schwelle, t["fehlt"] > 0, abstand, tid, j))
        paare.sort()

        benutzt_t, benutzt_d = set(), set()
        for _, _, _, tid, j in paare:
            if tid in benutzt_t or j in benutzt_d:
                continue
            t = self.tracks[tid]
            schritte = 1 + t["fehlt"]
            neu_v = ((mitten[j][0] - t["pos"][0]) / schritte, (mitten[j][1] - t["pos"][1]) / schritte)
            vx = 0.6 * neu_v[0] + 0.4 * t["v"][0]
            vy = 0.6 * neu_v[1] + 0.4 * t["v"][1]
            tempo, limit = math.hypot(vx, vy), self.max_tempo * groessen[j]
            if tempo > limit:                          # Tempolimit: unplausible Geschwindigkeit kappen
                vx, vy = vx * limit / tempo, vy * limit / tempo
            t["v"] = (vx, vy)
            t["pos"], t["groesse"], t["flaeche"], t["fehlt"] = mitten[j], groessen[j], flaechen[j], 0
            t["w"], t["h"] = masse[j]
            t["treffer"] += 1
            t["am_rand"] = rand[j]
            t["verlauf"].append((self.frame, mitten[j][0], mitten[j][1]))
            del t["verlauf"][:-8]
            if self._gleichmaessig(t["verlauf"], self.min_bewegung * groessen[j]):
                t["bewegt_bis"] = self.frame + self.gedaechtnis   # Einstufung als Fahrzeug erneuern
            ids[j] = tid
            benutzt_t.add(tid)
            benutzt_d.add(j)

        for tid in list(self.tracks):
            if tid in benutzt_t:
                continue
            t = self.tracks[tid]
            t["fehlt"] += 1
            # Fahrzeug hat den Bereich verlassen: am Rand verschwunden oder erwartete Bahn läuft hinaus
            ausgetreten = t["am_rand"] or not self._innerhalb(self._weg(t, t["fehlt"]), grenzen)
            if t["fehlt"] > self.max_luecke or ausgetreten:
                del self.tracks[tid]

        for t in self.tracks.values():
            # "bewegt" verfällt, wenn eine Weile keine gleichmäßige Bewegung mehr festgestellt wurde
            war_bewegt = t["bewegt"]
            t["bewegt"] = self.min_bewegung <= 0 or self.frame <= t["bewegt_bis"]
            t["neu_bewegt"] = t["bewegt"] and not war_bewegt and self.min_bewegung > 0
            if t["neu_bewegt"]:
                # wie viele der letzten Zuordnungen bilden den gleichmäßigen Bewegungsabschnitt?
                t["lauf_laenge"] = next((k for k in range(len(t["verlauf"]), 2, -1) if self._gleichmaessig(
                    t["verlauf"][-k:], self.min_bewegung * t["groesse"])), 3)

        for j in range(n):
            if ids[j] is None and konfs[j] >= neu_schwelle:
                self.tracks[self.naechste_id] = {
                    "pos": mitten[j], "v": (0.0, 0.0), "groesse": groessen[j], "flaeche": flaechen[j],
                    "fehlt": 0, "w": masse[j][0], "h": masse[j][1], "treffer": 1, "am_rand": rand[j],
                    "verlauf": [(self.frame, mitten[j][0], mitten[j][1])], "bewegt_bis": -1,
                    "bewegt": self.min_bewegung <= 0, "neu_bewegt": False, "lauf_laenge": 0}
                ids[j] = self.naechste_id
                self.naechste_id += 1
        return ids

    def geister(self, grenzen=None, max_fehlt=4):
        """Verfolgte Fahrzeuge, die im aktuellen Frame nicht erkannt wurden: (id, (x, y), breite, höhe)."""
        ergebnis = []
        for tid, t in self.tracks.items():
            if 1 <= t["fehlt"] <= max_fehlt and t["treffer"] >= 3 and t["bewegt"]:
                pos = self._weg(t, t["fehlt"], gedaempft=True)
                if self._innerhalb(pos, grenzen):
                    ergebnis.append((tid, pos, t["w"], t["h"]))
        return ergebnis


class UeberfahrtZaehler:
    """
    Erkennt, wann ein verfolgtes Fahrzeug eine Zähllinie überquert (je Linie nur einmal).

    - Statische Fehlerkennungen werden nicht gezählt: Gezählt wird nur, solange der Track als Fahrzeug
      gilt ("bewegt"). Wird ein Track erst nach einigen Frames als Fahrzeug eingestuft, prüft die Klasse
      nachträglich den Weg seines Beobachtungsfensters (Überquerung kurz nach dem Auftauchen).
    - Geisterbox: Wird ein verfolgtes Fahrzeug kurz vor der Linie nicht mehr erkannt, die vorhergesagte
      (gelbe) Box überquert die Linie aber trotzdem, zählt das Fahrzeug (pruefen_geist).
    - Doppelzählungen: Hätten sich die Boxen zweier Überquerungen an der Linie eng überlappt (ungefähr
      gleicher Ort und gleiche Zeit), wird nur die erste gezählt. Das betrifft nur Fälle mit Geisterbox,
      z. B. wenn dasselbe Fahrzeug nach einem Aussetzer als neuer Track wieder auftaucht. Zwei normal
      erkannte, getrennte Fahrzeuge werden weiterhin beide gezählt.

    Pro ausgewertetem Frame einmal neuer_frame() aufrufen (Zeitbezug für die Doppelzählungs-Prüfung).
    """

    def __init__(self, fenster=8, ueberlappung=0.3, zeitfenster=6):
        self.fenster = fenster
        self.ueberlappung = ueberlappung   # Anteil der kleineren Box, ab dem Boxen als "eng überlappend" gelten
        self.zeitfenster = zeitfenster     # so nah (in Frames) müssen zwei Überquerungen beieinander liegen
        self.frame = 0
        self.letzte = {}                   # Schlüssel -> (frame, letzter Zählpunkt)
        self.verlauf = {}                  # Schlüssel -> [(frame, Zählpunkt)] (Beobachtungsfenster)
        self.gezaehlt = set()              # (linie, schluessel)
        self.ereignisse = {}               # linie -> [(zeit, ort, breite, höhe, geisterbox)] zuletzt gezählt
        self.unterdrueckt = set()          # (linie, schluessel): wegen enger Überlappung nicht gezählt
        self.statistik = {"geisterbox": 0, "unterdrueckt": 0}

    def neuer_frame(self):
        self.frame += 1

    @staticmethod
    def _box_um(punkt, w, h):
        """Box um einen Zählpunkt (Unterkante minus 20 % Höhe, wie bei erkannten Fahrzeugen)."""
        return (punkt[0] - w / 2, punkt[1] - 0.8 * h, punkt[0] + w / 2, punkt[1] + 0.2 * h)

    @staticmethod
    def _ueberlappt(a, b, schwelle):
        """Anteil der Schnittfläche an der kleineren der beiden Boxen (x1, y1, x2, y2) >= schwelle?"""
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        if ix <= 0 or iy <= 0:
            return False
        kleiner = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return kleiner > 0 and ix * iy / kleiner >= schwelle

    def _doppelt(self, i, schluessel, zeit, ort, groesse, geist):
        """Hätte sich diese Überquerung eng mit einer kürzlich gezählten überlappt (nur Fälle mit Geisterbox)?"""
        if self.ueberlappung <= 0 or groesse is None:
            return False
        box = self._box_um(ort, *groesse)
        for (z, o, b, h, g, k) in self.ereignisse.get(i, ()):
            if k == schluessel or abs(zeit - z) > self.zeitfenster or not (geist or g):
                continue
            if self._ueberlappt(box, self._box_um(o, b, h), self.ueberlappung):
                return True
        return False

    def _ueberquerungen(self, schluessel, alt, punkt, linien, groesse, geist):
        """alt, punkt: (frame, (x, y)). Gibt die neu gezählten Überquerungen [(linie, richtung)] zurück."""
        ereignisse = []
        for i, (p1, p2) in enumerate(linien):
            if (i, schluessel) in self.gezaehlt or (i, schluessel) in self.unterdrueckt:
                continue
            if not segmente_schneiden(alt[1], punkt[1], p1, p2):
                continue
            s = schnittparameter(alt[1], punkt[1], p1, p2)
            s = 1.0 if s is None else s
            zeit = alt[0] + s * (punkt[0] - alt[0])                     # wann (Frame) die Linie überquert wurde
            ort = (alt[1][0] + s * (punkt[1][0] - alt[1][0]), alt[1][1] + s * (punkt[1][1] - alt[1][1]))
            if self._doppelt(i, schluessel, zeit, ort, groesse, geist):
                self.unterdrueckt.add((i, schluessel))
                self.statistik["unterdrueckt"] += 1
                continue
            self.gezaehlt.add((i, schluessel))
            liste = self.ereignisse.setdefault(i, [])
            liste.append((zeit, ort, *(groesse or (0.0, 0.0)), geist, schluessel))
            del liste[:max(0, len(liste) - 50)]                          # nur die letzten Überquerungen merken
            if geist:
                self.statistik["geisterbox"] += 1
            ereignisse.append((i, "mit_pfeil" if seite(punkt[1], p1, p2) > 0 else "gegen_pfeil"))
        return ereignisse

    def pruefen(self, schluessel, punkt, linien, bewegt=True, neu_bewegt=False, lauf_laenge=0, box=None):
        """
        Gibt eine Liste (linien_index, richtung) der neu erkannten Überquerungen zurück.
        lauf_laenge: Anzahl der letzten Zuordnungen (inkl. der aktuellen), die den gleichmäßigen
        Bewegungsabschnitt bilden. Beim Übergang zu "Fahrzeug" wird dessen ganzer Weg geprüft.
        box: (x1, y1, x2, y2) des Fahrzeugs, für die Prüfung auf Doppelzählung.
        """
        jetzt = (self.frame, punkt)
        ereignisse = []
        if bewegt:
            alt = None
            if neu_bewegt and self.verlauf.get(schluessel):
                v = self.verlauf[schluessel]
                alt = v[-(lauf_laenge - 1)] if 2 <= lauf_laenge - 1 <= len(v) else v[0]
            elif schluessel in self.letzte:
                alt = self.letzte[schluessel]
            if alt is not None:
                groesse = None if box is None else (box[2] - box[0], box[3] - box[1])
                ereignisse = self._ueberquerungen(schluessel, alt, jetzt, linien, groesse, False)
        self.letzte[schluessel] = jetzt
        v = self.verlauf.setdefault(schluessel, [])
        v.append(jetzt)
        del v[:-self.fenster]
        return ereignisse

    def pruefen_geist(self, schluessel, punkt, linien, box=None):
        """Wie pruefen, aber für die vorhergesagte Position (Geisterbox) eines im Moment nicht erkannten Fahrzeugs."""
        jetzt = (self.frame, punkt)
        alt = self.letzte.get(schluessel)
        groesse = None if box is None else (box[2] - box[0], box[3] - box[1])
        ereignisse = [] if alt is None else self._ueberquerungen(schluessel, alt, jetzt, linien, groesse, True)
        self.letzte[schluessel] = jetzt
        return ereignisse

    def anzahl_tracks(self):
        return len(self.letzte)

    def leeren(self):
        self.letzte.clear()
        self.verlauf.clear()
        self.gezaehlt.clear()
        self.ereignisse.clear()
        self.unterdrueckt.clear()


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
    ap.add_argument("--min-bewegung", type=float, default=0.25,
                    help="Ein Track gilt erst als Fahrzeug, wenn er sich gleichmäßig um so viele Boxgrößen "
                         "bewegt hat. Filtert statische Fehlerkennungen (Fahrbahnmarkierungen). "
                         "0 = Filter aus (nur --tracker einfach)")
    ap.add_argument("--ueberlappung", type=float, default=0.3,
                    help="Hätten sich die Boxen zweier Überquerungen an der Linie mindestens so stark überlappt "
                         "(Anteil der kleineren Box) und ist eine davon eine Geisterbox, wird nur die erste "
                         "gezählt. 0 = Prüfung aus (nur --tracker einfach)")
    ap.add_argument("--ohne-geisterzaehlung", action="store_true",
                    help="Fahrzeuge, die kurz vor der Linie nicht mehr erkannt werden, nicht anhand der "
                         "Geisterbox zählen")
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

    import torch                       # erst hier laden: macht Tracker/Geometrie ohne PyTorch importierbar (Tests)
    from ultralytics import YOLO

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

    # Tracker-Profil für ByteTrack (an --conf gekoppelt, tolerant gegenüber größeren Sprüngen);
    # wird nur gebraucht, wenn --tracker bytetrack gewählt ist
    tracker_datei = "bytetrack_zaehler.yaml"
    if args.tracker == "bytetrack":
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
    if args.tracker == "einfach":
        # Die Erkennung ist zustandslos: ein Modell für alle Bereiche spart Speicher und Startzeit
        modelle = [YOLO(args.model)] * len(bereiche)
    else:
        modelle = [YOLO(args.model) for _ in bereiche]     # ByteTrack hält Zustand pro Modell
    roh_modell = YOLO(args.model) if args.debug and args.tracker == "bytetrack" else None
    einfache_tracker = [EinfacherTracker(min_bewegung=args.min_bewegung) for _ in bereiche]

    namen = {k: "Fahrzeug" for k in FAHRZEUGE} if args.zusammenfassen else dict(FAHRZEUGE)
    typen = list(dict.fromkeys(namen.values()))
    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        if f.tell() == 0:
            csv.writer(f).writerow(["zeit", "linie", "richtung", *typen, "gesamt"])

    zaehler = UeberfahrtZaehler(ueberlappung=args.ueberlappung)   # Schlüssel: (bereich, track_id)
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
            geister_alle = []                  # (id, (x, y), breite, höhe) aller Bereiche für die Anzeige
            zaehler.neuer_frame()
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
                    ids = einfache_tracker[bi].update(boxen, konfs, args.conf, grenzen=(rx1, ry1, rx2, ry2))
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

                geister_bereich = []
                if args.tracker == "einfach":
                    geister_bereich = einfache_tracker[bi].geister((rx1, ry1, rx2, ry2))
                    geister_alle.extend(geister_bereich)

                for box, tid, kls, konf in zip(boxen, ids, klassen, konfs):
                    if tid is None:       # schwache Erkennung ohne passenden Track
                        continue
                    x1, y1, x2, y2 = box
                    mitte = (float((x1 + x2) / 2), float(y2 - (y2 - y1) * 0.2))
                    schluessel = (bi, tid)
                    # Typ = konfidenzgewichtete Mehrheit über alle bisherigen Frames dieses Fahrzeugs
                    stimmen[schluessel][namen[kls]] += konf
                    typ = max(stimmen[schluessel], key=stimmen[schluessel].get)

                    # statische Fehlerkennungen (Fahrbahnmarkierungen o. ä.) weder zählen noch zeigen
                    if args.tracker == "einfach":
                        track = einfache_tracker[bi].tracks[tid]
                        bewegt, neu_bewegt, lauf = track["bewegt"], track["neu_bewegt"], track["lauf_laenge"]
                    else:
                        bewegt, neu_bewegt, lauf = True, False, 0
                    for i, richtung in zaehler.pruefen(schluessel, mitte, pix, bewegt, neu_bewegt, lauf,
                                                       box=(x1, y1, x2, y2)):
                        summe[(i, richtung, typ)] += 1
                        minute[(i, richtung, typ)] += 1

                    if args.show and bewegt:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{typ} #{tid}", (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Fahrzeuge, die gerade nicht erkannt werden: überquert die Geisterbox die Linie, zählt das
                # Fahrzeug als das, was es zuvor (per Mehrheit über alle Frames) erkannt wurde
                if args.tracker == "einfach" and not args.ohne_geisterzaehlung:
                    for gid, (gx, gy), gw, gh in geister_bereich:
                        schluessel = (bi, gid)
                        abstimmung = stimmen.get(schluessel)
                        if not abstimmung:
                            continue
                        typ = max(abstimmung, key=abstimmung.get)
                        punkt = (float(gx), float(gy + 0.3 * gh))      # gleicher Zählpunkt wie bei Erkennungen
                        for i, richtung in zaehler.pruefen_geist(
                                schluessel, punkt, pix, (gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2)):
                            summe[(i, richtung, typ)] += 1
                            minute[(i, richtung, typ)] += 1

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
                if zaehler.anzahl_tracks() > 5000:
                    zaehler.leeren()
                    stimmen.clear()

            if args.show:
                for tid, (gx, gy), gw, gh in geister_alle:
                    cv2.rectangle(frame, (int(gx - gw / 2), int(gy - gh / 2)),
                                  (int(gx + gw / 2), int(gy + gh / 2)), (0, 255, 255), 1)
                    cv2.putText(frame, f"#{tid} (nicht erkannt)", (int(gx - gw / 2), int(gy - gh / 2) - 4),
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
        if args.tracker == "einfach":
            print(f"davon über Geisterbox gezählt: {zaehler.statistik['geisterbox']}, "
                  f"wegen enger Überlappung nicht doppelt gezählt: {zaehler.statistik['unterdrueckt']}")


if __name__ == "__main__":
    main()
