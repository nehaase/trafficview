"""
Tests für TrafficView. Sie brauchen weder PyTorch noch YOLO noch eine Netzwerkverbindung.

    python -m unittest -v test_trafficview

Die Simulationstests bilden typische Probleme öffentlicher Kameras nach (abgeschnittene Boxen am Rand des
Auswertebereichs, flackernde Fehlerkennungen von Fahrbahnmarkierungen, Aussetzer der Erkennung, große
Sprünge zwischen ausgewerteten Frames) und prüfen Zählgenauigkeit und Geisterboxen.
"""
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trafficview as tv
from trafficview import (EinfacherTracker, SofortZaehler, UeberfahrtZaehler, empfohlener_skip, lade_konfig,
                         neuer_csv_name, segmente_schneiden, seite)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
class Szene:
    """Zwei Fahrbahnen, Fahrzeuge fahren abwärts durch einen Auswertebereich der Größe w x h."""

    def __init__(self, seed, w=260, h=316, aussetzer=0.08, markierungen=True, spuren=((80, 9.0), (180, 13.0)),
                 fz=(50.0, 40.0), frames=900, abstand=16, unsichtbar_vor_linie=None, blindzone=None,
                 aussehen_wechselt_ab=None):
        self.w, self.h, self.frames = w, h, frames
        self.unsichtbar_vor_linie = unsichtbar_vor_linie   # Erkennung bricht so viele px vor der Linie ab
        self.blindzone = blindzone                         # (px davor, px dahinter): hier keine Erkennung
        self.aussehen_wechselt_ab = aussehen_wechselt_ab   # so viele px vor der Linie ändert sich die Boxform (Anhänger o. ä.)
        self.linie_y = h / 2.0
        self.fz_b, self.fz_h = fz
        self.aussetzer = aussetzer
        self.rng = random.Random(seed)
        # statische Fehlerkennungen: x, y, Breite, Höhe (liegen auf den Fahrbahnen)
        self.markierungen = [(70, 110, 70, 16), (190, 235, 60, 14), (130, 60, 40, 12)] if markierungen else []
        self.fahrzeuge, letzter = [], [-99] * len(spuren)
        for f in range(frames - 120):
            for i, (x, v0) in enumerate(spuren):
                if f - letzter[i] >= abstand and self.rng.random() < 0.05:
                    letzter[i] = f
                    self.fahrzeuge.append({"idx": len(self.fahrzeuge), "start": f,
                                           "x": x + self.rng.uniform(-6, 6), "v": v0 + self.rng.uniform(-2, 2)})

    def erkennungen(self, f):
        """Gibt (boxen, konfidenzen, herkunft) zurück; herkunft = Fahrzeugindex oder -1 (Markierung)."""
        boxen, konfs, herkunft = [], [], []
        for z in self.fahrzeuge:
            if f < z["start"]:
                continue
            y = -self.fz_h + z["v"] * (f - z["start"])
            if y - self.fz_h / 2 > self.h:
                continue
            y_zaehl = y + 0.3 * self.fz_h                   # Zählpunkt (wie im Programm: Unterkante - 20 % Höhe)
            if self.unsichtbar_vor_linie is not None and y_zaehl >= self.linie_y - self.unsichtbar_vor_linie:
                continue
            if self.blindzone and self.linie_y - self.blindzone[0] <= y_zaehl <= self.linie_y + self.blindzone[1]:
                continue
            b_, h_ = self.fz_b, self.fz_h
            if self.aussehen_wechselt_ab is not None and y_zaehl >= self.linie_y - self.aussehen_wechselt_ab:
                b_, h_ = b_ * 2.2, h_ * 0.5            # andere Boxform: der Tracker ordnet sie nicht dem alten Track zu
            x1, y1, x2, y2 = z["x"] - b_ / 2, y - h_ / 2, z["x"] + b_ / 2, y + h_ / 2
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(self.w, x2), min(self.h, y2)   # YOLO schneidet am Ausschnitt ab
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            anteil = (cx2 - cx1) * (cy2 - cy1) / (self.fz_b * self.fz_h)
            if anteil < 0.3 or self.rng.random() < self.aussetzer:
                continue
            konf = max(0.05, self.rng.gauss(0.35, 0.08) * anteil ** 0.5)
            if konf < 0.08:
                continue
            boxen.append([cx1, cy1, cx2, cy2]); konfs.append(konf); herkunft.append(z["idx"])
        for mx, my, mw, mh in self.markierungen:
            if self.rng.random() < 0.5:
                boxen.append([mx - mw / 2, my - mh / 2, mx + mw / 2, my + mh / 2])
                konfs.append(self.rng.uniform(0.09, 0.40)); herkunft.append(-1)
        return np.array(boxen, dtype=float).reshape(-1, 4), konfs, herkunft


def fahre(szene, geist=True, ueberlappung=0.3, **tracker_args):
    """Lässt die Szene durch Tracker und Zähler laufen, so wie es die Hauptschleife tut."""
    linie = [((0, szene.linie_y), (szene.w, szene.linie_y))]
    grenzen = (0, 0, szene.w, szene.h)
    tr, zaehl = EinfacherTracker(**tracker_args), UeberfahrtZaehler(ueberlappung=ueberlappung)
    ergebnis = {"gezaehlt": 0, "wahr": len(szene.fahrzeuge), "boxen": 0, "markierungsboxen": 0,
                "geister": 0, "geister_aussen": 0, "geister_max": 0.0, "geisterzaehlungen": 0, "unterdrueckt": 0}
    letzte_echte = {}
    for f in range(szene.frames):
        boxen, konfs, herkunft = szene.erkennungen(f)
        ids = tr.update(boxen, konfs, 0.2, grenzen=grenzen)
        zaehl.neuer_frame()
        for b, tid, h in zip(boxen, ids, herkunft):
            if tid is None:
                continue
            punkt = ((b[0] + b[2]) / 2, b[3] - (b[3] - b[1]) * 0.2)
            t = tr.tracks[tid]
            ergebnis["gezaehlt"] += len(zaehl.pruefen(tid, punkt, linie, t["bewegt"], t["neu_bewegt"],
                                                      t["lauf_laenge"], box=tuple(b)))
            if t["bewegt"]:
                ergebnis["boxen"] += 1
                ergebnis["markierungsboxen"] += (h < 0)
            letzte_echte[tid] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for tid, pos, gw, gh in tr.geister(grenzen):
            ergebnis["geister"] += 1
            ergebnis["geister_max"] = max(ergebnis["geister_max"], math.dist(pos, letzte_echte[tid]))
            ergebnis["geister_aussen"] += not (0 <= pos[0] <= szene.w and 0 <= pos[1] <= szene.h)
            if geist:
                box = (pos[0] - gw / 2, pos[1] - gh / 2, pos[0] + gw / 2, pos[1] + gh / 2)
                ergebnis["gezaehlt"] += len(zaehl.pruefen_geist(tid, (pos[0], pos[1] + 0.3 * gh), linie, box))
    ergebnis["geisterzaehlungen"] = zaehl.statistik["geisterbox"]
    ergebnis["unterdrueckt"] = zaehl.statistik["unterdrueckt"]
    return ergebnis


def sprung_szene(seed, sprung, aussetzer, rauschen=3.0):
    """Gegenverkehr auf zwei Fahrbahnen mit großen Sprüngen pro Frame (wie bei --skip)."""
    rng = random.Random(seed)
    fz = [(k // 2 * 6, 150 if k % 2 == 0 else 500, 40 if k % 2 == 0 else 380, 1 if k % 2 == 0 else -1) for k in range(40)]
    linie = [((0, 200), (700, 200))]
    tr, zaehl, n = EinfacherTracker(), UeberfahrtZaehler(), 0
    for frame in range(0, 40 * 6 // 2 + 60):
        boxen = []
        for start, x, y0, r in fz:
            t = frame - start
            y = y0 + r * sprung * t
            if t < 0 or not (0 <= y <= 480) or rng.random() < aussetzer:
                continue
            bx, by = x + rng.uniform(-rauschen, rauschen), y + rng.uniform(-rauschen, rauschen)
            boxen.append([bx - 20, by - 12, bx + 20, by + 12])
        boxen = np.array(boxen).reshape(-1, 4)
        ids = tr.update(boxen, None, 0.0, grenzen=(0, 0, 700, 480))
        zaehl.neuer_frame()
        for b, tid in zip(boxen, ids):
            t = tr.tracks[tid]
            n += len(zaehl.pruefen(tid, ((b[0] + b[2]) / 2, b[3] - (b[3] - b[1]) * 0.2), linie,
                                   t["bewegt"], t["neu_bewegt"], t["lauf_laenge"]))
    return n


# ----------------------------------------------------------------------------
# Einzelne Bausteine
# ----------------------------------------------------------------------------
class TestGeometrie(unittest.TestCase):
    def test_segmente(self):
        self.assertTrue(segmente_schneiden((5, 0), (5, 10), (0, 5), (10, 5)))
        self.assertFalse(segmente_schneiden((5, 0), (5, 4), (0, 5), (10, 5)))      # endet vor der Linie
        self.assertFalse(segmente_schneiden((15, 0), (15, 10), (0, 5), (10, 5)))   # neben dem Linienstück

    def test_seite(self):
        self.assertGreater(seite((5, 10), (0, 5), (10, 5)), 0)
        self.assertLess(seite((5, 0), (0, 5), (10, 5)), 0)


class TestCsvName(unittest.TestCase):
    def test_name_nach_datum_und_uhrzeit(self):
        from datetime import datetime
        with tempfile.TemporaryDirectory() as tmp:
            pfad = neuer_csv_name(datetime(2026, 11, 8, 14, 5, 9), tmp)
            self.assertEqual(os.path.basename(pfad), "tracking-data-2026-11-08-14-05-09.csv")

    def test_vorhandene_datei_wird_nie_ueberschrieben(self):
        from datetime import datetime
        zeit = datetime(2026, 11, 8, 14, 5, 9)
        with tempfile.TemporaryDirectory() as tmp:
            erste = neuer_csv_name(zeit, tmp)
            open(erste, "w").close()
            zweite = neuer_csv_name(zeit, tmp)
            open(zweite, "w").close()
            dritte = neuer_csv_name(zeit, tmp)
            self.assertEqual([os.path.basename(p) for p in (erste, zweite, dritte)],
                             ["tracking-data-2026-11-08-14-05-09.csv", "tracking-data-2026-11-08-14-05-09-2.csv",
                              "tracking-data-2026-11-08-14-05-09-3.csv"])


class TestBewegungspruefung(unittest.TestCase):
    def test_gleichmaessige_fahrt_gilt_als_bewegung(self):
        verlauf = [(f, 100.0, 20.0 + 10 * f) for f in range(5)]
        self.assertTrue(EinfacherTracker._gleichmaessig(verlauf, 12.0))

    def test_einzelner_sprung_gilt_nicht_als_bewegung(self):
        # z. B. Wechsel einer statischen Fehlerkennung zwischen zwei Markierungen
        for verlauf in ([(0, 70, 110), (1, 70, 110), (2, 130, 60), (3, 130, 60)],
                        [(0, 70, 110), (1, 130, 60), (2, 130, 60), (3, 130, 60), (4, 130, 60)]):
            self.assertFalse(EinfacherTracker._gleichmaessig(verlauf, 12.0))

    def test_stehendes_objekt_mit_zittern(self):
        rng = random.Random(1)
        verlauf = [(f, 70 + rng.uniform(-3, 3), 110 + rng.uniform(-3, 3)) for f in range(8)]
        self.assertFalse(EinfacherTracker._gleichmaessig(verlauf, 12.0))


class TestTracker(unittest.TestCase):
    GRENZEN = (0, 0, 260, 316)

    def test_austritt_beendet_track_und_zeigt_keine_geisterbox_ausserhalb(self):
        """Regression: Geisterboxen flogen aus dem Bereich heraus, wenn ein Fahrzeug ihn verließ."""
        tr = EinfacherTracker()
        for f in range(40):
            y = 20.0 + 12 * f                          # Fahrzeug fährt unten aus dem Bereich
            box = [60, max(0, y - 20), 110, min(316, y + 20)]
            tr.update(np.array([box]) if box[1] < 316 else np.zeros((0, 4)), [0.4] if box[1] < 316 else [],
                            0.2, grenzen=self.GRENZEN)
            for _, pos, _, _ in tr.geister(self.GRENZEN):
                self.assertTrue(0 <= pos[0] <= 260 and 0 <= pos[1] <= 316, f"Geisterbox außerhalb bei {pos}")
        self.assertEqual(len(tr.tracks), 0, "Track muss nach dem Austritt beendet sein")

    def test_statischer_track_uebernimmt_kein_vorbeifahrendes_fahrzeug(self):
        """Regression: Markierungs-Tracks 'teleportierten' auf Fahrzeuge und verfälschten die Zählung."""
        tr = EinfacherTracker()
        markierung = np.array([[35.0, 102.0, 105.0, 118.0]])             # 70 x 16, statisch
        for _ in range(8):
            tr.update(markierung, [0.3], 0.2, grenzen=self.GRENZEN)
        marker_id = next(iter(tr.tracks))
        fahrzeug = np.array([[130.0, 190.0, 180.0, 230.0]])              # 50 x 40, 90 px entfernt
        ids = tr.update(fahrzeug, [0.4], 0.2, grenzen=self.GRENZEN)
        self.assertNotEqual(ids[0], marker_id)

    def test_schwache_erkennung_startet_keinen_track_fuehrt_aber_einen_weiter(self):
        tr = EinfacherTracker()
        self.assertEqual(tr.update(np.array([[10.0, 10.0, 60.0, 50.0]]), [0.1], 0.2), [None])
        tid = tr.update(np.array([[10.0, 10.0, 60.0, 50.0]]), [0.5], 0.2)[0]
        self.assertEqual(tr.update(np.array([[10.0, 22.0, 60.0, 62.0]]), [0.1], 0.2), [tid])

    def test_geschwindigkeit_wird_begrenzt(self):
        tr = EinfacherTracker()
        tr.update(np.array([[0.0, 0.0, 40.0, 40.0]]), [0.5], 0.2)
        tr.update(np.array([[0.0, 70.0, 40.0, 110.0]]), [0.5], 0.2)        # Sprung von 70 px
        for t in tr.tracks.values():
            self.assertLessEqual(math.hypot(*t["v"]), tr.max_tempo * 40 + 1e-9)


class TestZaehler(unittest.TestCase):
    LINIE = [((0, 50), (100, 50))]

    @staticmethod
    def _schritt(z, aufruf):
        """Ein ausgewerteter Frame: Zeit weiter, dann der Aufruf."""
        z.neuer_frame()
        return aufruf()

    def test_ueberquerung_wird_einmal_gezaehlt(self):
        z = UeberfahrtZaehler()
        ereignisse = [self._schritt(z, lambda y=y: z.pruefen("a", (50, y), self.LINIE)) for y in (10, 30, 45, 55, 75, 95)]
        self.assertEqual(sum(len(e) for e in ereignisse), 1)

    def test_nicht_bewegter_track_zaehlt_nicht(self):
        z = UeberfahrtZaehler()
        self.assertEqual([self._schritt(z, lambda y=y: z.pruefen("m", (50, y), self.LINIE, bewegt=False))
                          for y in (40, 60)], [[], []])

    def test_nachtraegliche_pruefung_findet_ueberquerung_vor_der_einstufung(self):
        """Ein Fahrzeug, das die Linie schon in den ersten Frames überquert, geht nicht verloren."""
        z = UeberfahrtZaehler()
        punkte = [(50, 30), (50, 55), (50, 80)]
        e = [self._schritt(z, lambda p=p: z.pruefen("v", p, self.LINIE, bewegt=False)) for p in punkte[:2]]
        e.append(self._schritt(z, lambda: z.pruefen("v", punkte[2], self.LINIE, bewegt=True, neu_bewegt=True,
                                                   lauf_laenge=3)))
        self.assertEqual(sum(len(x) for x in e), 1)

    def test_richtung(self):
        z = UeberfahrtZaehler()
        self._schritt(z, lambda: z.pruefen("a", (50, 40), self.LINIE))
        ab = self._schritt(z, lambda: z.pruefen("a", (50, 60), self.LINIE))
        self._schritt(z, lambda: z.pruefen("b", (50, 60), self.LINIE))
        auf = self._schritt(z, lambda: z.pruefen("b", (50, 40), self.LINIE))
        self.assertNotEqual(ab[0][1], auf[0][1])


class TestGeisterZaehlung(unittest.TestCase):
    """Verschwindet ein Fahrzeug kurz vor der Linie und die Geisterbox überquert sie, zählt es."""
    LINIE = [((0, 100), (300, 100))]
    BOX = (50.0, 40.0)

    @staticmethod
    def _box(x, y, b=50.0, h=40.0):          # Box um den Zählpunkt (Unterkante - 20 % Höhe)
        return (x - b / 2, y - 0.8 * h, x + b / 2, y + 0.2 * h)

    def _fahrt(self, z, schluessel, x, ys, geist, box_groesse=BOX):
        ergebnisse = []
        for y in ys:
            z.neuer_frame()
            box = self._box(x, y, *box_groesse)
            ergebnisse.append(z.pruefen_geist(schluessel, (x, y), self.LINIE, box) if geist
                              else z.pruefen(schluessel, (x, y), self.LINIE, box=box))
        return sum(len(e) for e in ergebnisse)

    def test_geisterbox_ueberquerung_wird_einmal_gezaehlt(self):
        z = UeberfahrtZaehler()
        self.assertEqual(self._fahrt(z, "a", 100, (60, 75, 90), geist=False), 0)       # erkannt, vor der Linie
        self.assertEqual(self._fahrt(z, "a", 100, (101, 110, 118), geist=True), 1)     # Geisterbox überquert
        self.assertEqual(z.statistik["geisterbox"], 1)

    def test_dasselbe_fahrzeug_als_neuer_track_wird_nicht_doppelt_gezaehlt(self):
        """Geisterbox von Track A überquert die Linie, dasselbe Fahrzeug taucht als Track B auf und überquert sie."""
        for ueberlappung, erwartet in ((0.3, 1), (0.0, 2)):
            z = UeberfahrtZaehler(ueberlappung=ueberlappung)
            self._fahrt(z, "A", 100, (60, 75, 90), geist=False)
            n = self._fahrt(z, "A", 100, (101, 110), geist=True)
            n += self._fahrt(z, "B", 104, (95, 105, 115), geist=False, box_groesse=(110.0, 20.0))
            self.assertEqual(n, erwartet, f"Überlappung {ueberlappung}")

    def test_getrennte_fahrzeuge_werden_beide_gezaehlt(self):
        z = UeberfahrtZaehler()
        n = self._fahrt(z, "A", 100, (60, 75, 90), geist=False) + self._fahrt(z, "A", 100, (101, 110), geist=True)
        n += self._fahrt(z, "B", 250, (60, 75, 90, 105, 115), geist=False)         # andere Fahrbahn, gleiche Zeit
        n += self._fahrt(z, "C", 100, (-40, -25, -10, 5, 20, 35, 50, 65, 80, 95, 110, 125), geist=False)  # später
        self.assertEqual(n, 3)

    def test_zwei_normal_erkannte_fahrzeuge_werden_nicht_unterdrueckt(self):
        """Die Doppelzählungsprüfung betrifft nur Fälle mit Geisterbox."""
        z = UeberfahrtZaehler()
        n = self._fahrt(z, "A", 100, (60, 75, 90, 105, 120), geist=False)
        n += self._fahrt(z, "B", 105, (60, 75, 90, 105, 120), geist=False)         # überlappt fast vollständig
        self.assertEqual(n, 2)


# ----------------------------------------------------------------------------
# Interstate-Modus (keine Zähllinien: Fahrzeug zählt beim ersten Erkennen im Bereich)
# ----------------------------------------------------------------------------
class TestInterstateBausteine(unittest.TestCase):
    def test_empfohlener_skip(self):
        self.assertEqual(empfohlener_skip(None, 44, 4), 1)          # Tempo unbekannt: vorsichtig
        self.assertEqual(empfohlener_skip(0, 44, 4), 1)
        self.assertEqual(empfohlener_skip(27, 44, 4), 1)            # schnell: jedes Frame
        self.assertEqual(empfohlener_skip(19, 44, 4), 2)            # höchstens eine Boxgröße je ausgewertetem Frame
        self.assertEqual(empfohlener_skip(7, 44, 4), 4)             # langsam: durch max_skip begrenzt
        self.assertEqual(empfohlener_skip(7, 44, 1), 1)

    def test_sofortzaehler_zaehlt_jedes_fahrzeug_einmal(self):
        z = SofortZaehler()
        z.neuer_frame()
        self.assertTrue(z.pruefen("a"))
        for _ in range(5):
            z.neuer_frame()
            self.assertFalse(z.pruefen("a"))
        self.assertTrue(z.pruefen("b"))

    def test_sofortzaehler_wartet_bis_das_fahrzeug_als_solches_gilt(self):
        z = SofortZaehler()
        self.assertFalse(z.pruefen("m", gilt_als_fahrzeug=False))
        self.assertFalse(z.pruefen("m", gilt_als_fahrzeug=False))
        self.assertTrue(z.pruefen("m", gilt_als_fahrzeug=True))
        self.assertFalse(z.pruefen("m", gilt_als_fahrzeug=True))

    def test_sofortzaehler_vergisst_lange_nicht_gesehene_schluessel(self):
        z = SofortZaehler(aufbewahren=10)
        z.neuer_frame(); z.pruefen("alt")
        for _ in range(700):
            z.neuer_frame()
        self.assertNotIn("alt", z.gezaehlt)

    def test_tracker_autobahn_rechnet_geschwindigkeit_bei_skip_wechsel_um(self):
        tr = EinfacherTracker(autobahn=True, min_bewegung=0.0)
        for y in range(0, 200, 10):
            tr.update(np.array([[10.0, y, 54.0, y + 28.0]]), [0.5], 0.2)
        vorher = tr.fluss
        self.assertIsNotNone(vorher)
        v_vorher = next(iter(tr.tracks.values()))["v"]
        tr.skip_geaendert(2.0)
        self.assertAlmostEqual(tr.fluss[1], 2 * vorher[1])
        self.assertAlmostEqual(next(iter(tr.tracks.values()))["v"][1], 2 * v_vorher[1])

    def test_tracker_autobahn_loescht_am_rand_nicht_sofort_und_meldet_beendete(self):
        box = np.array([[100.0, 0.0, 144.0, 14.0]])                 # neu einfahrendes Fahrzeug am oberen Rand
        for autobahn, erwartet_track in ((True, 1), (False, 0)):
            tr = EinfacherTracker(autobahn=autobahn, min_bewegung=0.0)
            tr.update(box, [0.5], 0.2, grenzen=(0, 0, 300, 150))
            tr.update(np.zeros((0, 4)), [], 0.2, grenzen=(0, 0, 300, 150))    # ein Aussetzer
            self.assertEqual(len(tr.tracks), erwartet_track, f"autobahn={autobahn}")
        tr = EinfacherTracker(autobahn=True, min_bewegung=0.0, max_luecke=1)
        tr.update(box, [0.5], 0.2); tr.update(box, [0.5], 0.2)
        for _ in range(3):
            tr.update(np.zeros((0, 4)), [], 0.2)
        self.assertEqual(tr.beendet, [2])                           # Track mit 2 Sichtungen beendet

    def test_tracker_ordnet_langsames_fahrzeug_nicht_der_nachbarspur_zu(self):
        """Regression: Track sprang bei einem Aussetzer 85 px zur Seite auf ein Fahrzeug der Nachbarspur."""
        tr = EinfacherTracker(autobahn=True, min_bewegung=0.0, max_tempo=3.0, max_luecke=3)
        a, b = 148.0, 63.0
        for k in range(8):                                          # Fahrzeug A in Spur 2, 6 px/Frame
            tr.update(np.array([[a - 22, 10.0 + 6 * k, a + 22, 38.0 + 6 * k]]), [0.5], 0.2, grenzen=(0, 0, 300, 150))
        id_a = next(iter(tr.tracks))
        # A fällt aus, in der Nachbarspur taucht Fahrzeug B auf
        ids = tr.update(np.array([[b - 22, 0.0, b + 22, 9.0]]), [0.5], 0.2, grenzen=(0, 0, 300, 150))
        self.assertNotEqual(ids[0], id_a)

    def test_konfiguration_ohne_linien_ergibt_leere_linienliste(self):
        with tempfile.TemporaryDirectory() as tmp:
            pfad = os.path.join(tmp, "spuren.json")
            with open(pfad, "w") as f:
                json.dump({"rois": [[0.1, 0.2, 0.9, 0.6]], "linien": []}, f)
            with contextlib.redirect_stdout(io.StringIO()):
                linien, rois = lade_konfig(pfad, 0.55)
            self.assertEqual(linien, [])
            self.assertEqual(rois, [[0.1, 0.2, 0.9, 0.6]])


def interstate_lauf(seed, vmin, vmax, headway, skip=None, frames=1500, dropout=0.08, max_skip=4):
    """
    Dichter Verkehr auf drei Fahrspuren durch einen kleinen Bereich (300 x 150, Boxen 44 x 28), Zählung beim
    ersten Erkennen. skip=None: automatische Regelung wie in main(). Gibt (gezählt, wahr, Ø skip) zurück.
    """
    w, h, fb, fh = 300, 150, 44.0, 28.0
    rng = random.Random(seed)
    fz, naechste = [], {x: rng.uniform(0, 30) for x in (60, 150, 240)}
    for f in range(frames):
        for x in naechste:
            if f >= naechste[x]:
                v = rng.uniform(vmin, vmax)
                fz.append({"start": f, "x": x + rng.uniform(-5, 5), "v": v})
                naechste[x] = f + max(2.2 * fh / v, rng.expovariate(1.0 / headway))
    tr = EinfacherTracker(max_luecke=3, min_bewegung=0.0, max_tempo=3.0, autobahn=True)
    sofort, n = SofortZaehler(), 0
    f, aktuell, groesse_ema, genutzt = 0, (1 if skip is None else skip), None, []
    while f < frames + 250:
        boxen, konfs = [], []
        for z in fz:
            if f < z["start"]:
                continue
            y = -fh + z["v"] * (f - z["start"])
            if y - fh / 2 > h:
                continue
            x1, y1, x2, y2 = z["x"] - fb / 2, y - fh / 2, z["x"] + fb / 2, y + fh / 2
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            anteil = (cx2 - cx1) * (cy2 - cy1) / (fb * fh)
            if anteil < 0.3 or rng.random() < dropout:
                continue
            konf = max(0.05, rng.gauss(0.35, 0.08) * anteil ** 0.5)
            if konf >= 0.08:
                boxen.append([cx1, cy1, cx2, cy2]); konfs.append(konf)
        boxen = np.array(boxen, dtype=float).reshape(-1, 4)
        ids = tr.update(boxen, konfs, 0.2, grenzen=(0, 0, w, h))
        sofort.neuer_frame()
        groessen = []
        for b, tid in zip(boxen, ids):
            if tid is not None:
                groessen.append(max(b[2] - b[0], b[3] - b[1]))
                n += sofort.pruefen(tid, tr.tracks[tid]["bewegt"])
        if groessen:
            m = sum(groessen) / len(groessen)
            groesse_ema = m if groesse_ema is None else 0.95 * groesse_ema + 0.05 * m
        genutzt.append(aktuell)
        if skip is None:                                           # Regelung wie in main()
            tempo = math.hypot(*tr.fluss) / aktuell if tr.fluss is not None else None
            neu = empfohlener_skip(tempo, groesse_ema, max_skip)
            if neu != aktuell:
                tr.skip_geaendert(neu / aktuell)
                aktuell = neu
        f += aktuell
    return n, len(fz), sum(genutzt) / len(genutzt)


class TestInterstateSimulation(unittest.TestCase):
    SZENEN = {"mittel": (12, 26, 20), "schnell": (20, 34, 14), "langsam": (4, 10, 20)}
    SEEDS = range(3)

    def _fehler(self, szene, skip=None):
        erg = [interstate_lauf(s, *self.SZENEN[szene], skip=skip) for s in self.SEEDS]
        wahr = sum(e[1] for e in erg)
        return (sum(e[0] for e in erg) - wahr) / wahr, sum(e[2] for e in erg) / len(erg)

    def test_jedes_frame_zaehlt_genau(self):
        for szene in self.SZENEN:
            fehler, _ = self._fehler(szene, skip=1)
            self.assertLess(abs(fehler), 0.015, f"{szene}: {fehler:+.1%}")

    def test_automatische_skip_regelung_zaehlt_genau_und_spart_frames_bei_langsamem_verkehr(self):
        ergebnisse = {szene: self._fehler(szene) for szene in self.SZENEN}
        for szene, (fehler, _) in ergebnisse.items():
            self.assertLess(abs(fehler), 0.04, f"{szene}: {fehler:+.1%}")
        self.assertLess(ergebnisse["schnell"][1], 1.5)            # schneller Verkehr: (fast) jedes Frame
        self.assertGreater(ergebnisse["langsam"][1], 2.5)         # langsamer Verkehr: deutlich weniger Frames

    def test_zu_grosser_skip_bei_schnellem_verkehr_zaehlt_falsch(self):
        """Dokumentiert die Grenze, vor der die Regelung schützt."""
        fehler, _ = self._fehler("schnell", skip=4)
        self.assertLess(fehler, -0.15)


# ----------------------------------------------------------------------------
# Simulationen
# ----------------------------------------------------------------------------
class TestSimulation(unittest.TestCase):
    SEEDS = range(8)

    def _summe(self, **kw):
        erg = [fahre(Szene(s, **kw)) for s in self.SEEDS]
        return {k: sum(e[k] for e in erg) if k != "geister_max" else max(e[k] for e in erg) for k in erg[0]}

    def test_fahrbahnmarkierungen_und_aussetzer(self):
        r = self._summe(aussetzer=0.08)
        fehler = (r["gezaehlt"] - r["wahr"]) / r["wahr"]
        self.assertLess(abs(fehler), 0.03, f"Zählfehler {fehler:+.1%}")
        self.assertLess(r["markierungsboxen"] / r["boxen"], 0.02, "Markierungen werden als Fahrzeuge gezeigt")

    def test_viel_verkehr_mit_vielen_aussetzern(self):
        r = self._summe(aussetzer=0.25)
        self.assertLess(abs(r["gezaehlt"] - r["wahr"]) / r["wahr"], 0.04)

    def test_geisterboxen_bleiben_im_bereich_und_nahe_am_fahrzeug(self):
        r = self._summe(aussetzer=0.25)
        self.assertGreater(r["geister"], 0)
        self.assertEqual(r["geister_aussen"], 0)
        self.assertLess(r["geister_max"], 120.0, "Geisterbox 'fliegt' weit vom Fahrzeug weg")

    def _mit(self, szene_args, **fahre_args):
        erg = [fahre(Szene(s, **szene_args), **fahre_args) for s in self.SEEDS]
        wahr, gez = sum(e["wahr"] for e in erg), sum(e["gezaehlt"] for e in erg)
        return (gez - wahr) / wahr

    def test_fahrzeug_verschwindet_kurz_vor_der_linie(self):
        """Erkennung bricht 6 px vor der Linie ab: die gelbe Box überquert sie, das Fahrzeug wird gezählt."""
        self.assertLess(abs(self._mit(dict(unsichtbar_vor_linie=6))), 0.03)
        self.assertLess(self._mit(dict(unsichtbar_vor_linie=6), geist=False), -0.9)      # ohne die Funktion: fast nichts

    def test_dasselbe_fahrzeug_wechselt_kurz_vor_der_linie_sein_aussehen(self):
        """Neuer Track neben der Geisterbox des alten: das Fahrzeug darf nur einmal gezählt werden."""
        szene = dict(aussehen_wechselt_ab=15)
        self.assertLess(abs(self._mit(szene)), 0.04)
        self.assertGreater(self._mit(szene, ueberlappung=0.0), 0.4, "ohne Schutz müsste doppelt gezählt werden")
        self.assertLess(self._mit(szene, geist=False), -0.05, "ohne Geisterzählung fehlen Fahrzeuge")

    def test_blindzone_um_die_linie_ohne_doppelzaehlung(self):
        self.assertLess(abs(self._mit(dict(blindzone=(25, 25)))), 0.03)

    def test_ohne_stoerungen_exakt(self):
        r = self._summe(aussetzer=0.0, markierungen=False)
        self.assertEqual(r["gezaehlt"], r["wahr"])

    def test_grosse_spruenge_wie_bei_skip(self):
        # (Sprung px/Frame, Aussetzer, Mindestanzahl von 40) -- Mittel aus 8 Läufen
        for sprung, ausfall, minimum in ((15, 0.0, 39.5), (35, 0.2, 38.5), (50, 0.2, 38.0), (70, 0.2, 36.0)):
            mittel = sum(sprung_szene(s, sprung, ausfall) for s in self.SEEDS) / len(self.SEEDS)
            self.assertGreaterEqual(mittel, minimum, f"{sprung} px, {ausfall:.0%} Aussetzer: Ø {mittel:.1f}/40")


# ----------------------------------------------------------------------------
# Hauptschleife mit Attrappen für PyTorch/YOLO und einem erzeugten Testvideo
# ----------------------------------------------------------------------------
class _Tensor:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def tolist(self): return self.a.tolist()
    def int(self): return _Tensor(self.a.astype(int))


class _Boxen:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls, self.id = _Tensor(xyxy), _Tensor(conf), _Tensor(cls), None
    def __len__(self): return len(self.xyxy.a)


class _Ergebnis:
    def __init__(self, boxen): self.boxes = boxen


def _attrappen(aufrufe_max):
    """Falsches 'torch' und 'ultralytics': 'erkennt' helle Rechtecke im Bild und bricht nach N Aufrufen ab."""
    zaehler = {"n": 0}

    class YOLO:
        def __init__(self, *a, **k): pass

        def predict(self, bild, **k):
            zaehler["n"] += 1
            if zaehler["n"] > aufrufe_max:
                raise KeyboardInterrupt
            grau = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
            konturen, _ = cv2.findContours((grau > 200).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            xyxy = [[x, y, x + w, y + h] for x, y, w, h in (cv2.boundingRect(c) for c in konturen) if w * h > 100]
            return [_Ergebnis(_Boxen(np.array(xyxy, dtype=float).reshape(-1, 4), [0.6] * len(xyxy), [2] * len(xyxy)))]

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                  backends=types.SimpleNamespace())
    ultralytics = types.SimpleNamespace(YOLO=YOLO)
    cfg = types.SimpleNamespace(DEFAULT_CFG_DICT={})
    return {"torch": torch, "ultralytics": ultralytics, "ultralytics.cfg": cfg}


def _erzeuge_video(pfad, w=360, h=240, n=90):
    """Drei helle Fahrzeuge fahren abwärts (das dritte 'verschwindet' kurz vor der Mitte), dazu eine statische
    helle Markierung. Gibt False zurück, wenn OpenCV hier kein Video schreiben kann."""
    writer = cv2.VideoWriter(pfad, cv2.VideoWriter_fourcc(*"MJPG"), 25, (w, h))
    if not writer.isOpened():
        return False
    for f in range(n):
        bild = np.full((h, w, 3), 60, np.uint8)
        cv2.rectangle(bild, (100, 130), (170, 142), (255, 255, 255), -1)                      # statische "Markierung"
        for start, x, sichtbar_bis in ((5, 40, None), (30, 200, None), (20, 290, 114)):        # Fahrzeuge, abwärts
            y = -30 + 6 * (f - start)
            if f >= start and y < h and (sichtbar_bis is None or y + 30 <= sichtbar_bis):      # das 3. "verschwindet"
                cv2.rectangle(bild, (x, max(0, y)), (x + 40, min(h - 1, y + 30)), (255, 255, 255), -1)
        writer.write(bild)
    writer.release()
    return True


class TestHauptschleife(unittest.TestCase):
    N = 90

    def _starte(self, tmp, config, extra, aufrufe_max=None, csv_angeben=True):
        """Führt main() mit Attrappen aus und gibt (Summe der Zählungen, Ausgabetext) zurück."""
        with open(os.path.join(tmp, "spuren.json"), "w") as f:
            json.dump(config, f)
        argv = ["trafficview.py", os.path.join(tmp, "test.avi"), "--spuren", os.path.join(tmp, "spuren.json"),
                "--show", "--debug", "--skip", "1"] + (["--csv", os.path.join(tmp, "out.csv")] if csv_angeben else []) + extra
        ausgabe = io.StringIO()
        with mock.patch.dict(sys.modules, _attrappen(aufrufe_max=aufrufe_max or self.N)), \
                mock.patch.object(sys, "argv", argv), mock.patch.object(cv2, "imshow"), \
                mock.patch.object(cv2, "waitKey", return_value=-1), mock.patch.object(cv2, "destroyAllWindows"), \
                contextlib.redirect_stdout(ausgabe):
            tv.main()
        text = ausgabe.getvalue()
        self.assertIn("Endergebnis", text)
        zeilen = text.split("Endergebnis:")[1].splitlines()
        return sum(int(z.split()[-1]) for z in zeilen if z.startswith("  ")), text

    def _tmp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        alt_cwd = os.getcwd()
        self.addCleanup(os.chdir, alt_cwd)             # main() schreibt ggf. Dateien ins Arbeitsverzeichnis
        os.chdir(tmp.name)
        if not _erzeuge_video(os.path.join(tmp.name, "test.avi"), n=self.N):
            self.skipTest("OpenCV kann hier kein Testvideo schreiben")
        return tmp.name

    def test_ende_zu_ende_mit_zaehllinie(self):
        """Zwei Fahrzeuge fahren über die Linie, ein drittes verschwindet kurz davor (Geisterbox überquert die
        Linie), eine statische helle Markierung darf nicht gezählt werden."""
        tmp = self._tmp()
        config = {"rois": [[0.0, 0.0, 1.0, 1.0]], "linien": [{"name": "L", "p1": [0.0, 0.5], "p2": [1.0, 0.5]}]}
        for extra, erwartet, geister in (([], 3, 1), (["--ohne-geisterzaehlung"], 2, 0)):
            summe, text = self._starte(tmp, config, extra)
            self.assertEqual(summe, erwartet, f"{extra}: erwartet {erwartet} Fahrzeuge, Ausgabe:\n{text}")
            self.assertIn(f"davon über Geisterbox gezählt: {geister}", text)

    def test_ende_zu_ende_interstate_ohne_linien(self):
        """Keine Zähllinien in der Konfiguration: jedes Fahrzeug zählt beim ersten Erkennen im Bereich."""
        tmp = self._tmp()
        config = {"rois": [[0.0, 0.0, 1.0, 1.0]], "linien": []}
        # Standard: sofort zählen, also auch die statische Markierung (einmal) -> 3 Fahrzeuge + 1
        summe, text = self._starte(tmp, config, [])
        self.assertIn("Interstate-Modus: keine Zähllinien", text)
        self.assertIn("Bereich 1", text.split("Endergebnis:")[1])
        self.assertEqual(summe, 4, text)
        # Mit Bewegungsprüfung wird die statische Markierung nicht gezählt
        summe, text = self._starte(tmp, config, ["--min-bewegung", "0.25"])
        self.assertEqual(summe, 3, text)
        self.assertIn("Sichtungen je Fahrzeug", text)

    def test_interstate_kann_erzwungen_werden_und_ignoriert_dann_linien(self):
        tmp = self._tmp()
        config = {"rois": [[0.0, 0.0, 1.0, 1.0]], "linien": [{"name": "L", "p1": [0.0, 0.5], "p2": [1.0, 0.5]}]}
        summe, text = self._starte(tmp, config, ["--interstate"])
        self.assertEqual(summe, 4, text)
        self.assertIn("Bereich 1", text.split("Endergebnis:")[1])

    def test_jeder_lauf_schreibt_in_eine_neue_csv_datei(self):
        """Ohne --csv: pro Lauf eine neue Datei tracking-data-<Datum>-<Uhrzeit>.csv mit Kopfzeile."""
        tmp = self._tmp()
        config = {"rois": [[0.0, 0.0, 1.0, 1.0]], "linien": [{"name": "L", "p1": [0.0, 0.5], "p2": [1.0, 0.5]}]}
        ordner = os.path.join(tmp, "ausgabe")                       # wird bei Bedarf angelegt
        for _ in range(2):
            self._starte(tmp, config, ["--csv-ordner", ordner], csv_angeben=False)
        dateien = sorted(os.listdir(ordner))
        self.assertEqual(len(dateien), 2, dateien)
        for name in dateien:
            self.assertRegex(name, r"^tracking-data-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(-\d+)?\.csv$")
            with open(os.path.join(ordner, name), encoding="utf-8") as f:
                self.assertTrue(f.readline().startswith("zeit,linie,richtung"))
        self.assertFalse(os.path.exists(os.path.join(tmp, "verkehr.csv")))

    def test_interstate_ohne_bereich_bricht_mit_hinweis_ab(self):
        tmp = self._tmp()
        with self.assertRaises(SystemExit) as fehler:
            self._starte(tmp, {"rois": [], "linien": []}, [])
        self.assertIn("mindestens einen Auswertebereich", str(fehler.exception))

if __name__ == "__main__":
    unittest.main()
