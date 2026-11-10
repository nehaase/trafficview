#!/usr/bin/env python3
"""
Vergleicht die automatischen Zählungen von TrafficView (Option --ereignisse) mit einer manuellen Zählung
(annotieren.py) und gibt die Genauigkeit je Zählstelle aus.

    python vergleich.py auto.csv manuell.csv
    python vergleich.py auto.csv manuell.csv --toleranz 2 --liste --minuten --bericht bericht.csv

Beide Dateien brauchen die Spalten video_s (Sekunden im Video) und stelle_nr (Zählstelle 1, 2, ...). Optional:
typ (Pkw, Lkw, ...). Eine Zählung der Automatik gilt als richtig, wenn die manuelle Zählung derselben Stelle
innerhalb der Toleranz (Standard 1,5 s) liegt. Jede Zählung wird höchstens einmal zugeordnet.

Wichtig: Die Gesamtzahl allein täuscht. Übersehene und zusätzliche Fahrzeuge können sich ausgleichen, dann
stimmt die Summe, obwohl einzelne Fahrzeuge falsch sind. Deshalb werden auch Treffer, Übersehene und Zusätzliche
ausgegeben.
"""
import argparse
import csv
import statistics
import sys
from bisect import bisect_left


def lese_ereignisse(pfad):
    """Liest video_s und stelle_nr (Pflicht) sowie typ (optional) aus einer CSV-Datei."""
    ereignisse = []
    with open(pfad, newline="", encoding="utf-8") as f:
        for nr, zeile in enumerate(csv.DictReader(f), 2):
            try:
                ereignisse.append({"t": float(zeile["video_s"]), "stelle": int(float(zeile["stelle_nr"])),
                                   "typ": (zeile.get("typ") or "").strip(), "zeile": nr})
            except (KeyError, TypeError, ValueError):
                raise SystemExit(f"{pfad}, Zeile {nr}: die Spalten video_s und stelle_nr werden erwartet.")
    return ereignisse


def zuordnen(auto_t, manuell_t, toleranz):
    """
    Eins-zu-eins-Zuordnung nach kleinstem Zeitabstand innerhalb der Toleranz.
    Gibt eine Liste von Paaren (index_auto, index_manuell) zurück.
    """
    reihenfolge = sorted(range(len(manuell_t)), key=lambda j: manuell_t[j])
    sortiert = [manuell_t[j] for j in reihenfolge]
    kandidaten = []
    for i, a in enumerate(auto_t):
        k = bisect_left(sortiert, a - toleranz)
        while k < len(sortiert) and sortiert[k] <= a + toleranz:
            kandidaten.append((abs(a - sortiert[k]), i, reihenfolge[k]))
            k += 1
    kandidaten.sort()
    benutzt_a, benutzt_m, paare = set(), set(), []
    for _, i, j in kandidaten:
        if i not in benutzt_a and j not in benutzt_m:
            benutzt_a.add(i)
            benutzt_m.add(j)
            paare.append((i, j))
    return paare


def vergleiche(auto, manuell, toleranz=1.5):
    """Vergleicht je Zählstelle. Gibt {stelle: Kennzahlen} zurück (Stelle 0 = alle Stellen zusammen nicht enthalten)."""
    ergebnis = {}
    for stelle in sorted({e["stelle"] for e in auto} | {e["stelle"] for e in manuell}):
        a = sorted((e for e in auto if e["stelle"] == stelle), key=lambda e: e["t"])
        m = sorted((e for e in manuell if e["stelle"] == stelle), key=lambda e: e["t"])
        paare = zuordnen([e["t"] for e in a], [e["t"] for e in m], toleranz)
        zugeordnet_a = {i for i, _ in paare}
        zugeordnet_m = {j for _, j in paare}
        # Typen vergleichen, wenn beide eine echte Typangabe haben ("Fahrzeug" = zusammengefasst, nicht vergleichbar)
        typpaare = [(a[i]["typ"], m[j]["typ"]) for i, j in paare
                    if a[i]["typ"] not in ("", "Fahrzeug") and m[j]["typ"]]
        abweichungen = [a[i]["t"] - m[j]["t"] for i, j in paare]
        ergebnis[stelle] = {
            "manuell": len(m), "auto": len(a), "treffer": len(paare),
            "uebersehen": [m[j] for j in range(len(m)) if j not in zugeordnet_m],
            "zusaetzlich": [a[i] for i in range(len(a)) if i not in zugeordnet_a],
            "zeitabweichung_median": statistics.median(abweichungen) if abweichungen else None,
            "typ_richtig": sum(1 for x, y in typpaare if x == y), "typ_gesamt": len(typpaare),
        }
    return ergebnis


def kennzahlen(r):
    """Zählfehler, Trefferquote (recall) und Genauigkeit (precision) aus einem Ergebnis von vergleiche()."""
    return {
        "zaehlfehler": (r["auto"] - r["manuell"]) / r["manuell"] if r["manuell"] else None,
        "recall": r["treffer"] / r["manuell"] if r["manuell"] else None,
        "precision": r["treffer"] / r["auto"] if r["auto"] else None,
    }


def summiere(ergebnis):
    """Alle Stellen zusammen."""
    gesamt = {"manuell": 0, "auto": 0, "treffer": 0, "uebersehen": [], "zusaetzlich": [],
              "typ_richtig": 0, "typ_gesamt": 0}
    abweichungen = []
    for r in ergebnis.values():
        for k in ("manuell", "auto", "treffer", "typ_richtig", "typ_gesamt"):
            gesamt[k] += r[k]
        gesamt["uebersehen"] += r["uebersehen"]
        gesamt["zusaetzlich"] += r["zusaetzlich"]
        if r["zeitabweichung_median"] is not None:
            abweichungen.append(r["zeitabweichung_median"])
    gesamt["zeitabweichung_median"] = statistics.median(abweichungen) if abweichungen else None
    return gesamt


def minutenvergleich(auto, manuell):
    """Zählungen je Videominute: [(minute, manuell, auto)]."""
    minuten = {}
    for e in manuell:
        minuten.setdefault(int(e["t"] // 60), [0, 0])[0] += 1
    for e in auto:
        minuten.setdefault(int(e["t"] // 60), [0, 0])[1] += 1
    return [(m, v[0], v[1]) for m, v in sorted(minuten.items())]


def drucke_bericht(auto, manuell, toleranz, liste=False, minuten=False, ausgabe=sys.stdout):
    ergebnis = vergleiche(auto, manuell, toleranz)
    p = lambda *a: print(*a, file=ausgabe)
    p(f"Vergleich mit Toleranz {toleranz:g} s: {len(auto)} automatische, {len(manuell)} manuelle Zählungen\n")
    p(f"{'Stelle':>6s} {'manuell':>8s} {'auto':>6s} {'Treffer':>8s} {'übersehen':>10s} {'zusätzlich':>11s} "
      f"{'Zählfehler':>11s} {'Trefferquote':>13s} {'Genauigkeit':>12s}")
    zeilen = list(ergebnis.items()) + ([("alle", summiere(ergebnis))] if len(ergebnis) > 1 else [])
    for stelle, r in zeilen:
        k = kennzahlen(r)
        fmt = lambda x, vorzeichen=False: "-" if x is None else (f"{100 * x:+.1f} %" if vorzeichen else f"{100 * x:.1f} %")
        p(f"{stelle!s:>6s} {r['manuell']:8d} {r['auto']:6d} {r['treffer']:8d} {len(r['uebersehen']):10d} "
          f"{len(r['zusaetzlich']):11d} {fmt(k['zaehlfehler'], True):>11s} {fmt(k['recall']):>13s} {fmt(k['precision']):>12s}")
    gesamt = summiere(ergebnis)
    abw = gesamt["zeitabweichung_median"]
    if abw is not None:
        p(f"\nMedian der Zeitabweichung (auto - manuell): {abw:+.2f} s "
          f"({'Automatik zählt später' if abw > 0 else 'Automatik zählt früher'})")
    if gesamt["typ_gesamt"]:
        p(f"Typ stimmt bei {gesamt['typ_richtig']} von {gesamt['typ_gesamt']} zugeordneten Fahrzeugen "
          f"({100 * gesamt['typ_richtig'] / gesamt['typ_gesamt']:.1f} %)")
    kg = kennzahlen(gesamt)
    if kg["zaehlfehler"] is not None and kg["recall"] is not None and abs(kg["zaehlfehler"]) < 0.03 and kg["recall"] < 0.93:
        p("\nHinweis: Die Summe stimmt fast, aber viele einzelne Fahrzeuge sind nicht zugeordnet. Übersehene und "
          "zusätzliche Zählungen gleichen sich aus. Mit --liste siehst du die Zeitpunkte zum Nachschauen im Video.")
    if liste:
        for titel, schluessel in (("Übersehen (manuell gezählt, Automatik nicht)", "uebersehen"),
                                  ("Zusätzlich (Automatik gezählt, manuell nicht)", "zusaetzlich")):
            p(f"\n{titel}:")
            gefunden = False
            for stelle, r in ergebnis.items():
                for e in r[schluessel]:
                    gefunden = True
                    p(f"  Stelle {stelle}: {e['t']:8.2f} s ({int(e['t'] // 60)}:{e['t'] % 60:05.2f}) {e['typ']}")
            if not gefunden:
                p("  keine")
    if minuten:
        p("\nJe Minute (Video):")
        p(f"{'Minute':>7s} {'manuell':>8s} {'auto':>6s} {'Differenz':>10s}")
        for m, a_, b_ in minutenvergleich(auto, manuell):
            p(f"{m:7d} {a_:8d} {b_:6d} {b_ - a_:+10d}")
    return ergebnis


def schreibe_bericht(ergebnis, pfad):
    with open(pfad, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stelle", "manuell", "auto", "treffer", "uebersehen", "zusaetzlich", "zaehlfehler", "recall",
                    "precision", "zeitabweichung_median_s", "typ_richtig", "typ_gesamt"])
        for stelle, r in list(ergebnis.items()) + [("alle", summiere(ergebnis))]:
            k = kennzahlen(r)
            w.writerow([stelle, r["manuell"], r["auto"], r["treffer"], len(r["uebersehen"]), len(r["zusaetzlich"]),
                        *(("" if k[n] is None else round(k[n], 4)) for n in ("zaehlfehler", "recall", "precision")),
                        "" if r["zeitabweichung_median"] is None else round(r["zeitabweichung_median"], 3),
                        r["typ_richtig"], r["typ_gesamt"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Automatische und manuelle Zählung vergleichen")
    ap.add_argument("auto", help="Ereignisprotokoll von TrafficView (--ereignisse)")
    ap.add_argument("manuell", help="Manuelle Zählung (annotieren.py)")
    ap.add_argument("--toleranz", type=float, default=1.5, help="größter Zeitabstand (s) für eine Zuordnung")
    ap.add_argument("--liste", action="store_true", help="übersehene und zusätzliche Zählungen mit Zeitpunkt auflisten")
    ap.add_argument("--minuten", action="store_true", help="Zählungen je Videominute gegenüberstellen")
    ap.add_argument("--bericht", default=None, metavar="DATEI.csv", help="Kennzahlen zusätzlich als CSV speichern")
    args = ap.parse_args(argv)
    auto, manuell = lese_ereignisse(args.auto), lese_ereignisse(args.manuell)
    ergebnis = drucke_bericht(auto, manuell, args.toleranz, args.liste, args.minuten)
    if args.bericht:
        schreibe_bericht(ergebnis, args.bericht)
        print(f"\nBericht gespeichert: {args.bericht}")


if __name__ == "__main__":
    main()
