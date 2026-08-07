"""GAIRA V7 — Phase 10: delimited-text spectrum adapter (CSV / TSV / TXT / two-column ASCII).

One adapter covers all four because they differ only in delimiter and header conventions, and
guessing wrongly between them is the most common cause of a silently mangled spectrum. Detection
is explicit and every decision it makes is reported as a diagnostic, so a user can see that the
file was read as semicolon-delimited with a header rather than discovering it from a strange plot.
"""
from __future__ import annotations

import csv
import io
import re

import numpy as np

from .base import ParsedSpectrum, err, info, warn

DELIMITERS = (",", "\t", ";", "|", " ")
MAX_BYTES = 64 * 1024 * 1024
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# Column names seen in real exports, lower-cased and stripped of punctuation.
X_NAMES = {"wavenumber", "wavenumbers", "raman shift", "ramanshift", "shift", "wn", "x",
           "wavenumber cm-1", "raman shift cm-1", "cm-1", "cm1", "wavelength"}
Y_NAMES = {"intensity", "counts", "count", "signal", "y", "absorbance", "intensities",
           "raw intensity", "cps", "a.u.", "au", "amplitude"}


def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9 .\-]", "", s.strip().lower()).strip()


def _is_number(tok: str) -> bool:
    tok = tok.strip().replace("﻿", "")
    return bool(_NUM.match(tok))


class DelimitedTextAdapter:
    name = "delimited_text"
    extensions = (".csv", ".tsv", ".txt", ".dat", ".asc", ".spc.txt")

    def sniff(self, payload, filename=None) -> bool:
        try:
            text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) \
                else str(payload)
        except Exception:
            return False
        head = [ln for ln in text.splitlines()[:50] if ln.strip()]
        if len(head) < 2:
            return False
        for d in DELIMITERS:
            if sum(1 for ln in head if len([c for c in ln.split(d) if c.strip()]) >= 2) >= 2:
                return True
        return False

    # ── detection ────────────────────────────────────────────────────────────
    def _delimiter(self, lines: list[str]) -> tuple[str, str]:
        """Pick the delimiter that yields a consistent field count of at least two.

        Consistency matters more than field count: a decimal-comma European export split on ','
        gives four fields per line and looks great until the numbers are nonsense.
        """
        best, best_score, reason = None, -1.0, ""
        for d in DELIMITERS:
            counts = [len([c for c in ln.split(d) if c.strip() != ""]) for ln in lines]
            counts = [c for c in counts if c > 0]
            if not counts:
                continue
            mode = max(set(counts), key=counts.count)
            if mode < 2:
                continue
            consistency = counts.count(mode) / len(counts)
            score = consistency * 10 + min(mode, 4)
            if score > best_score:
                best, best_score = d, score
                reason = f"{mode} fields on {consistency:.0%} of sampled lines"
        if best is None:
            try:
                best = csv.Sniffer().sniff("\n".join(lines[:20])).delimiter
                reason = "csv.Sniffer fallback"
            except Exception:
                best, reason = None, "no consistent delimiter found"
        return best, reason

    def parse(self, payload, filename=None) -> ParsedSpectrum:
        diags = []
        raw = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        if len(raw) > MAX_BYTES:
            return ParsedSpectrum(np.array([]), np.array([]),
                                  [err("input.too_large",
                                       f"input is {len(raw)} bytes; the limit is {MAX_BYTES}")],
                                  self.name)
        text = raw.decode("utf-8-sig", errors="replace")
        if "\x00" in text:
            return ParsedSpectrum(np.array([]), np.array([]),
                                  [err("input.binary",
                                       "input contains NUL bytes; this is not delimited text")],
                                  self.name)
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if len(lines) < 3:
            return ParsedSpectrum(np.array([]), np.array([]),
                                  [err("input.too_few_lines",
                                       f"only {len(lines)} non-empty lines found")], self.name)

        delim, why = self._delimiter(lines[:200])
        if delim is None:
            return ParsedSpectrum(np.array([]), np.array([]),
                                  [err("input.no_delimiter", why)], self.name)
        diags.append(info("input.delimiter",
                          f"delimiter detected as {delim!r} ({why})", delimiter=delim))

        rows = [[c.strip() for c in ln.split(delim) if c.strip() != ""] for ln in lines]

        # header detection: the first row is a header if it is not all-numeric
        header, start = None, 0
        if rows and not all(_is_number(c) for c in rows[0][:2]):
            header, start = [_clean(c) for c in rows[0]], 1
            diags.append(info("input.header", f"header row detected: {rows[0][:4]}",
                              header=rows[0][:8]))
        else:
            diags.append(info("input.header", "no header row; first line parsed as data"))

        # column choice
        xi, yi, how = 0, 1, "first two columns"
        if header:
            xs = [i for i, h in enumerate(header) if h in X_NAMES]
            ys = [i for i, h in enumerate(header) if h in Y_NAMES]
            if xs and ys:
                xi, yi, how = xs[0], ys[0], f"header names {header[xs[0]]!r} / {header[ys[0]]!r}"
            elif len(header) > 2:
                diags.append(warn("input.ambiguous_columns",
                                  f"{len(header)} columns and no recognised wavenumber/intensity "
                                  f"names; using the first two", header=header[:8]))
        elif rows and len(rows[0]) > 2:
            diags.append(warn("input.ambiguous_columns",
                              f"{len(rows[0])} columns and no header; using the first two"))
        diags.append(info("input.columns", f"columns chosen by {how}",
                          x_index=xi, y_index=yi))

        xs_, ys_, bad = [], [], []
        for n, r in enumerate(rows[start:], start=start + 1):
            if len(r) <= max(xi, yi):
                bad.append(n); continue
            a, b = r[xi].replace("﻿", ""), r[yi].replace("﻿", "")
            try:
                # Parse BOTH before appending either. Appending as we go desynchronises the
                # columns whenever a wavenumber parses and its intensity does not, which
                # silently pairs each intensity with the wrong wavenumber — the worst possible
                # failure for a spectrum, because the result still looks like a spectrum.
                xv, yv = float(a), float(b)
            except ValueError:
                bad.append(n); continue
            xs_.append(xv); ys_.append(yv)
        if bad:
            sev = err if len(bad) > 0.10 * max(len(rows) - start, 1) else warn
            diags.append(sev("input.malformed_rows",
                             f"{len(bad)} of {len(rows) - start} data rows could not be parsed "
                             f"as two numbers", n_bad=len(bad), first_lines=bad[:10]))
        if len(xs_) < 2:
            diags.append(err("input.no_data", f"only {len(xs_)} usable data points"))
            return ParsedSpectrum(np.array(xs_), np.array(ys_), diags, self.name)

        x = np.asarray(xs_, float)
        y = np.asarray(ys_, float)
        return finalise(x, y, diags, self.name, {"delimiter": delim, "header": header,
                                                 "x_index": xi, "y_index": yi,
                                                 "n_rows": len(rows) - start})


def finalise(x: np.ndarray, y: np.ndarray, diags: list, source: str, detail: dict
             ) -> ParsedSpectrum:
    """Shared post-parse handling for every adapter: NaNs, ordering, duplicates.

    Each of these is a *reported* transformation. Nothing is repaired silently, and anything that
    changes the number of points says how many it changed.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        n = int((~finite).sum())
        sev = err if n > 0.20 * len(x) else warn
        diags.append(sev("input.non_finite",
                         f"{n} of {len(x)} points are NaN or infinite and were dropped",
                         n_dropped=n))
        x, y = x[finite], y[finite]
    if len(x) < 2:
        diags.append(err("input.no_data", "fewer than two finite points remain"))
        return ParsedSpectrum(x, y, diags, source, detail)

    if np.all(np.diff(x) < 0):
        diags.append(info("input.descending",
                          "wavenumber axis is descending; reversed to ascending"))
        x, y = x[::-1], y[::-1]
    elif not np.all(np.diff(x) > 0):
        order = np.argsort(x, kind="stable")
        diags.append(warn("input.unsorted",
                          "wavenumber axis is not monotonic; points were sorted ascending"))
        x, y = x[order], y[order]

    uniq, counts = np.unique(x, return_counts=True)
    if len(uniq) != len(x):
        dup = int(len(x) - len(uniq))
        diags.append(warn("input.duplicate_wavenumbers",
                          f"{dup} duplicate wavenumber values; intensities averaged per "
                          f"wavenumber", n_duplicates=dup))
        y = np.array([y[x == u].mean() for u in uniq])
        x = uniq
    detail["n_points_final"] = int(len(x))
    return ParsedSpectrum(x, y, diags, source, detail)
