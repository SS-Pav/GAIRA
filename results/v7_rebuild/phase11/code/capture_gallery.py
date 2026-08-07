"""GAIRA V7 — Phase 11: capture the screenshot gallery from the RUNNING app.

Walks the real user flow in headless Chromium and screenshots each screen. It also fails the run
if any page renders a Python traceback: Streamlit renders SERVER-side exceptions into the DOM, so
a browser-console listener never sees them, and checking the rendered text is the only way to
catch them. That check caught three real bugs during Phase 11.

    streamlit run streamlit_apps/gaira_v7_demo.py --server.port 8599
    python results/v7_rebuild/phase11/code/capture_gallery.py
"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "gallery"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8599"
errors = []


def shot(page, name, wait=1.6):
    time.sleep(wait)
    # Streamlit renders SERVER-side Python exceptions into the DOM; a browser-console listener
    # never sees them. Checking the rendered text is the only way to catch them.
    body = page.inner_text("body")
    for marker in ("Traceback:", "ValueError", "KeyError", "AttributeError", "TypeError",
                   "IndexError", "undefined"):
        if marker in body:
            errors.append(f"{name}: page shows {marker!r}")
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    print(f"  captured {name}")


def click_nav(page, label):
    page.get_by_role("button", name=label, exact=True).first.click()
    time.sleep(2.2)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1560, "height": 1000}, device_scale_factor=2)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
            if m.type == "error" else None)

    page.goto(URL, wait_until="networkidle", timeout=60000)
    page.wait_for_selector("text=Grounded AI for Raman Analysis", timeout=60000)
    shot(page, "01_home", 2.5)

    # Home → Analyze via the hero CTA
    page.get_by_role("button", name="Begin Analysis  →").first.click()
    page.wait_for_selector("text=Upload a spectrum", timeout=30000)
    shot(page, "02_upload_empty", 2.0)

    # pick a built-in reference spectrum
    page.get_by_text("— choose a built-in example —").first.click()
    time.sleep(0.8)
    page.get_by_text("Cholesterol (sterol)", exact=True).first.click()
    page.wait_for_selector("text=median step", timeout=30000)
    shot(page, "03_upload_loaded", 3.0)

    page.get_by_role("button", name="Preprocess Spectrum  →").first.click()
    page.wait_for_selector("text=Preprocessing", timeout=30000)
    time.sleep(1.2)
    shot(page, "04_preprocess_idle", 1.5)

    page.get_by_role("button", name="Run preprocessing").first.click()
    time.sleep(5.0)
    shot(page, "05_preprocess_done", 2.0)

    page.get_by_role("button", name="Analyze Spectrum  →").first.click()
    time.sleep(1.2)
    shot(page, "06_analysis_running", 0.9)
    page.wait_for_selector("text=bottom line", timeout=60000)
    shot(page, "07_results_hero", 3.5)

    # open a few sections
    for label, name in (("Chemical evidence — all sixteen axes", "08_chemistry"),
                        ("CSM contributions — the canonical 49 coordinates", "09_csm"),
                        ("Reconstruction and residual", "10_reconstruction"),
                        ("Confidence — why this number and not a higher one", "11_confidence"),
                        ("Provenance — the full evidence chain", "12_provenance")):
        try:
            page.get_by_text(label, exact=False).first.click()
            time.sleep(2.4)
            shot(page, name, 1.4)
            page.get_by_text(label, exact=False).first.click()
            time.sleep(0.7)
        except Exception as exc:
            errors.append(f"section {name}: {exc}")

    for label, name in (("Docs", "13_documentation"),
                        ("Architecture", "14_architecture"),
                        ("About", "15_about")):
        click_nav(page, label)
        shot(page, name, 2.0)

    b.close()

print("\nBROWSER ERRORS:" if errors else "\nNo browser errors.")
for e in errors[:25]:
    print("  " + e)
sys.exit(1 if errors else 0)
