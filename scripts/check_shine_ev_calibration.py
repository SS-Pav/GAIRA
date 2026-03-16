import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def extract_between(text: str, start_marker: str, end_marker: str) -> str | None:
    """Return the text between two markers if both are present."""
    start_index = text.find(start_marker)
    if start_index == -1:
        return None

    start_index += len(start_marker)
    end_index = text.find(end_marker, start_index)
    if end_index == -1:
        return None

    return text[start_index:end_index].strip()


def format_array(values: np.ndarray, digits: int = 3) -> str:
    """Format a short numeric array for readable terminal output."""
    rounded = [round(float(value), digits) for value in values]
    return str(rounded)


def load_representative_spectra(source_root: Path) -> list[Path]:
    """Pick one or two real spectrum files for a quick calibration sanity check."""
    candidates = sorted(
        path for path in source_root.rglob("s_*") if path.is_file() and path.name.startswith("s_")
    )
    if not candidates:
        return []

    selected = [candidates[0]]
    if len(candidates) > 1:
        selected.append(candidates[len(candidates) // 2])
    return selected


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from gaira.parsers.biosample.shine_ev_sers_parser import ShineEVSERSParser

    dataset_root = Path("/Volumes/SSD_SPG/GAIRA_DATA/raw/shine_ev_sers")
    if not dataset_root.exists():
        print(f"Dataset folder not found: {dataset_root}")
        return

    parser = ShineEVSERSParser(
        dataset_id="shine_ev_sers",
        dataset_root=dataset_root,
        db_path=project_root / "data" / "gaira.duckdb",
    )

    coefficients = np.polyfit(parser.PIXELS, parser.CALIBRATED_CM, 3)
    recovered_anchors = np.polyval(coefficients, parser.PIXELS)
    anchor_errors = recovered_anchors - parser.CALIBRATED_CM

    fig4_root = (
        dataset_root / "SERS-Hepatotoxicity_DATA_CODE_FIGURE" / "Figure4"
    )
    fig4d_path = fig4_root / "Fig4D" / "code" / "Fig4D.m"
    fig4c_plot_path = fig4_root / "Fig4C" / "code" / "plot_spectra.m"
    fig4f_plot_path = fig4_root / "Fig4F" / "code" / "plot_spectra.m"
    readme_path = dataset_root / "SERS-Hepatotoxicity_DATA_CODE_FIGURE" / "Readme.docx"
    mat_path = fig4_root / "Fig4C" / "data" / "combined_wavenumbers.mat"

    fig4d_text = fig4d_path.read_text()
    fig4c_plot_text = fig4c_plot_path.read_text()
    fig4f_plot_text = fig4f_plot_path.read_text()

    matlab_range = extract_between(fig4d_text, "range =", ";")
    matlab_xlim = extract_between(fig4d_text, "xlim([", "])")
    fig4c_offset = "x(range+162)" if "x(range+162)" in fig4c_plot_text else "not found"
    fig4f_offset = "x(range+162)" if "x(range+162)" in fig4f_plot_text else "not found"

    print("SHINE EV SERS calibration audit")
    print(f"Dataset root: {dataset_root}")
    print(f"Current parser-derived x range: {parser.calibrated_wavenumbers.min():.3f} to {parser.calibrated_wavenumbers.max():.3f} cm^-1")
    print()
    print("Calibration anchors from Figure4/Fig4D/code/Fig4D.m:")
    print(f"  Pixels: {parser.PIXELS.astype(int).tolist()}")
    print(f"  Calibrated cm^-1: {parser.CALIBRATED_CM.tolist()}")
    print()
    print("Anchor recovery error from the parser's cubic fit:")
    for pixel, expected_cm, recovered_cm, error_cm in zip(
        parser.PIXELS,
        parser.CALIBRATED_CM,
        recovered_anchors,
        anchor_errors,
    ):
        print(
            f"  pixel {int(pixel):>4}: expected {expected_cm:>7.1f}, "
            f"recovered {recovered_cm:>9.3f}, error {error_cm:>8.5f}"
        )
    print(f"  Max absolute anchor error: {np.max(np.abs(anchor_errors)):.6f} cm^-1")
    print()
    print("First 10 calibrated wavenumbers:")
    print(f"  {format_array(parser.calibrated_wavenumbers[:10])}")
    print("Last 10 calibrated wavenumbers:")
    print(f"  {format_array(parser.calibrated_wavenumbers[-10:])}")
    print()
    print("MATLAB workflow evidence:")
    print(f"  Fig4D uses cubic fit over 1:1650: {'x= polyval(fit,1:1650);' in fig4d_text}")
    print(f"  Fig4D plotted range statement: {matlab_range}")
    print(f"  Fig4D plot xlim: {matlab_xlim} cm^-1")
    print(f"  Fig4C plot uses offset subset: {fig4c_offset}")
    print(f"  Fig4F plot uses offset subset: {fig4f_offset}")
    print()

    representative_files = load_representative_spectra(parser.source_root)
    if representative_files:
        print("Representative per-spectrum files:")
        for spectrum_path in representative_files:
            spectrum_df = parser._read_spectrum_file(spectrum_path)
            if spectrum_df is None:
                print(f"  {spectrum_path}: could not parse")
                continue

            preview_df = pd.read_csv(spectrum_path, header=None, names=["pixel_index", "intensity"]).head(3)
            print(f"  {spectrum_path}")
            print(
                f"    rows={len(spectrum_df)}, pixel range={int(spectrum_df['pixel_index'].min())}-{int(spectrum_df['pixel_index'].max())}, "
                f"calibrated range={spectrum_df['wavenumber'].min():.3f}-{spectrum_df['wavenumber'].max():.3f} cm^-1"
            )
            print("    first rows:")
            for _, row in preview_df.iterrows():
                print(f"      pixel={int(row['pixel_index'])}, intensity={float(row['intensity'])}")
        print()

    if readme_path.exists():
        try:
            import subprocess

            result = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(readme_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            readme_text = result.stdout
            readme_hint = "wnIndex = pixTarget(j) - 159" in readme_text
            print("Readme evidence:")
            print(f"  Contains 'wnIndex = pixTarget(j) - 159': {readme_hint}")
            if readme_hint:
                print("  This indicates later analysis uses a shifted/cropped working window rather than the entire 1:1650 axis.")
            print()
        except Exception as exc:
            print(f"Could not inspect Readme.docx via textutil: {exc}")
            print()

    if mat_path.exists():
        try:
            mat_data = loadmat(mat_path)
            if "combined_wavenumbers" in mat_data:
                combined_wavenumbers = np.asarray(mat_data["combined_wavenumbers"]).ravel()
                print("combined_wavenumbers.mat inspection:")
                print(
                    f"  combined_wavenumbers length={len(combined_wavenumbers)}, "
                    f"min={combined_wavenumbers.min()}, max={combined_wavenumbers.max()}"
                )
                print(f"  first 10 values: {combined_wavenumbers[:10].tolist()}")
                print(f"  last 10 values: {combined_wavenumbers[-10:].tolist()}")
                print("  This MAT file appears to hold selected integer wavenumbers, not the full 1,650-point calibrated axis.")
                print()
        except Exception as exc:
            print(f"Could not inspect combined_wavenumbers.mat: {exc}")
            print()

    raw_min = float(parser.calibrated_wavenumbers.min())
    raw_max = float(parser.calibrated_wavenumbers.max())
    analysis_range = parser.calibrated_wavenumbers[161:898]

    print("Summary:")
    print(
        f"  The full native calibrated range of {raw_min:.1f} to {raw_max:.1f} cm^-1 is grounded in the source MATLAB code,"
    )
    print("  because Fig4D explicitly fits the anchor pixels with a cubic polynomial and evaluates it over 1:1650.")
    print(
        "  However, the published Figure 4 workflow does not appear to trust or use the entire extrapolated edge range."
    )
    print(
        f"  The scripts repeatedly subset the axis, and Fig4D uses range {matlab_range} with xlim [{matlab_xlim}] cm^-1."
    )
    print(
        f"  That subset corresponds roughly to {analysis_range.min():.1f} to {analysis_range.max():.1f} cm^-1 on the parser axis."
    )
    print(
        "  Conclusion: keep the full calibrated axis as the native raw storage range, but treat the low and high edges as extrapolated and less trustworthy for downstream comparison."
    )
    print(
        "  Recommendation: 450 to 1800 cm^-1 remains a sensible processed comparison window, and it is conservative relative to the paper's own 400 to 1700 cm^-1 plotting window."
    )


if __name__ == "__main__":
    main()
