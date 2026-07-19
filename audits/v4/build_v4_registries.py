"""GAIRA V4 — controlled-perturbation-evaluation + direct-grounding registries.

Reframes 'calibration' datasets as HELD-OUT controlled perturbation evaluations
(they must NOT fit axes/weights/centers/scales). Builds the definitive direct-
molecular-grounding source list, split by modality/substrate. Read-only.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
OUT = Path("/Users/surajpg/projects/GAIRA/data_audit"); OUT.mkdir(exist_ok=True)

# ── v4_controlled_perturbation_evaluation_registry.csv ──
# model_frozen_before_eval = YES for all; evaluation data EXCLUDED from axis/weight fitting.
pe = [
 dict(evaluation_id="adenine_concentration_response",challenge_type="dose_response_challenge",
      target="purine (G01)",matrix="aqueous",substrate="bAgNPs SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="RamanBioLib + adenine reference motif",
      evaluation_excluded_from_fitting="YES",expected_response="G01 up with conc",
      observed_bsv_response="G01 up (Spearman 0.83)",mss_response="adenine MSS fires",off_target="low",
      substrate_caveats="Ag-only; dampen unvalidated",verdict="supportive"),
 dict(evaluation_id="ergothioneine_dose_response",challenge_type="dose_response_challenge",
      target="redox (G10)",matrix="aqueous",substrate="cAg SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="ergothioneine motif/anchor",
      evaluation_excluded_from_fitting="YES",expected_response="G10 up with conc",
      observed_bsv_response="G10 up (Spearman 0.94)",mss_response="ergothioneine MSS fires",off_target="low",
      substrate_caveats="thiol boost heuristic",verdict="supportive"),
 dict(evaluation_id="hypoxanthine_spike_serum",challenge_type="enzyme_depletion_challenge/spike",
      target="purine-metabolite (G02)",matrix="commercial serum",substrate="Ag colloid SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="hypoxanthine reference",
      evaluation_excluded_from_fitting="YES",expected_response="G02 up on spike",
      observed_bsv_response="agree (small)",mss_response="hypoxanthine support",off_target="serum matrix",
      substrate_caveats="carotenoid 1517 overlap",verdict="supportive"),
 dict(evaluation_id="hypoxanthine_plus_uricase",challenge_type="enzyme_depletion_challenge",
      target="G02",matrix="serum+uricase",substrate="Ag colloid SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="hypoxanthine/UA reference",
      evaluation_excluded_from_fitting="YES",expected_response="hypox up; UA down",
      observed_bsv_response="agree",mss_response="mixed",off_target="serum matrix",
      substrate_caveats="single serum pool n=1x5",verdict="supportive"),
 dict(evaluation_id="uric_acid_uricase_depletion",challenge_type="enzyme_depletion_challenge",
      target="G02 (uric acid)",matrix="serum+uricase",substrate="Ag colloid SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="UA reference",
      evaluation_excluded_from_fitting="YES",expected_response="UA/G02 DOWN on uricase",
      observed_bsv_response="INCONSISTENT (6/11 axes wrong direction)",mss_response="mixed",off_target="high",
      substrate_caveats="n=5/5; serum matrix variability",verdict="INCONSISTENT (preserved, not laundered)"),
 dict(evaluation_id="15N_uric_acid_isotope",challenge_type="isotope_challenge",
      target="G02 band shift",matrix="aqueous +/- HSA",substrate="Ag colloid SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="UA reference",
      evaluation_excluded_from_fitting="YES",expected_response="15N band shift confirms UA assignment",
      observed_bsv_response="mechanistic (binding study)",mss_response="n/a",off_target="n/a",
      substrate_caveats="mechanistic not BSV",verdict="context (supports UA assignment mechanistically)"),
 dict(evaluation_id="53_serum_metabolite_spikes",challenge_type="analytical_challenge_set",
      target="multi-axis",matrix="Merck serum",substrate="Ag colloid SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="53 pure metabolite references",
      evaluation_excluded_from_fitting="YES",expected_response="per-metabolite axis response",
      observed_bsv_response="NOT run as a demo evaluation",mss_response="unused",off_target="unknown",
      substrate_caveats="serum matrix",verdict="not_evaluated (available, unwired)"),
 dict(evaluation_id="european_interinstrument_adenine",challenge_type="cross_platform_challenge",
      target="purine (G01) reproducibility",matrix="aqueous",substrate="cAg/cAu/sAg/sAu x 532/785",
      model_frozen_before_eval="YES",grounding_sources_allowed="adenine reference",
      evaluation_excluded_from_fitting="YES",expected_response="stable purine top-1 across platforms",
      observed_bsv_response="purine top-1 across all substrates; G01 CV~0.14 (substrate rules give no benefit)",
      mss_response="adenine fires",off_target="low",
      substrate_caveats="demo/prod blind to Au/planar/excitation",verdict="partially_supportive (identification robust; magnitude not)"),
 dict(evaluation_id="serum_protocol_comparison",challenge_type="cross_platform_challenge",
      target="protocol reproducibility",matrix="commercial serum",substrate="Ag colloid SERS",
      model_frozen_before_eval="YES",grounding_sources_allowed="n/a",
      evaluation_excluded_from_fitting="YES",expected_response="protocol stability",
      observed_bsv_response="not evaluated in demo",mss_response="n/a",off_target="n/a",
      substrate_caveats="5 protocols",verdict="not_evaluated"),
]
pd.DataFrame(pe).to_csv(OUT/"v4_controlled_perturbation_evaluation_registry.csv", index=False)

# ── v4_direct_grounding_sources.csv (by modality/substrate) ──
gs = [
 dict(source="RamanBioLib",publication="DOI 10.1002/jrs.1734 (140 components)",unique_analytes=141,
      measured_spectra=202,raman_or_sers="Raman(spontaneous)",substrate="CaF2/glass/metal-ring",excitation="785/1064/532/488...",
      pure_or_matrix="pure",raw_available="YES(parquet 272902 pts)",replicates="1/(compound x substrate x laser)",
      concentrations="n/a",in_duckdb="YES(reference 202)",used_demo="YES(202 table)",used_prod="YES(reference_spectra)",
      recommended_role="direct molecular grounding — Raman observation domain"),
 dict(source="amino_acid_raman_grounding",publication="curated AA panel",unique_analytes=20,
      measured_spectra=20,raman_or_sers="Raman",substrate="powder/CaF2",excitation="",
      pure_or_matrix="pure",raw_available="xlsx",replicates="1/analyte",concentrations="n/a",
      in_duckdb="YES(grounding 20)",used_demo="grounding",used_prod="grounding",
      recommended_role="direct molecular grounding — Raman observation domain"),
 dict(source="adenine_sers_control",publication="bAgNP adenine",unique_analytes=1,
      measured_spectra="12-17",raman_or_sers="SERS",substrate="bAgNPs",excitation=785,
      pure_or_matrix="pure",raw_available="YES(CSV)",replicates=5,concentrations="6-7 conc",
      in_duckdb="YES(grounding 16)",used_demo="YES(live)",used_prod="calibration/loaders",
      recommended_role="direct grounding (Ag-SERS) + dose-response perturbation challenge"),
 dict(source="sers_metabolite_63",publication="PMC6989628 (NIHMS1547448)",unique_analytes="63(verify)",
      measured_spectra=63,raman_or_sers="SERS",substrate="Au colloid (VERIFY — see metabolite63 audit)",excitation="verify",
      pure_or_matrix="pure",raw_available="xlsx(averaged)",replicates="1 avg/analyte",concentrations="physiological",
      in_duckdb="YES(grounding 64)",used_demo="grounding count",used_prod="grounding",
      recommended_role="direct molecular grounding — Au-SERS observation domain (substrate metadata must be preserved)"),
 dict(source="serum_ag_colloids (Gobbato, 53 pure metabolites)",publication="Gobbato/Bonifacio Trieste",unique_analytes=53,
      measured_spectra="265(pure) / 153 Raman powders",raman_or_sers="SERS(pure) + Raman(powder)",substrate="Ag colloid",excitation=785,
      pure_or_matrix="pure(SERS metabolites) + powder(Raman)",raw_available="YES(907 txt)",replicates=5,concentrations="physiological",
      in_duckdb="YES(serum_ag_colloids_grounding 368)",used_demo="SAEL contrasts",used_prod="grounding",
      recommended_role="direct grounding (Ag-SERS pure metabolites 265 + Raman 153); serum-context = perturbation eval, NOT grounding"),
 dict(source="ag_flakes_metabolites_23",publication="Spectrochim Acta A S1386142523012726",unique_analytes="23(verify)",
      measured_spectra="peak tables (verify)",raman_or_sers="SERS",substrate="Ag flakes",excitation="verify",
      pure_or_matrix="pure",raw_available="DOCX peak tables (see audit)",replicates="verify",concentrations="verify",
      in_duckdb="NO",used_demo="no",used_prod="no",
      recommended_role="direct PEAK-LEVEL SERS grounding (Ag-flake) — MSS band construction/collision, NOT full-spectrum"),
 dict(source="metabolite_sers63_support",publication="Fityk fits",unique_analytes="(support)",
      measured_spectra=0,raman_or_sers="fit products",substrate="",excitation="",
      pure_or_matrix="n/a",raw_available=".fit/.peaks",replicates="",concentrations="",
      in_duckdb="peaks",used_demo="no",used_prod="peak support",
      recommended_role="peak-evidence support (NOT measured spectra)"),
]
pd.DataFrame(gs).to_csv(OUT/"v4_direct_grounding_sources.csv", index=False)
print("wrote v4_controlled_perturbation_evaluation_registry (%d), v4_direct_grounding_sources (%d)" % (len(pe), len(gs)))
print("perturbation datasets: all model_frozen_before_eval=YES, excluded_from_fitting=YES")
print("direct grounding by modality: Raman(RamanBioLib 202 + AA 20), Ag-SERS(adenine + Gobbato 265 + Ag-flake peaks), Au-SERS(metabolite-63 63)")
