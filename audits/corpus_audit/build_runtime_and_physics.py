"""Emit runtime-usage, physics-atlas, and substrate/modality rule registries.
Encodes findings confirmed by direct code/DB inspection (traceable to file
paths + code locations). Read-only, deterministic.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT = Path("/Users/surajpg/projects/GAIRA/data_audit"); OUT.mkdir(exist_ok=True)

# ── runtime_dataset_usage.csv ──
runtime = [
 # dataset, v3_1_tab, loader, transformation, raw_or_cached, fit_or_projection, role, fallback, placeholder, src_module, domain_pack, retrieval_weight, substrate_rule, output_affected
 dict(dataset="grounding_molecule_bsv.csv (202/141 RamanBioLib)",v3_1_tab="Mode1 11-axis space; coverage",loader="data_loader.load_reference_points",transformation="8->11 remap",raw_or_cached="bundled CSV",fit_or_projection="viz only",grounding_or_calibration_or_biology="molecular_grounding",fallback_behavior="curated placeholder",placeholder_possible="yes",src_gaira_module="reference_metadata/reference_spectra (DB)",domain_pack="GAIRA_GROUNDING",retrieval_weight="n/a",substrate_rule="n/a",output_affected="biochemical space + coverage"),
 dict(dataset="warehouse_source_registry.csv (43/28 sources)",v3_1_tab="Mode1 Grounding Corpus Map",loader="data_loader._load_grounding_corpus_real",transformation="tiering",raw_or_cached="SSD CSV",fit_or_projection="viz",grounding_or_calibration_or_biology="grounding+literature",fallback_behavior="curated placeholder",placeholder_possible="yes",src_gaira_module="grounding_search (DB)",domain_pack="GAIRA_GROUNDING",retrieval_weight="n/a",substrate_rule="n/a",output_affected="corpus map"),
 dict(dataset="adenine_sers_control (6 conc)",v3_1_tab="Mode2 Adenine",loader="data_loader._load_adenine_real",transformation="crop+interp+build_report",raw_or_cached="SSD raw CSV (live)",fit_or_projection="projection",grounding_or_calibration_or_biology="calibration",fallback_behavior="sigmoid placeholder",placeholder_possible="yes",src_gaira_module="calibration/loaders",domain_pack="GAIRA_GROUNDING",retrieval_weight="n/a",substrate_rule="ag_sers_purine_amplify 0.65",output_affected="G01 dose radar"),
 dict(dataset="ergothioneine_serum ERG_calibration.csv (55)",v3_1_tab="Mode2 Ergothioneine; equivalence",loader="build_diabetes_equivalence / legacy dose CSV",transformation="build_report / cached SAEL",raw_or_cached="SSD raw + bundled",fit_or_projection="projection",grounding_or_calibration_or_biology="calibration",fallback_behavior="sigmoid placeholder",placeholder_possible="yes",src_gaira_module="calibration/loaders",domain_pack="GAIRA_SERUM",retrieval_weight="n/a",substrate_rule="ag_sers_thiol_amplify 1.20",output_affected="G10 dose"),
 dict(dataset="serum_ag_colloids / cspp (SAEL contrasts)",v3_1_tab="Mode2 Uric Acid",loader="data_loader.load_uric_acid_validation",transformation="8->11 remap of SAEL delta",raw_or_cached="bundled CSV",fit_or_projection="cached",grounding_or_calibration_or_biology="calibration",fallback_behavior="placeholder",placeholder_possible="yes",src_gaira_module="grounding(serum_ag_colloids_grounding)",domain_pack="GAIRA_SERUM",retrieval_weight="n/a",substrate_rule="ag_sers_carotenoid_overlap 0.85 (caveat)",output_affected="uric-acid contrasts (uricase inconsistent)"),
 dict(dataset="pilot4_1 serum_liver mean spectra (212)",v3_1_tab="Mode3 Serum Liver",loader="_load_serum_liver_from_spectra + global_coordinates",transformation="build_report + frozen calib",raw_or_cached="SSD processed CSV",fit_or_projection="fit(range)+projection",grounding_or_calibration_or_biology="biological",fallback_behavior="autoresearch BSV then placeholder",placeholder_possible="yes",src_gaira_module="inference.py (DB class summaries)",domain_pack="GAIRA_SERUM",retrieval_weight="serum ctx",substrate_rule="Ag colloid SERS",output_affected="serum global coords + effect sizes"),
 dict(dataset="pilot2 EV-diabetes sample_query_spectra (63)",v3_1_tab="Mode3 EV Diabetes",loader="_load_ev_diabetes_from_spectra + build_diabetes_equivalence",transformation="build_report + frozen calib + cohort z",raw_or_cached="SSD processed CSV",fit_or_projection="fit(range)+projection",grounding_or_calibration_or_biology="biological",fallback_behavior="autoresearch BSV then placeholder",placeholder_possible="yes",src_gaira_module="inference.py + ev_context",domain_pack="GAIRA_EV",retrieval_weight="EV ctx (G08/G09 x0.7)",substrate_rule="Ag colloid SERS",output_affected="EV cohort-relative + global"),
 dict(dataset="pilot3 SHINE class_mean_bsv (8 cohorts)",v3_1_tab="Mode3 SHINE (reduced)",loader="_load_shine_real",transformation="autoresearch 3-axis remap",raw_or_cached="SSD processed CSV",fit_or_projection="cached",grounding_or_calibration_or_biology="biological",fallback_behavior="placeholder",placeholder_possible="yes",src_gaira_module="autoresearch pilot3",domain_pack="GAIRA_EV",retrieval_weight="n/a",substrate_rule="n/a",output_affected="reduced-dim heatmap (11-axis radar removed)"),
 dict(dataset="global_coordinate_calibration_v1.json (frozen)",v3_1_tab="Mode3/4 global coords",loader="global_coordinates.load_calibration",transformation="apply frozen robust-z",raw_or_cached="bundled JSON",fit_or_projection="apply(never refit)",grounding_or_calibration_or_biology="calibration(derived)",fallback_behavior="GLOBAL COORDINATE UNAVAILABLE",placeholder_possible="no(explicit)",src_gaira_module="n/a(demo-only)",domain_pack="n/a",retrieval_weight="n/a",substrate_rule="n/a",output_affected="all global coords"),
 dict(dataset="interim/gaira.duckdb (185,686 biosample + 468 grounding + 202 ref)",v3_1_tab="NONE (demo never reads DB)",loader="n/a",transformation="n/a",raw_or_cached="n/a",fit_or_projection="n/a",grounding_or_calibration_or_biology="all",fallback_behavior="n/a",placeholder_possible="n/a",src_gaira_module="inference.py, grounding_search, ev/serum_context, sael, expected",domain_pack="all",retrieval_weight="reranking",substrate_rule="n/a",output_affected="production inference (NOT the demo)"),
 dict(dataset="GAIRA_BUILD substrate seeds (42 effects/47 registry)",v3_1_tab="NONE",loader="n/a",transformation="n/a",raw_or_cached="n/a",fit_or_projection="n/a",grounding_or_calibration_or_biology="substrate_physics",fallback_behavior="n/a",placeholder_possible="n/a",src_gaira_module="src/gaira/substrate/* (LOADABLE but imported by nothing)",domain_pack="n/a",retrieval_weight="n/a",substrate_rule="42 bounded [0.40,1.15]",output_affected="DORMANT (no runtime importer)"),
 dict(dataset="GAIRA_BUILD atlas phase4 YAMLs",v3_1_tab="NONE",loader="n/a",transformation="n/a",raw_or_cached="n/a",fit_or_projection="n/a",grounding_or_calibration_or_biology="physics_atlas",fallback_behavior="n/a",placeholder_possible="n/a",src_gaira_module="src/gaira/atlas/atlas_loader (no importer)",domain_pack="n/a",retrieval_weight="n/a",substrate_rule="band constraints",output_affected="DORMANT"),
]
pd.DataFrame(runtime).to_csv(OUT/"runtime_dataset_usage.csv", index=False)

# ── physics_atlas_registry.csv (8 demo atlas regions; literature-derived; UI-only) ──
def A(i, lo, hi, theme, assign, coll, sers, impl, effect, valid):
    return dict(atlas_entry_id=i, wavenumber_min=lo, wavenumber_max=hi, primary_theme=theme,
                candidate_assignments=assign, known_collisions=coll,
                raman_behavior="reference bands", sers_behavior=sers,
                ag_behavior="Ag-colloid prose only", au_behavior="not modeled",
                excitation_dependence="not modeled", adsorption_dependence="prose only",
                evidence_source="literature/curated prose (app.py atlas_details)",
                evidence_type="literature", confidence="curated",
                implemented_as_rule=impl, code_location="config.ATLAS_REGIONS + app.py atlas_details",
                runtime_effect=effect, validation_status=valid)
atlas = [
 A(1,400,700,"Skeletal/ring/metal-ligand","sterol 548; thione/thiol 490-510","many low-freq overlaps","metal-ligand dominated","no(UI text)","caveat/UI only","not_testable"),
 A(2,700,760,"Purine/nucleobase","adenine 725; guanine 670; hypoxanthine/UA","adenine/hypox/UA ~720-740","Ag-SERS x3-10 amplify","partially(via substrate rule 0.65)","informs substrate rule + caveat","suggestive"),
 A(3,760,900,"Ring breathing/AA/carbohydrate","Tyr 830/850; Trp 760; sugar/lactate 845","lactate vs sugar 845","moderate","no(UI)","caveat only","not_testable"),
 A(4,900,1150,"C-C/C-O/phosphate/glycan","glycan 1020-1150; PO2 1080; glucose 1125","phosphate vs glycan","phosphate weak on SERS","no(UI)","caveat only","not_testable"),
 A(5,1150,1350,"Nucleobase/protein/lipid mixed","adenine 1335; amide-III 1230-1300","highly mixed","SERS rebalances","no(UI)","caveat only","not_testable"),
 A(6,1350,1500,"CH deformation/nucleobase/lipid","CH2 1440; adenine 1485","1440 shared all lipids","minor shift","no(UI)","caveat only","not_testable"),
 A(7,1500,1700,"Aromatic/amide/C=C/unsat","amide-I 1640-70; Phe 1605; carotenoid 1517","amide-I vs lipid C=C; carotenoid vs UA","carotenoid matrix-dependent","partially(carotenoid caveat 0.85)","caveat + G02 caveat","suggestive"),
 A(8,1700,1800,"Carbonyl/lipid ester/oxidation","ester 1745; oxidation 1700-1800","oxidation artifacts","laser photoproducts on SERS","no(UI)","caveat only","not_testable"),
]
pd.DataFrame(atlas).to_csv(OUT/"physics_atlas_registry.csv", index=False)

# ── substrate_physics_rules.csv (consolidated) ──
rules = [
 ("ag_sers_purine_amplify","Ag colloid SERS","motif purine_720_735 / G01","multiply",0.65,"Ag purine over-amplification","demo/substrate_physics.py","yes","no","no","heuristic_multiplier","dampens G01; no dose-ordering benefit (ablation)"),
 ("ag_sers_carotenoid_overlap","Ag colloid SERS","axis G02","caveat",0.85,"carotenoid 1517 vs UA","demo/substrate_physics.py","yes(caveat)","no","no","caveat_generator","no numeric BSV change"),
 ("raman_amide_protein","Raman","motif amide-III / G06","none",1.0,"amide-III reliable","demo/substrate_physics.py","yes(no-op)","no","no","metadata_only","no effect"),
 ("raman_amide_i_lipid_overlap","Raman","axis G08","caveat",0.92,"amide-I vs lipid C=C","demo/substrate_physics.py","yes(caveat)","no","no","caveat_generator","no numeric BSV change"),
 ("ag_sers_thiol_amplify","Ag colloid SERS","motif thione_490_500 / G10","multiply",1.20,"thiol Ag affinity","demo/substrate_physics.py","yes","no","no","heuristic_multiplier","raises G10; suggestive not validated"),
 ("diabetes_g10_window_tighten","Ag/plasma","motif thione 490-505","spectral_mask","-","reduce Ag-oxide/citrate baseline","analysis/_diabetes_overrides.py","opt-in tool","no","partial","spectral_mask","most defensible; audit-reasoned"),
 ("diabetes_coband_thiol_gate","Ag SERS + 720 co-band","G10","gated_multiply","1.20/1.0","require imidazole co-band","analysis/_diabetes_overrides.py","opt-in tool","no","partial","evidence_gating","most defensible substrate rule"),
 ("domain_context_weights","EV/serum domain","G08/G09/G02","rank_weight",0.70,"matrix-expected downweight","gaira_core/domain_context.py","yes(ranking)","no","no","evidence_reranking","ranking + caveat only; never changes coords"),
 ("prod_substrate_engine_42","substrate family+band","axis/motif/band","compose_multiply","0.40-1.15","source-backed evidence registry","src/gaira/substrate/* + GAIRA_BUILD","no","loadable_no_importer","source-backed","evidence_reranking_engine","DORMANT — most rigorous but unused"),
 ("prod_atlas_band_constraints","wavenumber band","band","constrain/ambiguity","-","evidence-derived band atlas","src/gaira/atlas/atlas_loader.py","no","loadable_no_importer","evidence-derived","spectral_constraint_engine","DORMANT"),
]
rcols=["rule_id","substrate_or_mode","affected_band_or_axis","operation","multiplier_or_penalty","scientific_basis","code_location","used_in_v3_1","used_in_src_gaira","validated","classification","risk_or_note"]
pd.DataFrame(rules, columns=rcols).to_csv(OUT/"substrate_physics_rules.csv", index=False)
print("wrote runtime_dataset_usage.csv (%d), physics_atlas_registry.csv (8), substrate_physics_rules.csv (%d)" % (len(runtime), len(rules)))
