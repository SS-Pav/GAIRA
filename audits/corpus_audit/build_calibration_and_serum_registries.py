"""Emit serum Ag-colloid condition + calibration registries + axis calibration
coverage. Counts are from direct archive inspection (documented per row: source
path + how counted). Archive-derived counts are recorded as confirmed constants
because committed scripts must not extract archives into source dirs; the
extraction commands are documented in the serum/calibration audit reports.
Deterministic, read-only.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

OUT = Path("/Users/surajpg/projects/GAIRA/data_audit"); OUT.mkdir(exist_ok=True)
RAW = "/Volumes/SSD_Rad/GAIRA_DATA/raw"

# ── serum_ag_colloid_conditions.csv (Gobbato/Bonifacio Trieste dataset) ──
serum = [
    ("uricase_serum_hypox_spike","hypoxanthine","commercial(Sigma)","none","spike","spiked test",5,1,5,f"{RAW}/serum_ag_colloids/dataset uricase/*Serumspiked_Prot1.txt","calibration(perturbation)","supportive"),
    ("uricase_serum_hypox_spike_enzyme","hypoxanthine+uricase","commercial(Sigma)","uricase","spike","enzyme-treated",5,1,5,f"{RAW}/serum_ag_colloids/dataset uricase/*Serumspiked+Enzyme_Prot1.txt","calibration(depletion)","partially_supportive"),
    ("uricase_serum_neat","uric_acid(endogenous)","commercial(Sigma)","none","none","untreated control",5,1,5,f"{RAW}/serum_ag_colloids/dataset uricase/*SerumSigma_Prot1.txt","calibration(control)","supportive"),
    ("uricase_serum_neat_enzyme","uric_acid(uricase-removed)","commercial(Sigma)","uricase","none","enzyme control",5,1,5,f"{RAW}/serum_ag_colloids/dataset uricase/*SerumSigma+Enzyme_Prot1.txt","calibration(depletion)","inconsistent(historical SAEL)"),
    ("isotopic_UA_vs_15N","uric_acid vs 15N-uric_acid","aqueous","none","280uM","isotope control",15,2,5,f"{RAW}/serum_ag_colloids/isotopic/","calibration(isotope)","supportive(binding study)"),
    ("isotopic_UA_HSA_filter","UA-albumin binding + ultrafiltration","+HSA 50mg/mL","none","280uM","binding/filtration",58,None,None,f"{RAW}/serum_ag_colloids/isotopic/","mechanistic","context"),
    ("sers_metabolites_pure","53 pure metabolites (incl adenine,hypoxanthine,uric_acid,ergothioneine,glucose,urea,AAs)","aqueous","none","physiological","pure standard",265,53,5,f"{RAW}/serum_ag_colloids/SERS metabolites/","grounding+calibration","supportive"),
    ("sers_spiked_serum_53","53 metabolites spiked into serum + SerumSigma_na control","Merck serum","none","physiological","spiked",270,54,5,f"{RAW}/serum_ag_colloids/SERS spiked serum Merck/","calibration(spike)","supportive"),
    ("raman_metabolites_powder","51 metabolite powders","none(powder)","none","none","pure Raman standard",153,51,None,f"{RAW}/serum_ag_colloids/Raman metabolites/","grounding(Raman)","reference"),
    ("serum_fitting_refs","hypoxanthine / UA-free / UA-bound fit references","aqueous/+HSA","none","none","fit reference",30,3,10,f"{RAW}/serum_ag_colloids/SERS metabolites for fitting/","mechanistic","context"),
    ("serum_merck_baseline","commercial serum baseline","Merck serum","none","none","matrix control",15,3,5,f"{RAW}/serum_ag_colloids/SERS serum Merck/","calibration(control)","supportive"),
    ("donors_serum_81","81 healthy-donor serum","donor serum","none","none","donor cohort",81,81,1,f"{RAW}/serum_ag_colloids/donors serum SERS/","biological(reference cohort)","context"),
    ("digitized_literature","De Gelder2007/Kim1987/Stewart1999 digitized","n/a","none","none","literature-digitized",0,None,None,f"{RAW}/serum_ag_colloids/digitized literature spectra/","supporting_literature","not measured"),
    # cspp_serum companion
    ("cspp_fig7_background","serum background","pooled serum(SS)","none","0uM","background",50,None,None,f"{RAW}/cspp_serum/Figure-7_all-spectra-and-metadata.csv","calibration(control)","supportive"),
    ("cspp_fig7_ergothioneine","ergothioneine","pooled serum(SS)","none","25uM","spike",50,None,None,f"{RAW}/cspp_serum/Figure-7_all-spectra-and-metadata.csv","calibration(spike)","supportive"),
    ("cspp_fig7_hypoxanthine","hypoxanthine","pooled serum(SS)","none","50uM","spike",50,None,None,f"{RAW}/cspp_serum/Figure-7_all-spectra-and-metadata.csv","calibration(spike)","supportive"),
]
scols = ["condition_id","analyte_or_intervention","serum_background","enzyme","spike_concentration",
         "control_type","n_spectra","n_independent_samples","n_replicates","source_file","role_in_gaira","validation_status"]
pd.DataFrame(serum, columns=scols).to_csv(OUT/"serum_ag_colloid_conditions.csv", index=False)

# ── calibration_dataset_registry.csv ──
cal = [
  dict(calibration_id="adenine_bagnp",dataset="adenine_sers_control",target_analyte_or_behavior="adenine dose response",matrix="aqueous",modality="SERS",substrate="bAgNPs",excitation_nm=785,concentrations_or_conditions="10pg..10ug/mL (6-7 levels)+stability",n_levels=7,n_independent_samples=6,n_spectra=17,n_replicates=5,expected_direction="purine up with conc",observed_direction="G01 up (Spearman 0.83)",primary_gaira_axis="G01_purine_nucleotide",secondary_axes="G02,G04",specificity_test="partial",negative_control="background CSV",depletion_or_enzyme_test="no",isotope_test="no",cross_instrument_test="no",cross_substrate_test="no",currently_used_v3_1="YES (6 conc, live)",currently_used_src_gaira="unknown",validation_verdict="supportive",limitations="single substrate; substrate dampen ×0.65 not validated"),
  dict(calibration_id="adenine_european_ils",dataset="european_multi_instrument_adenine",target_analyte_or_behavior="adenine reproducibility across labs/substrates",matrix="aqueous",modality="SERS",substrate="sAg/sAu/cAg/cAu",excitation_nm="532;785",concentrations_or_conditions="43 conc; train/test/blank",n_levels=43,n_independent_samples="15 labs",n_spectra="3516 (ILS) / 7032 raw txt",n_replicates="a/b/c",expected_direction="purine up",observed_direction="G01 varies by substrate (CV~0.13)",primary_gaira_axis="G01_purine_nucleotide",secondary_axes="",specificity_test="no",negative_control="blank",depletion_or_enzyme_test="no",isotope_test="no",cross_instrument_test="YES(15 labs)",cross_substrate_test="YES(4 substrates,2 lasers)",currently_used_v3_1="NO (only ablation)",currently_used_src_gaira="unknown",validation_verdict="partially_supportive",limitations="demo blind to Au/planar/excitation — cannot exploit this dataset"),
  dict(calibration_id="ergothioneine_cag",dataset="ergothioneine_serum",target_analyte_or_behavior="ergothioneine dose (redox)",matrix="aqueous/serum-study",modality="SERS",substrate="cAg",excitation_nm=785,concentrations_or_conditions="0.0-2.0uM step0.2",n_levels=11,n_independent_samples=11,n_spectra=55,n_replicates=5,expected_direction="G10 up with conc",observed_direction="G10 up (Spearman 0.94)",primary_gaira_axis="G10_sulfur_thiol_redox",secondary_axes="G11",specificity_test="partial",negative_control="0uM",depletion_or_enzyme_test="no",isotope_test="no",cross_instrument_test="no",cross_substrate_test="no",currently_used_v3_1="YES (live, equivalence)",currently_used_src_gaira="unknown",validation_verdict="supportive",limitations="thiol boost ×1.20 heuristic; single substrate"),
  dict(calibration_id="serum_ag_colloid_uricase",dataset="serum_ag_colloids(Gobbato)",target_analyte_or_behavior="hypoxanthine spike + uricase depletion of UA",matrix="commercial serum",modality="SERS",substrate="Ag colloid",excitation_nm=785,concentrations_or_conditions="spike/enzyme 4-condition design",n_levels=4,n_independent_samples=1,n_spectra=20,n_replicates=5,expected_direction="UA down on uricase; hypox up on spike",observed_direction="hypox spike agree; uricase depletion INCONSISTENT",primary_gaira_axis="G02_purine_metabolite",secondary_axes="G01",specificity_test="yes(enzyme)",negative_control="neat serum",depletion_or_enzyme_test="YES(uricase)",isotope_test="YES(15N UA)",cross_instrument_test="no",cross_substrate_test="no",currently_used_v3_1="YES (as cached SAEL contrasts)",currently_used_src_gaira="grounding(serum_ag_colloids_grounding)",validation_verdict="partially_supportive(uricase inconsistent)",limitations="n=1 serum pool ×5 rep; uricase contrast inconsistent, preserved honestly"),
  dict(calibration_id="cspp_serum_spike",dataset="cspp_serum",target_analyte_or_behavior="ergothioneine + hypoxanthine serum spikes",matrix="pooled serum",modality="SERS",substrate="Ag colloid",excitation_nm=785,concentrations_or_conditions="Bkg/Erg25uM/Hyp50uM",n_levels=3,n_independent_samples="pooled",n_spectra=150,n_replicates="~50/cond",expected_direction="Erg->G10, Hyp->G02 up",observed_direction="agree",primary_gaira_axis="G10;G02",secondary_axes="",specificity_test="partial",negative_control="Bkg",depletion_or_enzyme_test="no",isotope_test="no",cross_instrument_test="no",cross_substrate_test="no",currently_used_v3_1="YES (SAEL contrasts)",currently_used_src_gaira="unknown",validation_verdict="supportive",limitations="single conc per analyte (conc confounded)"),
  dict(calibration_id="sers_metabolite_63",dataset="sers_metabolite_63(NIHMS1547448)",target_analyte_or_behavior="63-metabolite SERS reference panel",matrix="aqueous",modality="SERS",substrate="Ag",excitation_nm="",concentrations_or_conditions="1 per analyte",n_levels=1,n_independent_samples=63,n_spectra=63,n_replicates=1,expected_direction="reference",observed_direction="reference",primary_gaira_axis="multi",secondary_axes="",specificity_test="no",negative_control="no",depletion_or_enzyme_test="no",isotope_test="no",cross_instrument_test="no",cross_substrate_test="no",currently_used_v3_1="grounding(warehouse 64)",currently_used_src_gaira="grounding",validation_verdict="reference",limitations="one averaged spectrum per metabolite"),
  dict(calibration_id="amino_acid_raman_20",dataset="amino_acid_raman_grounding",target_analyte_or_behavior="20-analyte amino-acid/metabolite Raman panel",matrix="powder",modality="Raman",substrate="powder/CaF2",excitation_nm="",concentrations_or_conditions="1 per analyte",n_levels=1,n_independent_samples=20,n_spectra=20,n_replicates=1,expected_direction="reference",observed_direction="reference",primary_gaira_axis="G06,G07",secondary_axes="",specificity_test="no",negative_control="no",depletion_or_enzyme_test="no",isotope_test="no",cross_instrument_test="no",cross_substrate_test="no",currently_used_v3_1="grounding(warehouse 20)",currently_used_src_gaira="grounding",validation_verdict="reference",limitations="single spectrum per analyte; Raman not SERS"),
  dict(calibration_id="otc_drugs_raman",dataset="otc_drugs",target_analyte_or_behavior="3 OTC drugs (aspirin/ibuprofen/paracetamol)",matrix="pure substance",modality="Raman",substrate="none",excitation_nm="",concentrations_or_conditions="generic+trademark",n_levels=2,n_independent_samples=3,n_spectra=300,n_replicates=50,expected_direction="drug detection",observed_direction="not wired",primary_gaira_axis="n/a",secondary_axes="",specificity_test="no",negative_control="no",depletion_or_enzyme_test="no",isotope_test="no",cross_instrument_test="no",cross_substrate_test="no",currently_used_v3_1="NO",currently_used_src_gaira="drug_detection module",validation_verdict="not_tested",limitations="drug detection, out of biochemical scope"),
]
pd.DataFrame(cal).to_csv(OUT/"calibration_dataset_registry.csv", index=False)

# ── axis_calibration_coverage.csv ──
axcov = [
 ("G01_purine_nucleotide","adenine_bagnp; adenine_european_ils; serum_ag_colloid(adenine spike)","supportive","adenine dose Spearman 0.83; but substrate blindness limits cross-substrate"),
 ("G02_purine_metabolite","serum_ag_colloid_uricase; cspp_hypoxanthine","partially_supportive","hypoxanthine spike agree; uricase depletion INCONSISTENT (preserved)"),
 ("G03_pyrimidine_nucleotide","(none)","not_tested","no dedicated calibration analyte"),
 ("G04_nucleic_acid_phosphate","(none direct)","not_tested","only incidental in adenine"),
 ("G05_glycan_carbohydrate","serum_ag_colloid(glucose spike)","insufficient","glucose present in 53-metabolite spike but not isolated as a calibration"),
 ("G06_protein_peptide_backbone","amino_acid_raman_20(partial)","insufficient","reference panel only; no dose/perturbation"),
 ("G07_aromatic_residue","amino_acid_raman_20(Phe,Trp)","insufficient","reference only"),
 ("G08_lipid_acyl_membrane","(none)","not_tested","no lipid calibration series"),
 ("G09_sterol_neutral_lipid","(none)","not_tested","no sterol calibration series"),
 ("G10_sulfur_thiol_redox","ergothioneine_cag; cspp_ergothioneine; serum_ag_colloid(ergothioneine spike)","supportive","ergothioneine dose Spearman 0.94"),
 ("G11_metabolic_small_molecule","serum_ag_colloid(53 metabolites)","insufficient","many metabolites present but none run as a demo calibration series"),
]
pd.DataFrame(axcov, columns=["axis","calibration_datasets","verdict","note"]).to_csv(OUT/"axis_calibration_coverage.csv", index=False)

print("wrote serum_ag_colloid_conditions.csv (%d), calibration_dataset_registry.csv (%d), axis_calibration_coverage.csv (11)"
      % (len(serum), len(cal)))
print("calibration SERS spectra (curated): 17+3516+55+75+63 + serum-agcolloid-controlled ~ ; Raman: 20+300=320")
