from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/global_v2_forensic_candidate_review_batch_b"


ROWS = [
    {
        "candidate_id": "FC08",
        "title": "Dual-mode Identification of Ischemic Stroke based on Urine SERS Spectra and Carotid B-Ultrasound",
        "urls": "https://zenodo.org/records/19109120 | https://zenodo.org/records/19369604",
        "spectra_available": "yes",
        "sample_type": "human urine",
        "labels_present": "disease class; participant-level multimodal linkage",
        "usable_data_estimate": "10,100 urine SERS spectra from 101 participants reported in paper; newer Zenodo version splits data/code/results",
        "reconstruction_required": "yes",
        "likely_exclusions": "exclude code.rar, results.rar, README.docx, carotid B-ultrasound images if building spectra-only corpus, and superseded older monolithic version after verifying parity",
        "overlap_with_existing_gaira": "none",
        "final_decision": "core_training",
    },
    {
        "candidate_id": "FC12",
        "title": "Surface Enhanced Raman Spectroscopy and Machine Learning for Identification of Beta-Lactam Antibiotics Resistance Gene Fragment in Bacterial Plasmid",
        "urls": "https://zenodo.org/records/12740805",
        "spectra_available": "partial",
        "sample_type": "plasmid / gene-fragment capture assay",
        "labels_present": "assay condition only",
        "usable_data_estimate": "Raman zip contains one Au-Capture.csv with a few plotted spectral traces; non-Raman sidecars dominate release",
        "reconstruction_required": "no",
        "likely_exclusions": "exclude UV-Vis, electrophoresis, AFM, XPS, ML outputs, and assay-support characterization",
        "overlap_with_existing_gaira": "scope overlap with ramanbiolib",
        "final_decision": "reject",
    },
    {
        "candidate_id": "FC14",
        "title": "Specificity and strain-typing capabilities of Nanorod Array-Surface Enhanced Raman Spectroscopy for Mycoplasma pneumoniae detection",
        "urls": "https://zenodo.org/records/4941488 | https://figshare.com/articles/dataset/_PLS_DA_of_NA_SERS_specificity_and_sensitivity_in_discriminating_M_pneumoniae_strains_/493381?file=823019 | https://figshare.com/articles/dataset/_Cross_validated_PLS_DA_modeling_statistics_for_the_prediction_performance_for_NA_SERS_typing_of_individual_type_1_and_2_M_pneumoniae_clinical_isolates_/1467505?file=2154588",
        "spectra_available": "yes",
        "sample_type": "pathogen SERS / throat-swab-associated Mycoplasma panel",
        "labels_present": "species / strain; background; media control",
        "usable_data_estimate": "521 spectra in core CSV across 46 labeled groups including M. pneumoniae reference strains, clinical isolates, other mycoplasmas, background, and media controls",
        "reconstruction_required": "no",
        "likely_exclusions": "exclude figshare statistics/performance tables and separate background/media controls from core pathogen training lane if needed",
        "overlap_with_existing_gaira": "none",
        "final_decision": "core_training",
    },
    {
        "candidate_id": "FC17",
        "title": "Spectroscopic investigation of faeces with surface-enhanced Raman scattering: a case study with coeliac patients on gluten-free diet",
        "urls": "https://zenodo.org/records/5947010",
        "spectra_available": "yes",
        "sample_type": "human faeces",
        "labels_present": "CTR; CD; GFD; patient sex/age encoded in filenames",
        "usable_data_estimate": "27 patient-level faecal TXT spectra plus 3 pure-metabolite references, OTU tables, RData, and R code",
        "reconstruction_required": "no",
        "likely_exclusions": "exclude pure metabolite references from cohort ingest, OTU tables, RData copies, and R code",
        "overlap_with_existing_gaira": "none",
        "final_decision": "augmentation_only",
    },
    {
        "candidate_id": "FC23",
        "title": "Raman spectroscopic techniques to detect ovarian cancer biomarkers in blood plasma",
        "urls": "https://figshare.com/articles/dataset/Raman_spectroscopic_techniques_to_detect_ovarian_cancer_biomarkers_in_blood_plasma/6744206",
        "spectra_available": "yes",
        "sample_type": "human plasma",
        "labels_present": "ovarian cancer vs healthy; modality (Raman vs SERS)",
        "usable_data_estimate": "385 Raman txt + 385 SERS txt; folder counts imply about 28 healthy and 27 ovarian-cancer plasma donors with repeated spectra",
        "reconstruction_required": "moderate",
        "likely_exclusions": "exclude duplicate preprocessed copies if both raw and processed traces appear after unpacking; keep modality flag rather than mixing Raman/SERS blindly",
        "overlap_with_existing_gaira": "none",
        "final_decision": "core_training",
    },
    {
        "candidate_id": "FC24",
        "title": "ACS platelet workbook family",
        "urls": "https://figshare.com/articles/dataset/SERS_spectra_of_43_patients_with_ACS_xlsx/24747531?file=43481136 | https://figshare.com/articles/dataset/Suplementary_material_DIB_ACS_40_samples_xlsx/24564787?file=43183257",
        "spectra_available": "yes",
        "sample_type": "human platelet / ACS clinical samples",
        "labels_present": "sample ID; likely ACS cohort membership",
        "usable_data_estimate": "two workbook exports, each 284 rows x 424 columns on the main spectral sheet with many per-sample replicate traces",
        "reconstruction_required": "yes",
        "likely_exclusions": "exclude empty Sheet1 tabs and any duplicated spectral blocks between the two nearly overlapping workbooks",
        "overlap_with_existing_gaira": "none",
        "final_decision": "augmentation_only",
    },
    {
        "candidate_id": "FC25",
        "title": "Surface-enhanced and tip-enhanced Raman scattering in label-free characterization of erythrocyte membranes and extracellular vesicles",
        "urls": "https://figshare.com/articles/dataset/Surface-enhanced_SERS_and_tip-enhanced_TERS_Raman_scattering_in_label-free_characterization_of_erythrocyte_membranes_and_extracellular_vesicles_in_nano-scale_and_at_the_single-molecule_level_/24105993",
        "spectra_available": "yes",
        "sample_type": "RBC membrane ghosts / EV-focused nanoscale spectroscopy",
        "labels_present": "modality and material type only",
        "usable_data_estimate": "52 readily ingestable SERS txt files for RBC ghosts, 10 pure-standard txt references, one TERS OPJU project, one AFM size workbook",
        "reconstruction_required": "yes",
        "likely_exclusions": "exclude pure standard spectra from core ingest, AFM size sheet, and defer OPJU-only TERS content unless converted cleanly",
        "overlap_with_existing_gaira": "scope overlap with raman_knowledge_core",
        "final_decision": "grounding_only",
    },
    {
        "candidate_id": "FC32",
        "title": "UCLA saliva sEV gastric-cancer shard family",
        "urls": "https://figshare.com/articles/dataset/Health_control_01_sEV_Saliva_UCLA_ERCC_/20428395 | https://figshare.com/articles/dataset/GC_Patient_20_sEV_Saliva_UCLA_ERCC_/20427957 | https://figshare.com/articles/dataset/GC_Patient_19_sEV_Saliva_UCLA_ERCC_/20427954 | https://figshare.com/articles/dataset/GC_Patient_18_sEV_Saliva_UCLA_ERCC_/20427951 | https://figshare.com/articles/dataset/GC_Patient_17_sEV_Saliva_UCLA_ERCC_/20427948 | https://figshare.com/articles/dataset/GC_Patient_16_sEV_Saliva_UCLA_ERCC_/20427945 | https://figshare.com/articles/dataset/GC_Patient_15_sEV_Saliva_UCLA_ERCC_/20427939 | https://figshare.com/articles/dataset/GC_Patient_14_sEV_Saliva_UCLA_ERCC_/20427936 | https://figshare.com/articles/dataset/GC_Patient_13_sEV_Saliva_UCLA_ERCC_/20427933 | https://figshare.com/articles/dataset/GC_Patient_12_sEV_Saliva_UCLA_ERCC_/20427930 | https://figshare.com/articles/dataset/GC_Patient_11_sEV_Saliva_UCLA_ERCC_/20427927 | https://figshare.com/articles/dataset/GC_Patient_10_sEV_Saliva_UCLA_ERCC_/20427924 | https://figshare.com/articles/dataset/GC_Patient_9_sEV_Saliva_UCLA_ERCC_/20427921 | https://figshare.com/articles/dataset/GC_Patient_8_sEV_Saliva_UCLA_ERCC_/20427918 | https://figshare.com/articles/dataset/GC_Patient_7_sEV_Saliva_UCLA_ERCC_/20427909 | https://figshare.com/articles/dataset/GC_Patient_6_sEV_Saliva_UCLA_ERCC_/20427906 | https://figshare.com/articles/dataset/GC_Patient_5_sEV_Saliva_UCLA_ERCC_/20427903 | https://figshare.com/articles/dataset/GC_Patient_1_UG3/20282238 | https://figshare.com/articles/dataset/ERCC/20406102",
        "spectra_available": "yes",
        "sample_type": "human saliva small-EV cohort shards",
        "labels_present": "disease class; patient ID",
        "usable_data_estimate": "at least 2,231 txt spectra across 18 actual patient/control shard records; one ERCC metadata stub has no files",
        "reconstruction_required": "yes",
        "likely_exclusions": "exclude ERCC metadata-only stub, shard-level duplicates, and any malformed or non-spectral txt files after merge",
        "overlap_with_existing_gaira": "fragmented cohort family only",
        "final_decision": "augmentation_only",
    },
]


REPORT = """# Global v2 Forensic Candidate Review Batch B

None of the Batch B items are marked `already_ingested` in the current GAIRA deduplication registry.

### [FC08] Dual-mode Identification of Ischemic Stroke based on Urine SERS Spectra and Carotid B-Ultrasound

**Availability:**  
- Real spectra are available at the family level. The older Zenodo version (`19109120`) is one monolithic `Ischemic Stroke.rar` at about 1.49 GB. The newer version (`19369604`) is materially better organized: `data.rar` (~128 MB), `code.rar` (~84 MB), `results.rar` (~1.29 GB), and `README.docx`.  
- The paper description reports 10,100 urine SERS spectra and 481 carotid B-ultrasound images from 101 participants.  
- The family should be treated as one dataset, with the newer version preferred. For spectra ingest, the likely target is `data.rar`; `results.rar` is almost certainly derived outputs rather than primary spectra.

**Biological Content:**  
- Human urine from ischemic-stroke patients and healthy controls, paired with carotid imaging in the paper.  
- This is real biological mixture data, not a synthetic assay.  
- For Global v2, the value is specifically the urine SERS arm, not the imaging arm.

**Labels:**  
- Disease class: ischemic stroke vs healthy control.  
- Participant-level multimodal linkage to carotid B-ultrasound exists in the study design.  
- These labels are useful for biological representation learning because they tie biofluid molecular fingerprints to disease state.

**Quality:**  
- Likely sizable and useful, but reconstruction is required because the usable spectra are buried in RAR archives.  
- The biggest confound is mixed-modality packaging: code, results, and images are bundled with spectra.  
- SERS substrate effects will still matter, but this is a true human urine cohort rather than a toy assay.

**Exclusions:**  
- Exclude `code.rar`, `results.rar`, and `README.docx` from ingest.  
- Exclude carotid B-ultrasound images if the target corpus is spectra-only.  
- Exclude the older monolithic Zenodo version after confirming the newer release fully covers the same spectra.

**Paper Context:**  
- The actual experiment is dual-mode identification of ischemic stroke using urine SERS plus carotid B-ultrasound, with ML fusion reported to outperform either modality alone.  
- The clinically relevant ingest target is the urine SERS cohort, and the paper explicitly states the spectra count and participant count.  
- The newer Zenodo version materially improves packaging because it separates `data`, `code`, and `results`.

**GAIRA Decision:** core_training  
**Reason:** Large real human urine cohort with disease labels and reported 10,100 spectra. Reconstruction is required, but the biological signal is exactly the kind of patient biofluid data Global v2 should learn from.

### [FC12] Surface Enhanced Raman Spectroscopy and Machine Learning for Identification of Beta-Lactam Antibiotics Resistance Gene Fragment in Bacterial Plasmid

**Availability:**  
- A Raman payload exists, but it is much smaller and narrower than the title suggests.  
- The Zenodo release includes `2_Raman.zip`, `6_ML.zip`, AFM/XPS/UV-Vis/electrophoresis zips, and a readme.  
- The inspected Raman zip contains only one file, `Au-Capture.csv`, with a few repeated wavelength-intensity column pairs rather than a substantial labeled spectral archive.

**Biological Content:**  
- This is not organism-level pathogen spectroscopy. It is a capture-surface assay on digested bacterial plasmids / resistance-gene fragments.  
- The readme says spectra were measured on `Au-Capture`, a gold grating with immobilized ssDNA fragment, after enzymatic digestion and capture.  
- That makes it much closer to a nucleic-acid detection assay than to a bacterial/pathogen phenotype dataset.

**Labels:**  
- The scientific claim involves plasmids containing or lacking the resistance fragment and various complex background biomolecules.  
- The released Raman matrix does not expose a clean, rich sample-level label table.  
- In practice the released labels are too weak for biological representation learning.

**Quality:**  
- The release is heavily assay- and substrate-dominated.  
- Most of the deposited files are characterization or validation sidecars, not reusable spectra.  
- The tiny Raman payload is not enough to justify ingest for Global v2.

**Exclusions:**  
- Exclude UV-Vis, electrophoresis, AFM, XPS, and ML outputs.  
- Exclude all assay-support characterization.  
- Exclude `Au-Capture.csv` from the training corpus unless a future narrow DNA-capture support lane is explicitly needed.

**Paper Context:**  
- The experiment aims to detect beta-lactam resistance gene fragments in plasmids using a functionalized SERS capture substrate plus ML.  
- That aligns with the files: this is a targeted surface-capture assay, not a real pathogen spectral panel.  
- The release does not substantiate a strong organism-level pathogen ingest target.

**GAIRA Decision:** reject  
**Reason:** The deposited Raman payload is too small and too assay-specific, and it is not organism-level pathogen data. This will not materially improve Global v2.

### [FC14] Specificity and strain-typing capabilities of Nanorod Array-Surface Enhanced Raman Spectroscopy for Mycoplasma pneumoniae detection

**Availability:**  
- The core spectra are real and directly available as one large CSV, `NA-SERS specificity spectra.csv` (~10.5 MB).  
- The stats-only figshare companions are separate and should not be confused with the actual spectra release.  
- The Zenodo metadata explicitly says the CSV contains 521 spectra, and the CSV has 522 columns: one wavenumber column plus 521 spectral columns.

**Biological Content:**  
- Human respiratory pathogen–relevant SERS panel centered on *Mycoplasma pneumoniae*.  
- The CSV contains true pathogen classes rather than only one target plus blanks: headers include `M129`, `FH`, many clinical-isolate codes, `U. urealyticum`, `Bkg`, and `Media Ctl`.  
- This is serious pathogen content, though still NA-SERS and therefore platform-shaped.

**Labels:**  
- Species / strain identity is encoded directly in the CSV headers.  
- There are background and media-control columns, reference strains, and multiple clinical isolate groups.  
- These labels are useful for learning pathogen-level biochemical structure and discrimination.

**Quality:**  
- This is cleaner than most pathogen datasets because the core spectra are already consolidated in one matrix.  
- Technical confounds remain: nanorod-array SERS and throat-swab / culture-prep effects will shape the signal.  
- Still, the dataset has real class diversity and enough structure to be operationally useful.

**Exclusions:**  
- Exclude the two figshare performance/statistics tables from ingest; they are method support only.  
- Keep `Bkg` and `Media Ctl` outside the primary pathogen training set, or at least flag them separately.  
- Exclude any duplicated analytical summaries derived from the CSV.

**Paper Context:**  
- The paper claims NA-SERS can detect *M. pneumoniae*, distinguish it from 12 other human commensal/pathogenic mycoplasma species, and strain-type clinical isolates.  
- The released CSV is consistent with that claim because it contains many labeled spectral groups, not just a single binary panel.  
- The figshare companions are only model-performance context and are not needed for ingest.

**GAIRA Decision:** core_training  
**Reason:** This is a real pathogen panel with 521 labeled spectra across species, strain, and control groups. It is one of the strongest pathogen candidates in the list.

### [FC17] Spectroscopic investigation of faeces with surface-enhanced Raman scattering: a case study with coeliac patients on gluten-free diet

**Availability:**  
- Real raw spectra are available. The zip contains original per-spectrum ASCII `.txt` files, mirrored `RData` objects, OTU tables, and R code.  
- There are 27 faecal `.txt` spectra plus 3 pure-metabolite reference spectra (`bilirubin`, `hypoxanthine`, `xanthine`).  
- This looks like one spectrum per patient sample, not a dense replicate-heavy acquisition.

**Biological Content:**  
- Human faecal SERS from a coeliac-disease case study.  
- The cohort spans controls (`CTR`), active coeliac disease (`CD`), and gluten-free diet (`GFD`).  
- This is real biological mixture data and adds a domain not covered by typical serum/plasma archives.

**Labels:**  
- Disease/state labels are directly encoded in filenames: 8 `CTR`, 9 `CD`, and 10 `GFD`.  
- Filenames also encode sex and age-like metadata tokens.  
- The three-class structure is useful for biological-state learning, though the cohort is small.

**Quality:**  
- The release is unusually clean: original TXT spectra, mirrored R objects, OTU sidecars, and code are all separated.  
- The main limitation is scale: 27 patient-level spectra is small for a cohort dataset.  
- SERS substrate effects are still present, but this is clearly real stool chemistry, not an analytical toy.

**Exclusions:**  
- Exclude the pure metabolite reference spectra from the core cohort ingest and keep them only as support if needed.  
- Exclude OTU tables, `RData` duplicates, and `R_code.R` from training data ingest.  
- Keep only the 27 faecal TXT files for the biological cohort lane.

**Paper Context:**  
- The study asks whether faecal SERS can capture differences among controls, coeliac patients, and gluten-free-diet patients, alongside microbiome/OTU context.  
- The archive aligns with that: raw faecal spectra are present, and the OTU tables are clearly orthogonal sidecars.  
- It is a valid cohort, just not a large one.

**GAIRA Decision:** augmentation_only  
**Reason:** Real human faecal biology with meaningful three-state labels, but the cohort is too small to anchor Global v2. It is still a worthwhile diversity add-on.

### [FC23] Raman spectroscopic techniques to detect ovarian cancer biomarkers in blood plasma

**Availability:**  
- This is a real spectra release with two modality archives: `Raman dataset.zip` and `SERS dataset.zip`.  
- Both archives contain per-spectrum `.txt` files organized by class folders.  
- The inspected counts are 385 Raman txt files and 385 SERS txt files, split into 196 healthy and 189 ovarian-cancer spectra in each modality. The naming pattern suggests about 55 plasma donors total with repeated measurements.

**Biological Content:**  
- Human blood plasma from ovarian-cancer patients and healthy individuals.  
- This is real patient biofluid data and not just a calibration or biomarker standard set.  
- It is one of the clearest disease-plasma cohort datasets in the candidate pool.

**Labels:**  
- Disease class is explicit via folder names: `Healthy Individuals` and `Ovarian Cancer`.  
- Modality is also explicit: spontaneous Raman vs SERS.  
- Those labels are directly useful for both disease representation learning and cross-domain encoder training.

**Quality:**  
- The archive structure is strong: class-separated raw `.txt` spectra at scale.  
- The main confound is modality mixing, because Raman and SERS should not be blended without a modality flag.  
- Otherwise this is one of the cleaner real-biofluid datasets available.

**Exclusions:**  
- Exclude any duplicated preprocessed copies if they surface during deeper unpacking; the figshare description says raw and preprocessed spectra are present, though the main listings show txt-based spectral folders.  
- Do not collapse spontaneous Raman and SERS into one unlabeled pool.  
- Exclude any non-plasma sidecars if they appear in deeper inspection.

**Paper Context:**  
- The experiment is blood-plasma Raman/SERS analysis for ovarian-cancer biomarker detection.  
- What matters for ingest is that the release includes large numbers of per-spectrum plasma files for both modalities, with class labels already organized.  
- This aligns very well with the paper’s disease-cohort framing.

**GAIRA Decision:** core_training  
**Reason:** Real patient plasma cohort, good scale, clear disease labels, and both Raman and SERS modalities. This is a high-priority Global v2 training asset.

### [FC24] ACS platelet workbook family

**Availability:**  
- Both figshare records are genuine source-data workbooks, not PDFs.  
- Each workbook has the same main spectral sheet, `SERS spectra from ACS samples`, with dimensions about 284 rows × 424 columns, plus an empty `Sheet1`.  
- The main sheet contains wavenumber plus many per-sample replicate columns for `Sample 1` through `Sample 44`-style blocks. The two files are not byte-identical but look heavily overlapping.

**Biological Content:**  
- Human acute-coronary-syndrome clinical samples, with one record explicitly referencing platelets.  
- This is real clinical material, not a synthetic assay, but the exact cohort manifest and class structure are weakly documented in the deposited files.  
- It is likely platelet SERS from ACS patients rather than a balanced case-control archive.

**Labels:**  
- Sample IDs are explicit in the workbook.  
- Disease context is ACS, but per-sample clinical metadata are not cleanly exposed in the spectral sheet itself.  
- The labels are usable only after reconstruction and external mapping.

**Quality:**  
- The spectral matrices are real and dense, but the release is workbook-centric and not analysis-ready.  
- Metadata reconstruction will be needed, and the two workbooks likely contain partially redundant exports of the same study family.  
- This is usable, but not clean.

**Exclusions:**  
- Exclude the empty `Sheet1` tabs.  
- Exclude duplicated sample blocks after comparing the two workbooks.  
- Exclude any derived average/summary rows if they are mixed into the sheet.

**Paper Context:**  
- The practical signal here is that these files are not just figure tables: they are large spectral source matrices from an ACS study.  
- What matters for ingest is not the paper narrative but the need to reconstruct sample-level spectral blocks and resolve overlap between the 43-patient and 40-sample workbook versions.  
- This is realistic to reconstruct, but it is not an immediate ingest.

**GAIRA Decision:** augmentation_only  
**Reason:** Real human cardiovascular spectra are present, but the workbooks are messy, metadata-poor, and likely overlapping. Worth reconstructing, not a first-line core asset.

### [FC25] Surface-enhanced and tip-enhanced Raman scattering in label-free characterization of erythrocyte membranes and extracellular vesicles

**Availability:**  
- Real data exist, but they are split across multiple formats.  
- `sers.zip` contains 52 plain-text spectra, all readily ingestable and named as `Au60nm-RBCs ghosts_633 (...) .txt`.  
- `Standard spectra.zip` contains 10 pure-reference txt files, `ters.zip` contains a single `OPJU` Origin project, and `REVs size - AFM.xlsx` is an AFM size sidecar.

**Biological Content:**  
- Real biological material is involved: erythrocyte membrane ghosts and extracellular-vesicle-related nanoscale characterization.  
- The directly readable SERS payload is RBC membrane ghost spectra.  
- The EV side appears to be present mainly in the TERS/AFM sidecars rather than as an equally clean text-matrix release.

**Labels:**  
- Labels are weak: mostly material/modality rather than disease or cohort classes.  
- This is better suited to domain grounding than to supervised biological-state learning.  
- There is little patient/sample metadata in the released filenames.

**Quality:**  
- The text SERS files are clean and easy to ingest.  
- The strong confound is specialization: this is nanoscale, platform-driven SERS/TERS with specialized preparation, and the EV portion is less directly accessible because it sits in an Origin project.  
- It is useful, but easy to overrate.

**Exclusions:**  
- Exclude the 10 pure-standard spectra from the core biological lane.  
- Exclude `REVs size - AFM.xlsx` from spectral ingest.  
- Defer the `OPJU` TERS project unless converted cleanly; do not treat it as immediately ingestable raw spectra.

**Paper Context:**  
- The study is about label-free nanoscale characterization of erythrocyte membranes and extracellular vesicles with SERS and TERS.  
- The release matches that claim, but only part of it is in directly reusable txt form.  
- The most operationally usable files are the RBC-ghost SERS txt files; the EV/TERS part needs extra tool-specific work.

**GAIRA Decision:** grounding_only  
**Reason:** Real membrane/EV biophysical spectra are present, but the release is small, specialized, and only partly accessible as plain text. Good support data, not a central Global v2 training block.

### [FC32] UCLA saliva sEV gastric-cancer shard family

**Availability:**  
- Real spectra are available, but they are fragmented across many figshare records.  
- The 18 actual patient/control shard records contain at least 2,231 `.txt` spectra in total. One separate `ERCC` record has no downloadable files and is only a metadata stub.  
- Each shard is a single patient/control with tens to hundreds of spectra; reconstruction requires merging all shard-level releases into one cohort.

**Biological Content:**  
- Human saliva small-EV spectra from a gastric-cancer study.  
- This is real patient EV biology and one of the more biologically attractive cohort families in the list.  
- It is not an assay toy, but it is operationally difficult because the cohort is fragmented.

**Labels:**  
- Disease class and patient ID are explicit in the record titles.  
- The provided candidate family includes one healthy control shard and many gastric-cancer patient shards.  
- These are biologically meaningful labels, but the shard-level packaging creates patient imbalance and metadata assembly overhead.

**Quality:**  
- The spectra appear real and plentiful.  
- The main quality problem is not signal quality but packaging: no central cohort manifest, shard-by-shard downloads, likely inconsistent filenames, and potential missing controls beyond the provided list.  
- This is reconstructable, but it is labor-intensive.

**Exclusions:**  
- Exclude the `ERCC` metadata-only stub; it has no files.  
- Exclude any duplicate shard contents after merge.  
- Exclude malformed or non-spectral txt files if encountered during consolidation.

**Paper Context:**  
- The practical point is that this is a saliva sEV gastric-cancer cohort deposited one patient at a time.  
- The biologically valuable part is the shard collection as a whole, not any single figshare record.  
- The archive family is aligned with a real cohort, but only after reconstruction.

**GAIRA Decision:** augmentation_only  
**Reason:** High biological value and large spectrum count, but the fragmented deposition and incomplete manifest keep it out of the immediate core lane. It is a strong reconstruction target.

**Batch B decision summary**
- core_training: FC08, FC14, FC23
- augmentation_only: FC17, FC24, FC32
- grounding_only: FC25
- reject: FC12

**Top priorities to move forward with immediately from Batch B**
- FC23 for direct plasma disease-cohort ingestion.
- FC14 for pathogen-core ingestion from the already structured CSV.
- FC08 for targeted extraction of urine spectra from the newer split Zenodo release.
- FC32 as the strongest reconstruction project once shard consolidation bandwidth is available.
- FC17 as a small but clean faecal diversity add-on after the core items above.
"""


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    write_csv(OUT / "tables/candidate_review_master_batch_b.csv", ROWS)
    (OUT / "report").mkdir(parents=True, exist_ok=True)
    (OUT / "report/global_v2_forensic_candidate_review_batch_b.md").write_text(REPORT)


if __name__ == "__main__":
    main()
