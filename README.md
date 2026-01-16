# Dynamics-aware Evolutionary Profiling Uncouples Structural Rigidity from Functional Motion to Enable Enhanced Variant Interpretation

The identification of dynamic-conserved residues via the Dynamics-aware Evolutionary Profiling may fundamentally expand the boundaries of the druggable proteome and improve the resolution of genetic and evolutionary data.

This repository contains the source code and datasets for the large-scale statistical analysis presented in the manuscript. It encompasses the processing, scoring, and benchmarking of across two distinct cohorts.

We also provide an open-access web interface, [_ADEPT_ (Automated Dynamics-aware Evolutionary Profiling Tool)](https://www.karagolresearch.com/adept), for researchers to apply this framework to their own proteins. [![Web Server](https://img.shields.io/badge/Web%20Server-ADEPT-blue)]((https://www.karagolresearch.com/adept)) [![Mirror Link](https://img.shields.io/badge/Mirror%20Link-ADEPT-orange)]((https://www.tanerkaragol.com/dynamics-aware-evolution)) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Citation
If you use this framework in your research, please cite:

* Karagöl, T., & Karagöl, A. (2026). Dynamics-aware Evolutionary Profiling Uncouples Structural Rigidity from Functional Motion to Enable Enhanced Variant Interpretation.

## Usage

* **`codes/`**: Contains all Python and R scripts  required to reproduce the statistical analysis and figure generation.
* ! Note: Due to the high volume of input data, the runtime of the code may be lengthy. For faster results, we recommend using a high-performance environment.
* **`data-alpha-helical/`**: Analysis containing analysis profiles for the cross-species cohort of 93 alpha-helical proteins.
* **`data-human/`**: Analysis containing the targeted cohort of 58 human proteins (400–600 residues).
* You can also check our open-access web interface (_ADEPT_). https://www.karagolresearch.com/adept or https://www.tanerkaragol.com/dynamics-aware-evolution

## Abstract

Evolutionary conservation is a powerful part of mutational intolerance prediction, yet traditional pathogenicity metrics frequently conflate two distinct biophysical constraints: structural stability (rigidity) and functional mechanics (dynamics). We introduce Dynamics-Aware Evolutionary Profiling to resolve this ambiguity, integrating Molecular Dynamics with evolutionary conservation and coupling analysis across human/cross-species-proteome of 151 protein structures. By mathematically uncoupling biophysical forces, we define orthogonal metrics; the Rigid Conserved Score (RCS) for the structural scaffold, and the Dynamic Conserved Score (DCS) for flexible residues. Our analysis reveals a fundamental bifurcation in pathogenicity. RCS serves as a filter for lethal structural failure, isolating hydrophobic core residues whose mutation triggers unfolding. In contrast, DCS identified a rare population of residues that are evolutionarily highly-conserved but structurally mobile; these Dynamic-Conserved sites exhibit intermediate pathogenicity and are enriched in flexible hinge residues (Gly, Pro). Validation against 737 human variants from ClinVar demonstrates that DCS captures a distinct pathogenic mechanism regarding essential protein motion. Notably, DCS and RCS correctly flagged some pathogenic variants of NARS1 and PGK1 that were misclassified as benign or ambiguous by AlphaMissense. These results indicate that while the rigid core represents a stability bottleneck, DCS isolates functional sites likely driving allosteric regulation. We provide an open-access web interface (ADEPT) for these metrics. By isolating dynamic-conserved residues, this framework refines the interpretation of Variants of Uncertain Significance in dynamic regions and reveals tunable targets for rational drug design, moving beyond the static optimization of the folded state.

Keywords: Allosteric Regulation, Dynamic Conservation, Structure-Function Relationship, Rational Drug Design, Genetics, Variant analysis
