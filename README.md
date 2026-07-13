# MPhil Project - Reproducing Patch-Based Diffusion Models for Solving Inverse Problems in Computed Tomography Imaging

This repository combines the different git repositories either created or modified as part of this project. 
They have all been merged here with their commit histories maintained so my contributions can be easily traced.
The report and executive summary can be found in the [`report`](report) directory.

## Repository Structure

- [`report/`](report-source/output/pdf) - Contains the final report and executive summary.
- [`LION/`](LION) - The main repository within which my reproduction and extensions are documented. Full details on how to reproduce my experiments are documented locally [here](LION/scripts/paper_scripts/PaDIS-Reproduction/README.md), and as part of the new online documentation on [readthedocs](https://lion-padis-project.readthedocs.io/en/feature-padis_implementation/index.html). My contributions to this repository are detailed in a more fine-grained manner below.
- [`PaDIS-Lion-Compat/`](PaDIS-Lion-Compat) - A forked and modified version of the original repository developed by Hu et. al. to accompany their PaDIS paper. Modifications allow their codebase to now accept the LION model checkpoints and LIDC-IDRI geometry, whilst also re-enabling some of their deactivated methods to enable comparison. This was used to check consistency against the new public-compatible LION PaDIS implementations.
- [`report-source/`](report-source) - The source repository used to write and generate the report and executive summary.
- [`scripts/`](scripts) - Helper scripts to automate the process of merging these repositories into this format from their github sources whilst retaining commit history.
- `CT_Basics.md` - A more detailed summary of the computed tomography background, including areas which were presented as part of the Medical Imaging minor course.

## Primary Contributions to the LION Repository

- Increased the robustness of [conda environments](LION/README.md) across CUDA versions, and improved [package portability](/LION/LION/utils/paths.py) to enable easy use outside of DAMTP.
- Developed a script to enable reproducibile and fast downloads of the LIDC-IDRI dataset [here](LION/LION/data_loaders/LIDC_IDRI/README.md), whilst also improving the robustness of its post-processing [here](LION/LION/data_loaders/LIDC_IDRI/pre_process_lidc_idri.py).
- Implemented the NCSNN++ denoising model by Song et. al [here](LION/LION/models/diffusion/NCSNpp.py).
- Implemented PaDIS score and training loss calculations [here](LION/LION/losses/PaDIS.py).
- Added support for diffusion model training [here](LION/LION/optimizers/PaDISSolver.py).
- Implemented [DPS, Langevin, predictor-corrector, VE-DDNM, patch averaging, patch stitching and PaDIS-DPS](LION/LION/reconstructors/PaDIS.py) reconstruction methods.
- Introduced [three implementations of the original PaDIS paper](LION/LION/reconstructors/PaDIS.py) - one matching the paper description, one matching the form presented in their [original accompanying repository](PaDIS-Lion-Compat/README.md), and one which introduces the use of LION's [power iteration method](LION/LION/utils/math.py) for calculating Lipschitz constant terms for greater stability across geometries.
- Integrated the ability to perform experiments using [google colab](LION/scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/PaDIS_Colab_manual_reconstruction.ipynb), [google cloud compute](LION/scripts/paper_scripts/PaDIS-Reproduction/platforms/gcp/run_PaDIS_GCP_spot_training.sh), or [slurm-enabled HPC settings](LION/scripts/paper_scripts/PaDIS-Reproduction/platforms/slurm/submit_PaDIS_A100_pipeline.sh).
- Created a full pipeline [here](LION/scripts/paper_scripts/PaDIS-Reproduction/pipeline) which reproduces all experiments, figures and tables listed in the [report](report/report.pdf).
- Added [robust tests](LION/tests) for all of these functionalities, automating their execution with [github workflows](LION/.github/workflows).
- Produced the [first documentation for LION](LION/docs), covering all aspects of the package used as part of this work. This documentation was then published on readthedocs, and can be accessed [here](https://lion-padis-project.readthedocs.io/en/feature-padis_implementation/index.html).
- Created thorough README files covering the usage of the [LIDC-IDRI dataset downloader](LION/LION/data_loaders/LIDC_IDRI/README.md) and [PaDIS reproduction pipeline](LION/scripts/paper_scripts/PaDIS-Reproduction/README.md) discussed above. 

## Repository Helper Scripts

The [helper script `subtrees.sh`](scripts) allows files to be edited and committed within this repository, and then pushed back to their original github repositories. The same is also true in reverse. These can be used as follows:

```sh
# Show the configured subtrees and their latest local commits.
./scripts/subtrees.sh status

# Pull one component, or all components.
./scripts/subtrees.sh pull LION
./scripts/subtrees.sh pull all

# Push committed changes back to one component's configured upstream branch.
./scripts/subtrees.sh push LION
```

Accepted component names are `report-source`, `LION`, and
`PaDIS-Lion-Compat`.

## LLM Usage Declaration
OpenAI Codex models 5.4, 5.5, and 5.6 were used iteratively in a prompt-check-improve manner to improve code implementations, format figures and tables, draft docs and READMEs, format docstrings, and perform code refactorings.
