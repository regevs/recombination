# Code for "Long-read sequencing reveals pre-meiotic gene conversion in sperm" (Nature, 2026)

Regev Schweiger, Sangjin Lee, Chenxi Zhou, Tsun-Po Yang, Stacy Li, Rashesh Sanghvi, Matthew Neville, Katie Smith, Kirsty Roberts, Ayrun Nessa, Sam Wadge, Kerrin S Small, Peter J Campbell, Kristian Almstrup, Peter H Sudmant, Raheleh Rahbari and Richard Durbin.

This repository contains the code, workflow definitions, reproduction notebooks, and generated outputs for the paper.

The repository is organized in two reproducibility stages:

1. The Snakemake workflow documents the processed-data generation pipeline.
2. The reproduction notebooks generate the final figures, tables, and exported result files from processed data.

## Repository layout

```text
configs/     Study configuration files used by the workflow
exports/     Exported result files
figures/     Final reproduced figure files
notebooks/   Final reproduction notebooks, ordered 00-07
snakefiles/  Snakemake rule files used by the workflow
src/         Python helper modules used by the workflow and notebooks
tables/      Final reproduced table files
```

## Reproduction notebooks

Run the notebooks in order:

```text
notebooks/00_samples_pipeline_and_tables.ipynb
notebooks/01_recombination_landscape_and_co_nco.ipynb
notebooks/02_co_source_and_dsb_features.ipynb
notebooks/03_gc_bias.ipynb
notebooks/04_decode_analysis.ipynb
notebooks/05_individual_variation.ipynb
notebooks/06_nco_tract_lengths_and_positioning.ipynb
notebooks/07_supplementary_qc_and_model_checks.ipynb
```

The notebooks write final outputs to:

```text
figures/
tables/
exports/
```

## Figure outputs

| Paper item | Output file | Reproduction notebook |
| --- | --- | --- |
| Fig. 1b | `figures/event_count_summary.pdf` | `notebooks/00_samples_pipeline_and_tables.ipynb` |
| Fig. 2a | `figures/all_chroms.pdf` | `notebooks/01_recombination_landscape_and_co_nco.ipynb` |
| Fig. 2a colorbar | `figures/all_chroms_colorbar.pdf` | `notebooks/01_recombination_landscape_and_co_nco.ipynb` |
| Fig. 2a legend | `figures/all_chroms_legend.pdf` | `notebooks/01_recombination_landscape_and_co_nco.ipynb` |
| Fig. 2b | `figures/sperm_co_nco_genetic_lengths.pdf` | `notebooks/01_recombination_landscape_and_co_nco.ipynb` |
| Fig. 2c | `figures/blood_nco_genetic_lengths.pdf` | `notebooks/01_recombination_landscape_and_co_nco.ipynb` |
| Fig. 2d | `figures/nco_converted_marker_count_fit.pdf` | `notebooks/06_nco_tract_lengths_and_positioning.ipynb` |
| Fig. 2e | `figures/decode_gc_bias.pdf` | `notebooks/04_decode_analysis.ipynb` |
| Fig. 2f | `figures/decode_paternal_events.pdf` | `notebooks/04_decode_analysis.ipynb` |
| Fig. 3a-b | `figures/co_genetic_length_sample_comparisons.pdf` | `notebooks/05_individual_variation.ipynb` |
| Fig. 3c-d | `figures/nco_genetic_length_sample_comparisons.pdf` | `notebooks/05_individual_variation.ipynb` |
| Supplementary Fig. 1 | `figures/sequel_read_end_mismatch_calibration.pdf` | `notebooks/07_supplementary_qc_and_model_checks.ipynb` |
| Supplementary Fig. 2 | `figures/revio_read_end_mismatch_calibration.pdf` | `notebooks/07_supplementary_qc_and_model_checks.ipynb` |
| Supplementary Fig. 3 | `figures/sequel_bq_calibration.pdf` | `notebooks/07_supplementary_qc_and_model_checks.ipynb` |
| Supplementary Fig. 4 | `figures/revio_bq_calibration.pdf` | `notebooks/07_supplementary_qc_and_model_checks.ipynb` |
| Supplementary Fig. 6 | `figures/co_nco_telomere_distance_distribution.pdf` | `notebooks/06_nco_tract_lengths_and_positioning.ipynb` |

## Workflow

The canonical workflow entry point is:

```text
Snakefile
```

The study configs are:

```text
configs/Rahbari.yaml
configs/Sudmant.yaml
configs/CEPH.yaml
```

The workflow can be used to regenerate the processed data consumed by the reproduction notebooks. For example, the following targets correspond to the main read annotation, manual-validation BAMs, and PRDM9 allele calling steps:

```bash
snakemake -s Snakefile --configfile configs/Rahbari.yaml annotate_all_reads_structure_final
snakemake -s Snakefile --configfile configs/Rahbari.yaml create_dashboard_bams_final
snakemake -s Snakefile --configfile configs/Rahbari.yaml find_prdm9_alleles_final
```

See `config.md` for configuration fields and output structure.
