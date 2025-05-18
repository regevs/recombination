# Code for "Insights into non-crossover recombination from long-read sperm sequencing"

This repository includes all code necessary to generate the data and analysis presented in the paper. 

## Data
The data was generated using a snakemake pipeline. To generate the data:
```
$ snakemake -f Snakefile_new annotate_all_reads_structure_final
```

To generate data needed for manual validation of reads:
```
$ snakemake -f Snakefile_new create_dashboard_bams_final
```

To call the PRDM9 alleles:
```
$ snakemake -f Snakefile_new find_prdm9_alleles_final
```

## Analysis
The final results are available at these key notebooks:

### For manual curation
```
notebooks/20241125_read_dashboard_new_pipeline.ipynb
```
### For parameter tuning:
```
notebooks/20240207_read_trimming.ipynb
notebooks/20240122_calibrate_bq.ipynb
notebooks/20240538_calibrate_bq-revio.ipynb
notebooks/20241115_calibrate_bq-CEPH-and-Revio.ipynb
```
### For PRDM9 calling
```
notebooks/20240424_PRDM9_analysis.ipynb
```
### For figures and statistical analysis
```
notebooks/20250217_CO_only_analysis_new.ipynb
notebooks/20250217_NCO_only_analysis_new.ipynb
notebooks/20250216_CO_vs_NCO_new.ipynb
notebooks/20241206_CO_vs_NCO_blood_CEPH.ipynb
notebooks/20250110_preprint_numbers_new.ipynb
notebooks/20250509_all_chroms_figure-new.ipynb
notebooks/20250407_abc_revisited.ipynb
notebooks/20250225_complex_new.ipynb
notebooks/20250317_reviewer_comments.ipynb
```
```
