# Code for "Insights into non-crossover recombination from long-read sperm sequencing"

This repository includes all code necessary to generate the data and analysis presented in the paper. 

## Data
The data was generated using a snakemake pipeline. To generate the data:
```
$ snakemake annotate_all_reads_structure_final
```

To genereate data needed for manual validation of reads:
```
$ snakemake create_dashboard_bams_final
```

To call the PRDM9 alleles:
```
$ snakemake find_prdm9_alleles_final
```

## Analysis
The final results are available at these key notebooks:
```
notebooks/20240113_read_dashboard.ipynb (for manual curation)
notebooks/20240424_CO_only_analysis.ipynb
notebooks/20240425_NCO_only_analysis.ipynb
notebooks/20240427_CO_vs_NCO.ipynb.ipynb
notebooks/20240424_PRDM9_analysis.ipynb.ipynb
notebooks/20240504_preprint_numbers.ipynb
notebooks/20240509_all_chroms_figure.ipynb
notebooks/20240509_mixture_likelihood_again.ipynb
notebooks/20240618_complex.ipynb
```
