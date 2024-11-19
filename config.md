# How to configure runs

## Terminology

The largest unit is **sample_set** which is a set of samples from the same individual (and thus with the same genome), and that will be used together for de-novo assembly.

An **assmebly** is the result of the de-novo assembly of a sample set. This could be done in the pipeline or obtained from another source.

Within a sample set, there are **sample**-s, which in principle correspond to a single physical sample. A single sample will include one or more flow cells.

Each **flow_cell** corresponds to a reads data file derived from a flow cell. This is expected to be a `fastq.gz` file after read filtering.

Each flow cell should be defined a **flow_cell_type**, which defines a subset of QC parameters calibrated to fit the particular technology version this flow cell was run with. For example, Sequel II, Revio, or Revio with unbinned base quality scores.

## Definition
This is all defined in a `tsv` file pointed to in the config file (see below). The format is:
```
sample_set  sample  flow_cell   flow_cell_type  path
```
Here is an example:
```
sample_set  sample  flow_cell   flow_cell_type  path
NA123       NA123a  m84046_230701_234987_s1 sequel_ii   /path/to/m84046_230701_234987_s1.ccs.filtered.fastq.gz
NA123       NA123a  m84046_230701_010431_s2 revio   /path/to/m84046_230701_010431_s2.ccs.filtered.fastq.gz
NA123       NA123b  m84046_230701_010431_s2 revio   /path/to/m84046_230701_010431_s2.ccs.filtered.fastq.gz
NA555       NA555  m84046_230711_223630_s1 revio   /path/to/m84046_230711_223630_s1.ccs.filtered.fastq.gz
```
This contains two sample sets; the first sample set contains two samples, one sample with both a Sequel II and a Revio flow cell data files, and one sample with only Revio; and another sample set containing a single sample, containing a single flow cell.

In case you want to NOT generate de-novo assemblies, you can specify an additional optional file pointing to them:
```
sample_set  fasta_wildcard_path
NA123   /path/to/assembly
```
where `fasta_wildcard_path` is a string with `{haplotype}` that would be replaced to obtain the path the fasta file of this particular haplotype for this sample set.

# config file options
Defined in `yaml` as per snakemake config files:

```yaml
# This point to the table described above
data_table_path: /path/to/table.tsv

# Optional sublist of sample sets to work on - can omit if it's all, or give manually in the command line
sample_sets: X,Y,Z

# Optional path to assembly paths
assembly_table_path: /path/to/table.tsv

# Where the output of this run will be
output_dir_path: /path/to/output

# QC parameters that are common to all flow cell types
qc_parameters:    
  high_confidence_slack: 10
  certainty_threshold: 0.95
  map_qual_min: 60
  total_mismatches_max: 100
  total_clipping_max: 10
  phased_coverage_min: 3
  allele_coverage_min: 3
  GC_tract_mean: 30
  read_margin_in_bp: 5000

# QC parameters per flow cell type
flow_cell_types:
  sequel_ii: 
    base_qual_min_detection: 60
    base_qual_min_classification: 30
    read_trimming_detection: 1500
    read_trimming_classification: 500
  revio: 
    base_qual_min_detection: 40
    base_qual_min_classification: 30
    read_trimming_detection: 400
    read_trimming_classification: 200

# Paths to various programs needed
tools:
  hifiasm_path: /nfs/users/nfs_r/rs42/rs42/software/hifiasm/hifiasm
  bedtools_path: /nfs/users/nfs_r/rs42/rs42/software/bedtools
  minimap2_path: /nfs/users/nfs_r/rs42/rs42/software/minimap2-2.26_x64-linux/minimap2
  samtools_path: /nfs/users/nfs_r/rs42/rs42/software/samtools-1.18/samtools
  bcftools_path: /nfs/users/nfs_r/rs42/rs42/software/bcftools-1.18/bcftools
  trf_path: /nfs/users/nfs_r/rs42/rs42/software/trf409.linux64
  ragtag_path: ragtag.py
  sdust_path: /nfs/users/nfs_r/rs42/rs42/git/sdust/sdust

# Paths to various files needed
files:
  t2t_reference_path: /path/to/chm13v2.0.fasta
  grch37_reference_path: /path/to/human_g1k_v37_decoy.fasta
  grch38_reference_path: /path/to/Homo_sapiens_assembly38.fasta
```

# Output
The output directory will contain the following structure:
```yaml
assemblies/             
    sample_set/
        haplotype_1.fasta
        haplotype_2.fasta
        ...
    ...
T2T_scaffolds/
    sample_set/
        haplotype_1/
            Ragtag files...
        haplotype_2/
            ...
        chrN/
            haplotype_1/
                TRF files...
                sdust files...
            haplotype_2/
                ...            
        ...
alignments/
    sample_set/
        sample/
            flow_cell/
                T2T_scaffolds/
                    haplotype_1/
                        ...
                    haplotype_2/
                        ...
                T2T_reference/
                    ...
                grch37_reference/
                    ...
                grch38_reference/
                    ...

read_analysis/
    sample_set/
        sample/
            flow_cell/
                chrN/
                    read_refinement.parquet
                    high_confidence_snps.parquet
                    annotated_high_confidence_snps.parquet                    
                    haplotype_1/
                        phase_and_haplotag files...
                    haplotype_2/
                        phase_and_haplotag files...
            genome_coverage/
                haplotype_1/
                    calculate_genome_coverage_on_haplotype files...
                haplotype_2/
                    calculate_genome_coverage_on_haplotype files...
            reads/
                chrN/
                    all_reads.parquet
                    candidate_reads.parquet
                    classified_reads.parquet
                    all_reads_structure_annotated.parquet
                    haplotype_1/
                        coverage_for_all_events (candidate_reads) files...
                        coverage_for_all_events (all_reads) files...
                    haplotype_2
                        coverage_for_all_events (candidate_reads) files...
                        coverage_for_all_events (all_reads) files...
                    plots/
                        dashboard files...

```

