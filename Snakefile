from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import os
import subprocess
import pickle
import re

sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/himut/src")
sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/hapfusion/src")

import hapfusion
import hapfusion.bamlib

import himut
import himut.bamlib
import himut.phaselib

from src import liftover, annotate, diagnostics, dashboard, inference

aut_chrom_names = [f"chr{i}" for i in list(range(1, 23))]
chrom_names = aut_chrom_names + ["chrX", "chrY"]

aut_chrom_names_ragtag = [x + "_RagTag" for x in aut_chrom_names]
chrom_names_ragtag = [x + "_RagTag" for x in chrom_names]

# Binaries
bedtools_path = "/nfs/users/nfs_r/rs42/rs42/software/bedtools"
minimap2_path = "/nfs/users/nfs_r/rs42/rs42/software/minimap2-2.26_x64-linux/minimap2"
minidot_path = "/nfs/users/nfs_r/rs42/rs42/git/miniasm/minidot"
samtools_path = "/nfs/users/nfs_r/rs42/rs42/software/samtools-1.18/samtools"
bcftools_path = "/nfs/users/nfs_r/rs42/rs42/software/bcftools-1.18/bcftools"
trf_path = "/nfs/users/nfs_r/rs42/rs42/software/trf409.linux64"
ragtag_path = "ragtag.py"
deepvariant_command = "/software/singularity-v3.9.0/bin/singularity exec -B /lustre /lustre/scratch126/casm/team154pc/sl17/01.himut/02.results/02.germline_mutations/deepvariant.simg"
sdust_path = "/nfs/users/nfs_r/rs42/rs42/git/sdust/sdust"

# Data paths
root_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm")
scaffolds_path = root_path / "02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds"
hap_scaffolds_path = root_path / "01.data/05.ragtag/03.haplotype_specific_scaffolds"

# Hapfusion output paths
t2t_hapfusion_output_path = root_path / "02.results/01.read_alignment/01.ccs/03.T2T-CHM13"
hg19_hapfusion_output_path = root_path / "02.results/01.read_alignment/01.ccs/01.grch37"

# Samples to do
sample_ids = [
    "PD50477f",
    # "PD50508bf", -- ignore; merged two sampling dates just for phasing, but should be analyzed separately
    "PD50519d",
    # "PD47269d", -- don't use, not there
    "PD50508f",
    # "PD50511e", -- don't use, likely mixture
    "PD50523b",
    # "PD48473b", -- don't use, not there
    "PD50521b",
    "PD50508b",
    # "PD50521be", -- ignore; merged two sampling dates just for phasing, but should be analyzed separately
    "PD46180c",
    # "PD50502f", -- don't use, likely mixture
    "PD50521e",
    # "PD50511e_SS",  --- don't use
    "PD50489e",
]

sample_id_to_joint_id = {
    "PD50477f": "PD50477f",
    "PD50519d": "PD50519d",
    "PD50508f": "PD50508bf",
    "PD50523b": "PD50523b",
    "PD50521b": "PD50521be",
    "PD50508b": "PD50508bf",
    "PD46180c": "PD46180c",
    "PD50521e": "PD50521be",
    "PD50489e": "PD50489e",
}


# ------------------------------------------------------------------------------------------------------------------------
# Import other rules
#
include: "snakefiles/read_analysis.snk"
include: "snakefiles/tandem_repeats.snk"
#include: "snakefiles/nahr.snk"
include: "snakefiles/inference.snk"
include: "snakefiles/hotspots.snk"

# ------------------------------------------------------------------------------------------------------------------------
# Mapping to haplotypes
#

def denovo_hap_func(wildcards):
    joint_id = sample_id_to_joint_id[wildcards.focal_sample_id]
    denovo_reference = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/04.hifiasm/02.hifiasm_0.19.5-r592/") \
            / f"{joint_id}" / f"{joint_id}.asm.bp.hap{wildcards.haplotype}.p_ctg.fasta"
    return denovo_reference


rule gfa_to_fasta:
    input:
        gfa = "/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/04.hifiasm/02.hifiasm_0.19.5-r592/{something}/{prefix}.gfa",
    output:
        fasta = "/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/04.hifiasm/02.hifiasm_0.19.5-r592/{something}/{prefix}.fasta",
    run:
        shell(
            "awk '/^S/{{print \">\"$2;print $3}}' {input.gfa} > {output.fasta}"
        )

rule scaffold_haplotypes:
    input:
        query_fasta = denovo_hap_func,
        reference_fasta = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/03.t2t-chm13/chm13v2.0.fasta"),
    output:
        agp = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.agp",
        fasta = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.fasta",
        expanded_fasta = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.expanded.fasta",
        expanded_fai = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.expanded.fasta.fai",
    threads: 16
    resources:
        mem_mb=50000,
    run:
        # To infer gaps: " -r -g 2 -m 100000000 "
        output_directory = Path(output.fasta).parent
        shell(
            "{ragtag_path} scaffold {input.reference_fasta} {input.query_fasta}"
            " -o {output_directory} "
            " -u -w "            
            " --aligner {minimap2_path} "
            " -t {threads} "
        )
        with open(output.fasta, 'r') as infile:
            with open(output.expanded_fasta, 'w') as outfile:
                outfile.write(re.sub(r'N{100}', 'N'*30000, infile.read()))

        shell(
            "{samtools_path} faidx {output.expanded_fasta}"
        )

rule scaffold_haplotypes_final:
    input:
        fasta = [str(hap_scaffolds_path \
            / f"{focal_sample_id}" / f"haplotype_{haplotype}" / "ragtag.scaffold.expanded.fasta") \
            for focal_sample_id in ["PD50489e"] \
            for haplotype in [1,2]]            

# rule merge_and_index_fastq:
#     input:
#         gz_dir = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
#             / "{focal_sample_id}",
#     output:
#         fastq_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.ccs.filtered.fastq.gz",
#         fastq_gz_fxi = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.ccs.filtered.fastq.gz.fxi",
#     resources:
#         mem_mb=16000,
#     run:
#         shell("cat {input.gz_dir}/m*.ccs.filtered.fastq.gz > {output.fastq_gz}")
#         shell("rm -f {output.fastq_gz_fxi}")
#         shell("pyfastx index {output.fastq_gz}")

# rule merge_and_index_fastq_final:
#     input:
#         [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
#             / f"{focal_sample_id}" / f"{focal_sample_id}.ccs.filtered.fastq.gz") \ 
#             for focal_sample_id in sample_ids]


rule minimap2_to_haplotype:
    input:
        denovo_reference = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.expanded.fasta",
        fastq_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
            / "{focal_sample_id}" / "{focal_sample_id}.ccs.filtered.fastq.gz",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap{haplotype}.minimap2.sorted.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap{haplotype}.minimap2.sorted.bam.bai",
    threads: 32,
    resources:
        mem_mb=50000,
    run:
        shell(
            "{minimap2_path} "
            "-R \"@RG\\tID:{wildcards.focal_sample_id}\\tPL:PACBIO\\tSM:{wildcards.focal_sample_id}\\tPU:{wildcards.focal_sample_id}\\tPM:SEQUEL\" "
            "-t {threads} "
            "-ax map-hifi --cs=short --eqx --MD "
            "{input.denovo_reference} "
            "{input.fastq_gz} "
            "| {samtools_path} sort -o {output.bam}"
        )

        shell(
            "{samtools_path} index {output.bam}"
        )
        
rule filter_bam_for_primary:
    input:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap{haplotype}.minimap2.sorted.bam",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap{haplotype}.minimap2.sorted.primary_alignments.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap{haplotype}.minimap2.sorted.primary_alignments.bam.bai",
    threads: 16
    run:
        # 0x900 = SUPPLEMENTARY | SECONDARY (https://www.htslib.org/doc/samtools-flags.html)
        shell(
            "{samtools_path} view -@ {threads} -bh -F 0x900 {input.bam} > {output.bam}"
        )

        shell(
            "{samtools_path} index -@ {threads} {output.bam}"
        )


rule minimap2_to_haplotype_final:
    input:
        [Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / f"{focal_sample_id}" / f"{focal_sample_id}.hap{haplotype}.minimap2.sorted.primary_alignments.bam" \
            for focal_sample_id in sample_ids \
            for haplotype in [1,2]]






rule minimap2_to_haplotype_unscaffolded:
    input:
        reference_fasta = denovo_hap_func,
        fastq_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
            / "{focal_sample_id}" / "{focal_sample_id}.ccs.filtered.fastq.gz",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.unscaffolded.hap{haplotype}.minimap2.sorted.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.unscaffolded.hap{haplotype}.minimap2.sorted.bam.bai",
    threads: 32,
    resources:
        mem_mb=50000,
    run:
        shell(
            "{minimap2_path} "
            "-R \"@RG\\tID:{wildcards.focal_sample_id}\\tPL:PACBIO\\tSM:{wildcards.focal_sample_id}\\tPU:{wildcards.focal_sample_id}\\tPM:SEQUEL\" "
            "-t {threads} "
            "-ax map-hifi --cs=short --eqx --MD "
            "{input.reference_fasta} "
            "{input.fastq_gz} "
            "| {samtools_path} sort -o {output.bam}"
        )

        shell(
            "{samtools_path} index {output.bam}"
        )
        
rule filter_bam_for_primary_unscaffolded:
    input:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.unscaffolded.hap{haplotype}.minimap2.sorted.bam",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.unscaffolded.hap{haplotype}.minimap2.sorted.primary_alignments.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.unscaffolded.hap{haplotype}.minimap2.sorted.primary_alignments.bam.bai",
    threads: 16
    run:
        # 0x900 = SUPPLEMENTARY | SECONDARY (https://www.htslib.org/doc/samtools-flags.html)
        shell(
            "{samtools_path} view -@ {threads} -bh -F 0x900 {input.bam} > {output.bam}"
        )

        shell(
            "{samtools_path} index -@ {threads} {output.bam}"
        )


rule minimap2_to_haplotype_unscaffolded_final:
    input:
        [Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / f"{focal_sample_id}" / f"{focal_sample_id}.unscaffolded.hap{haplotype}.minimap2.sorted.primary_alignments.bam" \
            for focal_sample_id in sample_ids \
            for haplotype in [1,2]]


rule minimap2_to_T2T:
    input:
        reference = "/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/03.t2t-chm13/chm13v2.0.fasta",
        fastq_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
            / "{focal_sample_id}" / "{focal_sample_id}.ccs.filtered.fastq.gz",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam.bai",
    threads: 32,
    resources:
        mem_mb=50000,
    run:
        shell(
            "{minimap2_path} "
            "-R \"@RG\\tID:{wildcards.focal_sample_id}\\tPL:PACBIO\\tSM:{wildcards.focal_sample_id}\\tPU:{wildcards.focal_sample_id}\\tPM:SEQUEL\" "
            "-t {threads} "
            "-ax map-hifi --cs=short --eqx --MD "
            "{input.reference} "
            "{input.fastq_gz} "
            "| {samtools_path} sort -o {output.bam}"
        )

        shell(
            "{samtools_path} index {output.bam}"
        )
        
rule filter_bam_for_primary_t2t:
    input:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.primary_alignments.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.primary_alignments.bam.bai",
    threads: 16
    run:
        # 0x900 = SUPPLEMENTARY | SECONDARY (https://www.htslib.org/doc/samtools-flags.html)
        shell(
            "{samtools_path} view -@ {threads} -bh -F 0x900 {input.bam} > {output.bam}"
        )

        shell(
            "{samtools_path} index -@ {threads} {output.bam}"
        )

rule minimap2_to_T2T_final:
    input:
        [Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13/") \
            / f"{focal_sample_id}" / f"{focal_sample_id}.minimap2.sorted.primary_alignments.bam" \
            for focal_sample_id in sample_ids]



rule minimap2_to_grch37:
    input:
        reference = "/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/01.grch37/human_g1k_v37_decoy.fasta",
        fastq_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
            / "{focal_sample_id}" / "{focal_sample_id}.ccs.filtered.fastq.gz",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam.bai",
    threads: 32,
    resources:
        mem_mb=50000,
    run:
        shell(
            "{minimap2_path} "
            "-R \"@RG\\tID:{wildcards.focal_sample_id}\\tPL:PACBIO\\tSM:{wildcards.focal_sample_id}\\tPU:{wildcards.focal_sample_id}\\tPM:SEQUEL\" "
            "-t {threads} "
            "-ax map-hifi --cs=short --eqx --MD "
            "{input.reference} "
            "{input.fastq_gz} "
            "| {samtools_path} sort -o {output.bam}"
        )

        shell(
            "{samtools_path} index {output.bam}"
        )
        
rule filter_bam_for_primary_grch37:
    input:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.primary_alignments.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.primary_alignments.bam.bai",
    threads: 16
    run:
        # 0x900 = SUPPLEMENTARY | SECONDARY (https://www.htslib.org/doc/samtools-flags.html)
        shell(
            "{samtools_path} view -@ {threads} -bh -F 0x900 {input.bam} > {output.bam}"
        )

        shell(
            "{samtools_path} index -@ {threads} {output.bam}"
        )

rule minimap2_to_grch37_final:
    input:
        [Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37/") \
            / f"{focal_sample_id}" / f"{focal_sample_id}.minimap2.sorted.primary_alignments.bam" \
            for focal_sample_id in sample_ids]            



rule minimap2_to_grch38:
    input:
        reference = "/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/02.grch38/Homo_sapiens_assembly38.fasta",
        fastq_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/") \
            / "{focal_sample_id}" / "{focal_sample_id}.ccs.filtered.fastq.gz",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/02.grch38/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/02.grch38/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam.bai",
    threads: 32,
    resources:
        mem_mb=50000,
    run:
        shell(
            "{minimap2_path} "
            "-R \"@RG\\tID:{wildcards.focal_sample_id}\\tPL:PACBIO\\tSM:{wildcards.focal_sample_id}\\tPU:{wildcards.focal_sample_id}\\tPM:SEQUEL\" "
            "-t {threads} "
            "-ax map-hifi --cs=short --eqx --MD "
            "{input.reference} "
            "{input.fastq_gz} "
            "| {samtools_path} sort -o {output.bam}"
        )

        shell(
            "{samtools_path} index {output.bam}"
        )
        
rule filter_bam_for_primary_grch38:
    input:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/02.grch38/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.bam",
    output:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/02.grch38/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.primary_alignments.bam",
        bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/02.grch38/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.sorted.primary_alignments.bam.bai",
    threads: 16
    run:
        # 0x900 = SUPPLEMENTARY | SECONDARY (https://www.htslib.org/doc/samtools-flags.html)
        shell(
            "{samtools_path} view -@ {threads} -bh -F 0x900 {input.bam} > {output.bam}"
        )

        shell(
            "{samtools_path} index -@ {threads} {output.bam}"
        )

rule minimap2_to_grch38_final:
    input:
        [Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/02.grch38/") \
            / f"{focal_sample_id}" / f"{focal_sample_id}.minimap2.sorted.primary_alignments.bam" \
            for focal_sample_id in sample_ids]                        