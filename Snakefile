from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import os
import subprocess
import pickle

sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/himut/src")
sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/hapfusion/src")

import hapfusion
import hapfusion.bamlib

import himut
import himut.bamlib
import himut.phaselib

from src import liftover, annotate, diagnostics, dashboard

aut_chrom_names = [f"chr{i}" for i in list(range(1, 23))]
chrom_names = aut_chrom_names + ["chrX", "chrY"]
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


# ------------------------------------------------------------------------------------------------------------------------
# Depth calculations
#

rule t2t_depth:
    input:
        bam_path = str(t2t_hapfusion_output_path / "{sample_id}" / "chm13.{sample_id}.minimap2.primary_alignments.sorted.bam"),
    output:
        depth_path = str(t2t_hapfusion_output_path / "{sample_id}" / "chm13.{sample_id}.{chrom}.depth.txt.gz"),
    run:
        shell("{samtools_path} depth -r {wildcards.chrom} {input.bam_path} | gzip > {output.depth_path}")



rule t2t_depth_final:
    input:
        [
            str(t2t_hapfusion_output_path / f"{sample_id}" / f"chm13.{sample_id}.{chrom}.depth.txt.gz") \
            for sample_id in sample_ids \
            for chrom in chrom_names
        ],


# ------------------------------------------------------------------------------------------------------------------------
# Liftovers
#
hg19_genetic_map_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/04.genetic_maps/01.Bherer_etal_SexualDimorphismRecombination/Refined_EUR_genetic_map_b37")
t2t_lifted_genetic_map_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/04.genetic_maps/02.T2T_lifted_Bherer_etal_SexualDimorphismRecombination/Refined_EUR_genetic_map_b37")

liftover_binary_path = Path("/nfs/users/nfs_r/rs42/rs42/software/liftOver")
hg19_to_t2t_overchain_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/08.liftover/hg19-chm13v2.over.chain.gz")

rule genetic_map_liftovers:
    input:
        map_path = hg19_genetic_map_path / "male_{chrom}.txt",
    output:
        input_bed_path = t2t_lifted_genetic_map_path / "male_{chrom}_input.bed",
        output_bed_path = t2t_lifted_genetic_map_path / "male_{chrom}_output.bed",
        output_unmapped_path = t2t_lifted_genetic_map_path / "male_{chrom}_unmapped",
        output_map_path = t2t_lifted_genetic_map_path / "male_{chrom}_genetic_map.txt",
    run:
        # Prepare input bed file for liftover
        genetic_map = pd.read_csv(input.map_path, delim_whitespace=True)  
        (genetic_map            
            .assign(start_pos = np.concatenate([[0], genetic_map.pos.values[:-1]]))
            .assign(end_pos = genetic_map.pos.values)
            [["chr", "start_pos", "end_pos", "start_pos", "end_pos"]]
            .to_csv(output.input_bed_path, sep="\t", index=False, header=False)
        )

        # Liftover
        shell(
            f"{liftover_binary_path} "
            f"{output.input_bed_path} "
            f"{hg19_to_t2t_overchain_file} "
            f"{output.output_bed_path} "
            f"{output.output_unmapped_path} "
            f"-minMatch=0.1 "
        )

        # Create the updated genetic maps
        liftover.merge_genetic_map_and_liftover(
            input.map_path,
            output.output_bed_path,
            output.output_map_path,
        )


rule genetic_map_liftovers_final:
    input:
        [str(t2t_lifted_genetic_map_path / f"male_{chrom}_genetic_map.txt") \
            for chrom in aut_chrom_names
        ]

# ------------------------------------------------------------------------------------------------------------------------
# Calculating expected number of detected crossovers
#
rule generate_fake_reads:
    output:
        csv = str(t2t_hapfusion_output_path / "{sample_id}" / "fake_reads_for_qc" / "{chrom}.csv"),
    params:
        n_fake_reads = 1000
    run:
        reads_df = annotate.generate_and_annotate_fake_reads(
            wildcards.sample_id,
            wildcards.chrom,
            params.n_fake_reads,
        )
        reads_df.to_csv(
            output.csv,
            index=False,
        )

rule generate_fake_reads_final:
    input:
        [str(t2t_hapfusion_output_path / f"{sample_id}" / "fake_reads_for_qc" / f"{chrom}.csv") \
            for sample_id in sample_ids
            for chrom in aut_chrom_names
        ]


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
        fasta = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.fasta",
        fai = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.fasta.fai",
    threads: 16
    resources:
        mem_mb=50000,
    run:
        output_directory = Path(output.fasta).parent
        shell(
            "{ragtag_path} scaffold {input.reference_fasta} {input.query_fasta}"
            " -o {output_directory} "
            " -u -w "
            " --aligner {minimap2_path} "
            " -t {threads} "
        )
        shell(
            "{samtools_path} faidx {output.fasta}"
        )

rule scaffolad_haplotypes_final:
    input:
        fasta = [str(hap_scaffolds_path \
            / f"{focal_sample_id}" / f"haplotype_{haplotype}" / "ragtag.scaffold.fasta") \
            for focal_sample_id in ["PD50489e"] \
            for haplotype in [1,2]]            




rule minimap2_to_haplotype:
    input:
        denovo_reference = hap_scaffolds_path \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.fasta",
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
            for focal_sample_id in ["PD50489e"] \
            for haplotype in [1,2]]

