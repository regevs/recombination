from pathlib import Path
import pandas as pd
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

from src import liftover, annotate, diagnostics

aut_chrom_names = [f"chr{i}" for i in list(range(1, 23))]
chrom_names = aut_chrom_names + ["chrX", "chrY"]
chrom_names_ragtag = [x + "_RagTag" for x in chrom_names]

# Binaries
bedtools_path = "/nfs/users/nfs_r/rs42/rs42/software/bedtools"
minimap2_path = "/nfs/users/nfs_r/rs42/rs42/software/minimap2-2.26_x64-linux/minimap2"
minidot_path = "/nfs/users/nfs_r/rs42/rs42/git/miniasm/minidot"
samtools_path = "/nfs/users/nfs_r/rs42/rs42/software/samtools-1.18/samtools"
bcftools_path = "/nfs/users/nfs_r/rs42/rs42/software/bcftools-1.18/bcftools"
ragtag_path = "ragtag.py"
deepvariant_command = "/software/singularity-v3.9.0/bin/singularity exec -B /lustre /lustre/scratch126/casm/team154pc/sl17/01.himut/02.results/02.germline_mutations/deepvariant.simg"

# Hapfusion output paths
t2t_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13")
hg19_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37")
denovo_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds")

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
# Hapfusion Phasing
#
rule t2t_phase:
    input:
        t2t_alignment_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.bam",
        t2t_deepvariant_unphased_vcf = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.deepvariant_1.1.0.vcf.bgz",
    output:
        t2t_raw_phasing_pickle = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "debug" / "chm13.{focal_sample_id}.{chrom}.phasing_info.pcl",
    resources:
        mem_mb = 32000,
    run:
        chrom2hblock_lst = {}
        print("Calculating haplotype blocks...")

        edge_lst, edge2counts, hblock_lst = himut.phaselib.get_hblock(
            wildcards.chrom,
            himut.bamlib.get_tname2tsize(input.t2t_alignment_bam_file)[1][wildcards.chrom],
            str(input.t2t_alignment_bam_file),
            str(input.t2t_deepvariant_unphased_vcf),
            min_bq = 20, # defaults
            min_mapq = 20, 
            min_p_value = 0.0001,
            min_phase_proportion = 0.2,
            chrom2hblock_lst = chrom2hblock_lst,
        )

        print("Done, saving...")
        
        pickle.dump(
            {
                "edge_lst": edge_lst,
                "edge2counts": dict(edge2counts),
                "hblock_lst": hblock_lst,
            },
            open(output.t2t_raw_phasing_pickle, "wb"),
        )

rule t2t_phase_final:
    input:
        t2t_raw_phasing_pickle = [Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / f"{focal_sample_id}" / "debug" / f"chm13.{focal_sample_id}.{chrom}.phasing_info.pcl" \
            for focal_sample_id in ["PD50489e"] \
            for chrom in aut_chrom_names],

rule denovo_phase:
    input:
        denovo_alignment_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.primary_alignments.sorted.bam",
        denovo_unphased_vcf_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.deepvariant_1.1.0.vcf.bgz",
    output:
        denovo_raw_phasing_pickle = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds") \
            / "{focal_sample_id}" / "debug" / "{focal_sample_id}.{denovo_chrom}.phasing_info.pcl"
    resources:
        mem_mb = 32000,
    run:
        chrom2hblock_lst = {}
        print("Calculating haplotype blocks...")

        edge_lst, edge2counts, hblock_lst = himut.phaselib.get_hblock(
            wildcards.denovo_chrom,
            himut.bamlib.get_tname2tsize(input.denovo_alignment_bam_file)[1][wildcards.denovo_chrom],
            str(input.denovo_alignment_bam_file),
            str(input.denovo_unphased_vcf_file),
            min_bq = 20, # defaults
            min_mapq = 20, 
            min_p_value = 0.0001,
            min_phase_proportion = 0.2,
            chrom2hblock_lst = chrom2hblock_lst,
        )

        print("Done, saving...")
        
        pickle.dump(
            {
                "edge_lst": edge_lst,
                "edge2counts": dict(edge2counts),
                "hblock_lst": hblock_lst,
            },
            open(output.denovo_raw_phasing_pickle, "wb"),
        )

rule denovo_phase_final:
    input:
        t2t_raw_phasing_pickle = [Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds") \
            / f"{focal_sample_id}" / "debug" / f"{focal_sample_id}.{denovo_chrom + '_RagTag'}.phasing_info.pcl" \
            for focal_sample_id in ["PD50489e"] \
            for denovo_chrom in aut_chrom_names],

# ------------------------------------------------------------------------------------------------------------------------
# hiphase Phasing
#
rule t2t_hiphase:
    input:
        t2t_alignment_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.bam",
        t2t_deepvariant_unphased_vcf = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.deepvariant_1.1.0.vcf.bgz",
        t2t_reference = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/03.t2t-chm13/chm13v2.0.fasta"),
    output:
        t2t_phased_vcf_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.deepvariant_1.1.0.hiphase.vcf.gz",
        t2t_haplotagged_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.haplotagged.bam",
        t2t_haplotagged_bai_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.haplotagged.bam.bai",
    resources:
        mem_mb=10000,
    threads: 16
    run:
        shell(
            f"hiphase "
            f"--bam {input.t2t_alignment_bam_file} "
            f"--vcf {input.t2t_deepvariant_unphased_vcf} "
            f"--reference {input.t2t_reference} " 
            f"--output-vcf {output.t2t_phased_vcf_gz} " 
            f"--output-bam {output.t2t_haplotagged_bam_file} "
            f"--ignore-read-groups "
            f"--threads {threads} "
            f"--min-vcf-qual 20 "
            f"--min-mapq 20 "            
        )

        shell(
            "{samtools_path} index {output.t2t_haplotagged_bam_file}"
        ) 

rule t2t_hiphase_final:
    input:
        t2t_phased_vcf_gz = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / f"{focal_sample_id}" / f"chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.haplotagged.bam") \
            for focal_sample_id in sample_ids]


def denovo_reference_func(wildcards):
    joint_id = sample_id_to_joint_id[wildcards.focal_sample_id]
    denovo_reference = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/02.scaffold") \
            / f"{joint_id}" / f"{joint_id}.ragtag_scaffold.fasta"
    return denovo_reference

rule denovo_normalize_vcf:
    input:
        denovo_reference = denovo_reference_func,
        denovo_unphased_vcf_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.deepvariant_1.1.0.vcf.bgz",
    output:
        denovo_normed_unphased_vcf_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.deepvariant_1.1.0.normed.vcf.bgz",
    run:
        regions = ",".join([f"{chrom}_RagTag" for chrom in chrom_names])
        shell(
            f"{bcftools_path} norm "
            f"--check-ref s "
            f"--fasta-ref {input.denovo_reference} "
            f"--regions {regions}  "
            f"--do-not-normalize "
            f"--output-type z "
            f"--output {output.denovo_normed_unphased_vcf_file} "
            f"{input.denovo_unphased_vcf_file} "
        )

        shell(
            f"{bcftools_path} index {output.denovo_normed_unphased_vcf_file}"
        )

rule denovo_hiphase:
    input:
        denovo_reference = denovo_reference_func,
        denovo_alignment_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.primary_alignments.sorted.bam",
        denovo_normed_unphased_vcf_file = Path("/lustre/scratchq126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.deepvariant_1.1.0.normed.vcf.bgz",
    output:
        denovo_phased_vcf_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.deepvariant_1.1.0.hiphase.vcf.gz",
    resources:
        mem_mb=10000,
    threads: 16
    run:
        shell(
            f"hiphase "
            f"--bam {input.denovo_alignment_bam_file} "
            f"--vcf {input.denovo_normed_unphased_vcf_file} "
            f"--reference {input.denovo_reference} " 
            f"--output-vcf {output.denovo_phased_vcf_gz} "
            f"--ignore-read-groups "
            f"--threads {threads} "
            f"--min-vcf-qual 20 "
            f"--min-mapq 20 "            
        ) 

rule denovo_hiphase_final:
    input:
        denovo_phased_vcf_gz = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / f"{focal_sample_id}" / f"{focal_sample_id}.minimap2.deepvariant_1.1.0.hiphase.vcf.gz") \
            for focal_sample_id in sample_ids]

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
        fasta = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.fasta",
        fai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
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
        fasta = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
            / f"{focal_sample_id}" / f"haplotype_{haplotype}" / "ragtag.scaffold.fasta") \
            for focal_sample_id in ["PD50489e"] \
            for haplotype in [1,2]]            




rule minimap2_to_haplotype:
    input:
        denovo_reference = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
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

# ------------------------------------------------------------------------------------------------------------------------
# Analyze reads
#
import pysam
rule analyze_reads:
    input:
        bam_filename_1 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap1.minimap2.sorted.primary_alignments.bam",
        bam_filename_2 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap2.minimap2.sorted.primary_alignments.bam",
    output:
        csv_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "reads" / "{chrom}.read_refinement.csv",
    threads: 8,
    resources:
        mem_mb=64000,
    run:
        print("Running...")
        cdf = diagnostics.run_all_refine_cigars(
            input.bam_filename_1,
            input.bam_filename_2,
            wildcards.chrom,
            threads,
        )

        print("Writing...")        
        cdf.write_csv(
            output.csv_gz,
        )

rule analyze_reads_final:
    input:
        csv_gz = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
                / f"{focal_sample_id}" / "reads" / f"{chrom + '_RagTag'}.read_refinement.csv") \
                for focal_sample_id in ["PD50489e"] \
                for chrom in ["chr2"]]

rule phase_and_haplotag:
    input:
        csv_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "reads" / "{chrom}.read_refinement.csv",
        bam1 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap1.minimap2.sorted.primary_alignments.bam",
        bam2 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.hap2.minimap2.sorted.primary_alignments.bam",
    output:
        bam1 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap1.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.bam",
        bam_bai1 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap1.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.bam.bai",
        bam2 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap2.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.bam",
        bam_bai2 = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap2.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.bam.bai",
    resources:
        mem_mb=32000,
    run:
        diagnostics.phase_and_haplotag(
            input.csv_gz,
            input.bam1,
            input.bam2,
            output.bam1,
            output.bam2,
            certainty_threshold=float(wildcards.certainty),
            min_mapq=60,
            high_confidence_snp_slack=10,
        )

        shell(
            "{samtools_path} index {output.bam1}"
        )

        shell(
            "{samtools_path} index {output.bam2}"
        )

rule calculate_genome_coverage_on_haplotype:
    input:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.bam",
        bam_bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.bam.bai",
    output:
        bedgraph = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.coverage.bedgraph",
    resources:
        mem_mb=4000,
    run:
        shell(
            "{samtools_path} view --with-header --tag HP:{wildcards.haplotype} {input.bam} | "
            "{bedtools_path} genomecov -ibam - -bg > {output.bedgraph}"
        )

# rule calculate_genome_coverage_on_haplotype_per_position:
#     input:
#         bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.bam",
#         bam_bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.bam.bai",
#     output:
#         tsv = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.coverage_per_base.tsv.gz",
#     run:
#         shell(
#             "{samtools_path} view --with-header --tag HP:{wildcards.haplotype} {input.bam} | "
#             "{bedtools_path} genomecov -ibam - -dz | gzip -c > {output.tsv}"
#         )        

# rule calculate_reads_coverage_on_haplotype:
#     input:
#         bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.bam",
#         bam_bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.bam.bai",
#         bedgraph = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.coverage.bedgraph",
#     output:
#         bed = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
#             / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.coverage.per_read_min_{coverage}.bed",    
#     run:
#         shell(
#             "{bedtools_path} intersect "
#             "-abam {input.bam} "
#             "-b {input.bedgraph} "
#             "-wao -sorted "
#             "-bed | "
#             "awk '$16 >= {wildcards.coverage}' | "
#             "{bedtools_path} groupby -i stdin -g 1,2,3,4 -c 16 -o sum "
#             " > {output.bed}"
#         )

rule phase_and_haplotag_final:
    input:
        bam = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
                    / f"{focal_sample_id}" / f"{focal_sample_id}.{chrom}_RagTag.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged_{certainty}.coverage.bedgraph") \
                for focal_sample_id in ["PD50489e"] \
                for chrom in ["chr2"] \
                for haplotype in [1,2] \
                for certainty in [0.55, 0.8, 0.9, 0.95, 1]]

# ------------------------------------------------------------------------------------------------------------------------
# Compare the two haplotypes
#
rule compare_haplotypes:
    input:
        hap_1_fasta = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
            / "{focal_sample_id}" / "haplotype_1" / "ragtag.scaffold.fasta",
        hap_2_fasta = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
            / "{focal_sample_id}" / "haplotype_2" / "ragtag.scaffold.fasta",
    output:
        paf = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
            / "{focal_sample_id}" / "haplotype_alignment.paf",
    threads: 32,
    resources:
        mem_mb=64000,            
    run:
        shell(
            "{minimap2_path} -xasm5 "
            "-t {threads} "
            "{input.hap_1_fasta} {input.hap_2_fasta} "
            " > {output.paf}"
        )

rule paf_to_dot:
    input:
        paf = "{something}/{prefix}.paf",
    output:
        eps = "{something}/{prefix}.eps",
        pdf = "{something}/{prefix}.pdf",
    run:
        shell(
            "{minidot_path} -f 5 -d {input.paf} > {output.eps}"
        )
        shell("epstopdf {output.eps}")

rule compare_haplotypes_final:
    input:
        eps = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
            / f"{focal_sample_id}" / "haplotype_alignment.pdf") \ 
            for focal_sample_id in ["PD50489e"]]


# ------------------------------------------------------------------------------------------------------------------------
# Run DeepVariant again on the haplotagged aligment
#

rule deepvariant_haplotagged:
    input:
        t2t_haplotagged_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.haplotagged.bam",
        t2t_haplotagged_bai_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.haplotagged.bam.bai",
        t2t_reference = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/03.t2t-chm13/chm13v2.0.fasta"),
    output:
        vcf_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.haplotagged.deepvariant_1.1.0.vcf.gz",
    threads: 16,
    resources:
        mem_mb=64000,
    run:
        regions = ' '.join(chrom_names)
        shell(
            "ulimit -u 10000 && "
            "{deepvariant_command} /opt/deepvariant/bin/run_deepvariant "
            "--model_type PACBIO "
            "--ref {input.t2t_reference} "
            "--reads {input.t2t_haplotagged_bam_file} "
            "--use_hp_information "
            "--output_vcf {output.vcf_gz} "
            "--num_shards {threads} "
            "--regions {regions} "
        )

rule deepvariant_haplotagged_final:
    input:
        vcf_gz = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / f"{focal_sample_id}" / f"chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.haplotagged.deepvariant_1.1.0.vcf.gz") \
            for focal_sample_id in ["PD50489e"]]


rule deepvariant_hifiasm_haplotagged:
    input:
        bam = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.bam",
        bam_bai = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.bam.bai",
        denovo_reference = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/05.ragtag/03.haplotype_specific_scaffolds") \
            / "{focal_sample_id}" / "haplotype_{haplotype}" / "ragtag.scaffold.fasta",
    output:
        vcf_gz = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.{chrom}.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.deepvariant_1.1.0.csv.gz",
    threads: 16,
    resources:
        mem_mb=64000,
    run:
        shell(
            "ulimit -u 10000 && "
            "{deepvariant_command} /opt/deepvariant/bin/run_deepvariant "
            "--model_type PACBIO "
            "--ref {input.denovo_reference} "
            "--reads {input.bam} "
            "--use_hp_information "
            "--output_vcf {output.vcf_gz} "
            "--num_shards {threads} "
            "--regions {wildcards.chrom} "
        )

        # shell(
        #     f"{bcftools_path} index {output.vcf_gz}"
        # )

rule deepvariant_hifiasm_haplotagged_final:
    input:
        vcf_gz = [str(Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / f"{focal_sample_id}" / f"{focal_sample_id}.{chrom}_RagTag.hap{haplotype}.minimap2.sorted.primary_alignments.hifiasm_haplotagged.deepvariant_1.1.0.csv.gz") \
            for focal_sample_id in ["PD50489e"]
            for chrom in ["chr2"]
            for haplotype in [1,2]
            ]
