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

from src import liftover, annotate

aut_chrom_names = [f"chr{i}" for i in list(range(1, 23))]
chrom_names = aut_chrom_names + ["chrX", "chrY"]

# Binaries
samtools_path = "/nfs/users/nfs_r/rs42/rs42/software/samtools-1.18/samtools"

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
# Phasing
#
rule t2t_phase:
    input:
        t2t_alignment_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
            / "{focal_sample_id}" / "chm13.{focal_sample_id}.minimap2.primary_alignments.sorted.bam",
        t2t_deepvariant_phased_vcf = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
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
            str(input.t2t_deepvariant_phased_vcf),
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
        denovo_phased_vcf_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/") \
            / "{focal_sample_id}" / "{focal_sample_id}.minimap2.deepvariant_1.1.0.phased.vcf.bgz",
    output:
        denovo_raw_phasing_pickle = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds") \
            / "{focal_sample_id}" / "{focal_sample_id}.{denovo_chrom}.phasing_info.pcl"
    resources:
        mem_mb = 32000,
    run:
        chrom2hblock_lst = {}
        print("Calculating haplotype blocks...")

        edge_lst, edge2counts, hblock_lst = himut.phaselib.get_hblock(
            wildcards.denovo_chrom,
            himut.bamlib.get_tname2tsize(input.denovo_alignment_bam_file)[1][wildcards.denovo_chrom],
            str(input.denovo_alignment_bam_file),
            str(input.denovo_phased_vcf_file),
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
            / f"{focal_sample_id}" / f"{focal_sample_id}.{denovo_chrom + '_RagTag'}.phasing_info.pcl" \
            for focal_sample_id in ["PD50489e"] \
            for denovo_chrom in aut_chrom_names],