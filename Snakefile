from pathlib import Path
import pandas as pd
import numpy as np
import os

aut_chrom_names = [f"chr{i}" for i in list(range(1, 23))]
chrom_names = aut_chrom_names + ["chrX", "chrY"]

# Binaries
samtools_path = "/nfs/users/nfs_r/rs42/rs42/software/samtools-1.18/samtools"

# Hapfusion output paths
t2t_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13")
hg19_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37")
denovo_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds")

# ------------------------------------------------------------------------------------------------------------------------
# Depth calculations
#

rule t2t_depth:
    input:
        bam_path = str(t2t_hapfusion_output_path / "{sample_id}" / "chm13.{sample_id}.minimap2.primary_alignments.sorted.bam"),
    output:
        depth_path = str(t2t_hapfusion_output_path / "{sample_id}" / "chm13.{sample_id}.{chrom}.depth.txt"),
    shell:
        "{samtools_path} depth -r {wildcards.chrom} {input.bam_path} > {output.depth_path}"



rule t2t_depth_final:
    input:
        [
            str(t2t_hapfusion_output_path / f"{sample_id}" / f"chm13.{sample_id}.{chrom}.depth.txt") \
            for sample_id in ["PD46180c"] \
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
    run:
        # Prepare input bed file for liftover
        genetic_map = pd.read_csv(input.map_path, delim_whitespace=True)  
        (genetic_map            
            .assign(start_pos = np.concatenate([[0], genetic_map.pos.values[:-1]]))
            .assign(end_pos = genetic_map.pos.values)
            [["chr", "start_pos", "end_pos"]]
            .to_csv(output.input_bed_path, sep="\t", index=False, header=False)
        )

        # Liftover
        shell(
            f"{liftover_binary_path} "
            f"{output.input_bed_path} "
            f"{hg19_to_t2t_overchain_file} "
            f"{output.output_bed_path} "
            f"{output.output_unmapped_path} "
            f"-multiple "
        )

rule genetic_map_liftovers_final:
    input:
        [str(t2t_lifted_genetic_map_path / f"male_{chrom}_output.bed") \
            for chrom in aut_chrom_names
        ]