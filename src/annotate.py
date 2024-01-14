import pybedtools
import msprime
import pysam

import pandas as pd
import polars as pl
import numpy as np
import scipy.stats
from pathlib import Path

from . import liftover

aut_chrom_names = [f"chr{i}" for i in list(range(1, 23))]
chrom_names = aut_chrom_names + ["chrX", "chrY"]

t2t_access_mask_path = "/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/05.accessibility_masks/01.T2T/combined_mask.bed.gz"
t2t_access_mask = pybedtools.BedTool(t2t_access_mask_path)

rate_maps = {}
for chrom in aut_chrom_names:
    rate_maps[chrom] = \
        msprime.RateMap.read_hapmap(
            open(f"/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/04.genetic_maps/02.T2T_lifted_Bherer_etal_SexualDimorphismRecombination/Refined_EUR_genetic_map_b37/male_{chrom}_genetic_map.txt"),
            sequence_length=liftover.T2T_chromosome_sizes_in_bp[chrom],
        )

def annonate_with_mappability_mask(
    reads_df,
):
    assert "read_chrom" in reads_df.columns
    assert "read_start_pos_0based" in reads_df.columns
    assert "read_end_pos_0based" in reads_df.columns

    reads_bed = pybedtools.BedTool.from_dataframe(
        reads_df[["read_chrom", "read_start_pos_0based", "read_end_pos_0based"]]
    )
    
    intersection_df = (t2t_access_mask
        .intersect(reads_bed, wb=True)
        .to_dataframe()
        .set_axis(["chrom", "start", "end", "read_chrom", "read_start_pos_0based", "read_end_pos_0based"], axis=1)
    )

    intersection_df = (intersection_df
        .assign(interval_length = lambda df: df["end"] - df["start"])
        .groupby(by=["read_chrom", "read_start_pos_0based", "read_end_pos_0based"])[["interval_length"]]
        .sum()
        .reset_index()        
        .merge(reads_df, on=["read_chrom", "read_start_pos_0based", "read_end_pos_0based"], how="right")
        .fillna({"interval_length": 0})
        .assign(read_length = lambda df: df["read_end_pos_0based"] - df["read_start_pos_0based"])
        .assign(accessible_ratio = lambda df: df["interval_length"] / df["read_length"].astype(float))        
    )

    return intersection_df


def annotate_with_cm_ranges(
    reads_df
):
    assert "read_chrom" in reads_df.columns
    assert "read_start_pos_0based" in reads_df.columns
    assert "read_end_pos_0based" in reads_df.columns

    dfs = []
    
    for chrom in aut_chrom_names:
        rate_map = rate_maps[chrom]

        chr_reads_df = reads_df[reads_df.read_chrom == chrom][["read_chrom", "read_start_pos_0based", "read_end_pos_0based"]]

        start_poses_cm = rate_map.get_cumulative_mass(chr_reads_df["read_start_pos_0based"].values)
        end_poses_cm = rate_map.get_cumulative_mass(chr_reads_df["read_end_pos_0based"].values)

        # Apparently get_cumulative_mass results are in Morgan
        chr_reads_df["cM"] = (end_poses_cm - start_poses_cm) * 1e2
        dfs.append(chr_reads_df)

    reads_df = reads_df.merge(pd.concat(dfs), on=["read_chrom", "read_start_pos_0based", "read_end_pos_0based"])

    # The # of crossovers is assumed to be Poisson with lambda=n*p, n=cM distance, p=0.01
    reads_df["prob_crossover"] = scipy.stats.poisson.sf(0, reads_df["cM"] * 0.01)    

    return reads_df
    
sample_id_to_batch_sample_ids = {
    "PD50477f": "PD50477f",
    "PD50519d": "PD50519d",
    "PD50508f": "PD50508bf",
    "PD50508b": "PD50508bf",
    "PD50523b": "PD50523b",
    "PD50521b": "PD50521be",
    "PD50521e": "PD50521be",
    "PD46180c": "PD46180c",
    "PD50489e": "PD50489e",
}    
def get_all_hets(
    sample_id,
    chrom = None,
):
    hets = []
    batch_sample_id = sample_id_to_batch_sample_ids[sample_id]

    vcf = pysam.VariantFile(
        f"/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13/{sample_id}/chm13.{batch_sample_id}.minimap2.deepvariant_1.1.0.phased.vcf.bgz"
    )

    for rec in vcf.fetch(chrom):
        rec_sample = rec.samples[list(rec.samples.keys())[0]]
        hets.append([sample_id, rec.chrom, rec.pos, rec_sample["GT"][0], rec_sample["GT"][1], rec_sample.phased])

    het_df = pd.DataFrame(
        hets,
        columns=["sample_id", "chrom", "pos_1based", "haplotype_0", "haplotype_1", "phased"],
    )
    return het_df
    

def annotate_with_crossover_detection_prob(
    reads_df,
    all_hets_df,
    min_bp_between_hets = 500,
):
    assert "read_chrom" in reads_df.columns
    assert "read_start_pos_0based" in reads_df.columns
    assert "read_end_pos_0based" in reads_df.columns

    reads_bed = pybedtools.BedTool.from_dataframe(
        reads_df[["read_chrom", "read_start_pos_0based", "read_end_pos_0based"]]
    )

    all_hets_bed = pybedtools.BedTool.from_dataframe(
        pd.DataFrame({
            "chrom": all_hets_df["chrom"][all_hets_df["phased"]],
            "start_pos_0based": all_hets_df["pos_1based"][all_hets_df["phased"]] - 1,
            "end_pos_0based": all_hets_df["pos_1based"][all_hets_df["phased"]],
        })
    )

    intersection_df = (all_hets_bed
        .intersect(reads_bed, wb=True)
        .to_dataframe()
        .set_axis(["chrom", "snp_pos_0based", "snp_end_pos_0based", "read_chrom", "read_start_pos_0based", "read_end_pos_0based"], axis=1)
        .groupby(by=["read_chrom", "read_start_pos_0based", "read_end_pos_0based"])
        .agg({"snp_pos_0based": ["min", "max"]})
        .reset_index()
    )

    intersection_df.columns = ['_'.join([x for x in col if len(x)]).strip() for col in intersection_df.columns.values]

    intersection_df = intersection_df.merge(
        reads_df, 
        on=["read_chrom", "read_start_pos_0based", "read_end_pos_0based"],
        how="right",
    )

    intersection_df["phased_snps_span"] = intersection_df["snp_pos_0based_max"] - intersection_df["snp_pos_0based_min"]
    intersection_df["crossover_detection_prob"] = intersection_df["phased_snps_span"] / (intersection_df["read_end_pos_0based"] - intersection_df["read_start_pos_0based"])
    intersection_df = intersection_df.fillna({"crossover_detection_prob": 0.0})

    return intersection_df
    
def generate_and_annotate_fake_reads(
    sample_id,
    chrom,
    n_fake_reads,
):
    rng = np.random.default_rng()

    ccs_read_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs/")
    count_path = ccs_read_path / sample_id / (sample_id + ".ccs.filtered.fastqc")
    d = open(count_path).read().strip().split()
    fake_read_length = int(float(d[3]))

    # Create random positions
    random_positions = rng.integers(low = 0, high = liftover.T2T_chromosome_sizes_in_bp[chrom], size = n_fake_reads)

    # Remove if too close to the edge
    random_positions = random_positions[random_positions + fake_read_length < liftover.T2T_chromosome_sizes_in_bp[chrom]]

    reads_df = \
        pd.DataFrame({
            "sample_id": sample_id,
            "read_chrom": chrom,
            "read_start_pos_0based": random_positions,
            "read_end_pos_0based": random_positions + fake_read_length,
        })
    
    reads_df = annonate_with_mappability_mask(reads_df)
    reads_df = annotate_with_cm_ranges(reads_df) 
    reads_df = annotate_with_crossover_detection_prob(
        reads_df,
        get_all_hets(sample_id, chrom)
    )

    return reads_df
