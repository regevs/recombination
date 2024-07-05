import pybedtools
import msprime
import pysam

import pandas as pd
import polars as pl
import numpy as np
import scipy.stats
from pathlib import Path

import pyBigWig

from . import inference

aut_chrom_names = [f"chr{i}" for i in list(range(1, 23))]
chrom_names = aut_chrom_names + ["chrX", "chrY"]


grch37_chromosome_sizes_in_bp = {
    'chr1': 249250621,
    'chr2': 243199373,
    'chr3': 198022430,
    'chr4': 191154276,
    'chr5': 180915260,
    'chr6': 171115067,
    'chr7': 159138663,
    'chr8': 146364022,
    'chr9': 141213431,
    'chr10': 135534747,
    'chr11': 135006516,
    'chr12': 133851895,
    'chr13': 115169878,
    'chr14': 107349540,
    'chr15': 102531392,
    'chr16': 90354753,
    'chr17': 81195210,
    'chr18': 78077248,
    'chr19': 59128983,
    'chr20': 63025520,
    'chr21': 48129895,
    'chr22': 51304566,
}

rate_maps = {}
for chrom in aut_chrom_names:
    rate_maps[chrom] = \
        msprime.RateMap.read_hapmap(
            open(f"/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/07.references/04.genetic_maps/01.Bherer_etal_SexualDimorphismRecombination/Refined_EUR_genetic_map_b37/male_{chrom}.txt"),
            sequence_length=grch37_chromosome_sizes_in_bp[chrom],
        )


# ---------------------------------------------------------------------------
# Annotate read structure
#

# Detection prob functions
def crossover_prob_detection_full_read(
    read_length,
    snp_positions,
    GC_tract_mean,
):
    if len(snp_positions) < 4:
        return 0.0
    
    prob = 0.0
    for idx_trans in range(1, len(snp_positions)-2):
        prob += inference.old_likelihood_of_read(
            read_length = read_length,
            snp_positions_on_read = snp_positions,
            idx_transitions = [idx_trans],
            prob_CO = 1,
            prob_GC_component = 1,
            GC_tract_mean = GC_tract_mean,
            GC_tract_mean2 = 1000,
            recombination_rate_per_bp = 1e-8,
        ) / (1e-8 * read_length)
    
    return prob

def crossover_prob_detection_in_crossover_active_interval(
    read_length,
    snp_positions,
    GC_tract_mean,
):
    if len(snp_positions) < 4:
        return 0.0
    
    snp_positions = np.array(snp_positions)
    read_length = snp_positions[-2]-snp_positions[1]+1
    snp_positions = snp_positions[1:-1]-snp_positions[1]
    
    prob = 0.0
    for idx_trans in range(len(snp_positions)-1):
        this_prob = inference.old_likelihood_of_read(
            read_length = read_length,
            snp_positions_on_read = snp_positions,
            idx_transitions = [idx_trans],
            prob_CO = 1,
            prob_GC_component = 1,
            GC_tract_mean = GC_tract_mean,
            GC_tract_mean2 = 1000,
            recombination_rate_per_bp = 1e-8,
        ) / (1e-8 * (read_length-1))
        prob += this_prob
    
    return prob

def noncrossover_prob_detection_full_read(
    read_length,
    snp_positions,
    GC_tract_mean,
):
    if len(snp_positions) < 3:
        return 0.0
    
    prob = 0.0
    for idx_trans in range(0, len(snp_positions)-2):
        prob += inference.old_likelihood_of_read(
            read_length = read_length,
            snp_positions_on_read = snp_positions,
            idx_transitions = [idx_trans, idx_trans+1],
            prob_CO = 0,
            prob_GC_component = 1,
            GC_tract_mean = GC_tract_mean,
            GC_tract_mean2 = 1000,
            recombination_rate_per_bp = 1e-8,
        ) / (1e-8 * read_length)
    
    return prob

def noncrossover_prob_detection_in_noncrossover_active_interval(
    read_length,
    snp_positions,
    GC_tract_mean,
):
    if len(snp_positions) < 3:
        return 0.0
    
    snp_positions = np.array(snp_positions)
    read_length = snp_positions[-1]-snp_positions[0]+1
    snp_positions = snp_positions-snp_positions[0]    
    
    prob = 0.0
    for idx_trans in range(0, len(snp_positions)-2):
        prob += inference.old_likelihood_of_read(
            read_length = read_length,
            snp_positions_on_read = snp_positions,
            idx_transitions = [idx_trans, idx_trans+1],
            prob_CO = 0,
            prob_GC_component = 1,
            GC_tract_mean = GC_tract_mean,
            GC_tract_mean2 = 1000,
            recombination_rate_per_bp = 1e-8,
        ) / (1e-8 * read_length)
    
    return prob

def noncrossover_prob_detection_in_crossover_active_interval(
    read_length,
    snp_positions,
    GC_tract_mean,
):
    if len(snp_positions) < 5:
        return 0.0
    
    snp_positions = np.array(snp_positions)
    read_length = snp_positions[-2]-snp_positions[1]+1
    snp_positions = snp_positions[1:-1]-snp_positions[1]
    
    prob = 0.0
    for idx_trans in range(0, len(snp_positions)-2):
        prob += inference.old_likelihood_of_read(
            read_length = read_length,
            snp_positions_on_read = snp_positions,
            idx_transitions = [idx_trans, idx_trans+1],
            prob_CO = 0,
            prob_GC_component = 1,
            GC_tract_mean = GC_tract_mean,
            GC_tract_mean2 = 1000,
            recombination_rate_per_bp = 1e-8,
        ) / (1e-8 * read_length)
    
    return prob



def annotate_read_structure(
    snps_filename,
    sample_id,
    chrom,
    grch37_ref_starts_filename,
    grch38_ref_starts_filename,
    T2T_ref_starts_filename,
    cov_hap1_parquet_filename,
    cov_hap2_parquet_filename,
    AA_hotspots_filename,
    CL4_hotspots_filename,
    H3K4me3_filename,
    CTCF_filename,
    classified_reads_filename,
    GC_tract_mean = 30,
    min_mapq = 60,
    max_total_mismatches = 100,
    max_total_clipping = 10,
    read_margin_in_bp = 5000,
):
    # Read SNPs
    snps_df = pl.read_parquet(snps_filename)

    AA_hotspots_df = pl.read_csv(
        AA_hotspots_filename,
        null_values="NA",
    ).rename({
        "Chromosome": "chrom", 
        "Motif_Centre_Pos": "motif_center_pos",
        "Start_Pos": "hotspot_start_pos",
        "End_Pos": "hotspot_end_pos",    
    })

    CL4_hotspots_df = pl.read_csv(
        CL4_hotspots_filename,
        null_values="NA",
        separator=" ",
    ).rename({
        "Chromosome": "chrom", 
        "Motif_Centre_Pos": "motif_center_pos",
        "Start_Pos": "hotspot_start_pos",
        "End_Pos": "hotspot_end_pos",        
    })

    # Create reads dataframe
    reads_df = (snps_df
        .select(
            "read_name", 
            pl.col("read_length1").alias("read_length"), 
            "mapq1", 
            "mapq2", 
            "is_forward1", 
            "is_forward2", 
            "total_mismatches", 
            "num_common_insertions", 
            "num_common_deletions", 
            "total_clipping",
        )
        .unique()
        .with_columns(
            chrom = pl.lit(chrom), 
            sample_id = pl.lit(sample_id),
            grch37_chromosome_size_in_bp = pl.lit(grch37_chromosome_sizes_in_bp[chrom])
        )
    )

    # Find minimal coverage across each haplotype
    reads_df = (reads_df
        .join(
            pl.read_parquet(cov_hap1_parquet_filename).select("read_name", "min_coverage_hap1", "ref1_start"),
            on="read_name",
            how="left",
        )
        .join(
            pl.read_parquet(cov_hap2_parquet_filename).select("read_name", "min_coverage_hap2", "ref2_start"),
            on="read_name",
            how="left",
        )
        .with_columns(
            min_coverage_hap1 = pl.col("min_coverage_hap1").fill_null(0),
            min_coverage_hap2 = pl.col("min_coverage_hap2").fill_null(0),
        )
    )

    # Add SNP patterns
    reads_df = (reads_df
        .join(
            (snps_df
                .sort("read_name", "start")
                .filter("is_high_quality_snp")
                .group_by("read_name")
                .agg(
                    pl.col("start").alias("high_quality_snp_positions"),  
                    pl.col("fits1_more").alias("high_quality_snp_positions_alleles"),
                )
            ),
            on="read_name",
            how="left",
        )
        .join(
            (snps_df
                .sort("read_name", "start")
                .filter("is_mid_quality_snp")
                .group_by("read_name")
                .agg(
                    pl.col("start").alias("mid_quality_snp_positions"),
                    pl.col("fits1_more").alias("mid_quality_snp_positions_alleles"),
                )
            ),
            on="read_name",
            how="left",
        )
        .with_columns(
            pl.col("mid_quality_snp_positions").fill_null([]),
            pl.col("high_quality_snp_positions").fill_null([]),
            pl.col("high_quality_snp_positions_alleles").fill_null([]),
            pl.col("mid_quality_snp_positions_alleles").fill_null([]),
        )
    )

    ### Add CO and NCO active intervals
    reads_df = (reads_df
        .with_columns(
            CO_active_interval_start = pl.when(
                pl.col("high_quality_snp_positions").list.len() >= 4,
            ).then(
                pl.col("high_quality_snp_positions").list.get(1)
            ),
            CO_active_interval_end = pl.when(
                pl.col("high_quality_snp_positions").list.len() >= 4,
            ).then(
                pl.col("high_quality_snp_positions").list.get(-2)
            ),
            NCO_active_interval_start = pl.when(
                pl.col("high_quality_snp_positions").list.len() >= 3,
            ).then(
                pl.col("high_quality_snp_positions").list.get(0)
            ),
            NCO_active_interval_end = pl.when(
                pl.col("high_quality_snp_positions").list.len() >= 3,
            ).then(
                pl.col("high_quality_snp_positions").list.get(-1)
            ),
            mid_CO_active_interval_start = pl.when(
                pl.col("mid_quality_snp_positions").list.len() >= 4,
            ).then(
                pl.col("mid_quality_snp_positions").list.get(1)
            ),
            mid_CO_active_interval_end = pl.when(
                pl.col("mid_quality_snp_positions").list.len() >= 4,
            ).then(
                pl.col("mid_quality_snp_positions").list.get(-2)
            ),
            mid_NCO_active_interval_start = pl.when(
                pl.col("mid_quality_snp_positions").list.len() >= 3,
            ).then(
                pl.col("mid_quality_snp_positions").list.get(0)
            ),
            mid_NCO_active_interval_end = pl.when(
                pl.col("mid_quality_snp_positions").list.len() >= 3,
            ).then(
                pl.col("mid_quality_snp_positions").list.get(-1)
            ),
        )
        .with_columns(
            CO_active_interval_length_bp = pl.col("CO_active_interval_end") - pl.col("CO_active_interval_start"),
            NCO_active_interval_length_bp = pl.col("NCO_active_interval_end") - pl.col("NCO_active_interval_start"),
            mid_CO_active_interval_length_bp = pl.col("mid_CO_active_interval_end") - pl.col("mid_CO_active_interval_start"),
            mid_NCO_active_interval_length_bp = pl.col("mid_NCO_active_interval_end") - pl.col("mid_NCO_active_interval_start"),
        )
    )

    ### Add ref alignments
    grch37_refs_df = (
        pl.read_csv(
            grch37_ref_starts_filename,
            new_columns = ["read_name", "chrom", "grch37_reference_start"],
            infer_schema_length=0,
        )
        .cast({"grch37_reference_start": pl.Int64})
        .with_columns(
            chrom = pl.concat_str([pl.lit("chr"), pl.col("chrom")]),
        )
    )

    grch38_refs_df = (
        pl.read_csv(
            grch38_ref_starts_filename,
            new_columns = ["read_name", "chrom", "grch38_reference_start"],
            infer_schema_length=0,
        )
        .cast({"grch38_reference_start": pl.Int64})
    )

    T2T_refs_df = (
        pl.read_csv(
            T2T_ref_starts_filename,
            new_columns = ["read_name", "chrom", "T2T_reference_start"],
            infer_schema_length=0,
        )
        .cast({"T2T_reference_start": pl.Int64})
    )

    reads_df = (reads_df
        .join(grch37_refs_df, on=["read_name", "chrom"], how="left")
        .join(grch38_refs_df, on=["read_name", "chrom"], how="left")
        .join(T2T_refs_df, on=["read_name", "chrom"], how="left")
        .with_columns(
            grch37_reference_end = pl.col("grch37_reference_start") + pl.col("read_length"),
            grch38_reference_end = pl.col("grch38_reference_start") + pl.col("read_length"),
            T2T_reference_end = pl.col("T2T_reference_start") + pl.col("read_length"),
        )
    )

    ### Add cM
    rate_map = rate_maps[chrom]
    reads_df = (reads_df
        .with_columns(
            grch37_reference_start_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"]) * 1e2,
            grch37_reference_end_cM = rate_map.get_cumulative_mass(
                np.minimum(rate_map.right[-1]-1, reads_df["grch37_reference_start"] + reads_df["read_length"]),
                ) * 1e2,
            CO_active_interval_start_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["CO_active_interval_start"]) * 1e2,
            CO_active_interval_end_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["CO_active_interval_end"]) * 1e2,
            NCO_active_interval_start_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["NCO_active_interval_start"]) * 1e2,
            NCO_active_interval_end_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["NCO_active_interval_end"]) * 1e2,
            mid_CO_active_interval_start_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["mid_CO_active_interval_start"]) * 1e2,
            mid_CO_active_interval_end_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["mid_CO_active_interval_end"]) * 1e2,
            mid_NCO_active_interval_start_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["mid_NCO_active_interval_start"]) * 1e2,
            mid_NCO_active_interval_end_cM = rate_map.get_cumulative_mass(reads_df["grch37_reference_start"] + reads_df["mid_NCO_active_interval_end"]) * 1e2,
        )
        .with_columns(
            full_read_crossover_prob = (pl.col("grch37_reference_end_cM") - pl.col("grch37_reference_start_cM")) * 0.01,
            CO_active_interval_crossover_prob = (pl.col("CO_active_interval_end_cM") - pl.col("CO_active_interval_start_cM")) * 0.01,
            NCO_active_interval_crossover_prob = (pl.col("NCO_active_interval_end_cM") - pl.col("NCO_active_interval_start_cM")) * 0.01,
            mid_CO_active_interval_crossover_prob = (pl.col("mid_CO_active_interval_end_cM") - pl.col("mid_CO_active_interval_start_cM")) * 0.01,
            mid_NCO_active_interval_crossover_prob = (pl.col("mid_NCO_active_interval_end_cM") - pl.col("mid_NCO_active_interval_start_cM")) * 0.01,
        )
        .fill_nan(None)
    )

    ### Add cM between each pair of SNPs
    # For mid quality SNPs
    exploded_df = (reads_df
        .select(
            "read_name",
            "mid_quality_snp_positions",
            "grch37_reference_start",
        )
        .explode("mid_quality_snp_positions")
    )

    reads_df = (reads_df
        .join(
            (exploded_df
                .with_columns(
                    at_mid_quality_snp_cM = rate_map.get_cumulative_mass(exploded_df["grch37_reference_start"] + exploded_df["mid_quality_snp_positions"]) * 1e2,
                )
                .drop_nulls()
                .group_by("read_name", maintain_order=True)
                .agg("at_mid_quality_snp_cM")
            ),
            on="read_name",
            how="left",
        )
    )

    reads_df = (reads_df
        .with_columns(
            between_mid_quality_snps_cM = pl.lit([]).list.concat(
                [
                    "grch37_reference_start_cM",
                    "at_mid_quality_snp_cM",
                    "grch37_reference_end_cM",
                ]
            ).list.diff(null_behavior="drop")
        )
    )

    # For high quality SNPs
    exploded_df = (reads_df
        .select(
            "read_name",
            "high_quality_snp_positions",
            "grch37_reference_start",
        )
        .explode("high_quality_snp_positions")
    )

    reads_df = (reads_df
        .join(
            (exploded_df
                .with_columns(
                    at_high_quality_snp_cM = rate_map.get_cumulative_mass(exploded_df["grch37_reference_start"] + exploded_df["high_quality_snp_positions"]) * 1e2,
                )
                .drop_nulls()
                .group_by("read_name", maintain_order=True)
                .agg("at_high_quality_snp_cM")
            ),
            on="read_name",
            how="left",
        )
    )

    reads_df = (reads_df
        .with_columns(
            between_high_quality_snps_cM = pl.lit([]).list.concat(
                [
                    "grch37_reference_start_cM",
                    "at_high_quality_snp_cM",
                    "grch37_reference_end_cM",
                ]
            ).list.diff(null_behavior="drop")
        )
    )

    ### Add cM before and after read
    cols_df = (reads_df.select(
            pl.max_horizontal(
                pl.lit(0), 
                pl.col("grch37_reference_start") - read_margin_in_bp,
            ).alias("before_read_bp"),
            pl.min_horizontal(
                pl.col("grch37_chromosome_size_in_bp"), 
                pl.col("grch37_reference_end") + read_margin_in_bp,
            ).alias("after_read_bp"),
        )
    )
    reads_df = (reads_df
        .with_columns(
            before_read_cM = reads_df["grch37_reference_start_cM"] - rate_map.get_cumulative_mass(cols_df["before_read_bp"]) * 1e2,
            after_read_cM = rate_map.get_cumulative_mass(cols_df["after_read_bp"]) * 1e2 - reads_df["grch37_reference_end_cM"],
        )        
    )

    ### Add detection probabilities
    CO_prob_detection_full_read_df = (reads_df
        .select("read_name", "read_length", "high_quality_snp_positions")
        .map_rows(lambda row: (row[0], crossover_prob_detection_full_read(row[1], row[2], GC_tract_mean)))
        .rename({
            "column_0": "read_name",
            "column_1": "CO_prob_detection_full_read",
        })
    )

    # These should all be 1 or 0
    CO_prob_detection_CO_active_interval_df = (reads_df
        .select("read_name", "read_length", "high_quality_snp_positions")
        .map_rows(lambda row: (row[0], crossover_prob_detection_in_crossover_active_interval(row[1], row[2], GC_tract_mean)))
        .rename({
            "column_0": "read_name",
            "column_1": "CO_prob_detection_in_CO_active_interval",
        })
    )

    NCO_prob_detection_full_read_df = (reads_df
        .select("read_name", "read_length", "high_quality_snp_positions")
        .map_rows(lambda row: (row[0], noncrossover_prob_detection_full_read(row[1], row[2], GC_tract_mean)))
        .rename({
            "column_0": "read_name",
            "column_1": "NCO_prob_detection_full_read",
        })
    )

    NCO_prob_detection_NCO_active_region_df = (reads_df
        .select("read_name", "read_length", "high_quality_snp_positions")
        .map_rows(lambda row: (row[0], noncrossover_prob_detection_in_noncrossover_active_interval(row[1], row[2], GC_tract_mean)))
        .rename({
            "column_0": "read_name",
            "column_1": "NCO_prob_detection_in_NCO_active_interval",
        })
    )

    NCO_prob_detection_CO_active_region_df = (reads_df
        .select("read_name", "read_length", "high_quality_snp_positions")
        .map_rows(lambda row: (row[0], noncrossover_prob_detection_in_crossover_active_interval(row[1], row[2], GC_tract_mean)))
        .rename({
            "column_0": "read_name",
            "column_1": "NCO_prob_detection_in_CO_active_interval",
        })
    )      

    reads_df = (reads_df
        .join(CO_prob_detection_full_read_df, on="read_name", how="left")
        .join(NCO_prob_detection_full_read_df, on="read_name", how="left")
        .join(CO_prob_detection_CO_active_interval_df, on="read_name", how="left")
        .join(NCO_prob_detection_NCO_active_region_df, on="read_name", how="left")
        .join(NCO_prob_detection_CO_active_region_df, on="read_name", how="left")
    )

    ### Add AA hotspots
    AA_hotspots_df = (AA_hotspots_df
        .filter(pl.col("chrom") == chrom)
        .sort("motif_center_pos")
        .set_sorted("motif_center_pos")
    )
    
    AA_possible_hits = (reads_df
        .select("read_name", "grch38_reference_start", "grch38_reference_end")
        .sort("grch38_reference_start")
        .set_sorted("grch38_reference_start")
        .join_asof(
            AA_hotspots_df.select(
                AA_motif_center_pos=pl.col("motif_center_pos"), 
                AA_heat=pl.col("heat"),
                AA_motif_strand=pl.col("motif_strand"),
            ),
            left_on="grch38_reference_start",
            right_on="AA_motif_center_pos",
            strategy="forward",
        )
        .filter((pl.col("AA_motif_center_pos") >= pl.col("grch38_reference_start")) & 
                (pl.col("AA_motif_center_pos") < pl.col("grch38_reference_end")))
        .select("read_name", "AA_motif_center_pos", "AA_heat", "AA_motif_strand")
    )
    
    reads_df = (reads_df
        .join(
            AA_possible_hits, 
            on="read_name",
            how="left",
        )
    )

    ### Add CL4 hotspots
    CL4_hotspots_df = (CL4_hotspots_df
        .filter(pl.col("chrom") == chrom)
        .sort("motif_center_pos")
        .set_sorted("motif_center_pos")
    )
    
    CL4_possible_hits = (reads_df
        .select("read_name", "grch38_reference_start", "grch38_reference_end")
        .sort("grch38_reference_start")
        .set_sorted("grch38_reference_start")
        .join_asof(
            CL4_hotspots_df.select(
                CL4_motif_center_pos=pl.col("motif_center_pos"), 
                CL4_heat=pl.col("heat"),
                CL4_motif_strand=pl.col("motif_strand"),
            ),
            left_on="grch38_reference_start",
            right_on="CL4_motif_center_pos",
            strategy="forward",
        )
        .filter((pl.col("CL4_motif_center_pos") >= pl.col("grch38_reference_start")) & 
                (pl.col("CL4_motif_center_pos") < pl.col("grch38_reference_end")))
        .select("read_name", "CL4_motif_center_pos", "CL4_heat", "CL4_motif_strand")
    )
    
    reads_df = (reads_df
        .join(
            CL4_possible_hits, 
            on="read_name",
            how="left",
        )
    )

    # Add H3K4me3 signal
    with pyBigWig.open(H3K4me3_filename) as BW:
        rows = []
        for row in reads_df.iter_rows(named=True):
            if row["grch38_reference_start"] and row["CO_active_interval_start"]:
                ss = BW.stats(
                    row["chrom"], 
                    row["grch38_reference_start"] + row["CO_active_interval_start"],
                    row["grch38_reference_start"] + row["CO_active_interval_end"],
                    "sum"
                )[0]
                sm = BW.stats(
                    row["chrom"], 
                    row["grch38_reference_start"] + row["CO_active_interval_start"],
                    row["grch38_reference_start"] + row["CO_active_interval_end"],
                    "mean"
                )[0]
                rows.append([row["read_name"], ss, sm])
            
    H3K4me3_df = pl.DataFrame(rows, schema=["read_name", "H3K4me3_signal_sum", "H3K4me3_signal_mean"])
    reads_df = (reads_df
        .join(
            H3K4me3_df, 
            on="read_name",
            how="left",
        )
    )

    # Add CTCF signal
    with pyBigWig.open(CTCF_filename) as BW:
        rows = []
        for row in reads_df.iter_rows(named=True):
            if row["grch38_reference_start"] and row["CO_active_interval_start"]:
                ss = BW.stats(
                    row["chrom"], 
                    row["grch38_reference_start"] + row["CO_active_interval_start"],
                    row["grch38_reference_start"] + row["CO_active_interval_end"],
                    "sum"
                )[0]
                sm = BW.stats(
                    row["chrom"], 
                    row["grch38_reference_start"] + row["CO_active_interval_start"],
                    row["grch38_reference_start"] + row["CO_active_interval_end"],
                    "mean"
                )[0]
                rows.append([row["read_name"], ss, sm])
            
    CTCF_df = pl.DataFrame(rows, schema=["read_name", "CTCF_signal_sum", "CTCF_signal_mean"])
    reads_df = (reads_df
        .join(
            CTCF_df, 
            on="read_name",
            how="left",
        )
    )

    # Add read quality
    reads_df = (reads_df
        .with_columns(
            is_high_quality_read = (
                (pl.col("mapq1") >= min_mapq) & \
                (pl.col("mapq2") >= min_mapq) & \
                (pl.col("is_forward1") == pl.col("is_forward2")) & \
                (pl.col("total_mismatches") <= max_total_mismatches) & \
                (pl.col("total_clipping") <= max_total_clipping)
            )
        )
    )

    # Join with candidate classes
    cls_df = pl.read_parquet(classified_reads_filename)
    
    reads_df = (reads_df
        .join(
            cls_df.select("read_name", "class", "snp_positions_on_read", "idx_transitions", "high_quality_classification"),
            on="read_name",
            how="left",
        )
        .with_columns(
            high_quality_classification_class = pl.when("high_quality_classification").then("class")
        )                
    )

    # Re-call the class only on high quality SNPs, only in the active region, 
    # but use the filtering of reads to avoid problems like fake COs from low coverage etc.
    reads_df = (reads_df    
        .join(
            (reads_df
                .with_columns(
                    high_quality_snps_diff = pl.col("high_quality_snp_positions_alleles").list.diff(null_behavior="drop"),
                    mid_quality_snps_diff = pl.col("mid_quality_snp_positions_alleles").list.diff(null_behavior="drop"),
                )
                .with_columns(
                    high_quality_snps_transitions = pl.col("high_quality_snps_diff").list.eval(pl.element() != 0).cast(pl.List(int)),        
                    mid_quality_snps_transitions = pl.col("mid_quality_snps_diff").list.eval(pl.element() != 0).cast(pl.List(int)),        
                )
                .with_columns(
                    high_quality_snps_idx_transitions = pl.col("high_quality_snps_transitions").list.eval(pl.arg_where(pl.element() != 0)),
                    mid_quality_snps_idx_transitions = pl.col("mid_quality_snps_transitions").list.eval(pl.arg_where(pl.element() != 0)),
                )
                .with_columns(
                    high_quality_classification_in_detectable_class = pl.when( 
                        (pl.col("high_quality_classification_class").is_not_null()) &
                        (pl.col("high_quality_snp_positions_alleles").list.get(0) == pl.col("high_quality_snp_positions_alleles").list.get(1)) &
                        (pl.col("high_quality_snp_positions_alleles").list.get(-1) == pl.col("high_quality_snp_positions_alleles").list.get(-2))
                    ).then(
                        pl.when(
                            pl.col("high_quality_snps_transitions").list.sum() == 1
                        ).then(
                            pl.lit("CO")
                        ).when(
                            pl.col("high_quality_snps_transitions").list.sum() == 2
                        ).then(
                            pl.lit("NCO")
                        )
                    )
                )
                .select(
                    "read_name",
                    "high_quality_snps_idx_transitions", 
                    "mid_quality_snps_idx_transitions",
                    "high_quality_classification_in_detectable_class",
                )
            ),
            on="read_name",
        )        
    )       

    return reads_df