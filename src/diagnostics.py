import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines
import seaborn as sns
from pathlib import Path
import re
import sys
import joblib
import polars as pl

import tqdm
import pysam
import fastq as fq

from . import annotate


# sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/hapfusion/src")
# import hapfusion
# from hapfusion import bamlib

# sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/himut/src")
# import himut

def convert_quality_string_to_array(quality_string):
    offset = 33  # Offset value for ASCII characters in FASTQ format
    quality_array = np.frombuffer(quality_string.encode(), dtype=np.uint8)
    return quality_array - offset


def base_quality_scores_plot(
    sample_id,
    focal_event,
    ax,
):
    fastq_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/01.data/02.ccs") \
        / sample_id / f"{sample_id}.ccs.filtered.fastq.gz"
    # alignment_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds") \
    #     / sample_id / f"{sample_id}.minimap2.primary_alignments.sorted.bam"
    alignment_bam_file = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13") \
         / sample_id / f"chm13.{sample_id}.minimap2.primary_alignments.sorted.bam"
    
    for fo in fq.read(fastq_file):
        if fo.getHead() == "@" + focal_event.read_name:
            break

    read_sequencing_qualities = convert_quality_string_to_array(fo.getQual())

    for aln in pysam.AlignmentFile(alignment_bam_file):
        if aln.query_name == focal_event.read_name:
            ccs = bamlib.BAM(aln)
            ccs.get_cs2tpos2qbase()
            break

    hetsnp_positions = np.array(re.findall(r"chr[\d+]:(\d+)_", focal_event["hetsnps"])).astype(int)
    hetsnp_positions_in_read_coords_0based = [ccs.rpos2qpos[x] for x in hetsnp_positions]

    # Plot
    colors = []
    alphas = []
    sizes = []
    for pos_0based in np.arange(aln.query_length):
        if pos_0based in hetsnp_positions_in_read_coords_0based:
            if np.array(list(focal_event.ccs_hbit))[np.where(hetsnp_positions_in_read_coords_0based == pos_0based)[0][0]] == '0':
                colors.append("C1")
                alphas.append(1)
                sizes.append(10)
            else:
                colors.append("C2")
                alphas.append(1)
                sizes.append(10)
        else:
            colors.append("C0")
            alphas.append(1)
            sizes.append(0.01)

    ax.scatter(
        np.arange(aln.query_length),
        read_sequencing_qualities,
        alpha=alphas,
        c=np.array(colors),
        s=sizes,
    )

    ax.set_title("Calling quality scores at SNPs");
    ax.set_xlabel("Position");
    ax.set_ylabel("Base quality");
    ax.legend(handles=[
        matplotlib.lines.Line2D([0],[0], color='C0', label='Hom', marker='o', markersize=1, linestyle=''),
        matplotlib.lines.Line2D([0],[0], color='C1', label='SNP like haplotype 0', marker='o', markersize=5, linestyle=''),
        matplotlib.lines.Line2D([0],[0], color='C2', label='SNP like haplotype 1', marker='o', markersize=5, linestyle=''),
    ])

def diagnose_events(
    sample_id,
    focal_event,
):
    #
    fig, ax = plt.subplots(figsize=(9, 6))

    base_quality_scores_plot(
        sample_id,
        focal_event,
        ax=ax,
    )

#
# This function takes the cigar strings of a read aligned to two separate haplotype referencs
# and returns the partitioning of the read according to the two alignments
#
def refine_cigartuples(
    cigar_tuples_hap1, 
    cigar_tuples_hap2,
    read_length,
):
    consumes_query = [0, 1, 4, 7, 8]
    consumes_ref = [0, 2, 3, 7, 8]
    
    joint_tuples = []
    hap1_index_pointer = 0
    hap2_index_pointer = 0
    read_pointer = 0
    ref1_pointer = 0
    ref2_pointer = 0 
    
    n_nucs1 = cigar_tuples_hap1[hap1_index_pointer][1]
    n_nucs2 = cigar_tuples_hap2[hap2_index_pointer][1]
    
    op1 = cigar_tuples_hap1[hap1_index_pointer][0]
    op2 = cigar_tuples_hap2[hap2_index_pointer][0]       
    
    while True:
        # READ: xxx[xxxx]xxx
        # HAP1: xxx[xxxx]xxx
        # READ: xxx[xxxx]xxx
        # HAP2: xxx[xxxx]xxx

        # READ: xxx[xxxx]xxx
        # HAP1: xxx[xxxx]xxx
        # READ: xxx[xxxx]xxx
        # HAP2: xxx[yyyy]xxx
        
        # READ: xxx[xxxx]xxx
        # HAP1: xxx[xxxx]xxx
        # READ: xxx[xxxx]xxx
        # HAP2: xxx[----]xxx
        if (op1 in consumes_query) and (op2 in consumes_query):
            min_nucs = min(n_nucs1, n_nucs2)

            cons1 = (min_nucs if op1 in consumes_ref else 0)
            cons2 = (min_nucs if op2 in consumes_ref else 0)

            joint_tuples.append([
                read_pointer, 
                read_pointer + min_nucs,
                min_nucs,
                op1, 
                op2,
                ref1_pointer,
                ref1_pointer + cons1,
                cons1,
                ref2_pointer,
                ref2_pointer + cons2,
                cons2,
                hap1_index_pointer,
                hap2_index_pointer,
            ])
            
            n_nucs1 -= min_nucs
            n_nucs2 -= min_nucs
            read_pointer += min_nucs
            ref1_pointer += cons1
            ref2_pointer += cons2

            
        
        # READ: xxxx[xxxx]x
        # HAP1: xxxx[xxxx]x
        # READ: xxxx[]xxxxx
        # HAP2: xxxx[yyyyy]xxxxx
        
        # READ: xxxx[xxxx]x
        # HAP1: xxxx[----]x
        # READ: xxxx[]xxxxx
        # HAP2: xxxx[yyyyy]xxxxx
        if (op1 in consumes_query) and (op2 not in consumes_query):
            cons1 = 0
            cons2 = n_nucs2

            joint_tuples.append([
                read_pointer, 
                read_pointer,
                0,
                None, 
                op2,
                ref1_pointer,
                ref1_pointer + cons1,
                cons1,
                ref2_pointer,
                ref2_pointer + cons2,
                cons2,
                hap1_index_pointer,
                hap2_index_pointer,
            ])
            
            n_nucs2 = 0
            ref1_pointer += cons1
            ref2_pointer += cons2

        # READ: xxxx[]xxxxx
        # HAP1: xxxx[yyyyy]xxxxx
        # READ: xxxx[xxxx]x
        # HAP2: xxxx[xxxx]x
        
        # READ: xxxx[]xxxxx
        # HAP1: xxxx[yyyyy]xxxxx
        # READ: xxxx[xxxx]x
        # HAP2: xxxx[----]x
        if (op1 not in consumes_query) and (op2 in consumes_query):
            cons1 = n_nucs1
            cons2 = 0

            joint_tuples.append([
                read_pointer, 
                read_pointer,
                0,
                op1, 
                None,
                ref1_pointer,
                ref1_pointer + cons1,
                cons1,
                ref2_pointer,
                ref2_pointer + cons2,
                cons2,
                hap1_index_pointer,
                hap2_index_pointer,
            ])
            
            n_nucs1 = 0
            ref1_pointer += cons1
            ref2_pointer += cons2
            
        # READ: xxxx[]xxxxx
        # HAP1: xxxx[yyy]xxxxx
        # READ: xxxx[]xxxxx
        # HAP2: xxxx[yyyyy]xxxxx
        if (op1 not in consumes_query) and (op2 not in consumes_query):
            min_nucs = min(n_nucs1, n_nucs2)

            cons1 = min_nucs
            cons2 = min_nucs
            
            joint_tuples.append([
                read_pointer, 
                read_pointer,
                0,
                op1, 
                op2,
                ref1_pointer,
                ref1_pointer + cons1,
                cons1,
                ref2_pointer,
                ref2_pointer + cons2,
                cons2,
                hap1_index_pointer,
                hap2_index_pointer,
            ])
            
            n_nucs1 -= min_nucs
            n_nucs2 -= min_nucs
            ref1_pointer += cons1
            ref2_pointer += cons2
            
        # If done, quit
        if read_pointer >= read_length:
            break
            
        # Advance if needed
        if n_nucs1 == 0: 
            hap1_index_pointer += 1
            op1 = cigar_tuples_hap1[hap1_index_pointer][0]
            n_nucs1 = cigar_tuples_hap1[hap1_index_pointer][1]
        if n_nucs2 == 0:
            hap2_index_pointer += 1
            op2 = cigar_tuples_hap2[hap2_index_pointer][0]
            n_nucs2 = cigar_tuples_hap2[hap2_index_pointer][1]
            
    jtdf = pl.DataFrame(
        joint_tuples, 
        schema=["start", "end", "length", "op1", "op2", "ref1_start", "ref1_end", "ref1_gap", "ref2_start", "ref2_end", "ref2_gap", "cigar_ptr1", "cigar_ptr2"], 
        orient="row",
    )

    return jtdf

def run_all_refine_cigars(
    denovo_hap1_alignment_bam_file,
    denovo_hap2_alignment_bam_file,
    denovo_chrom,  
    n_threads,    
):
    denovo_hap1_bam = pysam.AlignmentFile(denovo_hap1_alignment_bam_file)
    denovo_hap2_bam = pysam.AlignmentFile(denovo_hap2_alignment_bam_file)

    read_lengths1 = {}
    read_lengths2 = {}
    cigar_tuples1 = {}
    cigar_tuples2 = {}
    is_forward1 = {}
    is_forward2 = {}
    mapq1 = {}
    mapq2 = {}
    ref_start1 = {}
    ref_start2 = {}
    base_qual1 = {}
    base_qual2 = {}

    s, e = None, None 

    for denovo_hap1_read_aln in denovo_hap1_bam.fetch(denovo_chrom, s, e):
        read_name = denovo_hap1_read_aln.query_name
        read_lengths1[read_name] = denovo_hap1_read_aln.query_length
        cigar_tuples1[read_name] = denovo_hap1_read_aln.cigartuples
        is_forward1[read_name] = denovo_hap1_read_aln.is_forward
        mapq1[read_name] = denovo_hap1_read_aln.mapping_quality
        ref_start1[read_name] = denovo_hap1_read_aln.reference_start
        base_qual1[read_name] = np.array(denovo_hap1_read_aln.query_qualities)
        
    for denovo_hap2_read_aln in denovo_hap2_bam.fetch(denovo_chrom, s, e):
        read_name = denovo_hap2_read_aln.query_name
        read_lengths2[read_name] = denovo_hap2_read_aln.query_length
        cigar_tuples2[read_name] = denovo_hap2_read_aln.cigartuples
        is_forward2[read_name] = denovo_hap2_read_aln.is_forward
        mapq2[read_name] = denovo_hap2_read_aln.mapping_quality
        ref_start2[read_name] = denovo_hap2_read_aln.reference_start
        base_qual2[read_name] = np.array(denovo_hap2_read_aln.query_qualities)

    s1, s2 = set(cigar_tuples1.keys()), set(cigar_tuples2.keys())
    common_reads = list(s1 & s2)

    def runme(read_name):
        jtdf = refine_cigartuples(
            cigar_tuples1[read_name], 
            cigar_tuples2[read_name],
            max(read_lengths1[read_name], read_lengths2[read_name])
        )
        
        jtdf = jtdf.with_columns(
            pl.lit(read_name).alias("read_name"),
            pl.lit(is_forward1[read_name]).alias("is_forward1"),
            pl.lit(is_forward2[read_name]).alias("is_forward2"),
            pl.lit(read_lengths1[read_name]).alias("read_length1"),
            pl.lit(read_lengths2[read_name]).alias("read_length2"),
            pl.lit(mapq1[read_name]).alias("mapq1"),
            pl.lit(mapq2[read_name]).alias("mapq2"),
            (pl.col("ref1_start") + ref_start1[read_name]).alias("ref1_start"),
            (pl.col("ref1_end") + ref_start1[read_name]).alias("ref1_end"),
            (pl.col("ref2_start") + ref_start2[read_name]).alias("ref2_start"),
            (pl.col("ref2_end") + ref_start2[read_name]).alias("ref2_end"),
            (pl.Series(name="qual_start1", values=base_qual1[read_name][np.array(jtdf["start"])])),
            (pl.Series(name="qual_start2", values=base_qual2[read_name][np.array(jtdf["start"])])),
        )
        
        return jtdf

    # alldfs = []
    # for read_name in common_reads:
    #     print(read_name)
    #     alldfs.append(runme(read_name))

    alldfs = joblib.Parallel(n_jobs=n_threads, verbose=1, backend="threading")(
        joblib.delayed(runme)(read_name) for read_name in list(common_reads)
    )

    print("Concat...")
    cdf = pl.concat(alldfs, how="vertical")
    
    return cdf

def filter_read_refinements(
    events_df,
    min_mapq = 60,
    max_total_mismatches = 20,
):
    # Take only reads where both are mapped to the forward strand. TODO: include others?
    events_df = events_df.filter(pl.col("is_forward1") & pl.col("is_forward2"))

    # Take only reads that have perfect mapping quality. TODO: include others?
    events_df = events_df.filter((pl.col("mapq1") >= min_mapq) & (pl.col("mapq2") >= min_mapq))

    # Take only reads without too many mismatches to both haplotypes (TODO: other indels?)
    events_df = (events_df
        .join(
            (events_df
                .filter((pl.col("op1") == 8) & (pl.col("op2") == 8))
                .group_by("read_name")
                .agg(
                    pl.col("length").sum().alias("total_mismatches"),
                )
            ),
            on="read_name",
            how="left",
        )
        .fill_null(0)
        .filter(pl.col("total_mismatches") <= max_total_mismatches)
    )

    return events_df

def extract_snps(
    events_df,
    high_confidence_snp_slack = 10,
):
    # Find SNPs
    events_df = (events_df
        .with_columns(
            ((pl.col("op1") != pl.col("op2")) & (pl.col("op1").is_in([7,8]) & pl.col("op2").is_in([7,8])) & (pl.col("length") == 1)).alias("is_snp")
        )
    )
   
    # Find high confidence SNPs
    high_confidence_snps = (events_df
        .filter(pl.col("is_snp"))
        .join(
            (events_df
                .select(["op1", "op2", "length", "read_name", "start", "end"])
                .filter((pl.col("op1") == pl.col("op2")) & (pl.col("op1") == 7) & (pl.col("length") >= high_confidence_snp_slack))
            ),
            left_on=["start", "read_name"],
            right_on=["end", "read_name"],
        )
        .join(
            events_df.filter(pl.col("length") == 0),        
            left_on=["start", "read_name"],
            right_on=["end", "read_name"],
            how="anti",
        )
        .drop(["op1_right", "op2_right", "length_right", "start_right"])
        .join(
            (events_df
                .select(["op1", "op2", "length", "read_name", "start", "end"])
                .filter((pl.col("op1") == pl.col("op2")) & (pl.col("op1") == 7) & (pl.col("length") >= high_confidence_snp_slack))
            ),
            left_on=["end", "read_name"],
            right_on=["start", "read_name"],
        )
        .join(
            events_df.filter(pl.col("length") == 0),        
            left_on=["end", "read_name"],
            right_on=["start", "read_name"],
            how="anti",
        )
        .select(["read_name", "start", "end"])
        .with_columns(pl.lit(True).alias("is_high_conf_snp"))
    )

    events_df = (events_df
        .join(high_confidence_snps, on=["read_name", "start", "end"], how="left")
        .with_columns(pl.col("is_high_conf_snp").fill_null(value=False))
    )

    snp_df = events_df.filter(pl.col("is_high_conf_snp"))

    # Which haplotype fits better?
    snp_df = (snp_df
        .with_columns(
            (pl.col("op1").is_in([0, 7]) | pl.col("op1").is_null()).cast(pl.Int32).alias("fits1"),
            (pl.col("op2").is_in([0, 7]) | pl.col("op2").is_null()).cast(pl.Int32).alias("fits2"),        
        ) 
        .with_columns(
            (pl.col("fits1") > pl.col("fits2")).cast(pl.Int32).alias("fits1_more"),
        )
        .drop(columns=["fits1", "fits2"])
    )

    return snp_df


def snps_to_read_stats(
    snps_df,
    filter,
    output_column_name,
):
    stats_df = (snps_df
        .filter(filter)
        .group_by("read_name")
        .agg(
            pl.col("fits1_more").mean().alias(output_column_name)
        )
    )

    return stats_df


# def estimate_phase_per_read(
#     read_refinement_filename,
#     min_mapq=60,
#     high_confidence_snp_slack = 10,
#     hap_and_certainty_to_bedgraph = {},
# ):
#     # Read the joint parquet
#     pdf = pl.scan_parquet(
#         read_refinement_filename,
#     )
    
#     print("Creating stats...")
#     # Take only reads where both are mapped to the forward strand. TODO: include others?
#     fdf = pdf.filter(pl.col("is_forward1") & pl.col("is_forward2"))

#     # Take only reads that have perfect mapping quality. TODO: include others?
#     fdf = fdf.filter((pl.col("mapq1") >= min_mapq) & (pl.col("mapq2") >= min_mapq))

#     # Create list of high confidence SNPs
#     high_confidence_snps = (fdf
#         .filter((pl.col("op1") != pl.col("op2")) & (pl.col("op1").is_in([7,8]) & pl.col("op2").is_in([7,8])) & (pl.col("length") == 1))
#         .join(
#             (fdf
#                 .select(["op1", "op2", "length", "read_name", "start", "end"])
#                 .filter((pl.col("op1") == pl.col("op2")) & (pl.col("op1") == 7) & (pl.col("length") >= high_confidence_snp_slack))
#             ),
#             left_on=["start", "read_name"],
#             right_on=["end", "read_name"],
#         )
#         .join(
#             fdf.filter(pl.col("length") == 0),        
#             left_on=["start", "read_name"],
#             right_on=["end", "read_name"],
#             how="anti",
#         )
#         .drop(["op1_right", "op2_right", "length_right", "start_right"])
#         .join(
#             (fdf
#                 .select(["op1", "op2", "length", "read_name", "start", "end"])
#                 .filter((pl.col("op1") == pl.col("op2")) & (pl.col("op1") == 7) & (pl.col("length") >= high_confidence_snp_slack))
#             ),
#             left_on=["end", "read_name"],
#             right_on=["start", "read_name"],
#         )
#         .join(
#             fdf.filter(pl.col("length") == 0),        
#             left_on=["end", "read_name"],
#             right_on=["start", "read_name"],
#             how="anti",
#         )
#         .select(["read_name", "start", "end"])
#         .with_columns(pl.lit(1).alias("is_high_conf_snp"))
#     )

#     # Does each segment fit hap1 and/or hap2
#     tmpdf = (fdf
#         .with_columns(
#             (pl.col("op1").is_in([0, 7]) | pl.col("op1").is_null()).cast(pl.Int32).alias("fits1"),
#             (pl.col("op2").is_in([0, 7]) | pl.col("op2").is_null()).cast(pl.Int32).alias("fits2"),        
#         ) 
#         .with_columns(
#             (pl.col("fits1") > pl.col("fits2")).cast(pl.Int32).alias("fits1_more"),
#         )
#         .join(high_confidence_snps, on=["read_name", "start", "end"], how="left")
#         .with_columns(pl.col("is_high_conf_snp").fill_null(value=0))
#     )

#     # Collect
#     tmpdf = tmpdf.collect(streaming=True).lazy()

#     print("Adding coverage...")
#     # Add information about coverage
#     for k, bedgraph_filename in hap_and_certainty_to_bedgraph.items():
#         haplotype, certainty = k
        
#         ref_start_column_name = f"ref{haplotype}_start"
#         coverage_column_name = f"hap{haplotype}_certainty_{certainty}_coverage"
        
#         # Open the coverage bedgraph
#         covdf = pl.scan_csv(
#             bedgraph_filename,
#             separator="\t",
#             has_header=False,
#             new_columns=["chrom", "start_pos_0based", "end_pos_0based", "coverage"],
#         )

#         # Create a dataframe that shows the coverage for SNPs
#         snp_cov_df = (tmpdf
#             .filter(pl.col("is_high_conf_snp") == 1)
#             .select([ref_start_column_name])
#             .sort(by=ref_start_column_name)
#             .unique([ref_start_column_name])
#             .join_asof(
#                 (covdf
#                     .set_sorted("start_pos_0based")
#                     .select(["start_pos_0based", "end_pos_0based", "coverage"])
#                 ),
#                 left_on=ref_start_column_name,
#                 right_on="start_pos_0based",
#                 strategy="backward",
#             )
#             .filter(pl.col(ref_start_column_name) >= pl.col("start_pos_0based"))
#             .filter(pl.col(ref_start_column_name) < pl.col("end_pos_0based"))
#         )

#         # Add the column to the main df
#         tmpdf = (tmpdf
#             .join(
#                 (snp_cov_df
#                     .select([ref_start_column_name, "coverage"])
#                     .with_columns(pl.lit(1).alias("is_high_conf_snp"))            
#                 ),
#                 on=[ref_start_column_name, "is_high_conf_snp"],
#                 how="left",
#             )
#             .with_columns(pl.col("coverage").fill_null(0))
#             .rename({"coverage": coverage_column_name})
#         )

#     tmpdf = tmpdf.collect(streaming=True)

#     #
#     # Create the stats per read
#     #

#     print("Summarizing...")
#     # Count # of operaations that are the same or not
#     df1 = (tmpdf
#         .group_by("read_name")
#         .agg(
#             (pl.col("op1") == pl.col("op2")).mean().alias("frac_ops_same"),
#             (pl.col("op1") != pl.col("op2")).sum().alias("n_ops_different"),
#         )
#     )   

#     # Count what fraction of different operations fit hap1 more
#     df2 = (tmpdf
#         .filter(pl.col("op1") != pl.col("op2"))
#         .group_by("read_name")
#         .agg(
#             pl.col("fits1_more").mean().alias("frac_fits1_more"),        
#         )    
#     )

#     # The same, but only for mismatches (hopefully SNPs)
#     df3 = (tmpdf
#         .filter(pl.col("op1") != pl.col("op2"))
#         .filter(pl.col("op1").is_in([7,8]) & pl.col("op2").is_in([7,8]))
#         .group_by("read_name")
#         .agg(
#             pl.col("fits1_more").mean().alias("frac_fits1_more_snps"),        
#         )    
#     )

#     # The same, but for high confidence SNPs
#     df4 = (tmpdf
#         .filter(pl.col("op1") != pl.col("op2"))
#         .filter(pl.col("op1").is_in([7,8]) & pl.col("op2").is_in([7,8]))
#         .filter(pl.col("is_high_conf_snp") == 1)
#         .group_by("read_name")
#         .agg(
#             pl.col("fits1_more").mean().alias("frac_fits1_more_snps_high_conf"),        
#         )    
#     )

#     hap_stats_df = (df1
#         .join(df2, on="read_name", how="outer")
#         .join(df3, on="read_name", how="outer")
#         .join(df4, on="read_name", how="outer")
#     )

#     return tmpdf, hap_stats_df


def phase_and_haplotag(
    high_confidence_snps_filename,
    input_bam_hap1_filename,
    input_bam_hap2_filename,
    output_bam_hap1_filename,
    output_bam_hap2_filename,
    certainty_threshold=0.95,
):
    # events_df = pl.scan_parquet(read_refinement_filename)
    # events_df = filter_read_refinements(events_df, min_mapq)
    # snps_df = extract_snps(events_df, high_confidence_snp_slack)

    # snps_df = snps_df.collect(streaming=True)
    snps_df = pl.read_parquet(high_confidence_snps_filename)

    hap_stats_df = snps_to_read_stats(
        snps_df,
        pl.col("is_high_conf_snp"),
        "frac_fits1_more_snps_high_conf",    
    )

    read_to_frac = dict(hap_stats_df.select(pl.col("read_name"), pl.col("frac_fits1_more_snps_high_conf").fill_null(0.5)).to_numpy())
    
    # Write a new bam file with the new tags, haplotype 1
    print("Writing hap1 bam...")
    bam = pysam.AlignmentFile(input_bam_hap1_filename)
    with pysam.AlignmentFile(output_bam_hap1_filename, "wb", header=bam.header) as outf:
        for read in tqdm.tqdm(bam.fetch()):
            if read.query_name in read_to_frac.keys():
                frac = read_to_frac[read.query_name]

                hap_tag = 0
                if frac >= certainty_threshold:
                    hap_tag = 1
                
                read.tags += [('HP', hap_tag)]
                outf.write(read)

    # Write a new bam file with the new tags, haplotype 2
    print("Writing hap2 bam...")
    bam = pysam.AlignmentFile(input_bam_hap2_filename)
    with pysam.AlignmentFile(output_bam_hap2_filename, "wb", header=bam.header) as outf:
        for read in tqdm.tqdm(bam.fetch()):
            if read.query_name in read_to_frac.keys():
                frac = read_to_frac[read.query_name]

                hap_tag = 0
                if frac <= (1-certainty_threshold):
                    hap_tag = 2
                
                read.tags += [('HP', hap_tag)]
                outf.write(read)


def add_phasing_coverage_annotation(
    snps_df,
    hap_and_certainty_to_bedgraph
):
    snps_df = snps_df.lazy()

    # Add information about coverage
    for k, bedgraph_filename in hap_and_certainty_to_bedgraph.items():
        haplotype, certainty = k
        
        ref_start_column_name = f"ref{haplotype}_start"
        coverage_column_name = f"hap{haplotype}_certainty_{certainty}_coverage"
        
        # Open the coverage bedgraph
        covdf = pl.scan_csv(
            bedgraph_filename,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start_pos_0based", "end_pos_0based", "coverage"],
        )

        # Create a dataframe that shows the coverage for SNPs
        snp_cov_df = (snps_df
            .select([ref_start_column_name])
            .sort(by=ref_start_column_name)
            .unique([ref_start_column_name])
            .join_asof(
                (covdf
                    .set_sorted("start_pos_0based")
                    .select(["start_pos_0based", "end_pos_0based", "coverage"])
                ),
                left_on=ref_start_column_name,
                right_on="start_pos_0based",
                strategy="backward",
            )
            .filter(pl.col(ref_start_column_name) >= pl.col("start_pos_0based"))
            .filter(pl.col(ref_start_column_name) < pl.col("end_pos_0based"))
        )

        # Add the column to the main df
        snps_df = (snps_df
            .join(
                (snp_cov_df
                    .select([ref_start_column_name, "coverage"])
                    .with_columns(pl.lit(True).alias("is_high_conf_snp"))            
                ),
                on=[ref_start_column_name, "is_high_conf_snp"],
                how="left",
            )
            .with_columns(pl.col("coverage").fill_null(0))
            .rename({"coverage": coverage_column_name})
        )
        
    return snps_df

trf_columns = ["start_pos_1based", "end_pos_1based", "repeat_length", "n_copies", "concensus_length", "percent_matches", "percent_indels", "alignment_score", "percent_A", "percent_C", "percent_G", "percent_T", "entropy", "concensus", "full_repeat", "flank_seq1", "flank_seq2"]
def add_tandem_repeat_finder_annotation(
    events_df,
    trf_hap1_filename,
    trf_hap2_filename,
):
    trf_hap1_df = (
        pl.scan_csv(
            trf_hap1_filename,
            has_header = False,
            separator = " ",
            comment_prefix = "@",
            new_columns = trf_columns,
        )
        .with_columns(
            (pl.col("start_pos_1based") - 1).alias("start_pos_0based"),
            pl.col("end_pos_1based").alias("end_pos_0based")
        )
    )

    trf_hap2_df = (
        pl.scan_csv(
            trf_hap2_filename,
            has_header = False,
            separator = " ",
            comment_prefix = "@",
            new_columns = trf_columns,
        )
        .with_columns(
            (pl.col("start_pos_1based") - 1).alias("start_pos_0based"),
            pl.col("end_pos_1based").alias("end_pos_0based")
        )
    )

    # Create a dataframe that shows the TRF information for high confidence SNPs
    for haplotype in [1, 2]:
        ref_start_column_name = f"ref{haplotype}_start"
        trf_df = [trf_hap1_df, trf_hap2_df][haplotype-1]

        snp_repeat_cov_df = (events_df.lazy()
            .select([ref_start_column_name])
            .sort(by=ref_start_column_name)
            .unique([ref_start_column_name])
            .join_asof(
                (trf_df
                    .set_sorted("start_pos_0based")
                    .select(["start_pos_0based", "end_pos_0based", "repeat_length", "n_copies"])
                ),
                left_on=ref_start_column_name,
                right_on="start_pos_0based",
                strategy="backward",
            )
            .filter(pl.col(ref_start_column_name) >= pl.col("start_pos_0based"))
            .filter(pl.col(ref_start_column_name) < pl.col("end_pos_0based"))
        )

        # Add the column to the main df
        events_df = (events_df.lazy()
            .join(
                (snp_repeat_cov_df
                    .select([ref_start_column_name, "repeat_length", "n_copies"])
                    .with_columns(pl.lit(True).alias("is_high_conf_snp"))            
                ),
                on=[ref_start_column_name, "is_high_conf_snp"],
                how="left",
            )
            .with_columns(
                pl.col("repeat_length").fill_null(0), 
                pl.col("n_copies").fill_null(0)
            )
            .rename({
                "repeat_length": f"trf_repeat_length_hap{haplotype}",
                "n_copies": f"trf_n_copies_hap{haplotype}"
            })
        )

    return events_df


sdust_columns = ["chrom", "start_pos_0based", "end_pos_0based"]
def add_sdust_annotation(
    events_df,
    sdust_hap1_filename,
    sdust_hap2_filename,
):
    sdust_hap1_df = (
        pl.scan_csv(
            sdust_hap1_filename,
            has_header = False,
            separator = "\t",
            new_columns = sdust_columns,
        )
    )

    sdust_hap2_df = (
        pl.scan_csv(
            sdust_hap2_filename,
            has_header = False,
            separator = "\t",
            new_columns = sdust_columns,
        )
    )

    # Create a dataframe that shows the TRF information for high confidence SNPs
    for haplotype in [1, 2]:
        ref_start_column_name = f"ref{haplotype}_start"
        sdust_df = [sdust_hap1_df, sdust_hap2_df][haplotype-1]

        snp_repeat_cov_df = (events_df.lazy()
            .select([ref_start_column_name])
            .sort(by=ref_start_column_name)
            .unique([ref_start_column_name])
            .join_asof(
                (sdust_df
                    .set_sorted("start_pos_0based")
                    .select(["start_pos_0based", "end_pos_0based"])
                    .with_columns(pl.lit(1).alias("sdust_repeat"))
                ),
                left_on=ref_start_column_name,
                right_on="start_pos_0based",
                strategy="backward",
            )
            .filter(pl.col(ref_start_column_name) >= pl.col("start_pos_0based"))
            .filter(pl.col(ref_start_column_name) < pl.col("end_pos_0based"))
        )

        # Add the column to the main df
        events_df = (events_df.lazy()
            .join(
                (snp_repeat_cov_df
                    .select([ref_start_column_name, "sdust_repeat"])
                    .with_columns(pl.lit(True).alias("is_high_conf_snp"))            
                ),
                on=[ref_start_column_name, "is_high_conf_snp"],
                how="left",
            )
            .with_columns(
                pl.col("sdust_repeat").fill_null(0), 
            )
            .rename({
                "sdust_repeat": f"sdust_repeat_length_hap{haplotype}",
            })
        )

    return events_df


def add_high_confidence_annotation(
    events_df,
    base_qual_min = 60,
):
    return events_df.with_columns(
        (((pl.col("trf_repeat_length_hap1") == 0) & (pl.col("trf_repeat_length_hap2") == 0)) & \
         ((pl.col("sdust_repeat_length_hap1") == 0) & (pl.col("sdust_repeat_length_hap2") == 0)) & 
         ((pl.col("qual_start1") >= base_qual_min) & (pl.col("qual_start2") >= base_qual_min))).alias("is_high_conf_snp")
    )

def add_high_quality_annotation(
    events_df,
    phased_coverage_min = 3,
):
    return events_df.with_columns((
            pl.col("is_high_conf_snp") & \
            (pl.col("hap1_certainty_0.95_coverage") >= phased_coverage_min) & 
            (pl.col("hap2_certainty_0.95_coverage") >= phased_coverage_min)
        ).alias("is_high_quality_snp")
    )
    
# ----------------------------------------------------------------------------------------------------
# Classify reads
#
def classify_read(snp_subset_df):
    # Only look at high quality SNPs
    snp_subset_df = snp_subset_df.filter(pl.col("is_high_quality_snp"))
    
    # Look at transitions
    transitions = np.diff(snp_subset_df["fits1_more"])!=0
    n_transitions = np.sum(transitions)
    
    if n_transitions == 0:
        what = "None"
    else:
        idx_transitions = np.where(transitions)[0]
        first_trans = idx_transitions[0]
        last_trans = idx_transitions[-1]
        if n_transitions == 1:
            if first_trans > 0 and last_trans < len(snp_subset_df)-1:
                what = "CO"
            else:
                what = "ambiguous"
        else:
            if n_transitions == 2:
                what = "GC"
            else:
                what = "CNCO"
        
    
    res_df = pl.DataFrame(
        {
            "read_name": snp_subset_df["read_name"][0], 
            "n_transitions": n_transitions, 
            "class": what,                
        }
    )
    
    return res_df

def classify_all_reads(
    snps_df,       
    candidates_df,
):
    # Take subsets of SNPs that belong only to candidate reads, and are of high quality
    cand_snps_df = (snps_df
        .join(candidates_df, on="read_name")
        .filter("is_high_quality_snp")
        .sort(by=["read_name", "start"])
    )

    # Figure out the transition points
    nzi = np.nonzero(
        (np.diff(cand_snps_df["fits1_more"])!=0) & \
        (cand_snps_df["read_name"][:-1] == cand_snps_df["read_name"][1:]).to_numpy()
    )[0]

    trans_df = pl.DataFrame({
        "read_name": cand_snps_df[nzi]["read_name"],
        "first_hap1": cand_snps_df[nzi]["ref1_start"], 
        "second_hap1": cand_snps_df[nzi+1]["ref1_start"],
        "first_hap2": cand_snps_df[nzi]["ref2_start"], 
        "second_hap2": cand_snps_df[nzi+1]["ref2_start"],
    })

    common_trans_df = (trans_df
        .group_by(["first_hap1", "second_hap1", "first_hap2", "second_hap2"])
        .count()
        .sort(by="count", descending=True)
        .filter(pl.col("count") > 1)
    )

    # Find reads with common transitions
    reads_with_common_transitions_df = (trans_df
        .join(
            common_trans_df,
            on=["first_hap1", "second_hap1", "first_hap2", "second_hap2"],
        )
        [["read_name"]]
        .unique()
        .with_columns(pl.lit(True).alias("has_common_transition"))
    )

    # Classify reads, add common transition information
    classified_df = (snps_df
        .join(
            candidates_df,
            on="read_name",
        )
        .group_by("read_name")
        .map_groups(classify_read)
        .join(
            reads_with_common_transitions_df,
            on="read_name",
            how="left",
        )
        .with_columns(
            pl.col("has_common_transition").fill_null(False)
        ) 
    )

    return classified_df
