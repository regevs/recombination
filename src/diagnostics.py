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

from . import annotate_old


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
    # is_forward1 = {}
    # is_forward2 = {}
    # mapq1 = {}
    # mapq2 = {}
    # ref_start1 = {}
    # ref_start2 = {}
    # base_qual1 = {}
    # base_qual2 = {}
    # ref_seq1 = {}
    # ref_seq2 = {}

    s, e = None, None 

    # for denovo_hap1_read_aln in tqdm.tqdm(denovo_hap1_bam.fetch(denovo_chrom, s, e)):
    #     read_name = denovo_hap1_read_aln.query_name
    #     read_lengths1[read_name] = denovo_hap1_read_aln.query_length
    #     cigar_tuples1[read_name] = denovo_hap1_read_aln.cigartuples
    #     is_forward1[read_name] = denovo_hap1_read_aln.is_forward
    #     mapq1[read_name] = denovo_hap1_read_aln.mapping_quality
    #     ref_start1[read_name] = denovo_hap1_read_aln.reference_start
    #     base_qual1[read_name] = np.array(denovo_hap1_read_aln.query_qualities)
    #     ref_seq1[read_name] = np.array(list(denovo_hap1_read_aln.get_reference_sequence()))
        
    # for denovo_hap2_read_aln in tqdm.tqdm(denovo_hap2_bam.fetch(denovo_chrom, s, e)):
    #     read_name = denovo_hap2_read_aln.query_name
    #     read_lengths2[read_name] = denovo_hap2_read_aln.query_length
    #     cigar_tuples2[read_name] = denovo_hap2_read_aln.cigartuples
    #     is_forward2[read_name] = denovo_hap2_read_aln.is_forward
    #     mapq2[read_name] = denovo_hap2_read_aln.mapping_quality
    #     ref_start2[read_name] = denovo_hap2_read_aln.reference_start
    #     base_qual2[read_name] = np.array(denovo_hap2_read_aln.query_qualities)
    #     ref_seq2[read_name] = np.array(list(denovo_hap2_read_aln.get_reference_sequence()))
        
    print("Refine reads..")
    for denovo_hap1_read_aln in tqdm.tqdm(denovo_hap1_bam.fetch(denovo_chrom, s, e)):
        read_name = denovo_hap1_read_aln.query_name
        read_lengths1[read_name] = denovo_hap1_read_aln.query_length
        cigar_tuples1[read_name] = denovo_hap1_read_aln.cigartuples
        
    for denovo_hap2_read_aln in tqdm.tqdm(denovo_hap2_bam.fetch(denovo_chrom, s, e)):
        read_name = denovo_hap2_read_aln.query_name
        read_lengths2[read_name] = denovo_hap2_read_aln.query_length
        cigar_tuples2[read_name] = denovo_hap2_read_aln.cigartuples
        
    s1, s2 = set(cigar_tuples1.keys()), set(cigar_tuples2.keys())
    common_reads_set = s1 & s2
    common_reads = list(common_reads_set)
    common_reads_to_index = dict([(read_name, i) for i, read_name in enumerate(common_reads)])    

    def runme(read_name):
        jtdf = refine_cigartuples(
            cigar_tuples1[read_name], 
            cigar_tuples2[read_name],
            max(read_lengths1[read_name], read_lengths2[read_name])
        )
        
        jtdf = jtdf.with_columns(
            pl.lit(read_name).alias("read_name"),
            pl.lit(read_lengths1[read_name]).alias("read_length1"),
            pl.lit(read_lengths2[read_name]).alias("read_length2"),
            # pl.lit(is_forward1[read_name]).alias("is_forward1"),
            # pl.lit(is_forward2[read_name]).alias("is_forward2"),
            # pl.lit(mapq1[read_name]).alias("mapq1"),
            # pl.lit(mapq2[read_name]).alias("mapq2"),
            # (pl.col("ref1_start") + ref_start1[read_name]).alias("ref1_start"),
            # (pl.col("ref1_end") + ref_start1[read_name]).alias("ref1_end"),
            # (pl.col("ref2_start") + ref_start2[read_name]).alias("ref2_start"),
            # (pl.col("ref2_end") + ref_start2[read_name]).alias("ref2_end"),
            # (pl.Series(name="qual_start1", values=base_qual1[read_name][np.array(jtdf["start"])])),
            # (pl.Series(name="qual_start2", values=base_qual2[read_name][np.array(jtdf["start"])])),
            # (pl.Series(name="ref_start1", values=ref_seq1[read_name][np.array(jtdf["start"])])),
            # (pl.Series(name="ref_start2", values=ref_seq2[read_name][np.array(jtdf["start"])])),
        )
        
        return jtdf

    # alldfs = []
    # for read_name in common_reads:
    #     print(read_name)
    #     alldfs.append(runme(read_name))

    alldfs = joblib.Parallel(n_jobs=n_threads, verbose=1, backend="threading")(
        joblib.delayed(runme)(read_name) for read_name in common_reads
    )

    print("Add columns to refined reads...")
    for denovo_hap1_read_aln in tqdm.tqdm(denovo_hap1_bam.fetch(denovo_chrom, s, e)):
        read_name = denovo_hap1_read_aln.query_name
        try:
            if read_name in common_reads_set:
                i = common_reads_to_index[read_name]
                jtdf = alldfs[i]
                alldfs[i] = jtdf.with_columns(
                    pl.lit(denovo_hap1_read_aln.is_forward).alias("is_forward1"),
                    pl.lit(denovo_hap1_read_aln.mapping_quality).alias("mapq1"),
                    pl.lit(denovo_hap1_read_aln.reference_name).alias("ref1_name"),
                    pl.Series(name="qual_start1", values=np.array(denovo_hap1_read_aln.query_qualities)[np.array(jtdf["start"])]),
                    pl.Series(name="refseq_start1", values=np.array(list(denovo_hap1_read_aln.get_reference_sequence()) + ['-'])[np.array(jtdf["ref1_start"])]).str.to_uppercase(),
                    (pl.col("ref1_start") + denovo_hap1_read_aln.reference_start).alias("ref1_start"),
                    (pl.col("ref1_end") + denovo_hap1_read_aln.reference_start).alias("ref1_end"),                    
                )
        except:
            print("prob at hap1", read_name)
            raise

    for denovo_hap2_read_aln in tqdm.tqdm(denovo_hap2_bam.fetch(denovo_chrom, s, e)):
        read_name = denovo_hap2_read_aln.query_name
        try:
            if read_name in common_reads_set:
                i = common_reads_to_index[read_name]
                jtdf = alldfs[i]
                alldfs[i] = jtdf.with_columns(
                    pl.lit(denovo_hap2_read_aln.is_forward).alias("is_forward2"),
                    pl.lit(denovo_hap2_read_aln.mapping_quality).alias("mapq2"),
                    pl.lit(denovo_hap2_read_aln.reference_name).alias("ref2_name"),
                    pl.Series(name="qual_start2", values=np.array(denovo_hap2_read_aln.query_qualities)[np.array(jtdf["start"])]),
                    pl.Series(name="refseq_start2", values=np.array(list(denovo_hap2_read_aln.get_reference_sequence()) + ['-'])[np.array(jtdf["ref2_start"])]).str.to_uppercase(),
                    (pl.col("ref2_start") + denovo_hap2_read_aln.reference_start).alias("ref2_start"),
                    (pl.col("ref2_end") + denovo_hap2_read_aln.reference_start).alias("ref2_end"),
                )
        except:
            print("prob at hap 2", read_name)
            raise
    print("Concat...")
    cdf = pl.concat(alldfs, how="vertical")
    
    return cdf

def filter_read_refinements(
    events_df,
    # min_mapq = 60,
):
    # Take only reads where both are mapped to the forward strand. TODO: include others?
    # events_df = events_df.filter(pl.col("is_forward1") & pl.col("is_forward2"))

    # Take only reads that have perfect mapping quality. TODO: include others?
    # events_df = events_df.filter((pl.col("mapq1") >= min_mapq) & (pl.col("mapq2") >= min_mapq))

    # Take only reads that match to the same chromosome 
    events_df = events_df.filter(pl.col("ref1_name") == pl.col("ref2_name"))

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
    )

    # Record the number of common insertions
    events_df = (events_df
        .join(
            (events_df
                .filter((pl.col("op1") == 1) & (pl.col("op2") == 1))
                .group_by("read_name")
                .agg(
                    pl.col("length").sum().alias("total_common_insertions"),
                    pl.col("length").len().alias("num_common_insertions"),
                )
            ),
            on="read_name",
            how="left",
        )
        .fill_null(0)
    )

    # Record the number of common deletions
    events_df = (events_df
        .join(
            (events_df
                .filter((pl.col("op1") == 2) & (pl.col("op2") == 2))
                .group_by("read_name")
                .agg(
                    pl.col("length").len().alias("num_common_deletions"),
                )
            ),
            on="read_name",
            how="left",
        )
        .fill_null(0)
    )

    # Record the number of any clippings
    events_df = (events_df
        .join(
            (events_df
                .filter(pl.col("op1").is_in([4,5]) | pl.col("op2").is_in([4,5]))
                .group_by("read_name")
                .agg(
                    pl.col("length").sum().alias("total_clipping"),
                    pl.col("length").len().alias("num_clipping"),
                )
            ),
            on="read_name",
            how="left",
        )
        .fill_null(0)
    )

    return events_df

def extract_high_conf_events(
    events_df,
    high_confidence_slack = 10,
):
    # Find SNPs
    events_df = (events_df
        .with_columns(
            ((pl.col("op1") != pl.col("op2")) & (pl.col("op1").is_in([7,8]) & pl.col("op2").is_in([7,8])) & (pl.col("length") == 1)).alias("is_snp"),
            ((pl.col("op1") == pl.col("op2")) & (pl.col("op1").is_in([8]) & pl.col("op2").is_in([8])) & (pl.col("length") == 1)).alias("is_two_sided_mismatch"),
        )
    )

    events_df = (events_df
        .with_columns(
            (pl.col("is_snp") | pl.col("is_two_sided_mismatch")).alias("is_interesting_event"),
        )
    )
   
    # Find high confidence events
    high_confidence_events = (events_df
        .filter(pl.col("is_interesting_event"))
        .join(
            (events_df
                .select(["op1", "op2", "length", "read_name", "start", "end"])
                .filter((pl.col("op1") == pl.col("op2")) & (pl.col("op1") == 7) & (pl.col("length") >= high_confidence_slack))
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
                .filter((pl.col("op1") == pl.col("op2")) & (pl.col("op1") == 7) & (pl.col("length") >= high_confidence_slack))
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
        .with_columns(pl.lit(True).alias("is_flanked_event"))
    )

    events_df = (events_df
        .join(high_confidence_events, on=["read_name", "start", "end"], how="left")
        .with_columns(pl.col("is_flanked_event").fill_null(value=False))
    )

    events_df = events_df.filter(pl.col("is_flanked_event"))

    # Which haplotype fits better?
    events_df = (events_df
        .with_columns(
            (pl.col("op1").is_in([0, 7]) | pl.col("op1").is_null()).cast(pl.Int32).alias("fits1"),
            (pl.col("op2").is_in([0, 7]) | pl.col("op2").is_null()).cast(pl.Int32).alias("fits2"),        
        ) 
        .with_columns(
            (pl.col("fits1") > pl.col("fits2")).cast(pl.Int32).alias("fits1_more"),
        )
        .drop(columns=["fits1", "fits2"])
    )

    return events_df


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
#             .select([ref_start_column_name])#             
#             .unique([ref_start_column_name])
#             .sort(by=ref_start_column_name)
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
    hap_and_certainty_to_bedgraph,
    condition,
):
    #snps_df = snps_df.lazy()

    # Add information about coverage
    for k, bedgraph_filename in hap_and_certainty_to_bedgraph.items():
        haplotype, certainty = k
        
        ref_start_column_name = f"ref{haplotype}_start"
        coverage_column_name = f"hap{haplotype}_certainty_{certainty}_coverage"
        
        # Open the coverage bedgraph
        covdf = pl.read_csv(
            bedgraph_filename,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start_pos_0based", "end_pos_0based", "coverage"],
        )

        # Create a dataframe that shows the coverage for SNPs
        snp_cov_df = (snps_df
            .filter(condition)
            .select([ref_start_column_name])
            .unique([ref_start_column_name])
            .sort(by=ref_start_column_name)
            .set_sorted(ref_start_column_name)
            .join_asof(
                (covdf                    
                    .select(["start_pos_0based", "end_pos_0based", "coverage"])
                    .set_sorted("start_pos_0based")
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
                    #.with_columns(pl.lit(True).alias("is_high_conf_snp"))            
                ),
                #on=[ref_start_column_name, "is_high_conf_snp"],
                on=[ref_start_column_name],
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
    condition,
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
            .filter(condition)
            .select([ref_start_column_name])
            .unique([ref_start_column_name])
            .sort(by=ref_start_column_name)
            .collect(streaming=True)
            .set_sorted(ref_start_column_name)
            .join_asof(
                (trf_df                    
                    .select(["start_pos_0based", "end_pos_0based", "repeat_length", "n_copies"])
                    .collect(streaming=True)
                    .set_sorted("start_pos_0based")
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
                (snp_repeat_cov_df.lazy()
                    .select([ref_start_column_name, "repeat_length", "n_copies"])
                    #.with_columns(pl.lit(True).alias("is_high_conf_snp"))            
                ),
                #on=[ref_start_column_name, "is_high_conf_snp"],
                on=[ref_start_column_name],
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

        #events_df = events_df.collect(streaming=True)

    return events_df


sdust_columns = ["chrom", "start_pos_0based", "end_pos_0based"]
def add_sdust_annotation(
    events_df,
    sdust_hap1_filename,
    sdust_hap2_filename,
    condition,
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
            .filter(condition)
            .select([ref_start_column_name])
            .unique([ref_start_column_name])
            .sort(by=ref_start_column_name)
            .collect(streaming=True)
            .set_sorted(ref_start_column_name)
            .join_asof(
                (sdust_df                    
                    .select(["start_pos_0based", "end_pos_0based"])
                    .with_columns(pl.lit(1).alias("sdust_repeat"))
                    .collect(streaming=True)
                    .set_sorted("start_pos_0based")
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
                (snp_repeat_cov_df.lazy()
                    .select([ref_start_column_name, "sdust_repeat"])
                    #.with_columns(pl.lit(True).alias("is_high_conf_snp"))            
                ),
                #on=[ref_start_column_name, "is_high_conf_snp"],
                on=[ref_start_column_name],
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

def add_allele_coverage_annotation(
    events_df,
    condition,
):
    df1 = (events_df
        .filter(condition)
        .filter("is_snp")
        .group_by("ref1_start", "op1")
        .len()
        .group_by("ref1_start")
        .agg(allele_coverage_hap1 = pl.col("len").min())
    )

    df2 = (events_df
        .filter(condition)
        .filter("is_snp")
        .group_by("ref2_start", "op2")
        .len()
        .group_by("ref2_start")
        .agg(allele_coverage_hap2 = pl.col("len").min())
    )

    events_df = (events_df
        .join(
            df1,
            on=["ref1_start"],
            how="left",
        )
        .join(
            df2,
            on=["ref2_start"],
            how="left",
        )
    )

    return events_df    

def add_high_confidence_annotation(
    events_df,
    output_column_prefix = "is_high_conf",
    base_qual_min = 60,
    read_trimming = 1500,
):
    event_column_name = output_column_prefix + "_event"
    snp_column_name = output_column_prefix + "_snp"

    events_df = events_df.with_columns(
        ((pl.col("is_flanked_event") == True) & \
         ((pl.col("trf_repeat_length_hap1") == 0) & (pl.col("trf_repeat_length_hap2") == 0)) & \
         ((pl.col("sdust_repeat_length_hap1") == 0) & (pl.col("sdust_repeat_length_hap2") == 0)) & 
         ((pl.col("start") > read_trimming) & (pl.col("end") < (pl.col("read_length1") - read_trimming))) & 
         ((pl.col("qual_start1") >= base_qual_min) & (pl.col("qual_start2") >= base_qual_min))).alias(event_column_name)
    )

    events_df = events_df.with_columns(
        (pl.col(event_column_name) & pl.col("is_snp")).alias(snp_column_name)
    )

    return events_df

def add_high_quality_annotation(
    events_df,
    input_column_prefix = "is_high_conf",
    output_column_prefix = "is_high_quality",
    phased_coverage_min = 3,
    allele_coverage_min = 3,
):
    events_df = events_df.with_columns((
            pl.col(input_column_prefix + "_event") & \
            (pl.col("hap1_certainty_0.95_coverage") >= phased_coverage_min) & 
            (pl.col("hap2_certainty_0.95_coverage") >= phased_coverage_min) &
            (pl.col("allele_coverage_hap1") >= allele_coverage_min) &
            (pl.col("allele_coverage_hap2") >= allele_coverage_min)
        ).alias(output_column_prefix + "_event")
    )
    
    events_df = events_df.with_columns(
        (pl.col(output_column_prefix + "_event") & pl.col(input_column_prefix + "_snp")).alias(output_column_prefix + "_snp")    )
    

    return events_df
    
# ----------------------------------------------------------------------------------------------------
# Classify reads
#
def classify_read(events_df):
    # Only look at high/mid quality SNPs
    #snp_subset_df = events_df.filter(pl.col("is_high_quality_snp"))
    snp_subset_df = events_df.filter(pl.col("is_mid_quality_snp"))
    
    # Look at transitions
    transitions = np.diff(snp_subset_df["fits1_more"])!=0
    n_transitions = np.sum(transitions)
    
    min_coverage_1 = None
    min_coverage_2 = None
    if n_transitions == 0:
        what = "None"
        idx_transitions = []
    else:
        idx_transitions = np.where(transitions)[0]
        first_trans = int(idx_transitions[0])
        last_trans = int(idx_transitions[-1])

        # Find the minimal coverages between the first and last transitions
        first_trans_snp_index = (events_df
            .with_row_count()
            .filter(
                (pl.col("start") == snp_subset_df.get_column("start")[first_trans]) & 
                (pl.col("end") == snp_subset_df.get_column("end")[first_trans])
            )
            .get_column("row_nr")[0]
        )

        last_trans_snp_index = (events_df
            .with_row_count()
            .filter(
                (pl.col("start") == snp_subset_df.get_column("start")[last_trans+1]) & 
                (pl.col("end") == snp_subset_df.get_column("end")[last_trans+1])
            )
            .get_column("row_nr")[0]
        )

        if "min_coverage_hap1" in events_df.columns:
            min_coverage_1 = (
                events_df[first_trans_snp_index:last_trans_snp_index+1]
                .filter(pl.col("min_coverage_hap1").is_not_null())
                ["min_coverage_hap1"].min()
            )

        if "min_coverage_hap2" in events_df.columns:
            min_coverage_2 = (
                events_df[first_trans_snp_index:last_trans_snp_index+1]
                .filter(pl.col("min_coverage_hap2").is_not_null())
                ["min_coverage_hap2"].min()
            )

        if n_transitions == 1:
            if first_trans > 0 and last_trans < len(snp_subset_df)-2:
                what = "CO"
            else:
                what = "ambiguous"
        else:
            if n_transitions == 2:
                what = "GC"
            else:
                what = "CNCO"
        
    
    res_df = pl.DataFrame(
        data={
            "read_name": [snp_subset_df["read_name"][0]], 
            "read_length": [snp_subset_df["read_length1"][0]],
            "n_transitions": [n_transitions], 
            "idx_transitions": [idx_transitions],
            "snp_positions_on_read": [snp_subset_df["start"]],
            "class": [what],
            "total_mismatches": [snp_subset_df["total_mismatches"][0]],
            "total_common_insertions": [snp_subset_df["total_common_insertions"][0]],
            "num_common_insertions": [snp_subset_df["num_common_insertions"][0]],
            "num_common_deletions": [snp_subset_df["num_common_deletions"][0]],
            "total_clipping": [snp_subset_df["total_clipping"][0]],
            "num_clipping": [snp_subset_df["num_clipping"][0]],
            "min_coverage_between_transitions_hap1": [min_coverage_1],
            "min_coverage_between_transitions_hap2": [min_coverage_2],
            "mapq1": [snp_subset_df["mapq1"][0]],
            "mapq2": [snp_subset_df["mapq2"][0]],
            "is_forward1": [snp_subset_df["is_forward1"][0]],
            "is_forward2": [snp_subset_df["is_forward2"][0]],
        },
        schema=[
            ('read_name', pl.String),
            ('read_length', pl.Int64),
            ('n_transitions', pl.Int64),
            ('idx_transitions', pl.List(pl.Int64)),
            ('snp_positions_on_read', pl.List(pl.Int64)),
            ('class', pl.String),
            ('total_mismatches', pl.Int64),
            ('total_common_insertions', pl.Int64),
            ('num_common_insertions', pl.Int64),
            ('num_common_deletions', pl.Int64),
            ('total_clipping', pl.Int64),
            ('num_clipping', pl.Int64),
            ('min_coverage_between_transitions_hap1', pl.Int64),
            ('min_coverage_between_transitions_hap2', pl.Int64),
            ('mapq1', pl.Int64),
            ('mapq2', pl.Int64),
            ('is_forward1', pl.Boolean),
            ('is_forward2', pl.Boolean)
        ],
    )
    
    return res_df

def classify_all_reads(
    snps_df,       
    candidates_df,
    cov1_df,
    cov2_df,
    high_quality_classification_condition = pl.lit(True),
):
    # If there are no candidates, return empty DF
    if len(candidates_df) == 0:
        return pl.DataFrame(
            schema=[
                ('read_name', pl.String),
                ('read_length', pl.Int64),
                ('n_transitions', pl.Int64),
                ('idx_transitions', pl.List(pl.Int64)),
                ('snp_positions_on_read', pl.List(pl.Int64)),
                ('class', pl.String),
                ('total_mismatches', pl.Int64),
                ('total_common_insertions', pl.Int64),
                ('num_common_insertions', pl.Int64),
                ('num_common_deletions', pl.Int64),
                ('total_clipping', pl.Int64),
                ('num_clipping', pl.Int64),
                ('min_coverage_between_transitions_hap1', pl.Int64),
                ('min_coverage_between_transitions_hap2', pl.Int64),
                ('has_common_transition', pl.Boolean),
                ('chrom', pl.String),
                ('sample_id', pl.String),
                ('mapq1', pl.Int64),
                ('mapq2', pl.Int64),
                ('is_forward1', pl.Boolean),
                ('is_forward2', pl.Boolean),
                ('high_quality_classification', pl.Boolean),
            ]
        )

    # Take subsets of SNPs that belong only to candidate reads, and are of high quality
    cand_snps_df = (snps_df
        .join(candidates_df, on="read_name")        
        .filter("is_mid_quality_snp")   #.filter("is_high_quality_snp")
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
    )

    classified_df = (classified_df
        .join(cov1_df, on=["read_name", "start", "end"], how="left")
        .join(cov2_df, on=["read_name", "start", "end"], how="left")
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

    # Add high quality classification
    classified_df = (classified_df
        .with_columns(
            high_quality_classification = high_quality_classification_condition,
        )
    )

    return classified_df
