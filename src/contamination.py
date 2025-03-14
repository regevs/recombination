import numpy as np
from pathlib import Path
import re
import sys
import joblib
import polars as pl

import tqdm
import pysam
import fastq as fq

grch38_chromosome_lengths_in_bp = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr11": 135086622,
    "chr10": 133797422,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
}

def compare_read_to_reference(
    alignment_bam_filename,
    reference_panel_vcf_filename,
    chrom,
    start,
    end,
):
    rows = []

    # Load the reference panel
    reference_panel_vcf = pysam.VariantFile(reference_panel_vcf_filename)

    # Open the alignment file
    alignment_bam = pysam.AlignmentFile(alignment_bam_filename)

    # Go through the reads
    for read_alignment in alignment_bam.fetch(contig=chrom, start=start, end=end, multiple_iterators = True):
        # Get the read alignment start and end
        read_alignment_start = read_alignment.reference_start
        read_alignment_end = read_alignment.reference_end

        # Find SNPs in the reference panel that overlap with the read alignment
        snps_in_read = {}
        for record in reference_panel_vcf.fetch(chrom, read_alignment_start, read_alignment_end, reopen=True):
            # Save the AF of the SNP
            snps_in_read[record.pos-1] = record.info["AF"][0]

        aligned_pairs = read_alignment.get_aligned_pairs(with_seq=True, with_cigar=True)
        for read_pos, ref_pos, ref_seq, cigar_op in aligned_pairs:
            if ref_pos in snps_in_read.keys():
                rows.append([
                    read_alignment.query_name,
                    read_pos,
                    ref_pos,
                    ref_seq,
                    int(cigar_op),
                    snps_in_read[ref_pos],
                ])

    df = pl.DataFrame(
        rows,
        schema = ["read_name", "start", "ref_start", "ref_seq", "op", "allele_freq"],
        orient = "row",
    )

    return df


def compare_read_to_reference_chrom(
    alignment_bam_filename,
    reference_panel_vcf_filename,
    chrom,
    chunk_size = 10_000_000,
    n_threads = -1
):
    def runme(start, end):
        return compare_read_to_reference(
            alignment_bam_filename,
            reference_panel_vcf_filename,
            chrom,
            start,
            end,
        )
    
    # Split chromsome lengths into chunks
    chunks = []
    chrom_length = grch38_chromosome_lengths_in_bp[chrom]
    for start in range(0, chrom_length, chunk_size):
        end = min(start + chunk_size, chrom_length)
        chunks.append((start, end))

    # Run the function in parallel
    dfs = joblib.Parallel(n_jobs=n_threads, verbose=50)(
        joblib.delayed(runme)(start, end) for start, end in chunks
    )

    df = pl.concat([df for df in dfs if len(df) > 0])
    return df




def find_variation_on_reads(
    alignment_bam_filename,
    chrom,
    start,
    end,
):
    rows = []

    # Open the alignment file
    alignment_bam = pysam.AlignmentFile(alignment_bam_filename)

    # Go through the reads
    for read_alignment in alignment_bam.fetch(contig=chrom, start=start, end=end, multiple_iterators = True):
        aligned_pairs = read_alignment.get_aligned_pairs(with_seq=True, with_cigar=True)
        for read_pos, ref_pos, ref_seq, cigar_op in aligned_pairs:
            if int(cigar_op) == 8:
                rows.append([
                    read_alignment.query_name,
                    read_pos,
                    ref_pos,
                    ref_seq,
                ])

    df = pl.DataFrame(
        rows,
        schema = ["read_name", "start", "ref_start", "ref_seq"],
        orient = "row",
    )

    return df




def find_variation_on_reads_chrom(
    alignment_bam_filename,
    chrom,
    chunk_size = 10_000_000,
    n_threads = -1
):
    def runme(start, end):
        return find_variation_on_reads(
            alignment_bam_filename,
            chrom,
            start,
            end,
        )
    
    # Split chromsome lengths into chunks
    chunks = []
    chrom_length = grch38_chromosome_lengths_in_bp[chrom]
    for start in range(0, chrom_length, chunk_size):
        end = min(start + chunk_size, chrom_length)
        chunks.append((start, end))

    # Run the function in parallel
    dfs = joblib.Parallel(n_jobs=n_threads, verbose=50)(
        joblib.delayed(runme)(start, end) for start, end in chunks
    )

    df = pl.concat([df for df in dfs if len(df) > 0])
    return df



def find_variation_on_reads_at_mid_quality_events(
    alignment_bam_filename,
    annotated_events_parquet,
    chrom,
    start,
    end,
):
    rows = []

    # Open the alignment file
    alignment_bam = pysam.AlignmentFile(alignment_bam_filename)

    # Open the annotated events
    annotated_events = pl.read_parquet(annotated_events_parquet)
    mid_quality_events = set(annotated_events
       .filter(pl.col("is_mid_quality_event"))
       .select("read_name", "start")
       .rows()
    )

    # Go through the reads
    for read_alignment in alignment_bam.fetch(contig=chrom, start=start, end=end, multiple_iterators = True):
        aligned_pairs = read_alignment.get_aligned_pairs(with_seq=True, with_cigar=True)
        for read_pos, ref_pos, ref_seq, cigar_op in aligned_pairs:
            # If this is a mid-quality event on the read, and it is different from the reference, add it
            if (read_alignment.query_name, read_pos) in mid_quality_events and int(cigar_op) == 8:
                rows.append([
                    read_alignment.query_name,
                    read_pos,
                    ref_pos,
                    ref_seq,
                ])

    df = pl.DataFrame(
        rows,
        schema = ["read_name", "start", "ref_start", "ref_seq"],
        orient = "row",
    )

    return df


def find_variation_on_reads_at_mid_quality_events_chrom(
    alignment_bam_filename,
    annotated_events_parquet,
    chrom,
    chunk_size = 10_000_000,
    n_threads = -1
):
    # Split chromsome lengths into chunks
    chunks = []
    chrom_length = grch38_chromosome_lengths_in_bp[chrom]
    for start in range(0, chrom_length, chunk_size):
        end = min(start + chunk_size, chrom_length)
        chunks.append((start, end))

    # Run the function in parallel
    dfs = joblib.Parallel(n_jobs=n_threads, verbose=50)(
        joblib.delayed(find_variation_on_reads_at_mid_quality_events)(
            str(alignment_bam_filename),
            str(annotated_events_parquet),
            str(chrom),
            int(start), 
            int(end)) \
                for start, end in chunks
    )

    df = pl.concat([df for df in dfs if len(df) > 0])
    return df


# def compare_reads_to_dataset_reference(
#     alignment_bam_filename,
#     reference_panel_parquet,
#     chrom,
#     start,
#     end,
#     minimal_n_times = 5,
# ):
#     rows = []

#     # Load the dataset reference panel
#     dataset_reference_panel = pl.read_parquet(reference_panel_parquet)

#     # Create a set from the ref_start and ref_seq columns
#     pos_in_panel = set(dataset_reference_panel
#         .filter(
#             (pl.col("n_times") >= minimal_n_times) & 
#             (pl.col("chrom") == chrom) &
#             (pl.col("ref_start") >= start) &
#             (pl.col("ref_start") < end)
#         )
#         .select("ref_start")
#         .unique()
#         ["ref_start"]
#     )
#     print("# in panel", len(pos_in_panel))

#     # Open the alignment file
#     alignment_bam = pysam.AlignmentFile(alignment_bam_filename)

#     # Go through the reads
#     for read_alignment in alignment_bam.fetch(contig=chrom, start=start, end=end, multiple_iterators = True):
#         # Create a DataFrame with the aligned pairs
#         aligned_pairs = read_alignment.get_aligned_pairs(with_seq=True, with_cigar=True)
#         for read_pos, ref_pos, ref_seq, cigar_op in aligned_pairs:
#             if ref_pos in pos_in_panel:
#                 rows.append([
#                     read_alignment.query_name,
#                     read_pos,
#                     ref_pos,
#                     ref_seq,
#                     int(cigar_op),
#                 ])
        

#     df = pl.DataFrame(
#         rows,
#         schema = ["read_name", "start", "ref_start", "ref_seq", "op"],
#         orient = "row",
#     )

#     return df


def compare_reads_to_dataset_reference(
    alignment_bam_filename,
    reference_panel_parquet,
    chrom,
    start,
    end,
    minimal_n_times = 5,
):
    rows = []

    # Load the dataset reference panel
    dataset_reference_panel = pl.read_parquet(reference_panel_parquet)

    # Create a set from the ref_start column
    pos_in_panel = set(dataset_reference_panel
        .filter(
            (pl.col("n_times") >= minimal_n_times) & 
            (pl.col("chrom") == chrom) &
            (pl.col("ref_start") >= start) &
            (pl.col("ref_start") < end)
        )
        .select("ref_start")
        .unique()
        ["ref_start"]
    )
    print("# in panel", len(pos_in_panel))

    # Open the alignment file
    alignment_bam = pysam.AlignmentFile(alignment_bam_filename)

    # Go through the reads
    for read_alignment in alignment_bam.fetch(contig=chrom, start=start, end=end, multiple_iterators = True):
        # Create a DataFrame with the aligned pairs
        aligned_pairs = read_alignment.get_aligned_pairs(with_seq=True, with_cigar=True)
        for read_pos, ref_pos, ref_seq, cigar_op in aligned_pairs:
            if ref_pos in pos_in_panel:
                rows.append([
                    read_alignment.query_name,
                    read_pos,
                    ref_pos,
                    ref_seq,
                    int(cigar_op),
                ])
        

    df = pl.DataFrame(
        rows,
        schema = ["read_name", "start", "ref_start", "ref_seq", "op"],
        orient = "row",
    )

    return df


def compare_reads_to_dataset_reference_chrom(
    alignment_bam_filename,
    reference_panel_parquet,
    chrom,
    chunk_size = 10_000_000,
    minimal_n_times = 5,
    n_threads = -1,
):
    def runme(start, end):
        return compare_reads_to_dataset_reference(
            alignment_bam_filename,
            reference_panel_parquet,
            chrom,
            start,
            end,
            minimal_n_times,
        )
    
    # Split chromsome lengths into chunks
    chunks = []
    chrom_length = grch38_chromosome_lengths_in_bp[chrom]
    for start in range(0, chrom_length, chunk_size):
        end = min(start + chunk_size, chrom_length)
        chunks.append((start, end))

    # Run the function in parallel
    dfs = joblib.Parallel(n_jobs=n_threads, verbose=50)(
        joblib.delayed(runme)(start, end) for start, end in chunks
    )

    df = pl.concat([df for df in dfs if len(df) > 0])
    return df