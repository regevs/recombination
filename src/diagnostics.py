import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines
import seaborn as sns
from pathlib import Path
import re
import sys

import pysam
import fastq as fq

sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/hapfusion/src")
import hapfusion
from hapfusion import bamlib

sys.path.append("/nfs/users/nfs_r/rs42/rs42/git/himut/src")
import himut

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
