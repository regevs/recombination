import matplotlib as mpl
import numpy as np
import polars as pl
from pathlib import Path

import pysam


def write_small_bam(
    tmpfilename, 
    input_bams, 
    focal_read,
    read_to_frac,
):
    with pysam.AlignmentFile(tmpfilename, "wb", header=input_bams[0].header) as outf:
        for bam in input_bams:
            for read in bam.fetch(
                focal_read.reference_name, 
                focal_read.reference_start, 
                focal_read.reference_end,
                multiple_iterators=True,
                ):
                
                if read.mapping_quality < 60:
                    continue

                frac = read_to_frac.get(read.query_name, 0.5)
                rgba_color = list(mpl.colormaps["coolwarm"](frac))
                rgb_color = np.array(mpl.colors.to_rgb(rgba_color))*255

                color_str = ','.join([str(x) for x in rgb_color])
                if read.query_name == focal_read.query_name:
                    color_str = "152,251,152"
                #color_str = f"rgb({color_str})"
                read.tags += [('YC', color_str)]
                read.tags += [('YD', str(frac))]
                outf.write(read)

    pysam.sort("-o", tmpfilename.replace(".bam", ".sorted.bam"), tmpfilename)
    pysam.index(tmpfilename.replace(".bam", ".sorted.bam"))

def write_all_bams(
    all_reads_parquet,
    candidate_reads_parquet,
    input_bam_filenames,
    output_dir_path,
    output_bam_filename,
    chrom,
    column,
):
    hap_stats_df = pl.read_parquet(all_reads_parquet)
    read_to_frac = dict(
        hap_stats_df.select(
            pl.col("read_name"), 
            pl.col(column).fill_null(0.5)
        ).to_numpy()    )
    
    candidates_df = pl.read_parquet(candidate_reads_parquet)

    input_bams = [pysam.AlignmentFile(input_bam_filename) for input_bam_filename in input_bam_filenames]
    for bam in input_bams:
        for read in bam.fetch(chrom, multiple_iterators=True):
            read_name = read.query_name
            if read_name in candidates_df["read_name"]:
                read_dir = Path(output_dir_path) / read_name.replace("/", ".")
                read_dir.mkdir(parents=True, exist_ok=True)

                # Write extra read info
                read_tsv_filename = str(read_dir / output_bam_filename) + ".metadata"
                open(read_tsv_filename, "w").write(str(dict(read.aligned_pairs)))            

                # Write the BAM
                read_bam_filename = str(read_dir / output_bam_filename)
                write_small_bam(
                    read_bam_filename, 
                    input_bams, 
                    read,
                    read_to_frac,
                )



def start_igv(
    focal_read_pos,
    bam_filename,
    bam_metadata_filename,
    reference_fasta_filename,
    chrom,
    slack = 30,
    maxHeight = 700,
):
    import igv_notebook

    aligned_pairs = eval(open(bam_metadata_filename).read())

    x = aligned_pairs[focal_read_pos] + 1
    locus = f"{chrom}:{x - slack}-{x + slack}"

    igv_browser = igv_notebook.Browser(
        {
            "reference": {
                "id": str(x),
                "name": str(x),
                "fastaPath": str(reference_fasta_filename),
                "indexPath": str(reference_fasta_filename) + ".fai",
            },
            "locus": locus,
            "showCenterGuide": True,
            "tracks": [
                {
                    "name": "Local BAM",
                    "path": bam_filename,
                    "indexPath": bam_filename + ".bai",
                    "format": "bam",
                    "type": "alignment",
                    "colorBy": "tag",
                    "colorByTag": "YC",
                    "maxHeight": maxHeight,
                    "autoHeight": True,
                    "sort": {
                        "chr": chrom,
                        "position": x,
                        "option": "TAG", 
                        "tag": "YD",
                        "direction": "DESC",
                    }
                }
            ]
        }
    )

    return igv_browser    
