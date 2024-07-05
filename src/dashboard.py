import matplotlib as mpl
import numpy as np
import polars as pl
from pathlib import Path

import pysam


def write_small_bam(
    tmpfilename, 
    bam, 
    focal_read,
    read_to_frac,
):
    with pysam.AlignmentFile(tmpfilename, "wb", header=bam.header) as outf:
        for read in bam.fetch(
            focal_read.reference_name, 
            focal_read.reference_start, 
            focal_read.reference_end):
            
            if read.mapping_quality < 60:
                continue

            frac = read_to_frac.get(read.query_name, 0.5)
            rgba_color = list(mpl.colormaps["coolwarm"](frac))
            rgb_color = np.array(mpl.colors.to_rgb(rgba_color))*255

            color_str = ','.join([str(x) for x in rgb_color])
            if read.query_name == focal_read.query_name:
                color_str = "152,251,152"
            read.tags += [('YC', color_str)]
            read.tags += [('YD', str(frac))]
            outf.write(read)

    pysam.sort("-o", tmpfilename.replace(".bam", ".sorted.bam"), tmpfilename)
    pysam.index(tmpfilename.replace(".bam", ".sorted.bam"))

def write_all_bams(
    all_reads_parquet,
    candidate_reads_parquet,
    input_bam_filename,
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
        ).to_numpy()
    )
    bam = pysam.AlignmentFile(input_bam_filename)
    candidates_df = pl.read_parquet(candidate_reads_parquet)

    # print("WTF???")    
    # #bam = pysam.AlignmentFile("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds/PD50489e/PD50489e.hap1.minimap2.sorted.primary_alignments.bam")
    # for read in bam.fetch(chrom, multiple_iterators=True):
    #     read_name = read.query_name
    #     if read_name in ["m64094e_220717_002414/118489792/ccs"]:
    #         print("YES", read_name)

    for read in bam.fetch(chrom, multiple_iterators=True):
        read_name = read.query_name
        # if read_name in ["m64094e_220717_002414/118489792/ccs"]:
        #     print("WWW", read_name)
        #     read_dir = Path(output_dir_path) / read_name.replace("/", ".")
        #     print(read_dir)
        #     read_dir.mkdir(parents=True, exist_ok=True)
        #     print(read_dir, read_dir.exists())

        # if read_name == "m64094e_220717_002414/118489792/ccs":
        #     print("WTF")
        if read_name in candidates_df["read_name"]:
            #print(read_name)

            read_dir = Path(output_dir_path) / read_name.replace("/", ".")
            read_dir.mkdir(parents=True, exist_ok=True)
            # if not read_dir.exists():
            #     print(read_dir, read_dir.exists())

            # Write extra read info
            read_tsv_filename = str(read_dir / output_bam_filename) + ".metadata"
            open(read_tsv_filename, "w").write(str(dict(read.aligned_pairs)))            

            # Write the BAM
            read_bam_filename = str(read_dir / output_bam_filename)
            write_small_bam(
                read_bam_filename, 
                bam, 
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
                    "maxHeight": 700,
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
