from pathlib import Path
import pandas as pd

t2t_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/03.T2T-CHM13")

def t2t_load_files(sample_id):
    data_path = t2t_hapfusion_output_path / sample_id

    reads_df = pd.read_csv(
        str(data_path / f"chm13.{sample_id}.hapfusion.txt"),
        comment="#",
        delim_whitespace=True,
        names=["coordinates", "read_name", "where", "state", "event", "haplotype", "hd/whd", "recombination_length", "ccs_hbit", "h0_hbit", "h1_hbit", "hetsnps", "else"],
        header=None,
        index_col=None,
    )
    reads_df["read_chrom"] = reads_df["coordinates"].str.split(':').str[0]
    reads_df["read_start_pos_0based"] = reads_df["coordinates"].str.split(':').str[1].str.split('-').str[0].astype(int)
    reads_df["read_end_pos_0based"] = reads_df["coordinates"].str.split(':').str[1].str.split('-').str[1].astype(int)

    log_df = pd.read_csv(
        str(data_path / "hapfusion.log"),
        delim_whitespace=True,
        index_col=0,
    )

    candidates_df = pd.read_csv(
        str(data_path / "hapfusion_candidates.txt"),
        delim_whitespace=True,
    )

    return {"reads": reads_df, "summary": log_df, "candidates": candidates_df}

hg19_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/01.grch37")

def hg19_load_files(sample_id):
    data_path = hg19_hapfusion_output_path / sample_id

    reads_df = pd.read_csv(
        str(data_path / f"hg19.{sample_id}.hapfusion.txt"),
        comment="#",
        delim_whitespace=True,
        header=None,
    )
    reads_df["read_chrom"] = reads_df[0].str.split(':').str[0]
    reads_df["read_start_pos_0based"] = reads_df[0].str.split(':').str[1].str.split('-').str[0].astype(int)
    reads_df["read_end_pos_0based"] = reads_df[0].str.split(':').str[1].str.split('-').str[1].astype(int)

    log_df = pd.read_csv(
        str(data_path / "hapfusion.log"),
        delim_whitespace=True,
        index_col=0,
    )

    candidates_df = pd.read_csv(
        str(data_path / "hapfusion_candidates.txt"),
        delim_whitespace=True,
    )

    return {"reads": reads_df, "summary": log_df, "candidates": candidates_df}


denovo_hapfusion_output_path = Path("/lustre/scratch126/casm/team154pc/sl17/03.sperm/02.results/01.read_alignment/01.ccs/04.hifiasm/02.hifiasm_0.19.5-r592/02.chromosome_length_scaffolds")

def denovo_load_files(sample_id):
    data_path = denovo_hapfusion_output_path / sample_id
    if not (data_path / f"{sample_id}.hapfusion.txt").exists():
        data_path = denovo_hapfusion_output_path / sample_id / sample_id

    reads_df = pd.read_csv(
        str(data_path / f"{sample_id}.hapfusion.txt"),
        comment="#",
        delim_whitespace=True,
    )
    reads_df["read_chrom"] = reads_df[0].str.split(':').str[0]
    reads_df["read_start_pos_0based"] = reads_df[0].str.split(':').str[1].str.split('-').str[0].astype(int)
    reads_df["read_end_pos_0based"] = reads_df[0].str.split(':').str[1].str.split('-').str[1].astype(int)

    log_df = pd.read_csv(
        str(data_path / "hapfusion.log"),
        delim_whitespace=True,
        index_col=0,
    )

    candidates_df = pd.read_csv(
        str(data_path / "hapfusion_candidates.txt"),
        delim_whitespace=True,
    )

    return {"reads": reads_df, "summary": log_df, "candidates": candidates_df}



