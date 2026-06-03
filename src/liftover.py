import numpy as np
import pandas as pd

T2T_chromosome_sizes_in_bp = {
    "chr1": 248387328,
    "chr2": 242696752,
    "chr3": 201105948,
    "chr4": 193574945,
    "chr5": 182045439,
    "chr6": 172126628,
    "chr7": 160567428,
    "chr8": 146259331,
    "chr9": 150617247,
    "chr10": 134758134,
    "chr11": 135127769,
    "chr12": 133324548,
    "chr13": 113566686,
    "chr14": 101161492,
    "chr15": 99753195,
    "chr16": 96330374,
    "chr17": 84276897,
    "chr18": 80542538,
    "chr19": 61707364,
    "chr20": 66210255,
    "chr21": 45090682,
    "chr22": 51324926,
    "chrX": 154259566,
    "chrY": 62460029
}

# Take a genetic map in one coordinate system, the liftover, and return
# the new genetic map in the new coordinate system
def merge_genetic_map_and_liftover(
    genetic_map_input_path,
    liftover_path,
    genetic_map_output_path,
):
    # I was unable to confirm if the genetic map coordinates are 0-based or 1-based but it probably
    # will not matter for any downstream analyses
    map_df = pd.read_csv(
        genetic_map_input_path,
        delim_whitespace=True,
        names=["chrom", "pos_0based", "rate_cM_per_Mb", "pos_cM"],
        header=0,
    )

    # Convert this into a interval -> rate dataframe
    interval_df = pd.DataFrame(
        {
            "chrom": map_df.chrom,
            "start_pos_0based": np.concatenate([[0], map_df["pos_0based"].values[:-1]]),
            "end_pos_0based": map_df["pos_0based"].values,
            "rate_cM_per_Mb": np.concatenate([[0], map_df["rate_cM_per_Mb"].values[:-1]]),
        }
    )

    liftover_df = pd.read_csv(
        liftover_path,
        delim_whitespace=True,
        names=["chrom", "new_start_pos_0based", "new_end_pos_0based", "start_pos_0based", "end_pos_0based"],
    )

    # Merge
    new_df = interval_df.merge(liftover_df)
    new_df = new_df.sort_values("new_start_pos_0based")

    # Calculate the rate per interval
    new_df["rate_cM_per_interval"] = new_df["rate_cM_per_Mb"] * (new_df["end_pos_0based"] - new_df["start_pos_0based"]) / 1e6

    # Calculate the new rate per Mb
    new_df["new_rate_cM_per_Mb"] = new_df["rate_cM_per_interval"] / (new_df["new_end_pos_0based"] - new_df["new_start_pos_0based"]) * 1e6

    # Create a new genetic map
    coords = pd.Series(
        np.sort(
            np.unique(
                np.concatenate([
                    [0], 
                    new_df["new_start_pos_0based"].values, 
                    new_df["new_end_pos_0based"].values
                    ]
                )
            )
        )
    ).rename("new_pos_0based")

    new_df = (new_df
        .merge(coords, left_on="new_start_pos_0based", right_on="new_pos_0based", how="right")
        [["chrom", "new_pos_0based", "new_rate_cM_per_Mb"]]
        .fillna({"chrom": new_df.chrom.values[0], "new_rate_cM_per_Mb": 0.0})
    )

    # # First and last intervals have no recombination
    # new_df.iat[0, -1] = 0.0
    # new_df.iat[-1, -1] = 0.0

    # Calculate the genetic position
    new_df["new_pos_cM"] = np.concatenate([
        [0],
        np.cumsum(
            np.diff(new_df["new_pos_0based"]) * new_df["new_rate_cM_per_Mb"][:-1] / 1e6
        ),
    ])

    # Write to file
    new_df.to_csv(
        genetic_map_output_path,
        index=False,
        sep="\t",
    )

    return new_df