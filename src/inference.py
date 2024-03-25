import numpy as np
import polars as pl
import numba
import scipy.optimize
import tqdm
from collections import Counter

from . import diagnostics


# ------------------------------------------------------------------------------------------------
#
# Extract SNP patterns from true reads
#
def extract_snp_patterns(
    focal_sample_id,
    denovo_chrom,
    events_parquet_filename,
    candidate_reads_parquet_filename,
    take_every = 1,
):
    events_df = pl.scan_parquet(events_parquet_filename)
    cand_df = pl.scan_parquet(candidate_reads_parquet_filename)

    read_names_df = (events_df
        .filter("is_high_quality_snp")
        .select("read_name")
        .unique("read_name")
        .join(cand_df.select("read_name"), on="read_name", how="anti")
    )
    
    random_subset_df = read_names_df.gather_every(take_every)

    classified_df = (events_df
        .join(random_subset_df, on="read_name")          
        .collect(streaming=True)
        .group_by("read_name")
        .map_groups(diagnostics.classify_read)
        .select(["read_name", "read_length", "idx_transitions", "snp_positions_on_read", "class"])    
        .with_columns(
            sample_id=pl.lit(focal_sample_id),
            chrom=pl.lit(denovo_chrom),
        ) 
    )

    return classified_df



# ------------------------------------------------------------------------------------------------
#
# Simulate SNP patterns given parameters
#
# @numba.njit
# def simulate_read_pattern(
#     read_length,
#     snp_positions_on_read,
#     prob_CO,
#     GC_tract_mean,
#     recombination_rate_per_bp,
#     random_seed,
# ):
#     np.random.seed(random_seed)

#     idx_transitions = []

#     geom_p = 1.0 / GC_tract_mean
#     prob_GC = 1 - prob_CO
    
#     prob_no_recomb = 1 - (recombination_rate_per_bp * read_length)

#     switches = []

#     if np.random.random() < prob_no_recomb:
#         # No recombination at all
#         pass
    
#     else:
#         # Otherwise pick a recombination breakpoint
#         breakpoint = np.random.randint(0, read_length)
#         switches.append(breakpoint) 

#         # Pick which event happened
#         if np.random.random() < prob_CO:
#             # Crossover
#             pass
#         else:
#             # Simple gene conversion
#             tract_length = np.random.geometric(geom_p)
#             if breakpoint + tract_length < read_length:
#                 switches.append(breakpoint + tract_length)

#         # Find the transitions
#         for switch in switches:
#             if switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]:
#                 pass
#             else:
#                 idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)    

#     # If a transition appears an even number of times, remove it
#     counter = numba.typed.Dict.empty(
#         key_type=numba.types.int64,
#         value_type=numba.types.int64,
#     )
#     for i in idx_transitions:
#         counter[i] = counter.get(i, 0) + 1

#     idx_transitions = sorted([num for num, freq in counter.items() if freq % 2 == 1])

#     return idx_transitions

# @numba.njit
# def simulate_many_read_patterns_numba(
#     read_length_list,
#     snp_positions_on_read_list,
#     prob_CO,
#     GC_tract_mean,
#     recombination_rate_per_bp,
#     random_seed,
# ):
#     res = [
#         simulate_read_pattern(
#             read_length_list[i],
#             snp_positions_on_read_list[i],
#             prob_CO,
#             GC_tract_mean,
#             recombination_rate_per_bp,
#             random_seed + i,
#         )
#         for i in range(len(read_length_list))
#     ]

#     return res


# @numba.njit
# def simulate_many_read_patterns_numba(
#     read_length_list,
#     snp_positions_on_read_list,
#     prob_CO,
#     GC_tract_mean,
#     recombination_rate_per_bp,
#     random_seed,
# ):
#     # Set seed
#     np.random.seed(random_seed)

#     # Set defaults
#     print("Starting")
#     res = [[] for i in range(len(read_length_list))]
    
#     print("Finding events")
#     # Find the recombinations that actually happened
#     probs_no_recomb = 1 - (recombination_rate_per_bp * read_length_list)
#     event_indices = np.where(np.random.random() < probs_no_recomb)[0]

#     print("Simulating")
#     # Simulate the ones who are
#     for i in event_indices:
#         res[i] = \
#             simulate_read_pattern(
#                 read_length_list[i],
#                 snp_positions_on_read_list[i],
#                 prob_CO,
#                 GC_tract_mean,
#                 1,      # Recombination happened for sure here
#                 random_seed + i,
#             )
        
#     return res

def simulate_many_read_patterns(
    read_length_list,
    snp_positions_on_read_list,
    prob_CO,
    prob_GC_component,
    GC_tract_mean,
    GC_tract_mean2,
    recombination_rate_per_bp,
    random_seed,
):
    # Set seed
    np.random.seed(random_seed)    
    
    # Find the recombinations that actually happened
    probs_recomb = recombination_rate_per_bp * np.array(read_length_list)
    event_indices = np.nonzero(np.random.random(len(read_length_list)) < probs_recomb)[0]    
    n_events = len(event_indices)
    n_noevents = len(read_length_list) - n_events
    res = np.empty(n_events, dtype=object) 
    
    events_read_length_list = read_length_list[event_indices]
    events_snp_positions_on_read_list = snp_positions_on_read_list[event_indices]

    X = np.random.random(n_events) < prob_CO
    CO_event_indices = np.nonzero(X)[0]
    GC_event_indices = np.nonzero(~X)[0]
    n_CO_events = len(CO_event_indices)
    n_GC_events = len(GC_event_indices)

    
    
    breakpoints_per_CO = np.random.randint(0, events_read_length_list[CO_event_indices])
    for i in range(n_CO_events):
        idx_transitions = []        
        snp_positions_on_read = events_snp_positions_on_read_list.item(CO_event_indices[i])

        # Only switch
        switch = breakpoints_per_CO[i]        
        if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
            idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)
        
        # Add it
        res[CO_event_indices[i]] = idx_transitions

    breakpoints_per_GC = np.random.randint(0, events_read_length_list[GC_event_indices])
    
    tract_lengths = np.where(
        np.random.random(n_GC_events) < prob_GC_component,
        np.random.geometric(1.0 / GC_tract_mean, n_GC_events),
        np.random.geometric(1.0 / GC_tract_mean2, n_GC_events)
    )
    for i in range(n_GC_events):
        idx_transitions = []
        snp_positions_on_read = events_snp_positions_on_read_list.item(GC_event_indices[i])

        # First switch
        switch = breakpoints_per_GC[i]
        if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
            idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

        # Second switch
        switch = breakpoints_per_GC[i] + tract_lengths[i]
        if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
            idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

        # Remove doubles
        if len(idx_transitions) == 2 and (idx_transitions[0] == idx_transitions[1]):
            idx_transitions = []

        # Add it
        res[GC_event_indices[i]] = idx_transitions
        
    
    subset_idx_transitions_array = res
    subset_snp_positions_on_read_array = snp_positions_on_read_list[event_indices]

    return subset_idx_transitions_array, subset_snp_positions_on_read_array, n_noevents


# ------------------------------------------------------------------------------------------------
#
# Likelihood
#
@numba.jit
def geom_sum(p, N):
    return N - (1-p) * (1 - np.power(1-p, N)) / p

@numba.jit
def mix_geom_sum(p_mix, p, p2, N):
    return p_mix * geom_sum(p, N) + (1-p_mix) * geom_sum(p2, N)

@numba.jit
def geom_sum2(p, N, D, M):
    return (np.power(1-p, D) - np.power(1-p, D+M)) * (1 - np.power(1-p, N)) / p 

@numba.jit
def mix_geom_sum2(p_mix, p, p2, N, D, M):
    return p_mix * geom_sum2(p, N, D, M) + (1-p_mix) * geom_sum2(p2, N, D, M)

@numba.jit
def geom_sum3(p, N, D):
    return np.power(1-p, D+1) * (1 - np.power(1-p, N)) / p

@numba.jit
def mix_geom_sum3(p_mix, p, p2, N, D):
    return p_mix * geom_sum3(p, N, D) + (1-p_mix) * geom_sum3(p2, N, D)

@numba.jit
def likelihood_of_read(
    read_length,
    snp_positions_on_read,
    idx_transitions,
    prob_CO,
    prob_GC_component,
    GC_tract_mean,
    GC_tract_mean2,
    recombination_rate_per_bp,
):
    geom_p = 1.0 / GC_tract_mean
    geom_p2 = 1.0 / GC_tract_mean2
    prob_GC = 1 - prob_CO
    n_transitions = len(idx_transitions)
    n_snps = len(snp_positions_on_read)
    
    # Assuming up to a single recombination event
    prob_no_recomb = 1 - (recombination_rate_per_bp * read_length)
    
    total_prob = 0.0
    
    #
    # 1. Add the prob of seeing this read given no recombination
    #
    
    # If there were any transitions, this is inconsistent with no recomb
    if n_transitions > 0:
        total_prob += 0.0
    
    # Otherwise, up to symmetry, the probability is 1
    else:
        total_prob += prob_no_recomb * 1.0
     
    #
    # 2. Add the prob of seeing this read given a crossover
    #
    
    # If there are more than 1 transitions, then this read is inconsistent with CO, 
    # no matter where on the read the breakpoint happened
    if n_transitions >= 2:
        total_prob += 0.0
    
    # If there are 0 transitions, then this read is consistent with CO only if it 
    # happened before the first SNP or after the last SNP
    elif n_transitions == 0:
        total_prob += (recombination_rate_per_bp * prob_CO * (snp_positions_on_read[0] + (read_length-1 - snp_positions_on_read[-1])))
    
    # If there is a single transition, CO could only happen between the SNPs in transition
    else:
        snp_pos_before_transition = snp_positions_on_read[idx_transitions[0]]
        snp_pos_after_transition = snp_positions_on_read[idx_transitions[0]+1]
        total_prob += (recombination_rate_per_bp * prob_CO * (snp_pos_after_transition - snp_pos_before_transition))
    
    #
    # 3. Add the prob of seeing this read given a simple gene conversion
    #
    
    # If there are more than 2 transitions, then this read is inconsistent with a gene conversion
    if n_transitions > 2:
        total_prob += 0.0

    # If there is one transition, the this means the second switch must have happened either before the first SNP
    # (including before the read) or after the last SNP (including after the read); and the observed transition must 
    # have happened between its two SNPs.
    #
    # x ----- x --.....-- x -----...
    #     N         D             
    #
    # The prob of a geometric variable to be above x is (1-p)^x, so
    # This is \sum_{n=0}^{N-1}{ (1-p)^(n+1+D) } which then in turn is 
    # = (1-p)^(D+1) * \sum_{n=0}^{N-1}{ (1-p)^n } = (1-p)^(D+1) * (1 - (1-p)^N) / p
    if n_transitions == 1:
        # Case 1: Start before first SNP
        N = snp_positions_on_read[idx_transitions[0]+1] - snp_positions_on_read[idx_transitions[0]]
        D = snp_positions_on_read[idx_transitions[0]] - snp_positions_on_read[0]
        total_prob += recombination_rate_per_bp * prob_GC * mix_geom_sum3(prob_GC_component, geom_p, geom_p2, N, D)

        # Case 2: End after the last SNP
        N = snp_positions_on_read[idx_transitions[0]+1] - snp_positions_on_read[idx_transitions[0]]
        D = snp_positions_on_read[-1] - snp_positions_on_read[idx_transitions[0]+1]
        total_prob += recombination_rate_per_bp * prob_GC * mix_geom_sum3(prob_GC_component, geom_p, geom_p2, N, D)
    
    # If there are 0 transitions, then this means the GC must have happened before the first SNP, after the 
    # last SNP, or between two SNPs. The breakpoint could have happened anywhere along the read, as long as
    # the second transition happens before the next SNP (or end of read).
    elif n_transitions == 0:
        # The probability of a geometric variable no more than N is its cdf: 1 - (1-p)^N
        # The probability of a geometric variable starting uniformly in [0,N) and lasting no more than N is
        # therefore: \sum_{n=0}^{N-1} {1 - (1-p)^(N-n)}, which, using the sum of geometric series, is
        #       = N - (1-p) \cdot (1 - (1-p)^N) / p
        
        # s = geom_sum(geom_p, snp_positions_on_read[0])
        # for i in range(0, n_snps-1):
        #     s += geom_sum(geom_p, snp_positions_on_read[i+1] - snp_positions_on_read[i])
        # s += geom_sum(geom_p, read_length - snp_positions_on_read[-1])
        s = mix_geom_sum(prob_GC_component, geom_p, geom_p2, snp_positions_on_read[0])
        for i in range(0, n_snps-1):
            s += mix_geom_sum(prob_GC_component, geom_p, geom_p2, snp_positions_on_read[i+1] - snp_positions_on_read[i])
        s += mix_geom_sum(prob_GC_component, geom_p, geom_p2, read_length - snp_positions_on_read[-1])

        total_prob += recombination_rate_per_bp * prob_GC * s
        
    # If there are 2 transitions, then the breakpoint must have happened in the range before the first 
    # transition SNP (say, range of length N); and tract should have finished in the range after the
    # last transition SNP (say, range o length M); let D be the range between those two SNPs:
    #
    # x ----- x --.....-- x ----- x
    #     N         D         M       
    #
    # The probability of a geometric variable obtaining a value between A and B is q(A,B) := (1 - (1-p)^B) - (1 - (1-p)^A) = (1-p)^A-(1-p)^B
    # If we enumerate going back from the first x backward, we need:
    # \sum_{n=0}^{N-1}{ q(n+D,n+D+M) } = 
    # \sum_{n=0}^{N-1}{ (1-p)^(n+D) - (1-p)^(n+D+M) } = (1-p)^D \sum_{n=0}^{N-1}{ (1-p)^n } - (1-p)^(D+M) \sum_{n=0}^{N-1} (1-p)^(n) } = 
    # = ((1-p)^D - (1-p)^(D+M)) \sum_{n=0}^{N-1}{ (1-p)^n } = ((1-p)^D - (1-p)^(D+M)) (1 -(1-p)^N)/p 
    else:
        N = snp_positions_on_read[idx_transitions[0]+1] - snp_positions_on_read[idx_transitions[0]]
        M = snp_positions_on_read[idx_transitions[-1]+1] - snp_positions_on_read[idx_transitions[-1]]
        D = snp_positions_on_read[idx_transitions[-1]] - snp_positions_on_read[idx_transitions[0]+1]
        total_prob += recombination_rate_per_bp * prob_GC * mix_geom_sum2(prob_GC_component, geom_p, geom_p2, N, D, M)

    return total_prob


@numba.jit
def log_likelihood_of_many_reads(
    read_length_list,
    snp_positions_on_read_list,
    idx_transitions_list,
    weights_list,
    prob_CO,
    prob_GC_component,
    GC_tract_mean,
    GC_tract_mean2,
    recombination_rate_per_bp,
):
    return np.sum(
        np.array([
            weights_list[i] *
            np.log(
                likelihood_of_read(
                    read_length_list[i],
                    snp_positions_on_read_list[i],
                    idx_transitions_list[i],
                    prob_CO,
                    prob_GC_component,
                    GC_tract_mean,
                    GC_tract_mean2,
                    recombination_rate_per_bp,
                )
            ) for i in range(len(read_length_list))
        ])
    )

# ------------------------------------------------------------------------------------------------
#
# Optimize
#
def maximum_likelihood_all_reads(
    read_length_list,
    snp_positions_on_read_list,
    idx_transitions_list,
    weights_list,
    prob_CO_range,
    prob_GC_component_range,
    GC_tract_mean_range,
    GC_tract_mean2_range,
    recombination_rate_per_bp_range,
):
    read_length_list = numba.typed.List(read_length_list)
    snp_positions_on_read_list = numba.typed.List(snp_positions_on_read_list)
    idx_transitions_list = numba.typed.List([np.array(x).astype(np.int32) for x in idx_transitions_list])
    weights_list = numba.typed.List(weights_list)

    def minimizeme(x):
        return -log_likelihood_of_many_reads(
            read_length_list,
            snp_positions_on_read_list,
            idx_transitions_list,
            weights_list,
            prob_CO = x[0],
            prob_GC_component = x[1],
            GC_tract_mean = x[2],
            GC_tract_mean2 = x[3],
            recombination_rate_per_bp = x[4],
        )

    def callback(x):
        with np.printoptions(precision=3, suppress=True):
            print(f"Current:\t{x}")        
    
    res = scipy.optimize.minimize(
        fun = minimizeme,
        x0 = [
            np.mean(prob_CO_range), 
            np.mean(prob_GC_component_range),
            np.mean(GC_tract_mean_range), 
            np.mean(GC_tract_mean2_range), 
            np.mean(recombination_rate_per_bp_range)
        ],
        method = "Nelder-Mead",
        bounds = [prob_CO_range, prob_GC_component_range, GC_tract_mean_range, GC_tract_mean2_range, recombination_rate_per_bp_range],
        #options = {"disp": True},
        options={'xatol': 1e-3},
        callback = callback,
    )

    return res

# ------------------------------------------------------------------------------------------------
#
# Exponential Likelihood
#

