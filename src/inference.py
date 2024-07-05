import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numba
import scipy.optimize
import tqdm
from collections import Counter
import warnings
import itertools

from . import diagnostics, IDs


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
# Simulate according to genetic map, more slowly
#
def simulate_read_patterns_genetic_map(
    read_genetic_length_in_cM_list,
    snp_positions_on_read_list,
    read_starts_list,
    read_ends_list,
    rate_map,
    random_seed,
    CO = True,
    GC_tract_mean = None,
):
    # Set seed
    np.random.seed(random_seed)  

    n_reads = len(read_genetic_length_in_cM_list)

    # Define a histogram RV, which draws a single value proportionally
    # to the genetic map
    H = scipy.stats.rv_histogram(
        [
            np.nan_to_num(rate_map.rate),
            np.concatenate([rate_map.left, [rate_map.right[-1]]]),
        ],
        density=True,
    )

    # Draw a pool of breakpoints
    all_breakpoints = np.sort(H.rvs(size=10_000_000).astype(int))

    # Decide for each read whether it saw a recombination or not
    probs_recomb = np.array(read_genetic_length_in_cM_list) * 1e-2
    event_indices = np.nonzero(np.random.random(n_reads) < probs_recomb)[0]    
    n_events = len(event_indices)
    n_noevents = n_reads - n_events
    res = np.empty(n_events, dtype=object) 
    
    events_snp_positions_on_read_list = snp_positions_on_read_list[event_indices]
    
    for i in range(n_events):
        event_idx = event_indices[i]
        
        # Find a breakpoint, conditional on a breakpoint being drawn
        left_i = np.searchsorted(all_breakpoints, read_starts_list[event_idx], side="left")
        right_i = np.searchsorted(all_breakpoints, read_ends_list[event_idx], side="left")
        if right_i > left_i:
            switch = np.random.choice(all_breakpoints[left_i:right_i])
        else:
            # Should rarely happen
            switch = (read_starts_list[event_idx] + read_ends_list[event_idx]) // 2
        
        # Change to read coords
        switch -= read_starts_list[event_idx]

        idx_transitions = []        
        snp_positions_on_read = events_snp_positions_on_read_list[i]        

        if CO:
            # Only switch
            if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
                idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

        else: # GC
            # First switch            
            if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
                idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

            # Second switch
            tract_length = np.random.geometric(1.0 / GC_tract_mean)
            switch = switch + tract_length
            if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
                idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

            # Remove doubles
            if len(idx_transitions) == 2 and (idx_transitions[0] == idx_transitions[1]):
                idx_transitions = []
        
        #print(i, event_idx, all_breakpoints[left_i], all_breakpoints[right_i], switch, snp_positions_on_read, idx_transitions)

        # Add it
        res[i] = idx_transitions

    return res, event_indices, n_noevents


# ------------------------------------------------------------------------------------------------
#
# Simulate according to CO probs between snps
#
import random 

@numba.jit
def weighted_random_choice(p):
    cumulative_p = np.cumsum(p)
    r = np.random.random()
    return np.searchsorted(cumulative_p, r)

@numba.jit
def simulate_read_patterns_probs(
    read_length_list,
    snp_positions_on_read_list,
    prob_CO_between_snps_list,
    prob_CO_before_read_list,
    prob_CO_after_read_list,
    q,
    m,
    GC_tract_mean,
    GC_tract_mean2,
    read_margin_in_bp = 5000,
    allow_flip = True,
):

    # Number of reads
    n_reads = len(read_length_list)

    # Total probability of recombination per read
    probs_recomb = np.zeros(n_reads)
    for i, x in enumerate(prob_CO_between_snps_list):
        probs_recomb[i] += np.sum(x)
        probs_recomb[i] += prob_CO_before_read_list[i]
        probs_recomb[i] += prob_CO_after_read_list[i]

    probs_recomb = probs_recomb / q

    # Decide for each read whether it saw a recombination or not
    event_indices = np.nonzero(np.random.random(n_reads) < probs_recomb)[0]    
    n_events = len(event_indices)
    n_noevents = n_reads - n_events
    #res = np.empty(n_events, dtype=object) 
    res = numba.typed.List()

    # Direction per event
    if allow_flip:
        directions_per_event = (np.random.random(n_events) < 0.5).astype(np.int64)
    else:
        directions_per_event = np.zeros(n_events, dtype=np.int64)

    # Event types
    event_types = (np.random.random(n_events) < q).astype(np.int64)
    n_CO = np.sum(event_types == 1)
    n_GC = np.sum(event_types == 0)

    # GC type
    GC_tract_types = (np.random.random(n_events) < m).astype(np.int64)
    n_first_GC = np.sum(GC_tract_types[event_types == 0] == 1)
    GC_tract_lengths = np.zeros(n_events, dtype=np.int64)
        
    print(f"Working on {n_events} events")
    for i in range(n_events):
        event_idx = event_indices[i]
        idx_transitions = [np.int64(x) for x in range(0)]  # weird hack for numba to recognize type

        # Orient according to the direction
        read_length = read_length_list[event_idx]
        snp_positions_on_read = snp_positions_on_read_list[event_idx]
        prob_CO_between_snps = prob_CO_between_snps_list[event_idx]
        prob_CO_before_read = prob_CO_before_read_list[event_idx]
        prob_CO_after_read = prob_CO_after_read_list[event_idx]
        if directions_per_event[i]:
            snp_positions_on_read = read_length - snp_positions_on_read[::-1]
            prob_CO_between_snps = prob_CO_between_snps[::-1]
            prob_CO_before_read, prob_CO_after_read = prob_CO_after_read, prob_CO_before_read        

        # Draw the breakpoint according to the probs
        p = np.concatenate((np.array([prob_CO_before_read]), prob_CO_between_snps, np.array([prob_CO_after_read])))
        p = p / p.sum()
        j = weighted_random_choice(p)

        if j == 0:
            switch = np.random.randint(-read_margin_in_bp, 0)
        elif j == 1:
            switch = np.random.randint(0, snp_positions_on_read[0])
        elif j == len(p)-2:
            switch = np.random.randint(snp_positions_on_read[-1], read_length)
        elif j == len(p)-1:
            switch = np.random.randint(read_length, read_length+read_margin_in_bp)
        else:
            switch = np.random.randint(snp_positions_on_read[j-2], snp_positions_on_read[j-1])      

        if event_types[i]:
            # Only switch
            if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
                idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

        else: # GC
            if GC_tract_types[i] == 1:
                this_GC_tract_mean = GC_tract_mean
            else:
                this_GC_tract_mean = GC_tract_mean2            

            # First switch            
            if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
                idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

            # Second switch
            tract_length = np.random.geometric(1.0 / this_GC_tract_mean)
            GC_tract_lengths[i] = tract_length

            switch = switch + tract_length
            if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
                idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

            # Remove doubles
            if len(idx_transitions) == 2 and (idx_transitions[0] == idx_transitions[1]):
                idx_transitions = [np.int64(x) for x in range(0)] #numba.typed.List()
        
        #print(i, event_idx, all_breakpoints[left_i], all_breakpoints[right_i], switch, snp_positions_on_read, idx_transitions)

        # Flip back if needed
        if directions_per_event[i]:
            flipped_idx_transitions = []
            for idx in range(len(idx_transitions)-1, -1, -1):
                flipped_idx_transitions.append(len(snp_positions_on_read) - 2 - idx_transitions[idx])
            idx_transitions = flipped_idx_transitions #numba.typed.List(flipped_idx_transitions)

        # Add it
        res.append(idx_transitions)

    return res, event_indices, n_noevents, n_CO, n_GC, n_first_GC, event_types, GC_tract_types, GC_tract_lengths

# def simulate_read_patterns_probs(
#     read_length_list,
#     snp_positions_on_read_list,
#     prob_CO_between_snps_list,
#     prob_CO_before_read_list,
#     prob_CO_after_read_list,
#     q,
#     m,
#     GC_tract_mean,
#     GC_tract_mean2,
#     read_margin_in_bp = 5000,
#     random_seed = None,
# ):
#     # Set seed
#     random.seed(random_seed)  

#     # Number of reads
#     n_reads = len(read_length_list)

#     # Total probability of recombination per read
#     probs_recomb = \
#         np.array([np.sum(x) for x in prob_CO_between_snps_list]) + \
#         np.array(prob_CO_before_read_list) + \
#         np.array(prob_CO_after_read_list)
    
#     probs_recomb = probs_recomb / q

#     # Decide for each read whether it saw a recombination or not
#     event_indices = np.nonzero(np.random.random(n_reads) < probs_recomb)[0]    
#     n_events = len(event_indices)
#     n_noevents = n_reads - n_events
#     #res = np.empty(n_events, dtype=object) 
#     res = []

#     # Direction per event
#     directions_per_event = (np.random.random(n_events) < 0.5).astype(int)

#     # Event types
#     event_types = (np.random.random(n_events) < q).astype(int)
#     n_CO = np.sum(event_types == 1)
#     n_GC = np.sum(event_types == 0)

#     # GC type
#     GC_tract_types = (np.random.random(n_events) < m).astype(int)
#     n_first_GC = np.sum(GC_tract_types[event_types == 0] == 1)
        
#     for i in range(n_events):
#         event_idx = event_indices[i]
#         idx_transitions = []

#         # Orient according to the direction
#         read_length = read_length_list[event_idx]
#         snp_positions_on_read = snp_positions_on_read_list[event_idx]
#         prob_CO_between_snps = prob_CO_between_snps_list[event_idx]
#         prob_CO_before_read = prob_CO_before_read_list[event_idx]
#         prob_CO_after_read = prob_CO_after_read_list[event_idx]
#         if directions_per_event[i]:
#             snp_positions_on_read = read_length - snp_positions_on_read[::-1]
#             prob_CO_between_snps = prob_CO_between_snps[::-1]
#             prob_CO_before_read, prob_CO_after_read = prob_CO_after_read, prob_CO_before_read        

#         # Draw the breakpoint according to the probs
#         p = np.concatenate([[prob_CO_before_read], prob_CO_between_snps, [prob_CO_after_read]])
#         p = p / p.sum()
#         j = np.random.choice(a=len(p), p=p)
#         if j == 0:
#             switch = np.random.randint(-read_margin_in_bp, 0)
#         elif j == 1:
#             switch = np.random.randint(0, snp_positions_on_read[0])
#         elif j == len(p)-2:
#             switch = np.random.randint(snp_positions_on_read[-1], read_length)
#         elif j == len(p)-1:
#             switch = np.random.randint(read_length, read_length+read_margin_in_bp)
#         else:
#             switch = np.random.randint(snp_positions_on_read[j-2], snp_positions_on_read[j-1])      

#         if event_types[i]:
#             # Only switch
#             if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
#                 idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

#         else: # GC
#             if GC_tract_types[i] == 0:
#                 this_GC_tract_mean = GC_tract_mean
#             else:
#                 this_GC_tract_mean = GC_tract_mean2

#             # First switch            
#             if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
#                 idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

#             # Second switch
#             tract_length = np.random.geometric(1.0 / this_GC_tract_mean)
#             switch = switch + tract_length
#             if not (switch <= snp_positions_on_read[0] or switch > snp_positions_on_read[-1]): 
#                 idx_transitions.append(np.searchsorted(snp_positions_on_read, switch) - 1)

#             # Remove doubles
#             if len(idx_transitions) == 2 and (idx_transitions[0] == idx_transitions[1]):
#                 idx_transitions = []
        
#         #print(i, event_idx, all_breakpoints[left_i], all_breakpoints[right_i], switch, snp_positions_on_read, idx_transitions)

#         # Flip back if needed
#         idx_transitions = np.array(idx_transitions)
#         if directions_per_event[i]:
#             idx_transitions = len(snp_positions_on_read) - 2 - idx_transitions[::-1]


#         # Add it
#         res.append(idx_transitions)

#     return res, event_indices, n_noevents, n_CO, n_GC, n_first_GC


# ------------------------------------------------------------------------------------------------
#
# Likelihood (old)
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
def old_likelihood_of_read(
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
def old_log_likelihood_of_many_reads(
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


def old_maximum_likelihood_all_reads(
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
# Likelihood
#
@numba.jit
def old_f1(lmb, A, B, C, D):
    return ((1-lmb)**(-B) - (1-lmb)**(-A)) * ((1-lmb)**(C) - (1-lmb)**(D)) / lmb

@numba.jit
def f1(lmb, A, B, C, D):
    assert A <= B <= C <= D
    return 1/lmb * (1-lmb)**(C-B) * (1 - (1-lmb)**(B-A)) * (1 - (1-lmb)**(D-C))

@numba.jit
def f2(m, lmb1, lmb2, A, B, C, D):
    return m * f1(lmb1, A, B, C, D) + (1-m) * f1(lmb2, A, B, C, D)

@numba.jit
def old_f3(lmb, A, B, C):
    return ((1-lmb)**(C-B+1) - (1-lmb)**(C-A+1))/lmb

@numba.jit
def f3(lmb, A, B, C):
    assert A<=B<=C
    return 1/lmb * (1-lmb)**(C-B) * (1 - (1-lmb)**(B-A))

@numba.jit
def f4(m, lmb1, lmb2, A, B, C):
    return m * f3(lmb1, A, B, C) + (1-m) * f3(lmb2, A, B, C)

@numba.jit
def f5(lmb, A, B):
    return (B-A) - (1 - (1-lmb)**(B-A))/lmb

@numba.jit
def f6(m, lmb1, lmb2, A, B):
    return m * f5(lmb1, A, B) + (1-m) * f5(lmb2, A, B)


@numba.jit
def likelihood_of_read_one_direction(
    read_length,
    snp_positions_on_read,
    idx_transitions,
    prob_CO_between_snps,
    prob_CO_before_read,
    prob_CO_after_read,
    q,
    m,
    GC_tract_mean,
    GC_tract_mean2,
    read_margin_in_bp = 5000,
):
    lmb1 = 1.0 / GC_tract_mean
    lmb2 = 1.0 / GC_tract_mean2
    n_transitions = len(idx_transitions)
    n_snps = len(snp_positions_on_read)
    R = read_margin_in_bp
    L = read_length

    rs = prob_CO_between_snps
    assert len(rs) == n_snps+1

    ps = snp_positions_on_read
    
    # Two switches
    if n_transitions == 2:
        # idx_transition is 0-based, but i,j, rs are 1-based
        i0 = idx_transitions[0]
        j0 = idx_transitions[1]
        i1 = i0 + 1
        
        ri = rs[i1]
        pi_diff = ps[i0+1] - ps[i0]    # ps are 0=based

        L2 = ri * (1-q) / (pi_diff * q) * f2(m, lmb1, lmb2, ps[i0], ps[i0+1], ps[j0], ps[j0+1])

    # One switch
    elif n_transitions == 1:
        # idx_transition is 0-based, but i, rs are 1-based
        i0 = idx_transitions[0]
        i1 = i0 + 1

        ri = rs[i1]
        r0 = rs[0]
        p1 = ps[0]
        pn = ps[-1]
        rm1 = prob_CO_before_read
        pi_diff = ps[i0+1] - ps[i0]    

        L1CO = ri
        L1NCO_left = \
            (r0 * (1-q)) / (p1*q) * f2(m, lmb1, lmb2, 0, p1, ps[i0], ps[i0+1]) + \
            (rm1 * (1-q)) / (R * q) * f2(m, lmb1, lmb2, -R, 0, ps[i0], ps[i0+1])
        L1NCO_right = \
            (ri * (1-q)) / (pi_diff * q) * f4(m, lmb1, lmb2, ps[i0], ps[i0+1], pn)
        
        L1 = L1CO + L1NCO_left + L1NCO_right
            

    # No switches
    elif n_transitions == 0:
        L0No = 1 - (prob_CO_before_read + rs.sum() + prob_CO_after_read) / q
        assert (0 <= L0No <= 1)

        L0CO = prob_CO_before_read + rs[0] + rs[-1] + prob_CO_after_read

        rm1 = prob_CO_before_read
        r0 = rs[0]
        p1 = ps[0]
        pn = ps[-1]
        rn = rs[-1]
        rp1 = prob_CO_after_read
        L0NCO_left = \
            (rm1 * (1-q)) / (R * q) * (f6(m, lmb1, lmb2, -R, 0) + f2(m, lmb1, lmb2, -R, 0, 0, p1)) + \
            (r0 * (1-q)) / (p1 * q) * f6(m, lmb1, lmb2, 0, p1)

        L0NCO_right = \
            (rn * (1-q)) / ((L - pn) * q) * (f6(m, lmb1, lmb2, pn, L) + f4(m, lmb1, lmb2, pn, L, L)) \
            + (rp1 * (1-q)) / (R * q) * (f6(m, lmb1, lmb2, L, L + R) + f4(m, lmb1, lmb2, L, L+R, L+R))

        L0NCO_between = 0
        for n_snp in range(n_snps-1):
            L0NCO_between += (rs[n_snp+1] * (1-q)) / ((ps[n_snp+1] - ps[n_snp]) * q) * f6(m, lmb1, lmb2, ps[n_snp], ps[n_snp+1])

        L0NCO_all = (rm1 * (1-q)) / (R * q) * f4(m, lmb1, lmb2, -R, 0, pn) + (r0 * (1-q)) / (p1 * q) * f4(m, lmb1, lmb2, 0, p1, pn)

        L0 = L0No + L0CO + L0NCO_left + L0NCO_right + L0NCO_between + L0NCO_all

    else:
        assert("Too many switches")

    #print(L0No, L0CO, L0NCO_left, L0NCO_right, L0NCO_between, L0NCO_all)
    return L2 + L1 + L0

@numba.jit
def likelihood_of_read(
    read_length,
    snp_positions_on_read,
    idx_transitions,
    prob_CO_between_snps,
    prob_CO_before_read,
    prob_CO_after_read,
    q,
    m,
    GC_tract_mean,
    GC_tract_mean2,
    read_margin_in_bp = 5000,
):
    L_forward = likelihood_of_read_one_direction(
        read_length,
        snp_positions_on_read,
        idx_transitions,
        prob_CO_between_snps,
        prob_CO_before_read,
        prob_CO_after_read,
        q,
        m,
        GC_tract_mean,
        GC_tract_mean2,
        read_margin_in_bp,
    )

    L_backward = likelihood_of_read_one_direction(
        read_length,
        read_length - snp_positions_on_read[::-1],
        len(snp_positions_on_read) - 2 - idx_transitions[::-1],
        prob_CO_between_snps[::-1],
        prob_CO_after_read,
        prob_CO_before_read,
        q,
        m,
        GC_tract_mean,
        GC_tract_mean2,
        read_margin_in_bp,
    )

    return (L_forward + L_backward) / 2


@numba.jit(parallel=True)
def log_likelihood_of_many_reads(
    read_length_list,
    snp_positions_on_read_list,
    idx_transitions_list,
    prob_CO_between_snps_list,
    prob_CO_before_read_list,
    prob_CO_after_read_list,
    weights_list,
    q,
    m,
    GC_tract_mean,
    GC_tract_mean2,
    read_margin_in_bp,
):
    S = 0.0
    for i in numba.prange(len(read_length_list)):
        S += weights_list[i] * \
            np.log(
                likelihood_of_read(
                    read_length_list[i],
                    snp_positions_on_read_list[i],
                    idx_transitions_list[i],
                    prob_CO_between_snps_list[i],
                    prob_CO_before_read_list[i],
                    prob_CO_after_read_list[i],
                    q,
                    m,
                    GC_tract_mean,
                    GC_tract_mean2,
                    read_margin_in_bp,
                )
            )
    return S



@numba.jit(parallel=True)
def component_likelihood_many_reads(
    read_length_list,
    snp_positions_on_read_list,
    idx_transitions_list,
    prob_CO_between_snps_list,
    prob_CO_before_read_list,
    prob_CO_after_read_list,
    q,
    m,
    GC_tract_mean,
    GC_tract_mean2,
    read_margin_in_bp,
):
    n_reads = len(read_length_list)

    prob_from_1 = np.zeros(n_reads, dtype=np.float64)

    for i in numba.prange(len(read_length_list)):
        # Pr(data | q, m=1, L1, L2)
        p1 = likelihood_of_read(
            read_length_list[i],
            snp_positions_on_read_list[i],
            idx_transitions_list[i],
            prob_CO_between_snps_list[i],
            prob_CO_before_read_list[i],
            prob_CO_after_read_list[i],
            q,
            1,
            GC_tract_mean,
            GC_tract_mean2,
            read_margin_in_bp,
        )

        # Pr(data | q, m=0, L1, L2)
        p2 = likelihood_of_read(
            read_length_list[i],
            snp_positions_on_read_list[i],
            idx_transitions_list[i],
            prob_CO_between_snps_list[i],
            prob_CO_before_read_list[i],
            prob_CO_after_read_list[i],
            q,
            0,
            GC_tract_mean,
            GC_tract_mean2,
            read_margin_in_bp,
        )

        # Pr(m=0 | data) = Pr(data | m=0) * P(m=0) / Pr(data)
        # Pr(m=1 | data) = Pr(data | m=1) * P(m=1) / Pr(data)
        prob_from_1[i] = p1 * m / (p1 * m + p2 * (1-m))

    return prob_from_1


# ------------------------------------------------------------------------------------------------
#
# Optimize
#
def maximum_likelihood_all_reads(
    read_length_list,
    snp_positions_on_read_list,
    idx_transitions_list,
    prob_CO_between_snps_list,
    prob_CO_before_read_list,
    prob_CO_after_read_list,
    weights_list,
    q_range,
    m_range,
    GC_tract_mean_range,
    GC_tract_mean2_range,
    read_margin_in_bp = 5000,
    x0 = None,
):
    assert q_range[0] > 0 and q_range[1] < 1 and q_range[0] <= q_range[1]
    assert m_range[0] >= 0 and m_range[1] <= 1 and m_range[0] <= m_range[1]
    assert GC_tract_mean_range[0] >= 1
    assert GC_tract_mean2_range[0] >= 1

    read_length_list = numba.typed.List(read_length_list)
    snp_positions_on_read_list = numba.typed.List(snp_positions_on_read_list)
    idx_transitions_list = numba.typed.List([np.array(x).astype(np.int32) for x in idx_transitions_list])    
    prob_CO_between_snps_list = numba.typed.List(prob_CO_between_snps_list)
    prob_CO_before_read_list = numba.typed.List(prob_CO_before_read_list)
    prob_CO_after_read_list = numba.typed.List(prob_CO_after_read_list)
    weights_list = numba.typed.List(weights_list)

    if x0 is None:
        x0 = [
            np.mean(q_range), 
            np.mean(m_range),
            np.mean(GC_tract_mean_range), 
            np.mean(GC_tract_mean2_range), 
        ]

    def minimizeme(x):        
        with np.printoptions(precision=3, suppress=True):
            print(f"Current:\t{x}\t", end="")   
            res = -log_likelihood_of_many_reads(
                read_length_list,
                snp_positions_on_read_list,
                idx_transitions_list,
                prob_CO_between_snps_list,
                prob_CO_before_read_list,
                prob_CO_after_read_list,
                weights_list,
                q = x[0],
                m = x[1],
                GC_tract_mean = x[2],
                GC_tract_mean2 = x[3],
                read_margin_in_bp = read_margin_in_bp,
            )
            
            print(f"{res}")   
            return res     
    
    
    res = scipy.optimize.minimize(
        fun = minimizeme,
        x0 = x0,
        method = "Nelder-Mead",
        bounds = [q_range, m_range, GC_tract_mean_range, GC_tract_mean2_range],
        options={'xatol': 1e-2},
    )

    return res

# ------------------------------------------------------------------------------------------------
#
# Find full distribution on tract length upper bound
#
@numba.jit
def tract_length_upper_bound_dist_from_read(
    read_length,
    snp_positions_on_read,
    prob_CO_between_snps,
    prob_CO_before_read,
    prob_CO_after_read,
    q,
    m,
    GC_tract_mean,
    GC_tract_mean2,
    read_margin_in_bp = 5000,
    max_dist_bin = 30000,
    max_n_converted = 10,
):
    n_snps = len(snp_positions_on_read)
    
    D = np.zeros(max_dist_bin, dtype=np.float64)
    C = np.zeros(max_n_converted, dtype=np.float64)
    if n_snps < 3:
        return D, C

    for first_transition in range(n_snps-2):
        for second_transition in range(first_transition+1, n_snps-1):
            n_converted = second_transition-first_transition
            upper_bound = snp_positions_on_read[second_transition+1] - snp_positions_on_read[first_transition]

            likelihood_of_pair = likelihood_of_read_one_direction(
                read_length,
                snp_positions_on_read,
                np.array([first_transition, second_transition]),
                prob_CO_between_snps,
                prob_CO_before_read,
                prob_CO_after_read,
                q,
                m,
                GC_tract_mean,
                GC_tract_mean2,
                read_margin_in_bp,
            )

            if upper_bound < max_dist_bin:
                D[upper_bound] += likelihood_of_pair

            if n_converted < max_n_converted:
                C[n_converted] += likelihood_of_pair

    D /= D.sum()
    C /= C.sum()

    return D, C


@numba.jit(parallel=True)
def tract_length_upper_bound_dist_from_many_reads(
    read_length_list,
    snp_positions_on_read_list,
    prob_CO_between_snps_list,
    prob_CO_before_read_list,
    prob_CO_after_read_list,
    weights_list,
    q,
    m,
    GC_tract_mean,
    GC_tract_mean2,
    read_margin_in_bp = 5000,
    max_dist_bin = 30000,
    max_n_converted = 10,
):
    D = np.zeros(max_dist_bin, dtype=np.float64)  
    C = np.zeros(max_n_converted, dtype=np.float64)  
    
    for i in numba.prange(len(read_length_list)):
        retD, retC = \
            tract_length_upper_bound_dist_from_read(
                read_length_list[i],
                snp_positions_on_read_list[i],
                prob_CO_between_snps_list[i],
                prob_CO_before_read_list[i],
                prob_CO_after_read_list[i],
                q,
                m,
                GC_tract_mean,
                GC_tract_mean2,
                read_margin_in_bp,
                max_dist_bin,
            )

        if not np.any(np.isnan(retD)):
            D += weights_list[i] * retD
        if not np.any(np.isnan(retC)):
            C += weights_list[i] * retC        

    D /= D.sum()
    C /= C.sum()

    return D, C

# ------------------------------------------------------------------------------------------------
#
# Permutation testing
#
def permutation_testing(pairs, n_perms=10, seed=42, method="AD"):
    with warnings.catch_warnings(action="ignore"):
        if method == "KS":
            test_statistic = lambda x,y: scipy.stats.ks_2samp(x, y, method="auto").statistic
        if method == "AD":
            test_statistic = lambda x,y: scipy.stats.anderson_ksamp([x, y]).statistic
        if method == "U":
            test_statistic = lambda x,y: scipy.stats.mannwhitneyu(x, y, use_continuity=False).statistic
            
            
        rng = np.random.default_rng(seed=seed)
        orig = np.sum([test_statistic(x,y) for x,y in pairs if len(x)>2 and len(y)>2])
        permuted = []
        for i in range(n_perms):
            sum_stats = 0
            for x,y in pairs:
                if len(x)>2 and len(y)>2:
                    z = np.concatenate([x,y])
                    px, py = np.split(rng.permutation(z), [len(x)])
                    sum_stats += test_statistic(px, py)
            permuted.append(sum_stats)
            
        pvalue = np.mean(np.array(permuted)>=orig)
        return pvalue

# ------------------------------------------------------------------------------------------------
#
# DSB analysis
#
def calculate_motif_distance_histogram(
    reads_df,
    motif_center_column,
    motif_strand_column,
    signal_column = None,
    max_dist=30000,
    grch38_recombining_interval_threshold = None,
):
    H = np.zeros(max_dist*2 + 1)
    xs = np.arange(-max_dist, max_dist+1)

    n_rows = 0
    for row in reads_df.iter_rows(named=True):
        if signal_column is None:
            w = 1.0
        else:
            w = row[signal_column]

        if grch38_recombining_interval_threshold is not None:
            if row["grch38_recombining_interval_length"] >= grch38_recombining_interval_threshold:
                continue

        n_rows += 1
        if row[motif_strand_column] == 1:
            H[
                row["grch38_recombining_interval_start_pos"] - row[motif_center_column] - (-max_dist):
                row["grch38_recombining_interval_end_pos"] - row[motif_center_column] - (-max_dist)
            ] += \
                (w / row["grch38_recombining_interval_length"])
        else:
            H[
                -(row["grch38_recombining_interval_end_pos"] - row[motif_center_column]) - (-max_dist):
                -(row["grch38_recombining_interval_start_pos"] - row[motif_center_column]) - (-max_dist)
            ] += \
                (w / row["grch38_recombining_interval_length"])
        
    H /= n_rows

    return xs, H

def calculate_motif_distance_to_converted_snps_histogram(
    reads_df,
    motif_center_column,
    motif_strand_column,
    signal_column = None,
    max_dist=30000,
):
    H = np.zeros(max_dist*2 + 1)
    xs = np.arange(-max_dist, max_dist+1)

    n_rows = 0
    for row in reads_df.iter_rows(named=True):
        n_rows += 1
        background_allele = row["high_quality_snp_positions_alleles"][0]
        
        n_converted_alleles = len([x for x in row["high_quality_snp_positions_alleles"] if x != background_allele])

        if signal_column is None:
            w = 1.0
        else:
            w = row[signal_column]

        for snp_pos, snp_allele in zip(row["high_quality_snp_positions"], row["high_quality_snp_positions_alleles"]):
            if snp_allele != background_allele:
                weight = w / n_converted_alleles
                dist_to_motif = snp_pos + row["grch38_reference_start"] - row[motif_center_column]
                if row[motif_strand_column] == 1:
                    H[dist_to_motif - (-max_dist)] += weight
                else:
                    H[-dist_to_motif - (-max_dist)] += weight
        
    H /= n_rows

    return xs, H    

def motif_distance_histogram_symmetry_permutation_testing(
    reads_df,
    motif_center_column,
    motif_strand_column,
    signal_column = None,
    max_dist=30000,
    n_perms=1000,
    hist_func = calculate_motif_distance_histogram,
    stat = "max_abs",
    grch38_recombining_interval_threshold = None,
):
    def symm_stat(H):
        H1 = H
        H2 = H[::-1]

        if stat == "max_abs":
            return np.max(np.abs(H1 - H2))
        if stat == "sum_abs":
            return np.sum(np.abs(H1 - H2))
        if stat == "max_abs_cumsum":
            return np.max(np.abs(np.cumsum(H1) - np.cumsum(H2)))   
        if stat == "sum_sq_cumsum":
            return np.sum(np.square(np.cumsum(H1) - np.cumsum(H2)))   
        if stat == "max_sq_cumsum":
            return np.max(np.square(np.cumsum(H1) - np.cumsum(H2)))   
        
    
    rng = np.random.default_rng()

    # Read stat
    S = symm_stat(hist_func(
        reads_df,
        motif_center_column,
        motif_strand_column,
        signal_column,
        max_dist,
        grch38_recombining_interval_threshold,
    )[1])

    permed = []
    for n_perm in range(n_perms):
        permed.append(symm_stat(
            hist_func(
                reads_df.with_columns(
                    pl.Series(
                        name=motif_strand_column,
                        values=rng.integers(2, size=len(reads_df)),
                    ),
                ),
                motif_center_column,
                motif_strand_column,
                signal_column,
                max_dist,
                grch38_recombining_interval_threshold,
            )[1]
        ))

    permed = np.array(permed)
    pvalue = np.mean(permed >= S)

    return pvalue

def motif_distance_histogram_diffs_permutation_testing(
    reads_df1,
    reads_df2,
    motif_center_column,
    motif_strand_column,
    signal_column = None,
    max_dist=30000,
    n_perms=1000,
    hist_func = calculate_motif_distance_histogram,
    stat="max_abs",
):
    def diff_stat(H1, H2):
        #return np.sum((np.cumsum(H) - np.cumsum(H[::-1]))**2)
        if stat == "max_abs":
            return np.max(np.abs(H1 - H2))
        if stat == "sum_abs":
            return np.sum(np.abs(H1 - H2))
        if stat == "sum_sq_cumsum":
            return np.sum(np.square(np.cumsum(H1) - np.cumsum(H2)))
        if stat == "max_sq_cumsum":
            return np.max(np.square(np.cumsum(H1) - np.cumsum(H2)))   
    
    rng = np.random.default_rng()

    # Read stat
    H1 = hist_func(
        reads_df1,
        motif_center_column,
        motif_strand_column,
        signal_column,
        max_dist,
    )[1]
    H2 = hist_func(
        reads_df2,
        motif_center_column,
        motif_strand_column,
        signal_column,
        max_dist,
    )[1]
    S = diff_stat(H1, H2)

    permed = []
    for n_perm in range(n_perms):
        shuffled_reads_df = pl.concat([reads_df1, reads_df2])
        shuffled_reads_df = shuffled_reads_df.sample(len(shuffled_reads_df), shuffle=True)
        shuffled_reads_df1 = shuffled_reads_df[:len(reads_df1)]
        shuffled_reads_df2 = shuffled_reads_df[len(reads_df1):]
        SH1 = hist_func(
            shuffled_reads_df1,
            motif_center_column,
            motif_strand_column,
            signal_column,
            max_dist,
        )[1]
        SH2 = hist_func(
            shuffled_reads_df2,
            motif_center_column,
            motif_strand_column,
            signal_column,
            max_dist,
        )[1]
        
        permed.append(diff_stat(SH1, SH2))

    permed = np.array(permed)
    pvalue = np.mean(permed >= S)

    return pvalue

###

def get_sample_pairwise_tests(
    sample_ids,
    test_df,
    test_field,
    n_perms = 1000,
    take_log10 = False,
):
    sample_id_to_signal = {}
    for sample_id in sample_ids:
        sdf = (test_df
            .filter(pl.col("sample_id") == sample_id)
        )
        signal = sdf[test_field]    
        sample_id_to_signal[sample_id] = signal.to_numpy()
        if take_log10:
            sample_id_to_signal[sample_id] = np.log10(sample_id_to_signal[sample_id])

    rows = []
    for sample_id1, sample_id2 in tqdm.tqdm(list(itertools.combinations(sorted(sample_ids), 2))):
        paper_label_id1 = IDs.sample_id_to_paper_label[sample_id1]
        paper_label_id2 = IDs.sample_id_to_paper_label[sample_id2]
        signal_1 = sample_id_to_signal[sample_id1]
        signal_2 = sample_id_to_signal[sample_id2]
        rows.append(
            [
                sample_id1, 
                sample_id2,
                paper_label_id1,
                paper_label_id2,
                len(signal_1), 
                len(signal_2),
                scipy.stats.ks_2samp(signal_1, signal_2).pvalue,
                permutation_testing([[signal_1, signal_2]], n_perms),
            ]
        )
        
    pairwise_df = pl.DataFrame(
        rows,
        schema=["sample_id1", "sample_id2", 
                "paper_label_id1", "paper_label_id2",
                "n1", "n2", 
                "ks_pvalue", 
                "AD_perm_pvalue"],
    ) 

    return pairwise_df   

def plot_pairwise_results(
    pairwise_df,
    sample_id_order,
    column = "AD_perm_pvalue",
    vmax = 0.1,
    show_paper_labels = True,
):
    mat = []
    for sample_id1 in sample_id_order:
        row = []
        for sample_id2 in sample_id_order:
            if sample_id1 == sample_id2:
                pval = 1
            else:
                pval = pairwise_df.filter(
                    ((pl.col("sample_id1") == sample_id1) & (pl.col("sample_id2") == sample_id2)) | 
                    ((pl.col("sample_id1") == sample_id2) & (pl.col("sample_id2") == sample_id1))
                )[column].item(0)
            row.append(pval)
        mat.append(row)
        
    if show_paper_labels:
        labels = [IDs.sample_id_to_paper_label[x] for x in sample_id_order]
    else:
        labels = sample_id_order
    mat = pd.DataFrame(mat, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(
        mat,
        vmin=0, 
        vmax=vmax,
        annot=True, 
        fmt=".2f",
        square=True,
    );

def get_sample_vs_rest_tests(
    sample_ids,
    test_df,
    test_field,
    n_perms = 1000,
):
    sample_id_to_signal = {}
    for sample_id in sample_ids:
        sdf = (test_df
            .filter(pl.col("sample_id") == sample_id)
        )
        signal = sdf[test_field]    
        sample_id_to_signal[sample_id] = signal.to_numpy()

    def runme(sample_id):
        paper_label_id = IDs.sample_id_to_paper_label[sample_id]

        lens_1 = sample_id_to_signal[sample_id]
        pairs = [
            [
                lens_1,
                sample_id_to_signal[other_sample_id],
            ]
            for other_sample_id in sample_ids if other_sample_id != sample_id 
        ]
        p_separate = permutation_testing(pairs, n_perms)

        rest = np.concatenate(
            [sample_id_to_signal[other_sample_id] for other_sample_id in sample_ids if other_sample_id != sample_id]
        )

        pairs = [
            [
                lens_1,
                rest
            ]            
        ]
        p_agg = permutation_testing(pairs, n_perms)

        p_ttest = scipy.stats.ttest_ind(lens_1, rest).pvalue

        return (
            sample_id, 
            paper_label_id,
            len(lens_1), 
            p_separate,
            p_agg,
            p_ttest,
        )
    
    import joblib
    rows = joblib.Parallel(n_jobs=-1, verbose=50)(
        joblib.delayed(runme)(sample_id) for sample_id in sorted(sample_ids)
    )
        
    sample_vs_rest_signal_df = pl.DataFrame(
        rows,
        schema=["sample_id", "paper_label_id", "n_in_sample", "AD_perm_pvalue", "AD_perm_pvalue_agg", "ttest_pvalue_agg"],
    ).sort("AD_perm_pvalue")
        
    return sample_vs_rest_signal_df   

def get_sample_vs_all_tests(
    sample_ids,
    test_df,
    all_df,
    test_field,
    n_perms = 1000,
    subsample_fraction = 0.01,
):
    sample_id_to_signal = {}
    for sample_id in sample_ids:
        sdf = (test_df
            .filter(pl.col("sample_id") == sample_id)
        )
        signal = sdf[test_field]    
        sample_id_to_signal[sample_id] = signal.to_numpy()

    all_signal = all_df[test_field].sample(fraction=subsample_fraction).to_numpy()

    rows = []
    for sample_id in tqdm.tqdm(sorted(sample_ids)):
        paper_label_id = IDs.sample_id_to_paper_label[sample_id]
        pairs = [
            [
                sample_id_to_signal[sample_id],
                all_signal
            ]            
        ]
        lens_1 = sample_id_to_signal[sample_id]
        rows.append(
            [
                sample_id, 
                paper_label_id,
                len(lens_1), 
                permutation_testing(pairs, n_perms),
            ]
        )
        
    sample_vs_rest_signal_df = pl.DataFrame(
        rows,
        schema=["sample_id", "paper_label_id", "n_in_sample", "AD_perm_pvalue"],
    ).sort("AD_perm_pvalue")
        
    return sample_vs_rest_signal_df   

def plot_boxplots_samples(
    test_df,
    test_column,
#    all_df,
    xlabel = "",
    take_log10=True,
    ticks = None,
    figsize=(6,7),
    label_to_color = None,
    ax = None,
    show_labels = True,
):

    if ax is None:
        fig, ax = plt.subplots(
            # nrows=len(IDs.sample_presentation_order), 
            # ncols=2, 
            figsize=figsize,
            #width_ratios=[1,10],
        )

    D = {}
    for sample_id in IDs.sample_ids:
        sdf = (test_df
            .filter(pl.col("sample_id") == sample_id)
        )
        signal = sdf[test_column]    
        D[sample_id] = signal.to_numpy()
        if take_log10:
            D[sample_id] = np.log10(D[sample_id])

    # D["All"] = all_df[test_column].to_numpy()
    # if take_log10:
    #     D["All"] = np.log10(D["All"])

#     sns.stripplot(
#         data = [D[k] for k in IDs.sample_id_order],
#         orient = "h",
#         s = 2,
#         alpha = 1,
#         ax = ax,
#         jitter = 0.4,
#         # showfliers = False,
# #        showmeans = True,
#     )

    color_palette = None
    if label_to_color is not None:
        color_palette = []
        for sample_id in IDs.sample_id_order:
            color_palette.append(
                label_to_color.get(
                    IDs.sample_id_to_paper_label[sample_id], 
                    "#f7f7f7"
                )
            )


    sns.boxplot(
        data = [D[k] for k in IDs.sample_id_order],
        orient = "h",
        ax = ax,
        showfliers = False,
        showmeans = True,
        palette = color_palette,
        medianprops = dict(
            linewidth=0.5, 
            color='black'
        ),
        meanprops = dict(
            marker='.', 
            markeredgecolor='black',
            markerfacecolor='black',
            markersize=10,
        ),
        whis=(10,90)
    )

    if ticks is not None:
        if take_log10:
            ax.set_xticks(
                ticks = ticks,
                labels = [f"$10^{{{x}}}$" for x in ticks],
            )
        else:
            ax.set_xticks(
                ticks = ticks,
                labels = [str(x) for x in ticks],
            )

    ax.set_xlabel(xlabel)

    sns.despine(
        ax=ax,
        top=True, right=True, left=True, bottom=False,
    )

    if show_labels:
        ax.set_yticks(
            np.arange(len(IDs.sample_id_order)),
            [IDs.sample_id_to_paper_label[k] for k in IDs.sample_id_order]
        )
    else:
        ax.set_yticks(
            [],
            []
        )

    return ax

def plot_histograms_samples(
    test_df,
    test_column,
    all_df,
    bins,
    ylim_max = 0.3,
    xlabel = "",
    take_log10=True,
    figsize=(6,7),
):

    fig, axs = plt.subplots(
        nrows=len(IDs.sample_presentation_order), 
        ncols=2, 
        figsize=figsize,
        width_ratios=[1,10],
    )

    D = {}
    for sample_id in IDs.sample_ids:
        sdf = (test_df
            .filter(pl.col("sample_id") == sample_id)
        )
        signal = sdf[test_column]    
        D[sample_id] = signal.to_numpy()
        if take_log10:
            D[sample_id] = np.log10(D[sample_id])

    D["All"] = all_df[test_column].to_numpy()
    if take_log10:
        D["All"] = np.log10(D["All"])

    for i, k in enumerate(IDs.sample_presentation_order):
        v = D[k]
        label = IDs.sample_id_to_paper_label[k]
        
        ax = axs[i,0]
        ax.axis("off");
        ax.set_xlim(0,1);
        ax.set_ylim(0,1);
        ax.text(0.7, 0.5, label, horizontalalignment='center',
            verticalalignment='center')    
        
        ax = axs[i,1]
        H = np.histogram(v, bins=bins)[0]
        H = H/H.sum()
        ax.bar(
            x=bins[:-1], 
            height=H, 
            width=bins[1]-bins[0],
            color=IDs.sample_to_color[k], 
            edgecolor="grey",
            linewidth=0.5,
        );

        ax.set_ylim(-0.04, ylim_max);
        ax.set_xlim(bins[0], bins[-1]);


        if i != len(D)-1:
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[['bottom', 'right', 'top', 'left']].set_visible(False)
        else:
            ax.set_xlabel(xlabel)
            # ax.set_xticks(
            #     ticks = np.arange(-2, 2),
            #     labels = [f"$10^{{{x}}}$" for x in np.arange(-2, 2)],
            # )
            ax.yaxis.tick_right()
            ax.spines[['bottom', 'top', 'left']].set_visible(False)
            ax.spines['right'].set_bounds((0, 0.1))
        
        #ax.yaxis.set_tick_params(labelleft=False)    
        
        # compute quantiles
        v = v[~np.isnan(v)]
        quantiles = np.percentile(v, [2.5, 10, 25, 75, 90, 97.5])

        # fill space between each pair of quantiles
        for j in range(len(quantiles) - 1):
            ax.fill_between(
                [quantiles[j], # lower bound
                quantiles[j+1]], # upper bound
                -0.04, # max y=0
                0, # max y=0.0002
                color=IDs.quantile_colors[j]
            )
            
        # mean
        ax.scatter([v.mean()], [-0.02], color='black', s=10)

    return axs
        
def generate_call_set(reads_df, focal_sample_ids, take_every=1, bootstrap=False, min_snps=2, sample_every=None):
    #
    # 1. Take all reads with any switches
    #
    # - High quality read (same strand, MAPQ, mismatches and clipping
    # - Has enough coverage on both haplotypes
    # - Mapped to nonzero cM
    # - Has more than min SNPs
    # - Is high quality classification (in addition to quality read + coverage, no common transition, and not
    #   in blacklist)
    # - Is not complex
    #
    cand_df = (reads_df
        .filter(pl.col("sample_id").is_in(focal_sample_ids))
        .filter("is_high_quality_read")
        .filter((pl.col("min_coverage_hap1") >= 3) & (pl.col("min_coverage_hap2") >= 3))
        .filter(pl.col("full_read_crossover_prob") > 0)
        .filter(pl.col("high_quality_snp_positions").list.len() >= min_snps)
        .filter(pl.col("high_quality_classification"))
        .filter(pl.col("high_quality_classification_class") != "CNCO")
        .select(
            "read_name",
            "read_length",
            "mid_quality_snp_positions",            
            "between_mid_quality_snps_cM",
            "between_high_quality_snps_cM",
            "before_read_cM",
            "after_read_cM",
            "high_quality_snp_positions",
            "CO_active_interval_crossover_prob",
            "mid_CO_active_interval_crossover_prob",
            "grch37_reference_start_cM",
            "at_mid_quality_snp_cM",
            "grch37_reference_end_cM",
            "high_quality_snp_positions_alleles",
            "mid_quality_snp_positions_alleles",
            "high_quality_snps_idx_transitions",
            "idx_transitions",
            weight = pl.lit(1),
        )
    )
    
    #
    # 2. Take a subset of reads without swithces
    #
    # - High quality read (same strand, MAPQ, mismatches and clipping
    # - Has enough coverage on both haplotypes
    # - Mapped to nonzero cM
    # - Has more than 1 SNP
    # - No switches
    # - Take every `take_every`
    #
    other_df = (reads_df
        .filter(pl.col("sample_id").is_in(focal_sample_ids))
        .filter("is_high_quality_read")
        .filter((pl.col("min_coverage_hap1") >= 3) & (pl.col("min_coverage_hap2") >= 3))
        .filter(pl.col("full_read_crossover_prob") > 0)
        .filter(pl.col("high_quality_snp_positions").list.len() >= min_snps)
        .filter(pl.col("idx_transitions").is_null())
        .gather_every(take_every)
        .select(
            "read_name",
            "read_length",
            "mid_quality_snp_positions",
            "between_mid_quality_snps_cM",
            "between_high_quality_snps_cM",
            "before_read_cM",
            "after_read_cM",
            "high_quality_snp_positions",
            "CO_active_interval_crossover_prob",
            "mid_CO_active_interval_crossover_prob",
            "grch37_reference_start_cM",
            "at_mid_quality_snp_cM",
            "grch37_reference_end_cM",
            "high_quality_snp_positions_alleles",
            "mid_quality_snp_positions_alleles",
            "high_quality_snps_idx_transitions",
            idx_transitions = pl.col("idx_transitions").fill_null([]),
            weight = pl.lit(take_every),
        )
    )
    
    # Combine
    callset_df = pl.concat([cand_df, other_df])

    if sample_every is not None:
        callset_df = callset_df.gather_every(sample_every)
    
    # Add between SNPs in bp
    callset_df = (callset_df                  
        .with_columns(
            pl.lit([]).list.concat([
                pl.lit(0),
                pl.col("high_quality_snp_positions"),
                pl.col("read_length"),
            ]).list.diff(null_behavior="drop").alias("between_high_quality_snps_bp")
        )
    )
    
    callset_df = callset_df.collect(streaming=True)
    
    # Bootstrap if needed
    if bootstrap:
        callset_df = callset_df.sample(n = len(callset_df), with_replacement = True)
    
    return callset_df
    

        

        
       
