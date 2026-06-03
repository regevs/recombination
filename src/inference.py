import numpy as np
import polars as pl
import numba
import scipy.optimize



# ------------------------------------------------------------------------------------------------
#
# Simulate according to CO probs between snps
#

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
    
    first_switches = np.zeros(n_events, dtype=np.int64)
    second_switches = np.zeros(n_events, dtype=np.int64)
        
    #print(f"Working on {n_events} events")
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

        first_switches[i] = switch

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
            second_switches[i] = switch

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

    return res, event_indices, n_noevents, n_CO, n_GC, n_first_GC, event_types, GC_tract_types, GC_tract_lengths, first_switches, second_switches

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
def f1(lmb, A, B, C, D):
    assert A <= B <= C <= D
    return 1/lmb * (1-lmb)**(C-B) * (1 - (1-lmb)**(B-A)) * (1 - (1-lmb)**(D-C))

@numba.jit
def f2(m, lmb1, lmb2, A, B, C, D):
    return m * f1(lmb1, A, B, C, D) + (1-m) * f1(lmb2, A, B, C, D)

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
    prob_factor,
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
                    prob_CO_between_snps_list[i] * prob_factor,
                    prob_CO_before_read_list[i] * prob_factor,
                    prob_CO_after_read_list[i] * prob_factor,
                    q,
                    m,
                    GC_tract_mean,
                    GC_tract_mean2,
                    read_margin_in_bp,
                )
            )
    return S


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
    prob_factor_range,
    read_margin_in_bp = 5000,
    x0 = None,
):
    assert q_range[0] > 0 and q_range[1] < 1 and q_range[0] <= q_range[1]
    assert m_range[0] >= 0 and m_range[1] <= 1 and m_range[0] <= m_range[1]
    assert prob_factor_range[0] > 0 and prob_factor_range[1] <= 1 and prob_factor_range[0] <= prob_factor_range[1]
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
            np.mean(prob_factor_range),
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
                prob_factor = x[4],
                read_margin_in_bp = read_margin_in_bp,
            )
            
            print(f"{res}")   
            return res     
    
    
    res = scipy.optimize.minimize(
        fun = minimizeme,
        x0 = x0,
        method = "Nelder-Mead",
        bounds = [q_range, m_range, GC_tract_mean_range, GC_tract_mean2_range, prob_factor_range],
        options={'xatol': 1e-2},
    )

    return res
        
def generate_call_set(reads_df, focal_sample_ids, take_every=1, bootstrap=False, min_snps=2, sample_every=None):
    #
    # 1. Take all reads with any switches
    #
    # - High quality read (same strand, MAPQ, mismatches and clipping)
    # - No contamination
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
        .filter(~pl.col("is_contamination"))
        .filter((pl.col("min_coverage_hap1") >= 3) & (pl.col("min_coverage_hap2") >= 3))
        .filter(pl.col("full_read_crossover_prob") > 0)
        .filter((pl.col("before_read_cM") > 0) & (pl.col("after_read_cM") > 0))
        .filter(pl.col("between_high_quality_snps_cM").list.min() > 0)
        .filter(pl.col("high_quality_snp_positions").list.len() >= min_snps)
        .filter(pl.col("high_quality_classification"))
        .filter(pl.col("high_quality_classification_class") != "CNCO")
        .collect(streaming=True)
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
    # 2. Take a subset of reads without switches
    #
    # - High quality read (same strand, MAPQ, mismatches and clipping)
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
        .filter((pl.col("before_read_cM") > 0) & (pl.col("after_read_cM") > 0))
        .filter(pl.col("between_high_quality_snps_cM").list.min() > 0)        
        .filter(pl.col("high_quality_snp_positions").list.len() >= min_snps)
        .filter(pl.col("idx_transitions").is_null())
        .gather_every(take_every)
        .collect(streaming=True)
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
            idx_transitions = pl.when(pl.col("idx_transitions").is_null())
                .then([])
                .otherwise(pl.col("idx_transitions")),
            weight = pl.lit(take_every),
        )
    )
    
    # Combine
    callset_df = pl.concat([cand_df, other_df])
    
    if sample_every is not None:
        callset_df = callset_df.gather_every(sample_every)
    
    # Add between SNPs in bp
    if len(callset_df):
        callset_df = (callset_df                  
            .with_columns(
                pl.lit([]).list.concat([
                    pl.lit(0),
                    pl.col("high_quality_snp_positions"),
                    pl.col("read_length"),
                ]).list.diff(null_behavior="drop").alias("between_high_quality_snps_bp")
            )
        )
        
        # Bootstrap if needed
        if bootstrap:
            callset_df = callset_df.sample(n = len(callset_df), with_replacement = True)
    
    return callset_df
    

        

        
       
