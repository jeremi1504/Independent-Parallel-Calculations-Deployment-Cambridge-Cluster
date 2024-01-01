###########################################################
###### PYTHON CODE not  compiled !!! 
###### Use it only when it executes only onece as the 
###### native python code is super slow 
###########################################################
## Create input file from the .txt Snakemake pipeline outputs


###########################################################
###### CYTHON CODE COMPILED !!! 
###### See setup.py for the compilation flags 
###########################################################

# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list as c_list
from libc.math cimport floor, ceil, fmax
from libcpp.map cimport map
from libcpp.pair cimport pair
from cython.parallel import parallel, prange

cdef class OnlineStats:
    cdef public int n
    cdef public double mean, M2

    def __cinit__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    cpdef update(self, double x):
        self.n += 1
        cdef double delta = x - self.mean
        self.mean += delta / self.n
        cdef double delta2 = x - self.mean
        self.M2 += delta * delta2

    cpdef tuple finalize(self):
        if self.n < 2:
            return float('nan'), float('nan')
        else:
            return self.mean, (self.M2 / (self.n - 1)) ** 0.5

def parse_distances(dist_str):
    return [float(d) for d in dist_str.split('-')]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double distance_similarity(const double[:] distances1, const double[:] distances2, int i, int j) nogil:
    if i >= distances1.shape[0] or j >= distances2.shape[0]:
        return 1.0
    cdef double dist1 = distances1[i]
    cdef double dist2 = distances2[j]
    if dist1 == 0 and dist2 == 0:
        return 1.0
    elif dist1 == 0 or dist2 == 0:
        return 0.0
    else:
        return min(dist1, dist2) / max(dist1, dist2)

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef float sequence_compare_cython(const vector[int] &seq1, const vector[int] &seq2,  const double[:] distances1, const double[:] distances2, const double[:,:] &matrix) nogil:
    cdef int i, j, k
    cdef float score, total_alignment_score, dist_sim

    cdef int gap_open_penalty = 12
    cdef int gap_extension_penalty = 1
    cdef int endpenalty = 3
    
    cdef int len_seq1 = seq1.size() // 3
    cdef int len_seq2 = seq2.size() // 3

    cdef vector[float] pairscore
    pairscore.resize((len_seq1+1) * (len_seq2+1), 0.0)

    for i in range(1, len_seq1+1):
        for j in range(1, len_seq2+1):
            score = 0.0
            for k in range(3):
                score += matrix[seq1[3*(i-1)+k], seq2[3*(j-1)+k]]
            dist_sim = distance_similarity(distances1, distances2, i-1, j-1)
            if score < 0:
                pairscore[i*(len_seq2+1) + j] = score
            else:
                pairscore[i*(len_seq2+1) + j] = score * dist_sim

    cdef vector[float] cumul_score
    cumul_score.resize((len_seq1+1) * (len_seq2+1), 0)

    for i in range(len_seq1+1):
        cumul_score[i*(len_seq2+1)] = -i*endpenalty
    for j in range(len_seq2+1):
        cumul_score[j] = -j*endpenalty

    cdef vector[float] gap_open_seq1
    cdef vector[float] gap_open_seq2
    gap_open_seq1.resize((len_seq1+1) * (len_seq2+1), 0)
    gap_open_seq2.resize((len_seq1+1) * (len_seq2+1), 0)

    cdef float localmax, temp
    for i in range(1, len_seq1+1):
        for j in range(1, len_seq2+1):
            localmax = cumul_score[((i-1)*(len_seq2+1)) + (j-1)] + pairscore[i*(len_seq2+1) + j]
            temp = cumul_score[((i-1)*(len_seq2+1)) + j] - gap_open_penalty
            if temp > localmax:
                localmax = temp
            temp = cumul_score[(i*(len_seq2+1)) + (j-1)] - gap_open_penalty
            if temp > localmax:
                localmax = temp
            temp = gap_open_seq1[((i-1)*(len_seq2+1)) + j] - gap_extension_penalty
            if temp > localmax:
                localmax = temp
            temp = gap_open_seq2[(i*(len_seq2+1)) + (j-1)] - gap_extension_penalty
            if temp > localmax:
                localmax = temp

            cumul_score[i*(len_seq2+1) + j] = localmax
            gap_open_seq1[i*(len_seq2+1) + j] = max(
                gap_open_seq1[((i-1)*(len_seq2+1)) + j] - gap_extension_penalty,
                cumul_score[((i-1)*(len_seq2+1)) + j] - gap_open_penalty
            )
            gap_open_seq2[i*(len_seq2+1) + j] = max(
                gap_open_seq2[(i*(len_seq2+1)) + (j-1)] - gap_extension_penalty,
                cumul_score[(i*(len_seq2+1)) + (j-1)] - gap_open_penalty
            )

    cdef float end_gap_1_max, end_gap_2_max
    end_gap_1_max = -9999
    for i in range(0, len_seq2+1):
        temp = cumul_score[len_seq1*(len_seq2+1) + len_seq2-i] - i*endpenalty
        if temp > end_gap_1_max:
            end_gap_1_max = temp
    
    end_gap_2_max = -9999
    for i in range(0, len_seq1+1):
        temp = cumul_score[(len_seq1-i)*(len_seq2+1) + len_seq2] - i*endpenalty
        if temp > end_gap_2_max:
            end_gap_2_max = temp
    
    if end_gap_1_max >= end_gap_2_max:
        total_alignment_score = end_gap_1_max
    else:
        total_alignment_score = end_gap_2_max
    
    return total_alignment_score

import pandas as pd
import numpy as np
import itertools


def testCompute(seq1_conv, seq2_conv, dist1, dist2, new_blosum_array):
    score = sequence_compare_cython(seq1_conv, seq2_conv, dist1, dist2, new_blosum_array)
    return score

def compare_sequences_generators(indexes, seqMap, distMap, df_new_blosum62):
    
    def convert_sequence(seq):
        seq_conv = np.zeros(len(seq), dtype=np.int32)
        for i in range(len(seq)):
            seq_conv[i] = new_blosum_alpha.index(seq[i])
        return seq_conv

    new_blosum_alpha = list(df_new_blosum62.columns)  
    new_blosum_array = df_new_blosum62.values

    for ix, iy in indexes:
        seq1 = seqMap[ix]
        seq2 = seqMap[iy]
        dist1 = distMap[ix]
        dist2 = distMap[iy]
        seq1_conv = convert_sequence(seq1.replace("-", ""))
        seq2_conv = convert_sequence(seq2.replace("-", ""))
        score = sequence_compare_cython(seq1_conv, seq2_conv, dist1, dist2, new_blosum_array)


def compare_sequences(sequences, distances, df_new_blosum62):
    sequences = sequences.values
    distances = distances.values
    indices = range(len(sequences))

    new_blosum_alpha = list(df_new_blosum62.columns)  
    new_blosum_array = df_new_blosum62.values

    def convert_sequence(seq):
        seq_conv = np.zeros(len(seq), dtype=np.int32)
        for i in range(len(seq)):
            seq_conv[i] = new_blosum_alpha.index(seq[i])
        return seq_conv

    results = []
    stats = {i: OnlineStats() for i in range(len(sequences))}
    
    for i, j in itertools.combinations(indices, 2):
        seq1 = sequences[i]
        seq2 = sequences[j]
        dist1 = distances[i]
        dist2 = distances[j]

        seq1_conv = convert_sequence(seq1.replace("-", ""))
        seq2_conv = convert_sequence(seq2.replace("-", ""))


        score = sequence_compare_cython(seq1_conv, seq2_conv, dist1, dist2, new_blosum_array)
        
        stats[i].update(score)
        stats[j].update(score)
        
        if score > 7: # threshold approximately corresponding to matching exactly one triplet 1:1          
            results.append((i, j, score))
    
    # with open('stats.csv', 'w') as f:
    #     f.write("Sequence ID,Mean,SD\n")
    #     for seq_id, stat in stats.items():
    #         mean, sd = stat.finalize()
    #         f.write(f"{seq_id},{mean},{sd}\n")

    df_results = pd.DataFrame(results, columns=["Source", "Target", "Weight"])
    return df_results
