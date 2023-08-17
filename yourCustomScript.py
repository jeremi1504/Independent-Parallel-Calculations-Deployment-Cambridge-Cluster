# import sys
# import numpy as np
# import Bio.Align.substitution_matrices as substitution_matrices
# import csv

import numpy as np
import Bio.Align.substitution_matrices as substitution_matrices
import typing as typing
from typing import Dict, List, Union
from sys import argv
import os
blosum62 = substitution_matrices.load("BLOSUM62")

    
# Define a function for scoring triplets
def triplet_score(triplet1, triplet2):
    return blosum62[triplet1[0]][triplet2[0]] + blosum62[triplet1[1]][triplet2[1]] + blosum62[triplet1[2]][triplet2[2]]

# Define a function for scoring entire sequences
def sequence_compare(config):
    outputPath, index1, seq1, index2, seq2 = config["output_path"], config["source_index"], config["source"], config["target_index"], config["target"]
    # index1 = sys.argv[1]
    # index2 = sys.argv[3]
    # seq1 = sys.argv[2]
    # seq2 = sys.argv[4]
    # define penalties
    gappenalty = 12        # gap penalty for a 3-letter gap within the sequence 
    endpenalty = 3         # gap penalty for a 3-letter gap at the ends of the sequence
    gap_ext = 3            # gap penalty for extension of a gap within the sequence
    
    # format sequences into 3-letter triplets
    seq1list = seq1.split("-")
    seq2list = seq2.split("-")
    
    len_seq1list = len(seq1list)
    len_seq2list = len(seq2list)

    ## Pairwise alignment score calculation using an iterative method ##

    # create matrix with pairwise matching scores using BLOSUM62
    pairscore = np.zeros((len_seq1list + 1, len_seq2list + 1))
    for i in range(1, len_seq1list + 1):
        for j in range(1, len_seq2list + 1):
            if i>j: continue
            pairscore[i, j] = triplet_score(seq1list[i - 1], seq2list[j - 1])
    # print(pairscore)
    
    ## Aligning the start of the sequences
    
    # Create matrices to store gap locations and cumulative scores
    gaplocation = np.zeros((len_seq1list + 1, len_seq2list + 1))
    cum_score = np.zeros((len_seq1list + 1, len_seq2list + 1))
    
    # Add penalty to gaps in the opening
    for i in range(1, len_seq1list + 1):
        cum_score[i, 0] = -endpenalty - (i - 1) * gap_ext  
    for j in range(1, len_seq2list + 1):
        cum_score[0, j] = -endpenalty - (j - 1) * gap_ext
    
    # Aligning middle of the sequences
    for i in range(1, len_seq1list + 1):
        for j in range(1, len_seq2list + 1):
            if i>j: continue
            # Check if we have a gap in either sequence (or both)
            gap_penalty_seq1 = cum_score[i - 1, j] - (gappenalty if gaplocation[i - 1, j] == 0 else -gap_ext)
            gap_penalty_seq2 = cum_score[i, j - 1] - (gappenalty if gaplocation[i, j - 1] == 0 else -gap_ext)

            # Compute the scores for sequence extensions (gap extensions)
            extension_score_seq1 = gaplocation[i - 1, j] * -gap_ext
            extension_score_seq2 = gaplocation[i, j - 1] * -gap_ext
            
            localmax = max(
                cum_score[i - 1, j - 1] + pairscore[i, j],
                gap_penalty_seq1 + extension_score_seq1,
                gap_penalty_seq2 + extension_score_seq2,
            )
            
            if cum_score[i - 1, j - 1] + pairscore[i, j] == localmax:
                gaplocation[i, j] = 0  # seq1 and seq2 match
            elif gap_penalty_seq1 - gap_ext + extension_score_seq1 == localmax:
                gaplocation[i, j] = 1  # gap opening in seq1
            else:
                gaplocation[i, j] = 2  # gap opening in seq2
            cum_score[i, j] = localmax
    # print(cum_score)
    # print(gaplocation)
    ## Aligning the last triplets of the sequences 
    
    # if there are ending gaps in seq1 
    end_gap_pos_1 = []
    for i in range(0, len_seq2list + 1):
        end_gap_pos_1.append(cum_score[len_seq1list, len_seq2list - i] - i * endpenalty/2)
    end_gap_1_max = max(end_gap_pos_1)
    end_gap_1_index = len_seq2list - end_gap_pos_1.index(end_gap_1_max)
    end_gap_1_max -= (len_seq2list - end_gap_1_index) * endpenalty/2

    # if there are ending gaps in seq2
    end_gap_pos_2 = []
    for i in range(0, len_seq1list + 1):
        end_gap_pos_2.append(cum_score[len_seq1list - i, len_seq2list] - i * endpenalty/2)
    end_gap_2_max = max(end_gap_pos_2)
    end_gap_2_index = len_seq1list - end_gap_pos_2.index(end_gap_2_max)
    end_gap_2_max -= (len_seq1list - end_gap_2_index) * endpenalty/2

    # determine ending gap position
    if end_gap_1_max >= end_gap_2_max:
        total_alignment_score = end_gap_1_max
    else:
        total_alignment_score = end_gap_2_max
    with open(outputPath, 'a') as f:
        f.write(str(index1)+","+str(index2)+","+str(total_alignment_score)+"\n")

    return total_alignment_score

def readInputData(filepath):
    import csv
    array_of_dicts = []
    outputPath = getOutputPath(inputPath)

    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            array_of_dicts.append({
                'output_path': outputPath,
                'source_index': int(row['source_index']),
                'source': row['source'],
                'target_index': int(row['target_index']),
                'target': row['target'],
            })

    return array_of_dicts

def getOutputPath(filepath):
    return filepath.replace("input", "output")

def buildOutputDir(outputPath):
    with open(outputPath, 'w') as f:
        f.write('source_index,source,target_index,target\n')
        
def executeInParallel(inputArray):
     for config in inputArray:
        sequence_compare(config)

# if __name__ == '__main__':
inputPath =  argv[1]
outputPath = getOutputPath(inputPath)
buildOutputDir(outputPath)
inputArray = readInputData(inputPath)
num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
print(f"Num cores: {num_cores}")
for config in inputArray:
    sequence_compare(config["output_path"], config["source_index"], config["source"], config["target_index"], config["target"])

# from multiprocessing import Pool

# Assuming the SLURM_CPUS_PER_TASK environment variable is set

# executeInParallel(inputArray)

# with Pool(processes=num_cores) as pool:
#     results = pool.map(sequence_compare, inputArray)