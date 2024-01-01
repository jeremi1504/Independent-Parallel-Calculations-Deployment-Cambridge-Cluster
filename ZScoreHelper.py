
import glob
import pandas as pd
from copy import deepcopy
import numpy as np
import re
import ZScorePW


def calculate_start_dist(row):
    dists = row['start_list'].split("-")
    c2h2_start_dist_list = ["0"]
    for i, dist in enumerate(dists[1:]):
        c2h2_start_dist_list.append(str(int(abs(float(dists[i]) - float(dists[i+1])))))
            
    return '-'.join(c2h2_start_dist_list)


def split_misassembled_ZNFs(df):
    #df["c2h2_start_dist"] = ""
    rows_to_drop = list()
    new_rows = list()
    df["c2h2_start_dist"] = df.apply(calculate_start_dist, axis=1)
    for row_index, row in df.iterrows():
            
        if row['krab_close'] == True:
            dists = row['c2h2_start_dist'].split("-")
            index = list()
            for i, dist in enumerate(dists[1:]):
                if float(dist) > 3000:
                    index.append(i + 1)

            if index:
                split_frame = np.split(row["frame"].split("-"), index)
                split_c2h2_dist = np.split(row["c2h2_dist"].split("-"), index)
                split_c2h2_start_list = np.split(row["c2h2_start_dist"].split("-"), index)
                
                split_3AA = np.split(row["3AA"].split("-"), index)
                split_3AA_pos = np.split(row["3AA_pos"].split("-"), index)
                split_ZNFs = np.split(row["ZNFs"].split("-"), index)
                #print(row["Ensembl Gene ID"])
                #print(index)
                #print(split_c2h2_dist)
                previous_length = 0
                for i in range(len(split_frame)):
                    new_row = deepcopy(row)
                    
                    new_row["frame"] = '-'.join(list(split_frame[i]))
                    new_row["c2h2_dist"] = '-'.join(list(split_c2h2_dist[i]))
                    new_row["c2h2_start_dist"] = '-'.join(list(split_c2h2_start_list[i]))
                    new_row["3AA"] = '-'.join(list(split_3AA[i]))
                    new_row["3AA_pos"] = '-'.join(list(split_3AA_pos[i]))
                    new_row["ZNFs"] = '-'.join(list(split_ZNFs[i]))
                    new_row["# of ZNFs"] = len(split_frame[i])
                    count = 0
                    for znf in list(split_3AA[i]):
                        if (znf == 'XXX') or (znf == '(XXX)'):
                            count += 1
                    new_row["# of ZNFs cannonical"] = len(split_frame[i]) - count
                    length = sum(map(float, split_c2h2_start_list[i][1:])) + previous_length
                    if i > 0:
                        previous_length += float(split_c2h2_start_list[i][0])
                        
                    if row["strand"] == "+":
                        new_row["start"] = row["start"] + previous_length
                        new_row["end"] = new_row["start"] + length
                        previous_length += length
                    else:
                        new_row["end"] = row["end"] - previous_length
                        new_row["start"] = new_row["end"] - length
                        previous_length += length

                    new_rows.append(new_row)
                rows_to_drop.append(row_index)

    if rows_to_drop:
        df.drop(rows_to_drop, inplace=True)
    if new_rows:
        new_rows = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows], ignore_index=True)
    return df

#make sure the sort is numeric
def filter_overlapping(df):
    df.sort_values(["Species", "chrom", "start", "end"], ascending=True, inplace=True)
    df["remove"] = False
    
    previous_max_start = 0
    previous_max_end = -1
    previous_chr = ""
    previous_species = ""
    
    for index, row in df.iterrows():
        if row["Species"] == previous_species:
            if row["chrom"] == previous_chr:
                if (previous_max_end != -1) and ((row["start"] >= previous_max_end) or ((previous_max_end - row["start"]) / (previous_max_end - previous_max_start) < 0.1)):
                    previous_max_start = row["start"]
                    previous_max_end = row["end"]
                    max_len = (row["end"] - row["start"])
                    best_id = index
                else:
                    previous_max_start = min(previous_max_start, row["start"])
                    previous_max_end = max(previous_max_end, row["end"])
                    if (row["end"] - row["start"]) > max_len:
                        max_len = (row["end"] - row["start"])
                        #remove previous best id
                        df.loc[best_id, "remove"] = True
                        #update best id to current one
                        best_id = index
                    else:
                        df.loc[index, "remove"] = True
                        
            else:
                previous_max_start = row["start"]
                previous_max_end = row["end"]
                previous_chr = row["chrom"]
                best_id = index
                max_len = (row["end"] - row["start"])
        else:
            #new species
            previous_max_start = row["start"]
            previous_max_end = row["end"]
            previous_chr = row["chrom"]
            previous_species = row["Species"]
            best_id = index
            max_len = (row["end"] - row["start"])
    
    df = df[df["remove"] == False]
    df = df[df["# of ZNFs cannonical"] >= 3]

    df.reset_index(inplace=True, drop=True)

    return df

def clean_text(text):
    text = text.replace('(', '').replace(')', '')
           
    # Remove starting patterns "XXX-" (can be multiple)
    text = re.sub(r'^(XXX-)+', '', text)

    # Remove trailing patterns "-XXX" (can be multiple)
    text = re.sub(r'(-XXX)+$', '', text)
    text = text.replace('-', '')
    return text


def applyClearnig(df_test):
    df_test = split_misassembled_ZNFs(df_test)
    df_test = filter_overlapping(df_test)
    df_test = df_test[df_test["remove"] == False]
    df_test = df_test[df_test["# of ZNFs cannonical"] >= 3]
    df_test.reset_index(inplace=True, drop=True)
    df = df_test.rename(columns={'3AA': 'x3aa'})
    df['x3aa'] = df['x3aa'].str.replace('(', '').str.replace(')', '')
    df = df[~df['chrom'].str.startswith('CHR_')]

    # Check how much the frequency of amino acids in the dataset differs from frequencies that were used to create BLOSUM62
    df['x3aa'] = df['x3aa'].str.replace('-', '').str.replace('X', '')
    aa = pd.Series(list(df['x3aa'].str.cat().upper()))
    total_aa = aa.count()
    aa_percentages = (aa.value_counts() / total_aa).round(3)
    aa_percentages_df = aa_percentages.reset_index()
    aa_percentages_df.columns = ['AA', 'Dataset']

    # Original frequencies from BLOSUM62 (copied from Eddy 2004)
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    blosum62 = [0.074, 0.052, 0.045, 0.054, 0.025, 0.034, 0.054, 0.074, 0.026, 0.068, 0.099, 0.058, 0.025, 0.047, 0.039, 0.057, 0.051, 0.013, 0.034, 0.073]
    blosum62_freqs = pd.DataFrame({'AA': amino_acids, 'Blosum62': blosum62})

    # Merge the two DataFrames
    df = pd.merge(aa_percentages_df, blosum62_freqs, on='AA', how='left')
    
    return df

def generate_matrix(df):
    # BLOSUM62 matrix 
    blosum62 = {
        'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
        'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
        'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
        'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
        'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
        'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
        'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
        'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
        'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
        'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
        'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
        'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
        'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
        'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
        'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
        'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
        'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
        'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
        'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
        'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4},}


    # Background frequencies of amino acids used in 1992 to calculate BLOSUM62 (data from Eddy 2004)
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    frequencies = [0.074, 0.052, 0.045, 0.054, 0.025, 0.034, 0.054, 0.074, 0.026, 0.068, 0.099, 0.058, 0.025, 0.047, 0.039, 0.057, 0.051, 0.013, 0.034, 0.073]
    background_freq = dict(zip(amino_acids, frequencies))


    # Scaling factor for BLOSUM62 (as in the original matrix)
    lambda_val = 0.3176

    # Compute q_ij
    p = np.array(frequencies)

    # Create an empty matrix for q_ij
    q = np.zeros((len(amino_acids), len(amino_acids)))

    # Calculate q_ij values
    for i, aa1 in enumerate(amino_acids):
        for j, aa2 in enumerate(amino_acids):
            q[i, j] = p[i] * p[j] * np.exp(lambda_val * blosum62[aa1][aa2])

    # Normalize q_ij so that its elements sum to 1
    q /= np.sum(q)
    amino_acids_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    dataset_frequencies = dict(zip(df['AA'], df['Dataset']))

    new_frequencies = [dataset_frequencies[aa] if aa in dataset_frequencies else 0 for aa in amino_acids_order]
    new_background_freq = dict(zip(amino_acids, new_frequencies))

    # Compute e_ij (expected frequency matrix based on new background frequencies)
    new_p = np.array(new_frequencies)
    e = np.outer(new_p, new_p)

    # Adjust the BLOSUM62 substitution matrix using the formula:
    # S'_ij = (1/lambda) * log(q_ij / e_ij)

    new_blosum62 = {}

    for i, aa1 in enumerate(amino_acids):
        new_blosum62[aa1] = {}
        for j, aa2 in enumerate(amino_acids):
            # Guard against log(0)
            if q[i, j] > 0 and e[i, j] > 0:
                new_blosum62[aa1][aa2] = (1/lambda_val) * np.log(q[i, j] / e[i, j])
            else:
                new_blosum62[aa1][aa2] = -np.inf  # You can also assign a very large negative number


    # Convert the adjusted BLOSUM62 matrix into a DataFrame and round to the nearest integer
    df_new_blosum62 = pd.DataFrame(new_blosum62)

    ## CORRECTION NOT TO PUNISH THE EXACT MATCHES FOR COMMON LETTERS
    for col in df_new_blosum62.columns:
        if df_new_blosum62.loc[col, col] < 1:
            df_new_blosum62.loc[col, col] = 3

    return df_new_blosum62 #df_new_blosum62.round(0).astype(int)

def df_to_tuple_dict(df):
    tuple_dict = {}
    for aa1 in df.columns:
        for aa2 in df.index:
            tuple_dict[(aa1, aa2)] = df[aa1][aa2]
    return tuple_dict

def generate_matrix_as_tupledict(df):
    return df_to_tuple_dict(generate_matrix(df))

def loadDistances(fileName: str = "HM_KRAB.csv"):
    ZNF_seq = pd.read_csv(fileName)

    ZNF_seq['3AA'] = ZNF_seq['3AA'].astype(str)
    sequences = ZNF_seq['3AA']
    sequences = sequences.apply(clean_text)

    ZNF_seq['c2h2_dist'] = ZNF_seq['c2h2_dist'].astype(str)
    distances = ZNF_seq['c2h2_dist'].apply(lambda x: np.array([float(i) for i in x.split('-', 1)[-1].split('-')], dtype=float))
    return distances, ZNF_seq, sequences

def loadRawData():
    df_total = pd.DataFrame()
    genes_to_process = [file for file in glob.glob("Data/Human_mouse/*.txt")]
    for genes in genes_to_process:
        df = pd.read_table(genes)
        df = df[df["# of ZNFs cannonical"] >= 3]
        #df = df[df["krab_close"] == True]
        df_total = pd.concat([df_total, df], ignore_index=True)

    df_total.loc[(df_total["Gene symbol"] == "Not found") | (df_total["Gene symbol"] == "."), "Gene symbol"] = ""

    df_test = df_total.copy(deep=True)
    return df_test

def AddColToBrokenChain(df_new_blosum62):
    df_new_blosum62['X'] = 0
    new_row = pd.Series(0, index=df_new_blosum62.columns)
    new_row['X'] = 1
    df_new_blosum62.loc['X'] = new_row
    new_blosum_alpha = list(df_new_blosum62.columns)  
    new_blosum_array = df_new_blosum62.values
    return df_new_blosum62, new_blosum_alpha, new_blosum_array

def getMatrixPipeline():
    dfRaw = loadRawData()
    dfCleaned = applyClearnig(dfRaw)
    df_new_blosum62 = generate_matrix(dfCleaned)
    new_blosum62_tuple = df_to_tuple_dict(df_new_blosum62)
    return new_blosum62_tuple, AddColToBrokenChain(df_new_blosum62)
            

def convert_sequence(seq, new_blosum_alpha):
    seq_conv = np.zeros(len(seq), dtype=np.int32)
    for i in range(len(seq)):
        seq_conv[i] = new_blosum_alpha.index(seq[i])
    return seq_conv
                                
# if __name__ =="__main__":
#     distances, ZNF_seq, sequences = loadDistances()
#     new_blosum62_tuple, (df_new_blosum62, new_blosum_alpha, new_blosum_array) = getMatrixPipeline()
#     seq = ZNF_seq['3AA'].astype(str)
#     seq = seq.apply(clean_text)
#     #ZScorePW.testFunction(seq[i1], seq[i2], distances[i1], distances[i2], new_blosum_alpha, df_new_blosum62)
#     res = ZScorePW.compare_sequences(seq, distances, df_new_blosum62)
#     print(res)