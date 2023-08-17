from mpi4py import MPI
import itertools
import os
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
#########################################################################
# HELPER FUNCTION FOR DATA LOADINT INTO THE SYSTEM
# DATA ARRAY MUST BE LOADED AND PARTITIONED BEFORE STARTING CALCULAITONS
# IN PRACTICE IT WILL WORK FOR 'FOR' LOOP WITHOUT INTERNAL CONDITIONS

def create_result_directory(name):
    # timestamp = "remember_to_copy"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    directory_name = f"{name}_{timestamp}"

    # Check if directory already exists
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory {directory_name} created successfully!")
    else:
        print(f"Directory {directory_name} already exists.")
    return directory_name

def loadData():
    import csv
    res = []
    with open('inputData.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            tmp = row[0].split(",")
            res.append((int(tmp[0]), tmp[1]))
    return res[:100]

with open('results.csv', 'w') as f:
    f.write('source,target,score\n')

dir_name = create_result_directory("results")

#########################################################################
if rank == 0:
    dir_name = create_result_directory("results")

    # data = getCombinations(readData()) #list(itertools.combinations([1,2,3,4], 2))
    data = list(itertools.combinations(loadData(), 2))

    # determine the size of each sub-task
    ave, res = divmod(len(data), nprocs)
    counts = [ave + 1 if p < res else ave for p in range(nprocs)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]

    # converts data into a list of arrays 
    data = [data[starts[p]:ends[p]] for p in range(nprocs)]
else:
    data = None

data = comm.scatter(data, root=0)

print(f"rank: {rank}, numprocess: {nprocs}")
file_path = os.path.join(dir_name, f"input_{rank}.csv")
with open(file_path, 'w') as f:
    f.write('source_index,source,target_index,target\n')

with open(file_path, 'a') as f:
    for (ix, x), (iy, y) in data:
        f.write(f'{ix},{x},{iy},{y}\n')

os.system("python yourCustomScript.py {} ".format(file_path))
