from mpi4py import MPI
import itertools
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#########################################################################
# HELPER FUNCTION FOR DATA LOADINT INTO THE SYSTEM
# DATA ARRAY MUST BE LOADED AND PARTITIONED BEFORE STARTING CALCULAITONS
# IN PRACTICE IT WILL WORK FOR 'FOR' LOOP WITHOUT INTERNAL CONDITIONS

with open('results.csv', 'w') as f:
    f.write('source,target,score\n')

def loadData():
    import csv
    res = []
    with open('inputData.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            tmp = row[0].split(",")
            res.append((int(tmp[0]), tmp[1]))
    return res[:100]

#########################################################################
if rank == 0:
    
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

# for x, y in data:
#     os.system('echo "{} - {}" >> results.csv '.format(x, y))

for (ix, x), (iy, y) in data:
    os.system("python yourCustomScript.py {} {} {} {}".format(ix, x, iy, y))
