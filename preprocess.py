import pandas as pd
import numpy as np
import tqdm
import os

def preprocess(dataname):
    file = "data/protein_links.txt"

    with open(os.path.join("data",dataname),"r") as f:
        data = pd.read_csv(f,sep='\t')
        genes = data.columns.values
    genes = genes[0].split(',')[1:]
    num_genes = len(genes)
    graph = np.zeros([num_genes,num_genes])
    f = open(file,'r')
    lines = f.readlines()
    v = 0
    for l in tqdm.tqdm(range(1,len(lines))):
        line =  lines[l]
        v1,v2, weight = line.split('\t')
        weight = int(weight[:-1])
        if v == v1:
            i1 = i
        else:
            if v1 in genes:
                i1 = genes.index(v1)
            else:
                continue
        if v2 in genes:
            i2 = genes.index(v2)
        else:
            continue
        graph[i1,i2] = weight
        v = v1
        i = i1
    # print(graph)
    graph_path = os.path.join("graph","graph_"+dataname[:-4]+".npy")
    np.save(graph_path,graph)