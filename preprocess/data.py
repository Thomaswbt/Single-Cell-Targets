import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
import json

def process_mat(DATA_PATH, NAME):
    '''
    Output GBM CSV shape: #samples (948) * #gene (12916)
    Output IDH-MUT CSV shape: #samples (809) * #gene (13197)
    '''
    print("Processing expression data from {}".format(DATA_PATH))
    genes, samples, data = [], [], []
    lines = open(DATA_PATH).readlines()
    line_cnt = 0
    for line in tqdm(lines, total=len(lines)):
        line_split = line.split('\t')
        if line_cnt == 0:
            samples += line_split[1:]
            print("first 5 samples: {}".format(samples[:5]))
        else:
            line_data = []
            assert len(line_split) == len(samples)+1, "Invalid data line!"
            genes.append(line_split[0])
            line_data += [float(item) for item in line_split[1:]]
            data.append(line_data)
        line_cnt += 1
    data = np.array(data).T
    print("Expression data shape: {}".format(data.shape))
    expression_df = pd.DataFrame(data=data, columns=genes, index=samples)
    print('Saving {} expression data csv...'.format(NAME))
    os.makedirs('./processed', exist_ok=True)
    expression_df.to_csv('./processed/{}.csv'.format(NAME))

def extract_genes():
    '''
    Output # GBM genes: 12916
    Output # IDH-MUT genes: 13197
    Output # total genes: 14334
    '''
    GBM_PATH = './SCP936/expression/GBM_raw_counts.txt'
    IDH_PATH = './SCP936/expression/IDH-MUT_raw_counts.txt'
    GBM_genes, IDH_genes = [], []
    lines = open(GBM_PATH).readlines()
    for id, line in enumerate(tqdm(lines, total=len(lines))):
        line_split = line.split('\t')
        if id > 0:
            GBM_genes.append(line_split[0])
    lines = open(IDH_PATH).readlines()
    for id, line in enumerate(tqdm(lines, total=len(lines))):
        line_split = line.split('\t')
        if id > 0:
            IDH_genes.append(line_split[0])    
    print('# Genes of GBM matrix: {}\n# Genes of IDH-mut matrix: {}'.format(len(GBM_genes), len(IDH_genes)))
    os.makedirs('./genes', exist_ok=True)
    total_genes = set(GBM_genes + IDH_genes)
    print('# Total genes: {}'.format(len(total_genes)))
    with open('./genes/GBM_genes.pkl', 'wb') as f:
        pickle.dump(GBM_genes, f)
    with open('./genes/IDH-MUT_genes.pkl', 'wb') as f:
        pickle.dump(IDH_genes, f)
    with open('./genes/total_genes.pkl', 'wb') as f:
        pickle.dump(total_genes, f)


def process_targets():
    """
    Process the target scores 
    Find intersection with the genes from the GBM matrix.

    """
    TARGET_PATH = './EFO_0005543-associated-diseases.json'
    GBM_GENE_PATH = './genes/GBM_genes.pkl'
    with open(TARGET_PATH, 'r') as f:
        targets = json.load(f)
    # print(targets[:5])
    print("# Known target genes: {}".format(len(targets)))
    with open(GBM_GENE_PATH, 'rb') as f:
        GBM_genes = pickle.load(f)
    print("# genes in GBM matrix: {}".format(len(GBM_genes)))
    GBM_genes_set = set(GBM_genes)  # For faster search
    overlap_genes = []  # Overlapping genes for GBM and targets
    index = []  # the index of each overlapping gene in the original GBM matrix
    score_items = ["overallAssociationScore", "geneticAssociations", "somaticMutations", "drugs", "pathwaysSystemsBiology", \
        "textMining", "rnaExpression", "animalModels"]
    # overall, genetic, somatic, drugs, pathways, text, rna, animals = [], [], [], [], [], [], [], []
    # score_lists = [[] for _ in score_items]
    overall_scores = []
    invalid_cnt = [0 for _ in score_items]
    idx2gene, gene2idx = dict(), dict()
    cnt = 0

    for target in targets:
        if target["symbol"] in GBM_genes_set:
            idx = GBM_genes.index(target["symbol"])
            index.append(idx)
            overlap_genes.append(target)
            overall_scores.append(target["overallAssociationScore"])
            idx2gene[cnt] = target["symbol"]
            gene2idx[target["symbol"]] = cnt
            cnt += 1
            for i, score_item in enumerate(score_items):
                if target[score_item] != "No data":
                    pass
                else:
                    invalid_cnt[i] += 1

    print("# Overlapping genes: {}".format(len(overlap_genes)))
    for item, cnt in zip(score_items, invalid_cnt):
        print("# Invalid items of {}: {}".format(item, cnt))
    print(index[:5])
    print(overall_scores[:5])
    # print(idx2gene[3])
    # print(gene2idx['PTEN'])

    with open("./targets/overall_scores.pkl", 'wb') as f:
        pickle.dump(overall_scores, f)
    with open("./targets/index.pkl", 'wb') as f:
        pickle.dump(index, f)
    with open("./targets/idx2gene.pkl", 'wb') as f:
        pickle.dump(idx2gene, f)
    with open("./targets/gene2idx.pkl", 'wb') as f:
        pickle.dump(gene2idx, f)        


def process_embeddings():
    NORM_PATH = './embeddings/GBM_gene_embed_0.npy'
    REG_PATH = './embeddings/GBM_gene_embed_1e-2.npy'
    INDEX_PATH = "./targets/index.pkl"

    with open(INDEX_PATH, 'rb') as f:
        target_index = pickle.load(f)
        print(len(target_index))

    norm_embed = np.load(NORM_PATH)
    reg_embed = np.load(REG_PATH)        
    print(norm_embed.shape)
    print(reg_embed.shape)
    
    norm_embed = norm_embed[target_index]
    reg_embed = reg_embed[target_index]
    print(norm_embed.shape)
    print(reg_embed.shape)
    print(norm_embed[0])
    print(reg_embed[0])
    np.save('./targets/GBM_gene_embed_0.npy', norm_embed)
    np.save('./targets/GBM_gene_embed_1e-2.npy', reg_embed)


def process_labels(NAME):
    # with open('./genes/{}_genes.pkl'.format(NAME),'rb') as f:
    #     cell_names = pickle.load(f)
    lines = open('./SCP936/expression/{}_raw_counts.txt'.format(NAME), 'r').readlines()
    cell_names = lines[0].split('\t')[1:]
    cell_names[-1] = cell_names[-1].strip()
    print('# cells: {}'.format(len(cell_names)))
    cell2idx = dict()
    label2idx = {"Immune": 0, "Glial":1, "Malignant": 2}
    for i, gene in enumerate(cell_names):
        cell2idx[gene] = i
    labels = ["" for _ in range(len(cell_names))]

    lines = open('./SCP936/cluster/UMAP_{}.txt'.format(NAME), 'r').readlines()

    for i, line in enumerate(tqdm(lines, total=len(lines))):
        line_split = line.split('\t')
        if i < 2:
            continue
        else:
            idx = cell2idx[line_split[0]]
            labels[idx] = label2idx[line_split[3].strip()]
            if i == 3:
                print(line_split)
                print(label2idx[line_split[3].strip()])
    
    print(cell_names[:5])
    print(labels[:5])

    df = pd.read_csv('./processed/{}.csv'.format(NAME), index_col=0)
    df.insert(df.shape[1], 'CELL_TYPE', labels)
    df.to_csv('./processed_types/{}.csv'.format(NAME))
    # with open('./processed/{}_types.pkl'.format(NAME), 'wb') as f:
    #     pickle.dump(labels, f)


def main():
    GBM_PATH = './SCP936/expression/GBM_raw_counts.txt'
    IDH_PATH = './SCP936/expression/IDH-MUT_raw_counts.txt'
    assert os.path.exists(GBM_PATH), "GBM_PATH is invalid!"
    assert os.path.exists(IDH_PATH), "IDH_PATH is invalid!"
    process_mat(GBM_PATH, 'GBM')
    process_mat(IDH_PATH, 'IDH-MUT')


if __name__ == '__main__':
    # main()
    # extract_genes()
    # process_targets()
    # process_embeddings()
    process_labels('GBM')
    process_labels('IDH-MUT')