#### Usage:

```shell
pip install . # Install modified scETM
python run.py # Training
```



##### Note:

- Put `GBM.csv`, `IDH-MUT.csv` and `protein_links.txt` into a `data` folder.

- When running `run.py` for the first time, it would preprocess the `protein_links.txt` and construct a graph (represented by an adjacent matrix), which takes around 15 minutes.

- Added a regularization term for the PPI graph in `src/scETM/scETM.py`, which computes
  $\mathcal L_{reg} = \sum_{i\neq j} w_{ij} \|\rho_i - \rho_j\|_2^2 and 
  $\mathcal L = \mathcal L_{ori} + \lambda_{reg}\mathcal L_{reg}$,
  where $i,j$ are two genes and $w_{ij}$ are the weight of the edge $(i,j)$.
