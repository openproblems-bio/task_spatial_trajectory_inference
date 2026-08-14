import os
import random
import warnings

# stlearn imports tensorflow, which logs its device setup to stderr on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import stlearn as st

## VIASH START
par = {
  'input': 'resources_test/task_spatial_trajectory_inference/dlpfc_151673/dataset.h5ad',
  'output': 'output.h5ad',
  'n_comps': 50,
  'n_neighbors': 200,
  'resolution': 0.8,
  'eps': 1500.0,
  'seed': 0,
}
meta = {
  'name': 'stlearn'
}
## VIASH END

# warnings raised by stlearn internals, not actionable from here
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', ad.ImplicitModificationWarning)

seed = par['seed']
np.random.seed(seed)
random.seed(seed)


def build_cell_type_int(adata):
    """Map cell_type strings to integer labels, as stLearn expects numeric cluster labels."""
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str).astype('category')
    unique_cell_types = adata.obs['cell_type'].cat.categories

    ct_to_num = {str(ct): str(i) for i, ct in enumerate(unique_cell_types)}
    num_to_ct = {str(i): str(ct) for i, ct in enumerate(unique_cell_types)}

    adata.obs['cell_type_int'] = (
        adata.obs['cell_type'].astype(str).map(ct_to_num).astype('category')
    )
    return num_to_ct


def select_root(adata):
    """Return the least differentiated cluster, i.e. the one whose cells express the largest
    number of genes on average. This is the same proxy stLearn uses to pick the root cell."""
    n_expressed = np.asarray((adata.layers['counts'] > 0).sum(axis=1)).reshape(-1)
    scores = (
        pd.DataFrame({'n_expressed': n_expressed, 'cluster': adata.obs['cell_type_int'].values})
        .groupby('cluster', observed=True)['n_expressed']
        .mean()
    )
    return str(scores.idxmax())


def filter_branches(available_paths, root):
    """Keep the paths starting from the root and drop branches contained in a longer one."""
    valid = [path for path in available_paths.values() if path[0] == root]
    valid.sort(key=len, reverse=True)

    unique = []
    for branch in valid:
        if not any(set(branch).issubset(set(kept)) for kept in unique):
            unique.append(branch)
    return unique


print('Reading input files', flush=True)
adata = ad.read_h5ad(par['input'])

adata.X = adata.layers['normalized']
if sp.issparse(adata.X):
    adata.X = adata.X.toarray()

# stLearn reads the spatial coordinates from these slots
adata.obsm['spatial'] = adata.obsm['X_spatial']
adata.obs['imagerow'] = adata.obsm['X_spatial'][:, 1]
adata.obs['imagecol'] = adata.obsm['X_spatial'][:, 0]

print('Embed and cluster the cells', flush=True)
st.em.run_pca(adata, n_comps=par['n_comps'])
sc.pp.neighbors(adata, n_neighbors=par['n_neighbors'], use_rep='X_pca')
st.tl.clustering.leiden(adata, resolution=par['resolution'], random_state=seed)

num_to_ct = build_cell_type_int(adata)

print('Determine root cell', flush=True)
root = select_root(adata)
print(f'Root cluster: {num_to_ct[root]}', flush=True)

# use_raw is False because the dataset does not carry a raw layer
adata.uns['iroot'] = st.spatial.trajectory.set_root(
    adata,
    use_label='cell_type_int',
    cluster=int(root),
    use_raw=False,
)

print('Calculate the pseudotime and the trajectory branches', flush=True)
st.spatial.trajectory.pseudotime(
    adata,
    eps=par['eps'],
    use_rep='X_pca',
    use_label='cell_type_int',
)
branches = filter_branches(adata.uns.get('available_paths', {}), int(root))

if not branches:
    raise RuntimeError(
        f'No valid branches found from root cluster {root}. Try adjusting --resolution or --eps.'
    )

print('Generate predictions', flush=True)
# cells not covered by any branch keep a NaN pseudotime
adata.obs['pseudotime_inferred'] = np.nan

for branch in branches:
    branch_labels = [str(node) for node in branch]
    try:
        st.spatial.trajectory.pseudotimespace_global(
            adata,
            use_label='cell_type_int',
            list_clusters=branch_labels,
        )
    except Exception as exc:
        print(f'Skipping branch {branch}: {exc}', flush=True)
        continue

    # the first branch a cell belongs to takes precedence
    unfilled = (
        adata.obs['cell_type_int'].isin(branch_labels)
        & adata.obs['pseudotime_inferred'].isna()
    )
    adata.obs.loc[unfilled, 'pseudotime_inferred'] = adata.obs.loc[unfilled, 'dpt_pseudotime']

n_assigned = adata.obs['pseudotime_inferred'].notna().sum()
print(f'Pseudotime assigned to {n_assigned}/{adata.n_obs} cells', flush=True)

print('Write output AnnData to file', flush=True)
output = ad.AnnData(
    obs=adata.obs[['pseudotime_inferred']],
    uns={
        'dataset_id': adata.uns['dataset_id'],
        'normalization_id': adata.uns['normalization_id'],
        'method_id': meta['name'],
    },
)
output.write_h5ad(par['output'], compression='gzip')
