import anndata as ad
import scipy.sparse as sp
import spaTrack as spt

## VIASH START
par = {
  'input': 'resources_test/task_spatial_trajectory_inference/cxg_mouse_pancreas_atlas/dataset.h5ad',
  'output': 'output.h5ad',
  'alpha1': 0.5,
  'alpha2': 0.5,
}
meta = {
  'name': 'spatrack'
}
## VIASH END

print('Reading input files', flush=True)
adata = ad.read_h5ad(par['input'])

adata.X = adata.layers['normalized']
if sp.issparse(adata.X):
    adata.X = adata.X.toarray()
adata.obs['cluster'] = adata.obs['cell_type'].astype(str).astype('category')

print('Calculate cell transition probability', flush=True)
adata.obsp['trans'] = spt.get_ot_matrix(
    adata,
    data_type='spatial',
    alpha1=par['alpha1'],
    alpha2=par['alpha2'],
)

print('Determine starting cells', flush=True)
adata = spt.assess_start_cluster(adata)
start_cluster = list(adata.uns['entropy value order'].index)[0]
start_cells = spt.set_start_cells(adata, select_way='cell_type', cell_type=start_cluster)

print('Generate predictions', flush=True)
adata.obs['pseudotime_inferred'] = spt.get_ptime(adata, start_cells)

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
