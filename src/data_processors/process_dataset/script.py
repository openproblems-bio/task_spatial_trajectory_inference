import sys
import random
import numpy as np
import anndata as ad
import openproblems as op

## VIASH START
par = {
    'input': 'resources_test/common/cxg_mouse_pancreas_atlas/dataset.h5ad',
    'seed': None,
    'obs_batch': 'batch',
    'obs_label': 'cell_type',
    'obs_ptime': 'pseudotime_true',
    'obsm_spatial': 'X_spatial',
    'layer_counts': 'counts',
    'output_dataset': 'dataset.h5ad',
    'output_solution': 'solution.h5ad'
}
meta = {
    'resources_dir': 'target/executable/data_processors/process_dataset',
    'config': 'target/executable/data_processors/process_dataset/.config.vsh.yaml'
}
## VIASH END

# import helper functions
sys.path.append(meta['resources_dir'])
from subset_h5ad_by_format import subset_h5ad_by_format

# read viash config
config = op.project.read_viash_config(meta["config"])

# set seed if need be
if par["seed"]:
    print(f">> Setting seed to {par['seed']}", flush=True)
    random.seed(par["seed"])

# read the dataset
print(">> Load data", flush=True)
input = ad.read_h5ad(par['input'])
print("input:", input, flush=True)

# map the source slots of the common dataset onto the dest slots expected by the
# task-specific file formats (file_dataset.yaml / file_solution.yaml)
slot_mapping = {
    "layers": {
        "counts": par["layer_counts"],
    },
    "obs": {
        "cell_type": par["obs_label"],
        "batch": par["obs_batch"],
        "pseudotime_true": par["obs_ptime"],
    },
    "obsm": {
        "X_spatial": par["obsm_spatial"],
    },
}

print(">> Creating input data for the methods", flush=True)
output_dataset = subset_h5ad_by_format(
    input,
    config,
    "output_dataset",
    slot_mapping
)

print(">> Creating solution data", flush=True)
output_solution = subset_h5ad_by_format(
    input,
    config,
    "output_solution",
    slot_mapping
)

print(">> Writing data", flush=True)
output_dataset.write_h5ad(par["output_dataset"])
output_solution.write_h5ad(par["output_solution"])
