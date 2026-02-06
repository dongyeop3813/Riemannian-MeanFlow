import os
import pickle
import torch
from typing import Any
import string
import numpy as np

from .openfold_fns import residue_constants

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + " "
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = {i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = ["atom_positions", "aatype", "atom_mask", "residue_index", "b_factors"]

NUM_TOKENS = residue_constants.restype_num
MASK_TOKEN_INDEX = residue_constants.restypes_with_x.index("X")
CA_IDX = residue_constants.atom_order["CA"]


def write_pkl(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location="cpu"):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as handle:
                return pickle.load(handle)
    except Exception as e:
        if verbose:
            print(f"Failed to read {read_path}: {e}")
        raise e


def parse_chain_feats(chain_feats, scale_factor=1.0, center=True):
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, CA_IDX]
    bb_pos = chain_feats["atom_positions"][:, CA_IDX]
    if center:
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
        centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
        scaled_pos = centered_pos / scale_factor
    else:
        scaled_pos = chain_feats["atom_positions"] / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, CA_IDX]
    return chain_feats


def read_clusters(file_path, synthetic=False):
    pdb_to_cluster = {}
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            for chain in line.split(" "):
                if not synthetic:
                    pdb = chain.split("_")[0].strip()
                else:
                    pdb = chain.strip()
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster
