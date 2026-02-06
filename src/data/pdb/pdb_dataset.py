from pathlib import Path

import torch

import random
import numpy as np
import tree
import pandas as pd

from .openfold_fns import seq_to_aatype, atom37_to_frames, Rigid
from .read_and_parse import read_pkl, parse_chain_feats, read_clusters
from .filter import (
    filter_with_sequence_length,
    filter_with_max_coil_percent,
    filter_large_structures,
)

to_numpy = lambda x: x.detach().cpu().numpy()


class PDBDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, dataset_config: dict):
        super(PDBDataset, self).__init__()

        self.dataset_config = dataset_config
        self.split = split

        # Configuration for dataset filtering.
        self.filter_cfg = dataset_config.filter

        # If this flags are set, the dataset will be augmented with the redesigned and synthetic data.
        self.use_redesigned = dataset_config.use_redesigned
        self.use_synthetic = dataset_config.use_synthetic

        # Parsing the path to the data/metadata files.
        self.data_dir = Path(self.dataset_config.data_dir)
        self.setup_paths(self.data_dir, self.dataset_config)

        # Building the metadata table.
        self.metadata_csv = self.build_metadata_table()

        # Create the split (train/test), only metadata of (train/test) is loaded at this point.
        self.create_split(self.split, self.dataset_config, self.metadata_csv)

        self.cache = {}
        self.rng = np.random.default_rng(seed=self.dataset_config.seed)

    def setup_paths(self, data_dir, config):
        self.csv_path = data_dir / config.csv_path
        self.cluster_path = data_dir / config.cluster_path

        if config.use_redesigned:
            self.redesigned_csv_path = data_dir / config.redesigned_csv_path

        if config.use_synthetic:
            self.synthetic_csv_path = data_dir / config.synthetic_csv_path
            self.synthetic_cluster_path = data_dir / config.synthetic_cluster_path

    def build_metadata_table(self):
        self.raw_csv = pd.read_csv(self.csv_path)
        metadata_csv = filter_metadata(self.raw_csv, self.filter_cfg)
        metadata_csv = metadata_csv.sort_values("modeled_seq_len", ascending=False)

        self.pdb_to_cluster = read_clusters(self.cluster_path, synthetic=False)
        self.max_cluster = max(self.pdb_to_cluster.values())
        self.missing_pdbs = 0

        def cluster_lookup(pdb):
            pdb = pdb.upper()
            if pdb not in self.pdb_to_cluster:
                self.pdb_to_cluster[pdb] = self.max_cluster + 1
                self.max_cluster += 1
                self.missing_pdbs += 1
            return self.pdb_to_cluster[pdb]

        metadata_csv["cluster"] = metadata_csv["pdb_name"].map(cluster_lookup)

        if self.use_redesigned:
            self.redesigned_csv = pd.read_csv(self.redesigned_csv_path)
            metadata_csv = metadata_csv.merge(
                self.redesigned_csv, left_on="pdb_name", right_on="example"
            )
            metadata_csv = metadata_csv[metadata_csv.best_rmsd < 2.0]

        if self.use_synthetic:
            self.synthetic_csv = pd.read_csv(self.synthetic_csv_path)
            self.synthetic_pdb_to_cluster = read_clusters(
                self.synthetic_cluster_path, synthetic=True
            )

            # offset all the cluster numbers by the number of real data clusters
            num_real_clusters = metadata_csv["cluster"].max() + 1

            def synthetic_cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self.synthetic_pdb_to_cluster:
                    raise ValueError(
                        f"Synthetic example {pdb} not in synthetic cluster file!"
                    )
                return self.synthetic_pdb_to_cluster[pdb] + num_real_clusters

            self.synthetic_csv["cluster"] = self.synthetic_csv["pdb_name"].map(
                synthetic_cluster_lookup
            )

            metadata_csv = pd.concat([metadata_csv, self.synthetic_csv])

        return metadata_csv

    def create_split(self, split, dataset_config, metadata_csv):
        if split == "train":
            self.csv = metadata_csv
            print(f"Training: {len(self.csv)} examples")

        elif split == "test":
            # Meaning of this code block is unclear to me.

            if dataset_config.max_eval_length is None:
                eval_lengths = metadata_csv.modeled_seq_len
            else:
                eval_lengths = metadata_csv.modeled_seq_len[
                    metadata_csv.modeled_seq_len <= dataset_config.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, dataset_config.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = metadata_csv[metadata_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby("modeled_seq_len").sample(
                dataset_config.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values("modeled_seq_len", ascending=False)
            self.csv = eval_csv
            print(f"Validation: {len(self.csv)} examples with lengths {eval_lengths}")

        else:
            raise ValueError(f"Invalid split: {split}")

        self.csv["index"] = list(range(len(self.csv)))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx) -> dict[str]:
        csv_row = self.csv.iloc[idx]

        path = self.data_dir / csv_row["processed_path"]
        seq_len = csv_row["modeled_seq_len"]

        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self.dataset_config.cache_num_res
        if use_cache and path in self.cache:
            return self.cache[path]

        processed_row = process_csv_row(path)
        processed_row["pdb_name"] = csv_row["pdb_name"]

        if self.dataset_config.use_redesigned:
            best_seq = csv_row["best_seq"]
            if not isinstance(best_seq, float):
                best_aatype = torch.tensor(seq_to_aatype(best_seq)).long()
                assert processed_row["aatypes_1"].shape == best_aatype.shape
                processed_row["aatypes_1"] = best_aatype

        if len(set(to_numpy(processed_row["aatypes_1"]))) == 1:
            raise ValueError(f"Example {path} has only one amino acid.")

        # Save to cache.
        if use_cache:
            self.cache[path] = processed_row

        return processed_row


def filter_metadata(data_csv, filter_cfg):
    # Filter by oligomeric detail.
    data_csv = data_csv[data_csv.oligomeric_detail.isin(filter_cfg.oligomeric)]

    # Filter by number of chains.
    data_csv = data_csv[data_csv.num_chains.isin(filter_cfg.num_chains)]

    data_csv = filter_with_sequence_length(
        data_csv,
        filter_cfg.min_num_res,
        filter_cfg.max_num_res,
    )

    data_csv = filter_with_max_coil_percent(data_csv, filter_cfg.max_coil_percent)

    data_csv = filter_large_structures(data_csv, filter_cfg.rog_quantile)

    return data_csv


def process_csv_row(processed_file_path):
    processed_feats = read_pkl(processed_file_path)
    processed_feats = parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats["modeled_idx"]
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats["modeled_idx"]
    processed_feats = tree.map_structure(
        lambda x: x[min_idx : (max_idx + 1)], processed_feats
    )

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }
    chain_feats = atom37_to_frames(chain_feats)
    rigids_1 = Rigid.from_tensor_4x4(chain_feats["rigidgroups_gt_frames"])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats["b_factors"][:, 1]
    res_mask = torch.tensor(processed_feats["bb_mask"]).int()

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats["chain_index"]
    res_idx = processed_feats["residue_index"]
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = (
        np.array(random.sample(all_chain_idx, len(all_chain_idx)))
        - np.min(all_chain_idx)
        + 1
    )
    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f"Found NaNs in {processed_file_path}")

    return {
        "res_plddt": res_plddt,
        "aatypes_1": chain_feats["aatype"],
        "rotmats_1": rotmats_1,
        "trans_1": trans_1,
        "res_mask": res_mask,
        "chain_idx": new_chain_idx,
        "res_idx": new_res_idx,
    }


class PDBSequenceDataset(PDBDataset):
    def __getitem__(self, idx) -> dict[str]:
        csv_row = self.csv.iloc[idx]

        path = self.data_dir / csv_row["processed_path"]
        seq_len = csv_row["modeled_seq_len"]

        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self.dataset_config.cache_num_res
        if use_cache and path in self.cache:
            return self.cache[path]

        processed_row = self.process_csv_row(path)
        processed_row["pdb_name"] = csv_row["pdb_name"]

        if self.dataset_config.use_redesigned:
            best_seq = csv_row["best_seq"]
            if not isinstance(best_seq, float):
                best_aatype = torch.tensor(seq_to_aatype(best_seq)).long()
                assert processed_row["aatypes_1"].shape == best_aatype.shape
                processed_row["aatypes_1"] = best_aatype

        if len(set(to_numpy(processed_row["aatypes_1"]))) == 1:
            raise ValueError(f"Example {path} has only one amino acid.")

        # Save to cache.
        if use_cache:
            self.cache[path] = processed_row

        return processed_row

    def process_csv_row(self, processed_file_path):
        processed_feats = read_pkl(processed_file_path)
        processed_feats = parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats["modeled_idx"]
        processed_feats = tree.map_structure(
            lambda x: x[min_idx : (max_idx + 1)], processed_feats
        )

        # Run through OpenFold data transforms.
        chain_feats = {
            "aatype": torch.tensor(processed_feats["aatype"]).long(),
            "all_atom_positions": torch.tensor(
                processed_feats["atom_positions"]
            ).double(),
            "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
        }

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats["chain_index"]
        res_idx = processed_feats["residue_index"]
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = (
            np.array(random.sample(all_chain_idx, len(all_chain_idx)))
            - np.min(all_chain_idx)
            + 1
        )
        for i, chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        return {
            "aatypes_1": chain_feats["aatype"],
            "chain_idx": new_chain_idx,
            "res_idx": new_res_idx,
        }


if __name__ == "__main__":
    from omegaconf import OmegaConf

    dataset_config = {
        "seed": 123,
        "data_dir": "data/pdb",
        "csv_path": "./metadata/pdb_metadata.csv",
        "cluster_path": "./metadata/pdb.clusters",
        "test_set_pdb_ids_path": None,
        "max_cache_size": 100_000,
        "cache_num_res": 0,
        "inpainting_percent": 1.0,
        "add_plddt_mask": False,
        "max_eval_length": 256,
        "redesigned_csv_path": "./metadata/pdb_redesigned.csv",
        "use_redesigned": True,
        "synthetic_csv_path": "./metadata/distillation_metadata.csv",
        "synthetic_cluster_path": "./metadata/distillation.clusters",
        "use_synthetic": True,
        # Eval parameters
        "samples_per_eval_length": 5,
        "num_eval_lengths": 8,
        # Filtering
        "filter": {
            "max_num_res": 384,
            "min_num_res": 60,
            "max_coil_percent": 0.5,
            "rog_quantile": 0.96,
            "oligomeric": ["monomeric"],
            "num_chains": [1],
        },
    }

    dataset = PDBDataset(
        split="test",
        dataset_config=OmegaConf.create(dataset_config),
    )
    print(dataset[0])
    print(dataset[0].keys())
