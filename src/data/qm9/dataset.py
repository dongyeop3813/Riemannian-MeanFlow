import torch

from pathlib import Path
from torch.nn.functional import one_hot


class QM9Dataset(torch.utils.data.Dataset):

    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super(QM9Dataset, self).__init__()

        # unpack some configs regarding the prior
        self.prior_config = prior_config
        self.dataset_config = dataset_config

        # get the processed data directory
        processed_data_dir: Path = Path(dataset_config["processed_data_dir"])

        # load the marginal distributions of atom types and the conditional distribution of charges given atom type
        marginal_dists_file = processed_data_dir / "train_data_marginal_dists.pt"
        p_a, p_c, p_e, p_c_given_a = torch.load(marginal_dists_file)

        # add the marginal distributions as arguments to the prior sampling functions
        if self.prior_config["a"]["type"] == "marginal":
            self.prior_config["a"]["kwargs"]["p"] = p_a

        if self.prior_config["e"]["type"] == "marginal":
            self.prior_config["e"]["kwargs"]["p"] = p_e

        if self.prior_config["c"]["type"] == "marginal":
            self.prior_config["c"]["kwargs"]["p"] = p_c

        if self.prior_config["c"]["type"] == "c-given-a":
            self.prior_config["c"]["kwargs"]["p_c_given_a"] = p_c_given_a

        if dataset_config["dataset_name"] in ["geom", "qm9", "geom_5conf"]:
            data_file = processed_data_dir / f"{split}_data_processed.pt"
        else:
            raise NotImplementedError("unsupported dataset_name")

        # load data from processed data directory
        data_dict = torch.load(data_file)

        self.positions = data_dict["positions"]
        self.atom_types = data_dict["atom_types"]
        self.atom_charges = data_dict["atom_charges"]
        self.bond_types = data_dict["bond_types"]
        self.bond_idxs = data_dict["bond_idxs"]
        self.node_idx_array = data_dict["node_idx_array"]
        self.edge_idx_array = data_dict["edge_idx_array"]

    def __len__(self):
        return self.node_idx_array.shape[0]

    def __getitem__(self, idx):
        node_start_idx = self.node_idx_array[idx, 0]
        node_end_idx = self.node_idx_array[idx, 1]
        edge_start_idx = self.edge_idx_array[idx, 0]
        edge_end_idx = self.edge_idx_array[idx, 1]

        # get data pertaining to nodes for this molecule
        positions = self.positions[node_start_idx:node_end_idx]
        atom_types = self.atom_types[node_start_idx:node_end_idx].float()
        atom_charges = self.atom_charges[node_start_idx:node_end_idx].long()

        # remove COM from positions
        positions = positions - positions.mean(dim=0, keepdim=True)

        # get data pertaining to edges for this molecule
        bond_types = self.bond_types[edge_start_idx:edge_end_idx].int()
        bond_idxs = self.bond_idxs[edge_start_idx:edge_end_idx].long()

        # reconstruct adjacency matrix
        n_atoms = positions.shape[0]
        adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int32)

        # fill in the values of the adjacency matrix specified by bond_idxs
        adj[bond_idxs[:, 0], bond_idxs[:, 1]] = bond_types

        # get upper triangle of adjacency matrix
        upper_edge_idxs = torch.triu_indices(
            n_atoms, n_atoms, offset=1
        )  # has shape (2, n_upper_edges)
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]

        # get lower triangle edges by swapping source and destination of upper_edge_idxs
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

        # one-hot encode edge labels and atom charges
        edge_labels = one_hot(
            edge_labels.to(torch.int64), num_classes=5
        ).float()  # hard-coded assumption of 5 bond types

        try:
            atom_charges = one_hot(
                atom_charges + 2, num_classes=6
            ).float()  # hard-coded assumption that charges are in range [-2, 3]
        except Exception as e:
            print("an atom charge outside of the expected range was encountered")
            print(
                f"max atom charge: {atom_charges.max()}, min atom charge: {atom_charges.min()}"
            )
            raise e

        return {
            "positions": positions,
            "atom_types": atom_types,
            "atom_charges": atom_charges,
            "bond_types": bond_types,
            "bond_idxs": bond_idxs,
            "edges": edges,
            "edge_labels": edge_labels,
        }


def collate_fn(batch):
    result = {
        "positions": [],
        "atom_types": [],
        "atom_charges": [],
        "bond_types": [],
        "bond_idxs": [],
        "edges": [],
        "edge_labels": [],
    }

    for x in batch:
        for key in result.keys():
            result[key].append(x[key])

    for key in result.keys():
        result[key] = torch.stack(result[key])

    return result


if __name__ == "__main__":
    dataset_config = {
        "processed_data_dir": "data/qm9",
        "raw_data_dir": "data/qm9_raw",
        "dataset_name": "qm9",
    }
    prior_config = {
        "a": {
            "align": False,
            "kwargs": {},
            "type": "gaussian",
        },
        "c": {
            "align": False,
            "kwargs": {},
            "type": "gaussian",
        },
        "e": {
            "align": False,
            "kwargs": {},
            "type": "gaussian",
        },
        "x": {
            "align": True,
            "kwargs": {},
            "std": 1.0,
            "type": "centered-normal",
        },
    }

    dataset = QM9Dataset(
        split="train", dataset_config=dataset_config, prior_config=prior_config
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, collate_fn=collate_fn
    )

    x = next(iter(dataloader))

    print(type(x))
