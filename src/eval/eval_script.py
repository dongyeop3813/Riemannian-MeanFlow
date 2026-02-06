from pathlib import Path
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


from src.eval.sei_eval import SeiEval

import argparse
import numpy as np
import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.eval.fbd import FBD
from src.eval.kmer_freq import KmerFreq
from src.eval.promoter_eval_fn import CategoricalEntropy
from src.manifold.conversion import sphere_to_simplex


def get_metrics(
    sei: SeiEval,
    model_sample: torch.Tensor,
    test_sample: torch.Tensor,
    project_to_onehot: bool,
):
    dataloader = DataLoader(
        TensorDataset(model_sample, test_sample), batch_size=128, shuffle=False
    )

    fbd = FBD()
    kmer_freq = KmerFreq()
    entropy = CategoricalEntropy()

    mse_list: list[float] = []

    for batch_idx, (x, y) in enumerate(dataloader):
        x = jnp.array(x)  # This is simplex representation
        y = jnp.array(y)

        prob_x = x
        ohe_x = jnp.argmax(x, axis=-1)
        ohe_x = jnp.eye(4)[ohe_x]

        # Update kmer frequency correlation
        kmer_freq.update(true=np.asarray(y), pred=np.asarray(ohe_x))

        # Update entropy
        entropy.update(np.asarray(prob_x))

        if project_to_onehot:
            mse, (pred_feat, target_feat) = sei.eval_sp_mse(
                ohe_x, y, batch_idx, return_features=True
            )
        else:
            mse, (pred_feat, target_feat) = sei.eval_sp_mse(
                x, y, batch_idx, return_features=True
            )

        mse_list.append(mse)
        fbd.update(pred_feat, target_feat)

    fbd_value = fbd.compute()
    kmer_corr = kmer_freq.compute()
    entropy_value = entropy.compute()

    return (
        jnp.mean(jnp.concatenate(mse_list, axis=0)),
        fbd_value,
        kmer_corr,
        entropy_value,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--project_to_onehot", action="store_true", default=False)
    args = parser.parse_args()

    sei = SeiEval()
    sei.to_device("gpu")
    sample_dir = Path(args.sample_dir)

    csv_path = sample_dir / "eval.csv"
    with open(csv_path, "w") as f:
        f.write("n_steps,mse,fbd,kmer_corr,entropy\n")

    for steps in [1, 2, 4, 8, 16, 32, 64, 100]:
        print("step", steps)
        sample_file = sample_dir / f"n_steps_{steps}.npz"
        ckpt_data = np.load(sample_file)
        model_sample = torch.from_numpy(ckpt_data["model_sample"])
        test_sample = torch.from_numpy(ckpt_data["test_data"])
        print("model_sample", model_sample, model_sample.shape)
        print("test_sample", test_sample, test_sample.shape)

        mse, fbd, kmer_corr, entropy = get_metrics(
            sei, model_sample, test_sample, args.project_to_onehot
        )
        print("mse", mse)
        with open(csv_path, "a") as f:
            f.write(f"{steps},{mse.item()},{fbd},{kmer_corr},{entropy}\n")
