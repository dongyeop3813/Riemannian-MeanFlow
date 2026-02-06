"""
Sequence design via backpropagation through SEI model.

Optimizes DNA sequences to achieve target H3K4me3 scores by backpropagating
through the SEI model and updating continuous logits that represent sequence
probabilities.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np

import rootutils

rootutils.setup_root(
    Path.cwd(), indicator=".project-root", pythonpath=True, cwd=True
)

from src.eval.sei_eval import SeiEval


class SequenceDesigner:
    """Design DNA sequences by optimizing for target SEI profiles."""

    IDX_TO_NUC = {0: "A", 1: "C", 2: "G", 3: "T"}
    NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __init__(self, sei_eval: SeiEval, seq_length: int = 1024):
        """
        Initialize the sequence designer.

        Parameters:
            - `sei_eval`: The SEI evaluation model.
            - `seq_length`: Length of sequences to design (default: 1024).
        """
        self.sei_eval = sei_eval
        self.seq_length = seq_length

    def logits_to_soft_onehot(
        self, logits: jnp.ndarray, temperature: float = 1.0
    ) -> jnp.ndarray:
        """Convert logits (B, L, 4) to soft one-hot via softmax."""
        return jax.nn.softmax(logits / temperature, axis=-1)

    def get_sei_profile_from_logits(
        self, logits: jnp.ndarray, temperature: float = 1.0
    ):
        """Differentiable SEI profile from logits."""
        soft_seq = self.logits_to_soft_onehot(logits, temperature)
        return self.sei_eval.get_sei_profile(soft_seq)

    def sequence_to_logits(self, sequence: str) -> jnp.ndarray:
        """Convert a DNA sequence string to logits (1, L, 4)."""
        indices = jnp.array([[self.NUC_TO_IDX[nuc] for nuc in sequence.upper()]])
        # Use large values for one-hot-like logits
        onehot = jax.nn.one_hot(indices, 4)
        # Convert to logits (inverse softmax approximation)
        logits = jnp.log(onehot + 1e-8)
        return logits

    def logits_to_sequence(self, logits: jnp.ndarray) -> str:
        """Convert logits (B, L, 4) to DNA sequence string (first batch element)."""
        indices = jnp.argmax(logits[0], axis=-1)
        return "".join([self.IDX_TO_NUC[int(i)] for i in indices])

    def design_sequence(
        self,
        target_h3k4me3: float,
        n_steps: int = 500,
        lr: float = 0.1,
        temperature: float = 1.0,
        temperature_anneal: bool = False,
        final_temperature: float = 0.1,
        entropy_weight: float = 0.0,
        init_logits: jnp.ndarray | None = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> dict:
        """
        Optimize a sequence to achieve target H3K4me3 score.

        Parameters:
            - `target_h3k4me3`: Target H3K4me3 score to optimize towards.
            - `n_steps`: Number of optimization steps.
            - `lr`: Learning rate for Adam optimizer.
            - `temperature`: Softmax temperature (lower = sharper).
            - `temperature_anneal`: Whether to anneal temperature during optimization.
            - `final_temperature`: Final temperature if annealing.
            - `entropy_weight`: Weight for entropy regularization (encourages diversity).
            - `init_logits`: Initial logits (B, L, 4). If None, random initialization.
            - `seed`: Random seed for initialization.
            - `verbose`: Whether to print progress.

        Returns:
            Dictionary with:
                - `logits`: Final optimized logits.
                - `final_onehot`: Hard one-hot encoding of final sequence.
                - `final_seq_indices`: Final sequence as indices.
                - `final_seq_str`: Final sequence as string.
                - `final_h3k4me3`: Predicted H3K4me3 of final sequence.
                - `loss_history`: List of loss values during optimization.
        """
        print("A")
        # Initialize logits
        if init_logits is not None:
            logits = init_logits
        else:
            key = jax.random.PRNGKey(seed)
            logits = jax.random.normal(key, (1, self.seq_length, 4)) * 0.1
        print("B")

        # Get pure model representation for use in JAX transformations
        graphdef, state = self.sei_eval.get_pure_model()

        # Define loss function using pure interface (no NNX state mutation inside trace)
        def loss_fn(logits, temp):
            soft_seq = self.logits_to_soft_onehot(logits, temp)
            pred_prof, _ = self.sei_eval.get_sei_profile_pure(soft_seq, graphdef, state)
            mse = ((pred_prof - target_h3k4me3) ** 2).mean()

            if entropy_weight > 0:
                # Entropy regularization to encourage diversity
                entropy = -(soft_seq * jnp.log(soft_seq + 1e-8)).sum(axis=-1).mean()
                return mse - entropy_weight * entropy
            return mse

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(logits)
        print("C")

        # Update step (no jit - NNX model captured in closure causes tracing issues)
        def step(logits, opt_state, temp):
            loss, grads = jax.value_and_grad(loss_fn)(logits, temp)
            print("D")
            print("grads", grads)
            updates, opt_state = optimizer.update(grads, opt_state, logits)
            logits = optax.apply_updates(logits, updates)
            return logits, opt_state, loss

        # Optimization loop
        loss_history = []
        for i in range(n_steps):
            # Temperature annealing
            if temperature_anneal:
                progress = i / max(n_steps - 1, 1)
                temp = temperature + (final_temperature - temperature) * progress
            else:
                temp = temperature
            print("D1")
            logits, opt_state, loss = step(logits, opt_state, temp)
            print("E")
            loss_history.append(float(loss))

            if verbose and i % 50 == 0:
                print(f"Step {i:4d}, Loss: {loss:.6f}, Temperature: {temp:.3f}")

        # Convert final logits to hard sequence
        final_seq_indices = jnp.argmax(logits, axis=-1)  # (B, L)
        final_onehot = jax.nn.one_hot(final_seq_indices, 4)  # (B, L, 4)
        final_seq_str = self.logits_to_sequence(logits)

        # Evaluate final sequence with hard one-hot
        final_h3k4me3, _ = self.sei_eval.get_sei_profile(final_onehot)

        if verbose:
            print(f"\nFinal H3K4me3: {float(final_h3k4me3[0]):.6f}")
            print(f"Target H3K4me3: {target_h3k4me3:.6f}")

        return {
            "logits": logits,
            "final_onehot": final_onehot,
            "final_seq_indices": final_seq_indices,
            "final_seq_str": final_seq_str,
            "final_h3k4me3": float(final_h3k4me3[0]),
            "loss_history": loss_history,
        }

    def optimize_from_reference(
        self,
        reference_seq: jnp.ndarray,
        target_h3k4me3: float | None = None,
        maximize: bool = True,
        **kwargs,
    ) -> dict:
        """
        Optimize starting from a reference sequence.

        Parameters:
            - `reference_seq`: Reference one-hot sequence (B, L, 4) or (L, 4).
            - `target_h3k4me3`: Target score. If None, uses reference score (+ or -).
            - `maximize`: If target is None, whether to maximize or minimize H3K4me3.
            - `**kwargs`: Additional arguments passed to `design_sequence`.

        Returns:
            Same as `design_sequence`.
        """
        if reference_seq.ndim == 2:
            reference_seq = reference_seq[None, ...]

        # Get reference H3K4me3 score
        ref_h3k4me3, _ = self.sei_eval.get_sei_profile(reference_seq)
        ref_score = float(ref_h3k4me3[0])
        print(f"Reference H3K4me3: {ref_score:.6f}")

        if target_h3k4me3 is None:
            # Set target based on maximize/minimize
            if maximize:
                target_h3k4me3 = ref_score + 0.5  # Try to increase
            else:
                target_h3k4me3 = max(0, ref_score - 0.5)  # Try to decrease

        # Convert reference to logits for initialization
        # Use log of one-hot as initial logits
        init_logits = jnp.log(reference_seq + 1e-8)

        return self.design_sequence(
            target_h3k4me3=target_h3k4me3,
            init_logits=init_logits,
            **kwargs,
        )


def main():
    parser = argparse.ArgumentParser(description="Design DNA sequences via SEI backprop")
    parser.add_argument(
        "--target-h3k4me3",
        type=float,
        default=0.5,
        help="Target H3K4me3 score to optimize towards",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Number of optimization steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature",
    )
    parser.add_argument(
        "--temperature-anneal",
        action="store_true",
        help="Anneal temperature during optimization",
    )
    parser.add_argument(
        "--final-temperature",
        type=float,
        default=0.1,
        help="Final temperature if annealing",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.0,
        help="Entropy regularization weight",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Sequence length to design",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for designed sequence (FASTA format)",
    )

    args = parser.parse_args()

    print("Initializing SEI model...")
    sei_eval = SeiEval()
    sei_eval.to_device(args.device)

    print(f"\nDesigning sequence with target H3K4me3 = {args.target_h3k4me3}")
    designer = SequenceDesigner(sei_eval, seq_length=args.seq_length)

    result = designer.design_sequence(
        target_h3k4me3=args.target_h3k4me3,
        n_steps=args.n_steps,
        lr=args.lr,
        temperature=args.temperature,
        temperature_anneal=args.temperature_anneal,
        final_temperature=args.final_temperature,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        verbose=True,
    )

    print(f"\nDesigned sequence (first 100bp): {result['final_seq_str'][:100]}...")
    print(f"Final H3K4me3 score: {result['final_h3k4me3']:.6f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f">designed_seq_h3k4me3_{result['final_h3k4me3']:.4f}\n")
            # Write sequence in 80-character lines
            seq = result["final_seq_str"]
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")
        print(f"\nSequence saved to {output_path}")

    return result


if __name__ == "__main__":
    main()
