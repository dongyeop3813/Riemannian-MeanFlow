import numpy as np
from scipy.stats import pearsonr

import flax.nnx as nnx


def compute_kmer_freq(seq, k=6):
    """Compute k-mer frequency from one-hot encoded sequences using integer encoding.

    This is an optimized version using integer encoding and np.bincount
    instead of dictionary-based counting.

    Args:
        seq: (B, L, 4) one-hot encoded sequences or (B, L) label sequences
        k: k-mer size

    Returns:
        np.ndarray of shape (4^k,) containing k-mer counts
    """
    seq = np.asarray(seq)

    # Convert one-hot to label if needed
    if seq.ndim == 3:
        seq = np.argmax(seq, axis=-1)

    # (B, L-k+1, k)
    windows = np.lib.stride_tricks.sliding_window_view(seq, window_shape=k, axis=-1)

    # (B*(L-k+1), k)
    windows = windows.reshape(-1, k)

    # Encode k-mers as integers (base-4 encoding)
    # Example: [0,1,2,3,0,1] -> 0*4^5 + 1*4^4 + 2*4^3 + 3*4^2 + 0*4^1 + 1*4^0
    powers = 4 ** np.arange(k - 1, -1, -1)
    kmer_ids = windows @ powers  # shape: (N,)

    # Count using bincount (much faster than np.unique with tuples)
    counts = np.bincount(kmer_ids, minlength=4**k)

    return counts


class KmerFreq(nnx.metrics.Metric):
    """K-mer frequency metric that computes Pearson correlation between
    true and generated sequence k-mer distributions.

    Uses integer encoding for fast k-mer counting.
    """

    def __init__(self, k=6):
        self.k = k
        self.n_kmers = 4**k
        self.true_counts = np.zeros(self.n_kmers, dtype=np.int64)
        self.pred_counts = np.zeros(self.n_kmers, dtype=np.int64)

    def update(self, true, pred):
        """Update frequency counts with new batch of sequences.

        Args:
            true: True sequences, shape (B, L, 4) one-hot or (B, L) labels
            pred: Generated sequences, shape (B, L, 4) one-hot or (B, L) labels
        """
        true_freq = compute_kmer_freq(true, k=self.k)
        pred_freq = compute_kmer_freq(pred, k=self.k)

        self.true_counts += true_freq
        self.pred_counts += pred_freq

    def update_with_true_freq(self, true_freq, pred):
        """Update frequency counts with precomputed true frequencies.

        Use this when the same true sequences are compared against
        multiple generated sequences to avoid redundant computation.

        Args:
            true_freq: Precomputed true k-mer counts, shape (4^k,)
            pred: Generated sequences, shape (B, L, 4) one-hot or (B, L) labels
        """
        pred_freq = compute_kmer_freq(pred, k=self.k)

        self.true_counts += true_freq
        self.pred_counts += pred_freq

    def compute(self):
        """Compute Pearson correlation between true and generated k-mer frequencies.

        Returns:
            Pearson correlation coefficient (float)
        """
        # Normalize to frequencies
        true_total = self.true_counts.sum()
        pred_total = self.pred_counts.sum()

        if true_total == 0 or pred_total == 0:
            return 0.0

        true_freqs = self.true_counts / true_total
        pred_freqs = self.pred_counts / pred_total

        # Only consider k-mers that appear in at least one distribution
        mask = (self.true_counts > 0) | (self.pred_counts > 0)
        if mask.sum() < 2:
            return 0.0

        corr, _ = pearsonr(true_freqs[mask], pred_freqs[mask])
        return corr

    def reset(self):
        """Reset frequency counts."""
        self.true_counts = np.zeros(self.n_kmers, dtype=np.int64)
        self.pred_counts = np.zeros(self.n_kmers, dtype=np.int64)
