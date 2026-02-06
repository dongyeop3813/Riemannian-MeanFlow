import flax.nnx as nnx


def create_stratified_metric(
    prefix: str = "",
    num_bins: int = 5,
    return_as_dict: bool = False,
):
    metrics = {}

    for i in range(num_bins):
        metrics[f"{prefix}loss_bin{i}"] = nnx.metrics.Average(f"{prefix}loss_bin{i}")

    if return_as_dict:
        return metrics
    else:
        return nnx.metrics.MultiMetric(**metrics)
