from src.utils.time_sampler import (
    UniformTimeSampler,
    LogitNormalTimeSampler,
    TimeSampler,
    OrderedIntervalSampler,
    UnorderedIntervalSampler,
    OrderedThreePointSampler,
    UnorderedThreePointSampler,
)


def create_time_sampler(cfg, seed: int) -> TimeSampler:
    if cfg.type == "uniform":
        time_sampler = UniformTimeSampler(seed, cfg.max_t)

    elif cfg.type == "lognorm":
        time_sampler = LogitNormalTimeSampler(seed, cfg.mu, cfg.sigma, cfg.max_t)

    else:
        raise ValueError(f"Invalid time sampler type: {cfg.type}")

    # If we need only single time point sample, return the current time sampler
    need_interval_sample = "interval_sample" in cfg
    need_three_sample = "three_sample" in cfg
    if not need_interval_sample and not need_three_sample:
        return time_sampler

    # If we need interval sample or three point sample, build it with the current time sampler.
    if need_interval_sample and cfg.interval_sample == "ordered":
        time_sampler = OrderedIntervalSampler(
            time_sampler, cfg.boundary_ratio, cfg.get("static_layout", False)
        )
    elif need_interval_sample and cfg.interval_sample == "unordered":
        time_sampler = UnorderedIntervalSampler(
            time_sampler, cfg.boundary_ratio, cfg.get("static_layout", False)
        )
    elif need_three_sample and cfg.three_sample == "ordered":
        time_sampler = OrderedThreePointSampler(
            time_sampler,
            cfg.boundary_ratio,
            cfg.get("static_layout", False),
            cfg.get("mid_point", False),
            cfg.get("min_length", 0.0),
        )
    elif need_three_sample and cfg.three_sample == "unordered":
        time_sampler = UnorderedThreePointSampler(
            time_sampler,
        )
    else:
        raise ValueError(f"Invalid time sampler type: {cfg.interval_sample}")

    return time_sampler
