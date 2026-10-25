"""Flower server for federated learning with FedAvg aggregation.

Configures the FedAvg strategy and wraps it in a ServerApp for use with
`flwr run` or simulation mode.

See fedler.md Phase 2 for design rationale.
"""
from __future__ import annotations

from typing import Any

import flwr as fl
from flwr.common import Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics weighted by number of examples."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}

    aggregated: dict[str, float] = {}
    metric_keys = ["val_acc", "melanoma_sensitivity", "bcc_sensitivity", "composite"]

    for key in metric_keys:
        weighted_sum = sum(n * m.get(key, 0.0) for n, m in metrics)
        aggregated[key] = weighted_sum / total

    return aggregated


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Factory function called by Flower to configure the server."""
    cfg = context.run_config

    num_rounds = int(cfg.get("num-rounds", 10))
    fraction_fit = float(cfg.get("fraction-fit", 1.0))
    fraction_evaluate = float(cfg.get("fraction-evaluate", 1.0))
    min_fit_clients = int(cfg.get("min-fit-clients", 3))
    min_evaluate_clients = int(cfg.get("min-evaluate-clients", 3))
    min_available_clients = int(cfg.get("min-available-clients", 3))

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
