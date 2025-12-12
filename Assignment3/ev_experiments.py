"""
Experiment utilities for the EV Stag Hunt model.

Contains policy factory functions, trial runners, multiprocessing helpers,
and standalone plotting routines. Depends on `ev_core` for the model.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import networkx as nx
import pickle

from ev_core import (
    EVStagHuntModel,
    set_initial_adopters,
    final_mean_adoption_vs_ratio,
    phase_sweep_X0_vs_ratio,
)
from ev_plotting import (
    plot_fanchart,
    plot_spaghetti,
    plot_density,
    plot_ratio_sweep,
    plot_phase_plot,
)


# -----------------------------
# Policy factories
# -----------------------------

def policy_subsidy_factory(start: int, end: int, delta_a0: float = 0.3, delta_beta_I: float = 0.0) -> Callable:
    """Create a policy that temporarily boosts coordination payoffs.

    Raises `a0` and/or `beta_I` during `[start, end)` and reverts after.
    Returns a closure `policy(model, step)`.
    """

    def policy(model, step):
        if not hasattr(policy, "base_a0"):
            policy.base_a0 = model.a0
        if not hasattr(policy, "base_beta_I"):
            policy.base_beta_I = model.beta_I

        if start <= step < end:
            model.a0 = policy.base_a0 + delta_a0
            model.beta_I = policy.base_beta_I + delta_beta_I
        else:
            model.a0 = policy.base_a0
            model.beta_I = policy.base_beta_I

    return policy


def policy_infrastructure_boost_factory(start: int, boost: float = 0.2, once: bool = True) -> Callable:
    """Create a policy that injects infrastructure at a specific step."""

    def policy(model, step):
        if step < start:
            return
        if once:
            if not hasattr(policy, "done"):
                model.infrastructure = float(np.clip(model.infrastructure + boost, 0.0, 1.0))
                policy.done = True
        else:
            model.infrastructure = float(np.clip(model.infrastructure + boost, 0.0, 1.0))

    return policy

def policy_carbon_tax(start: int, delta_b: float = -0.3) -> Callable:
    """
    Permanently change b at a given time step (carbon tax).

    delta_b should be NEGATIVE to make defection (ICE) less attractive.
    """
    def policy(model, step):
        # Do nothing before the tax kicks in
        if step < start:
            return

        # Apply the tax once, then never touch b again
        if not hasattr(policy, "done"):
            policy.base_b = model.b          # original payoff to D
            model.b = policy.base_b + delta_b
            policy.done = True               # flag so we don't re-apply

    return policy

def get_network_metrics(G) -> Dict:
    """Calculates structural network metrics (Clustering, Connectivity, Components)."""
    try:
        avg_clustering = nx.average_clustering(G)
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0.0
        density = nx.density(G)
        
        components = list(nx.connected_components(G))
        num_components = len(components)
        largest_component_size = len(max(components, key=len)) if components else 0
        
    except Exception:
        return {} # Returns empty if graph is invalid

    return {
        "avg_clustering": avg_clustering,
        "avg_degree": avg_degree,
        "density": density,
        "num_components": num_components,
        "largest_component_size": largest_component_size
    }
# -----------------------------
# Trial runner
# -----------------------------

def run_timeseries_trial(
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
    policy: Optional[Callable] = None,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run a single simulation and return X(t), I(t), and the model dataframe."""

    scenario = {
        # Either provide `ratio` to pin the initial a_I/b, or explicit `a0`.
        # Defaults here mirror the classroom-friendly values.
        # If `ratio` is present, we compute `a0 = ratio*b - beta_I*I0`.
        "a0": 2.0,
        "ratio": None,
        "beta_I": 3.0,
        "b": 1.0,
        "g_I": 0.1,
        "I0": 0.05,
        "network_type": "random",
        "n_nodes": 100,
        "p": 0.05,
        "m": 2,
        "collect": True,
        "X0_frac": 0.0,
        "init_method": "random",
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    # Compute a0 from ratio if provided to preserve initial payoff ratio
    a0_for_model = scenario["a0"]
    if scenario.get("ratio") is not None:
        a0_for_model = float(scenario["ratio"]) * float(scenario["b"]) - float(scenario["beta_I"]) * float(scenario["I0"])

    model = EVStagHuntModel(
        a0=a0_for_model,
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        I0=scenario["I0"],
        seed=seed,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        collect=True,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    if scenario.get("X0_frac", 0.0) > 0.0:
        set_initial_adopters(
            model,
            scenario["X0_frac"],
            method=scenario.get("init_method", "random"),
            seed=seed,
        )

    for t in range(T):
        if policy is not None:
            policy(model, t)
        model.step()

    df = model.datacollector.get_model_vars_dataframe().copy()
    net_stats = get_network_metrics(model.G)
    if seed is not None:
        net_stats["seed"] = seed
    return df["X"].to_numpy(), df["I"].to_numpy(), df, net_stats


def _timeseries_trial_worker(args_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Worker for parallel trials that reconstructs closures for policies."""
    T = args_dict["T"]
    scenario_kwargs = args_dict.get("scenario_kwargs", {})
    seed = args_dict.get("seed", None)
    policy_spec = args_dict.get("policy", None)
    strategy_choice_func = args_dict.get("strategy_choice_func", "imitate")
    tau = args_dict.get("tau", 1.0)

    policy = None
    if isinstance(policy_spec, dict):
        ptype = policy_spec.get("type")
        if ptype == "subsidy":
            policy = policy_subsidy_factory(**policy_spec["params"])
        elif ptype == "infrastructure":
            policy = policy_infrastructure_boost_factory(**policy_spec["params"])
        elif ptype == "carbon_tax":       
            policy = policy_carbon_tax(**policy_spec["params"])

    X, I, _df, net_stats = run_timeseries_trial(
        T=T,
        scenario_kwargs=scenario_kwargs,
        seed=seed,
        policy=policy,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )
    return X, I, net_stats


# -----------------------------
# Experiment: Intervention trials + plotting
# -----------------------------

def collect_intervention_trials(
    n_trials: int = 10,
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    subsidy_params: Optional[Dict] = None,
    max_workers: int = 1,
    seed_base: int = 42,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Run baseline and subsidy trials; return raw trajectories and summary dataframes."""

    scenario = scenario_kwargs or {}
    subsidy = subsidy_params or {"start": 30, "end": 80, "delta_a0": 0.3, "delta_beta_I": 0.0}

    baseline_args = []
    subsidy_args = []
    network_stats = []
    
    for i in range(n_trials):
        seed = seed_base + i
        baseline_args.append(
            {
                "T": T,
                "scenario_kwargs": scenario,
                "seed": seed,
                "policy": None,
                "strategy_choice_func": strategy_choice_func,
                "tau": tau,
            }
        )
        subsidy_args.append(
            {
                "T": T,
                "scenario_kwargs": scenario,
                "seed": seed,
                "policy": {"type": "carbon_tax", "params": subsidy},
                "strategy_choice_func": strategy_choice_func,
                "tau": tau,
            }
        )

    baseline_X, baseline_I = [], []
    subsidy_X, subsidy_I = [], []

    # Run sequentially or concurrently
    Executor = ThreadPoolExecutor if max_workers == 1 else ProcessPoolExecutor
    with Executor(max_workers=max_workers) as ex:
        baseline_futs = [ex.submit(_timeseries_trial_worker, args) for args in baseline_args]
        subsidy_futs = [ex.submit(_timeseries_trial_worker, args) for args in subsidy_args]
        for fut in as_completed(baseline_futs):
            X, I, stats = fut.result()
            baseline_X.append(X)
            baseline_I.append(I)
            network_stats.append(stats)
        for fut in as_completed(subsidy_futs):
            X, I, _ = fut.result()
            subsidy_X.append(X)
            subsidy_I.append(I)

    # Align order by seed (as_completed may scramble)
    baseline_X = sorted(baseline_X, key=lambda arr: tuple(arr))
    subsidy_X = sorted(subsidy_X, key=lambda arr: tuple(arr))

    # Summary stats
    def summarize(X_list: List[np.ndarray]) -> pd.DataFrame:
        mat = np.vstack(X_list)
        df = pd.DataFrame({
            "X_mean": mat.mean(axis=0),
            "X_q005": np.quantile(mat, 0.05, axis=0),
        })
        return df

    baseline_df = summarize(baseline_X)
    subsidy_df = summarize(subsidy_X)
    network_stats_df = pd.DataFrame(network_stats)

    return baseline_X, baseline_I, subsidy_X, subsidy_I, baseline_df, subsidy_df, network_stats


def traces_to_long_df(baseline_X: List[np.ndarray], subsidy_X: List[np.ndarray]) -> pd.DataFrame:
    """Convert trajectory lists to a tidy DataFrame: [group, trial, time, X]."""
    rows = []
    for trial, X in enumerate(baseline_X):
        for t, x in enumerate(X):
            rows.append(("baseline", trial, t, float(x)))
    for trial, X in enumerate(subsidy_X):
        for t, x in enumerate(X):
            rows.append(("subsidy", trial, t, float(x)))
    return pd.DataFrame(rows, columns=["group", "trial", "time", "X"])


def ratio_sweep_df(
    X0_frac: float = 0.40,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    T: int = 250,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
) -> pd.DataFrame:
    """Compute X* vs ratio and return as a DataFrame."""
    scenario = {
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "I0": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    X_means = final_mean_adoption_vs_ratio(
        X0_frac,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    return pd.DataFrame({"ratio": ratio_values, "X_mean": X_means})


def phase_sweep_df(
    max_workers: int | None = None,
    backend: str = "process",
    X0_values: Optional[np.ndarray] = None,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    T: int = 250,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
) -> pd.DataFrame:
    """Compute tidy DataFrame of X* over (X0, ratio)."""
    if X0_values is None:
        X0_values = np.linspace(0.0, 1.0, 21)
    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    scenario = {
        "I0": 0.05,
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    X_final = phase_sweep_X0_vs_ratio(
        X0_values,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        max_workers=max_workers or 1,
        backend=backend,
    )

    rows = []
    for i, X0 in enumerate(X0_values):
        for j, ratio in enumerate(ratio_values):
            rows.append((float(X0), float(ratio), float(X_final[j, i])))
    return pd.DataFrame(rows, columns=["X0", "ratio", "X_final"])


def run_ratio_sweep_plot(
    X0_frac: float = 0.40,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    T: int = 250,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
    out_path: Optional[str] = None,
) -> str:
    """Sweep ratio values and plot final adoption X* vs a_I/b for a fixed X0.

    Calls the core computation helper and saves a simple line plot.
    Returns the path to the saved image.
    """
    scenario = {
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "I0": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    X_means = final_mean_adoption_vs_ratio(
        X0_frac,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ratio_values, X_means, color="tabblue" if hasattr(plt, "tabblue") else "C0", lw=2)
    ax.set_xlabel("a_I / b (ratio)")
    ax.set_ylabel("Final adoption X*")
    ax.set_title(f"X* vs ratio for X0={X0_frac:.2f}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    if out_path is None:
        out_path = os.path.join(os.getcwd(), "ev_ratio_sweep.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_phase_plot_X0_vs_ratio_network(
    max_workers: int | None = None,
    backend: str = "process",
    X0_values: Optional[np.ndarray] = None,
    ratio_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    batch_size: int = 16,
    init_noise_I: float = 0.04,
    T: int = 250,
    strategy_choice_func: str = "logit",
    tau: float = 1.0,
    out_path: Optional[str] = None,
):
    """Produce a heatmap of X* over (X0, a_I/b) using core sweep helper.

    Saves a figure similar to the original model script and returns the path.
    """
    # Defaults aligned with the original phase plot
    if X0_values is None:
        X0_values = np.linspace(0.0, 1.0, 21)
    if ratio_values is None:
        ratio_values = np.linspace(0.8, 3.5, 41)

    scenario = {
        "I0": 0.05,
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    X_final = phase_sweep_X0_vs_ratio(
        X0_values,
        ratio_values,
        I0=scenario["I0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        init_noise_I=init_noise_I,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
        max_workers=max_workers or 1,
        backend=backend,
    )

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        X_final,
        origin="lower",
        extent=[X0_values[0], X0_values[-1], ratio_values[0], ratio_values[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="plasma",
    )
    plt.colorbar(im, label="Final adopters X*")
    plt.xlabel("X0 (initial adoption)")
    plt.ylabel("a_I / b (initial payoff ratio)")
    plt.title("Network phase plot: X* over X0 and a_I/b")

    # Overlay initial threshold X = b/a_I => X = 1/ratio
    X_thresh = 1.0 / ratio_values
    X_thresh_clipped = np.clip(X_thresh, 0.0, 1.0)
    plt.plot(
        X_thresh_clipped,
        ratio_values,
        color="white",
        linestyle="--",
        linewidth=1.5,
        label="X = b / a_I (initial)",
    )

    plt.legend(loc="upper right")

    if out_path is None:
        out_path = os.path.join(os.getcwd(), "ev_phase_plot.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


def run_intervention_example(
    n_trials: int = 10,
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    subsidy_params: Optional[Dict] = None,
    max_workers: int = 1,
    seed_base: int = 42,
    strategy_choice_func: str = "imitate",
    tau: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Convenience: collect trials, plot, and return summary + image path."""

    baseline_X, baseline_I, subsidy_X, subsidy_I, baseline_df, subsidy_df = collect_intervention_trials(
        n_trials=n_trials,
        T=T,
        scenario_kwargs=scenario_kwargs,
        subsidy_params=subsidy_params,
        max_workers=max_workers,
        seed_base=seed_base,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )
    # Use DataFrame-based plotting to ensure outputs go to plots/
    traces_df = traces_to_long_df(baseline_X, subsidy_X)
    img_path = plot_fanchart(traces_df)
    return baseline_df, subsidy_df, img_path

def save_scenario_data(label: str, name: str, **data):
    """
    Save arbitrary keyword-argument data to a pickle file inside plots_{label}/.

    Example:
        save_scenario_data("InitialAdoption0.3", "intervention",
                           baseline_X=..., subsidy_X=..., baseline_df=...)

    Produces:
        plots_InitialAdoption0.3/intervention_data.pkl
    """
    folder = f"data_{label}"
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{name}_data.pkl")

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"[saved] {path}")
    return path

# -----------------------------
# CLI Entrypoint
# -----------------------------

def main():
    # Defaults aligned with original ev_stag_mesa_model.run_intervention_example
    n_trials = 200  # use fewer than 500 for speed while keeping shape. Shortened
    T = 200 # shortened
    strategy_choice_func = "imitate"
    tau = 1.0
    max_workers = 4
    seed_base = 100

    base_scenario = dict(
        # Preserve initial ratio by computing a0 from ratio, matching the original
        ratio=2.3,
        beta_I=2.0,
        b=1.0,
        g_I=0.10,
        I0=0.05,
        network_type="BA",
        n_nodes=150 , # Set this low just to make runtime smaller
        m=2,
        collect=True,
        X0_frac=0.40,
        init_method="random",
        p=0.05,
    )

    subsidy_early = dict(start=0, delta_b=-0.3) 
    subsidy_late = dict(start=50, delta_b=-0.3)
    subsidy_strong = dict(start=0, delta_b=-0.4)
    subsidy_weak = dict(start=0, delta_b=-0.2) 
    scenarios = [
        # ("InitialAdoption0.3",
        #     {**base_scenario, "X0_frac": 0.3},
        #     subsidy_early),

        # ("InitialAdoption0.5",
        #     {**base_scenario, "X0_frac": 0.5},
        #     subsidy_early),

        # ("InitialInfrastructure0.15",
        #     {**base_scenario, "I0": 0.15},
        #     subsidy_early),

        # ("InitialInfrastructure0.25",
        #     {**base_scenario, "I0": 0.25},
        #     subsidy_early),

        # ("HighBetaI3.0",
        #     {**base_scenario, "beta_I": 3.0},
        #     subsidy_early),

        # ("LowBetaI1.0",
        #     {**base_scenario, "beta_I": 1.0},
        #     subsidy_early),

        ("BA_strong_b",
            {**base_scenario, "network_type": "BA"},
            subsidy_strong),

        ("BA_weak_b",
            {**base_scenario, "network_type": "BA"},
            subsidy_weak)

        # ("ER_late",
        #     {**base_scenario, "network_type": "random"},
        #     subsidy_late),

        # ("BA",
        #     {**base_scenario, "network_type": "BA"},
        #     subsidy_early),

        # ("Grids",
        #     {**base_scenario, "network_type": "grid"},
        #     subsidy_early),
    ]

    for label, scenario, subsidy in scenarios:
        print(f"\n=== Running scenario: {label} ===")

        #Intervention trials + fanchart
        baseline_X, baseline_I, subsidy_X, subsidy_I, baseline_df, subsidy_df, network_df = collect_intervention_trials(
            n_trials=n_trials,
            T=T,
            scenario_kwargs=scenario,
            subsidy_params=subsidy,
            max_workers=max_workers,
            seed_base=seed_base,
            strategy_choice_func=strategy_choice_func,
            tau=tau,
        )

        print("Baseline DF shape:", baseline_df.shape)
        print("Subsidy DF shape:", subsidy_df.shape)
        print("Baseline final X_mean:", float(baseline_df["X_mean"].iloc[-1]))

        save_scenario_data(
            label,
            "intervention",
            baseline_X=baseline_X,
            baseline_I=baseline_I,
            subsidy_X=subsidy_X,
            subsidy_I=subsidy_I,
            baseline_df=baseline_df,
            subsidy_df=subsidy_df,
            network_df=network_df
        )

        # Build traces for spaghetti/density from the same trajectories
        traces_df = traces_to_long_df(baseline_X, subsidy_X)

        save_scenario_data(
            label,
            "spaghetti_density",
            traces_df=traces_df,
        )

        print(f"Saved intervention + spaghetti/density data for {label}")

        # Also run the phase plot of X* over (X0, a_I/b) and save it
        phase_df = phase_sweep_df(
            max_workers=max_workers,
            backend="thread",
            X0_values=np.linspace(0.0, 1.0, 21),
            ratio_values=np.linspace(0.8, 3.5, 31),
            batch_size=8,
            T=200,
            strategy_choice_func=strategy_choice_func,
            tau=tau,
            scenario_kwargs=scenario
        )
        save_scenario_data(label, "phase", phase_df=phase_df)
        print(f"Saved phase data for {label}")

        # Ratio sweep computed to DF then plotted
        sweep_df = ratio_sweep_df(
            X0_frac=scenario.get("X0_frac", 0.40),
            ratio_values=np.linspace(0.8, 3.5, 31),
            scenario_kwargs=scenario,
            T=200,
            batch_size=8,
            strategy_choice_func=strategy_choice_func,
            tau=tau
        )
        save_scenario_data(label, "ratio_sweep", sweep_df=sweep_df)
        print(f"Saved ratio sweep data for {label}")


if __name__ == "__main__":
    main()