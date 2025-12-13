# EV Stag Hunt Model — Assignment 3
The simulation model was adapted from the base code provided for the assignment (Lees, 2025). For full context and explanation of parametres, refer to the original repository.

## Original Code
Lees, M. (2025). Model-Based Decisions Assignment 3 Code [Source code]. GitHub. https://github.com/mhlees/Model-Based-Decisions-Code/tree/main/Assignment%203

## What’s Included
- `ev_core.py` — Core Mesa model and helpers: agents, network generation, step dynamics, initial adopters, and computation utilities for ratio/phase sweeps.
- `ev_experiments.py` — Main entry point. Runs the simulation scenarios required for baseline, different topologies and policies.
- `ev_plotting.py` — Plotting file from original file structure, mainly there to assist with 'ev_experiments'.
- `plots/` — Output directory for generated figures.
- `data/` - Output directory for generated data.
- `metrics.ipynb` - Calculates metrics based on runs from `ev_experiments`
- `plotting.ipynb` - Creates plots based on runs from `ev_experiments`

## Quick Start
- Install dependencies from the repo root:
  - `pip install -r requirements.txt`
- Run the example experiments from this folder:
  - `cd Assignment3\`
  - `python ev_experiments.py`
- Analyse results
  - Run metrics `metrics.ipynb`
  - Run plots from `plotting.ipynb'
- Outputs:
  - Data is saved under `data/` by default. 
  - `plots/` contains generated figures from the experiments