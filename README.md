# Prediction Markets

## Overview

This repository contains the data and code necessary to reproduce the results presented in the paper:

> **"Probabilistic Prediction Markets with Intermittent Contributions"** by Michael Vitali, Pierre Pinson.
>
> **Journal:** Submitted at IEEE Transactions on Energy Markets, Policy and Regulation
>
> **Abstract:** Although both data availability and the demand for accurate forecasts are increasing, collaboration between stakeholders is often constrained by data ownership and competitive interests. In contrast to recent proposals within cooperative game-theoretical frameworks, we place ourselves in a more general framework, based on prediction markets. There, independent agents trade forecasts of uncertain future events in exchange for rewards. We introduce and analyse a prediction market that (i) accounts for the historical performance of the agents, (ii) adapts to time-varying conditions, while (iii) permitting agents to enter and exit the market at will. The proposed design employs robust regression models to learn the optimal forecasts' combination whilst handling missing submissions. Moreover, we introduce a pay-off allocation mechanism that considers both in-sample and out-of-sample performance while satisfying several desirable economic properties. Case-studies using simulated and real-world data allow demonstrating the effectiveness and adaptability of the proposed market design.

---

## Summary

Forecast providers and energy companies can benefit from collaboration to improve renewable energy forecasts. Although several approaches have been proposed for settings where data can be freely shared, such cooperation is often hindered by privacy, ownership, and competition concerns. Prediction markets offer a promising alternative framework to enable collaboration without direct data sharing. However, existing solutions fail to consider some essential aspects of real-world applications such as:
- real-time implementation;
- historical contributions of the participants;
- the ability to accommodate intermittent participation.

In this paper, we introduce a **new prediction market** that addresses these limitations through two main contributions:

- We design a **market operator** that combines agents' forecasts through a *robust online regression model*. This approach adapts to time-varying conditions, handles missing submissions allowing agents to enter or leave the market at any time.
- We propose a **pay-off allocation mechanism** that accounts for *both in-sample and out-of-sample* performance, combining time-varying Shapley values with accuracy-based scoring. Furthermore, the mechanism is designed to satisfy several desirable *economic properties*.

---

## Getting Started (Prerequisites)

To ensure full reproducibility, you must have the following software installed:

* **Operating System:** Tested on macOS.
* **Julia:** Version **1.11.x**

> **Note on the real-world test:** Pre-computed model predictions are included in `real_world_test/saved_models/`. Running `real_world_test/rqr_validation.jl` and `real_world_test/main_rewards.jl` requires only Julia. The forecasting model training code is not included in this repository.

---

## Repository Contents

| Folder/File | Description |
| :--- | :--- |
| **`/data_generation`** | Functions to generate synthetic test cases (time-invariant, abrupt, and time-variant environments). |
| **`/functions`** | Utility functions shared across algorithms and payoff mechanisms. |
| **`/online_algorithms`** | Proposed algorithms: online quantile regression (QR), adaptive robust quantile regression (RQR), and a robust optimisation benchmark. |
| **`/payoff`** | Payoff allocation mechanisms: Shapley values, leave-one-out, and proportion of variance. |
| **`/real_world_test`** | End-to-end real-world case study on offshore wind enegy production in Belgium. |
| **`/real_world_test/saved_models`** | Pre-computed predictions from QRF, XGBoost, and NN models for three NWP sources (ECMWF, NOAA, DWD). |
| **`real_world_test/rqr_validation.jl`** | Grid search script for hyperparameter validation of the RQR algorithm on real data. |
| **`real_world_test/main_rewards.jl`** | Runs the prediction market on real-world data and produces reward analysis plots. |
| **`main_rewards.jl`** | Runs and plots reward allocation on synthetic test cases (invariant, abrupt, or variant environment). |
| **`main_convergence.jl`** | Runs and plots algorithm convergence on synthetic test cases. |
| **`main_bias_variance_metrics.jl`** | Runs and plots bias/variance decomposition metrics on synthetic test cases. |

---

## Reproduction Guide

**1. Clone the repository:**
```bash
git clone https://github.com/MichaelVitali/prediction_markets.git
cd prediction_markets
```

**2. Install Julia dependencies:**
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

**3. Run synthetic experiments:**

Each script is self-contained. Environment settings (quantiles, number of forecasters, horizon, environment type) are configured at the top of each file.

```bash
# Reward allocation across algorithms
julia main_rewards.jl

# Algorithm convergence analysis
julia main_convergence.jl

# Bias-variance decomposition metrics
julia main_bias_variance_metrics.jl
```

**4. Run the real-world case study:**

Pre-computed predictions are included — no model retraining is required.

```bash
# (Optional) Validate RQR hyperparameters via grid search
julia real_world_test/rqr_validation.jl

# Run market and generate reward plots
julia real_world_test/main_rewards.jl
```
