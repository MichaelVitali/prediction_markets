# 🔬 Project Title: Prediction Markets

## 📖 Overview

This repository contains the data and code necessary to reproduce the results presented in the paper:

> **"Prediction Markets with Intermittent Contributions"** by Michael Vitali, Pierre Pinson.
>
> **Conference:** Submitted at PSCC 2026
>
> **Abstract:** Although both data availability and the demand for accurate forecasts are increasing, collaboration between stakeholders is often constrained by data ownership and competitive interests. In contrast to recent proposals within cooperative game-theoretical frameworks, we place ourselves in a more general framework, based on prediction markets. There, independent agents trade forecasts of uncertain future events in exchange for rewards. We introduce and analyse a prediction market that (i) accounts for the historical performance of the agents, (ii) adapts to time-varying conditions, while (iii) permitting agents to enter and exit the market at will. The proposed design employs robust regression models to learn the optimal forecasts' combination whilst handling missing submissions. Moreover, we introduce a pay-off allocation mechanism that considers both in-sample and out-of-sample performance while satisfying several desirable economic properties. Case-studies using simulated and real-world data allow demonstrating the effectiveness and adaptability of the proposed market design.

---

## Summary
Forecast providers and energy companies can benefit from collaboration to improve renewable energy forecasts. Although several approaches have been proposed for settings where data can be freely shared, such cooperation is often hindered by privacy, ownership, and competition concerns. Prediction markets offer a promising alternative framework to enable collaboration without direct data sharing. However, existing solutions fail to consider some essential aspects of real-world applications such as:
- real-time implementation;
- historical contributions of the participants;
- the ability to accommodate intermittent participation.

In this paper, we introduce a **new prediction market** that addresses these limitations through two main contributions:

- We design a **market operator** that combines agents’ forecasts through a *robust online regression model*. This approach adapts to time-varying conditions, handles missing submissions allowing agents to enter or leave the market at any time.
- We propose a **pay-off allocation mechanism** that accounts for *both in-sample and out-of-sample* performance, combining time-varying Shapley values with accuracy-based scoring. Furthermore, the mechanism is designed to satisfy several desirable *economic properties*.

Furthermore, in this repo you can find all reproducibility code to recreate the symulated environments and test the implemented algorithms and pay-off allocation functions.

## 🛠️ Getting Started (Prerequisites)

To ensure full reproducibility, you must have the following software installed:

* **Operating System:** Tested on macOS.
* **Julia:** Version **1.11.x**

## 📂 Repository Contents

| Folder/File | Description |
| :--- | :--- |
| **`/data_generation`** | Structure that contains all the functions to generate the synthetic test cases. |
| **`/functions`** | Contains utility functions used by the algorithms. |
| **`online_algorithms`** | Contains all the algorithms proposed and developed. |
| **`/payoff`** | Payoff contains all the implemented payoff allocation functions. |
| **`/real_world_test`** | Contains the code to run the proposed method on real world scenario. |
| **`README.md`** | This documentation file. |
| **`main_rewards`** | Main used to run and plot the methods on synthetic test cases and showcase methods calculated rewards. |
| **`main_convergence`** | Main used to run and plot the methods on synthetic test cases and showcase methods convergence. |
| **`main_bias_variance_metrics`** | Main used to run and plot the methods on synthetic test cases and showcase bias and variance metrics. |

## ⚙️ Reproduction Guide

**1. Clone the repository:**
```bash
git clone https://github.com/MichaelVitali/prediction_markets.git
cd prediction_markets
```

**2. Run main files**
```bash
/path/to/julia file_name.jl
```
