using LinearAlgebra
using Convex
using DataStructures
using Plots

include("functions/functions.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
using .UtilsFunctions
using .DataGeneration
using .AdaptiveRobustRegression

n_experiments = 100
T = 10000
q = 0.9
n_forecasters = 3

exp_weights = zeros((n_forecasters, T))
exp_weights[:, 1] = initialize_weights(n_forecasters)
D = zeros(n_forecasters, n_forecasters)
true_weights = nothing

for i in 1:n_experiments
    weights_exp = zeros((n_forecasters, T))
    weights_exp[:, 1] = initialize_weights(n_forecasters)
    realizations, forecasters_preds, true_weights = generate_time_invariant_data(T, q)
    global true_weights = true_weights
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)
    D_exp = zeros(n_forecasters, n_forecasters)

    alpha = zeros(Int, n_forecasters, T)
    for t in 1:T
        alpha[:, t] = Int.(rand(n_forecasters) .< 0.1)
    end

    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = realizations[t]

        new_w, new_D = online_adaptive_robust_quantile_regression(forecasters_preds_t, y_true, weights_exp[:, t-1], D_exp, alpha[:, t], q)
        weights_exp[:, t] = new_w
        D_exp = new_D
    end

    global exp_weights .+= weights_exp
    global D .+= D_exp
end

exp_weights ./= n_experiments
D ./= n_experiments
display(exp_weights)
display(D)

# Plot weights for this algorithm only (no subfigures)
plt = plot(1:T, exp_weights', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
    xlabel="Time", ylabel="Weights", title="Weights History Over Time")

# Optionally plot true_weights if available
if true_weights !== nothing
    plot!(plt, 1:T, true_weights, label=["w1" "w2" "w3"])
end

display(plt)
