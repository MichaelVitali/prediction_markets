using Revise
using LinearAlgebra
using Combinatorics
using LossFunctions
using Statistics
using Distributions
using Plots
using StatsPlots

include("../functions/functions.jl")
using .UtilsFunctions


function get_subsets(coalition)

    subsets = combinations(coalition)
    return subsets

end

function get_subsets_excluding_players(coalition, player)

    coalition_no_player = [p for p in coalition if p != player]
    combs = get_subsets(coalition_no_player)
    subsets = []

    for comb in combs
        if "f0" in comb
            push!(subsets, comb)
        end
    end
    return subsets

end

function calculate_quantile_loss(y_true, y_preds, quantiles)

    losses = Array{Float64}(undef, 1)
    for (i, q) in enumerate(quantiles)
        loss = mean(QuantileLoss(q), y_preds[i, :], y_true)
        push!(losses, loss)
    end

    return mean(losses)

end

# Function to add a key-value pair to all dictionaries in an array
function add_keys_values_to_dicts(array_of_dicts, key, values)

    array_of_dicts_copy = deepcopy(array_of_dicts)
    for (i, dict) in enumerate(array_of_dicts_copy)
        dict[key] = values[i][key]
    end

    return array_of_dicts_copy
end

function calculate_shapley_values(forecasters_dists, forecasters_weigths::Dict, list_quantiles::Array, y_true::Array, T)

    forecasters_list = collect(keys(forecasters_dists[1]))
    shapley_values = Dict(k => 0.0 for k in forecasters_list)
    n_forecasters = length(forecasters_list) - 1

    for forecaster in forecasters_list[2:end]
        value_forecaster = 0.0
        subsets = get_subsets_excluding_players(forecasters_list, forecaster)

        for sub in subsets
            w = length(sub) - 1
            factor = factorial(w) * factorial((n_forecasters - w - 1)) / factorial(n_forecasters)
            
            dist_subset = [Dict(k => v for (k, v) in dist if k in sub) for dist in forecasters_dists]
            weights_subset = [forecasters_weigths[f] for f in sub]
            avg_quants_subset = quantile_averaging_dist_multiple_times(dist_subset, list_quantiles, weights_subset, T)
            
            dist_subset_f = add_keys_values_to_dicts(dist_subset, forecaster, forecasters_dists)
            weights_subset_f = push!(weights_subset, forecasters_weigths[forecaster])
            avg_quants_subset_f = quantile_averaging_dist_multiple_times(dist_subset_f, list_quantiles, weights_subset_f, T)

            loss_subset = calculate_quantile_loss(y_true, avg_quants_subset, list_quantiles)
            loss_subset_f = calculate_quantile_loss(y_true, avg_quants_subset_f, list_quantiles)
            
            value_sub = factor * (loss_subset - loss_subset_f)
            value_forecaster += value_sub
        end

        shapley_values[forecaster] = value_forecaster
    end

    return shapley_values
end




####### Main #######
quantiles = [0.1, 0.5, 0.9]
forecasters_dists = [
    Dict(
        "f0" => Normal(30, 50),
        "f1" => Normal(20, 5),
        "f2" => Normal(40, 5),
        "f3" => Normal(60, 10),
    ),
    Dict(
        "f0" => Normal(30, 50),
        "f1" => Normal(20, 5),
        "f2" => Normal(40, 5),
        "f3" => Normal(60, 10),
    )
]

forecasters_weigths = Dict([
    ("f0", 1),
    ("f1", 1),
    ("f2", 1),
    ("f3", 1),
])

y_true = [40, 40]
values = calculate_shapley_values(forecasters_dists, forecasters_weigths, quantiles, y_true, 2)