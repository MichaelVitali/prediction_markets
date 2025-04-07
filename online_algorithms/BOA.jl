using Revise
using LinearAlgebra
using Distributions
using Plots
using DataFrames
using LossFunctions
using DataStructures
using Statistics

include("../data_generation/DataGeneration.jl")
include("../functions/functions.jl")
using .UtilsFunctions
using .DataGeneration

function initialize_weights(n_experts::Integer)

    weigths = fill(1 / n_experts, n_experts)

    return weigths

end

function bernstein_online_aggregation(forecasters_preds, forecaster_weights, y_true, T, q)

    forecasters_names = collect(keys(forecasters_preds))
    weights_history = Matrix{Float64}(undef, length(forecaster_weights), T)
    weights_history[:, 1] = forecaster_weights
    avg_quantiles_history = zeros(T)
    learning_rate = 0.01

    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in forecasters_names]
        agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
        #agg_loss = QuantileLoss(q)(agg_quantile_t, y_true[t])
        #agg_loss = sum((agg_quantile_t .- y_true[t]).^2) / length(agg_quantile_t)

        #losses = [QuantileLoss(q)(forecasters_preds[f][t], y_true[t]) for f in forecasters_names]
        #losses = [sum((forecasters_preds[f][t] .- y_true[t]).^2) / length(forecasters_preds[f][t]) for f in forecasters_names]

        #lks = losses .- agg_loss
        lks = quantile_loss_gradient(y_true[t], agg_quantile_t, q) .* (forecasters_preds_t .- agg_quantile_t)

        numerators = weights_history[:, t-1] .* exp.(-learning_rate .* lks .* (1 .- (learning_rate .* lks)))
        weights_history[:, t] = numerators ./ sum(numerators)
        avg_quantiles_history[t] = agg_quantile_t
    end

    return weights_history, avg_quantiles_history
end

function optimal_weights(forecaster_preds, y_true)

    X = hcat([(y_true .- forecaster_preds[f]) for f in keys(forecaster_preds)]...)
    covariance_matrix = cov(X)
    id = ones(size(X, 2))
    optimal_weights = (covariance_matrix \ id) / (id' * (covariance_matrix \  id))

    return optimal_weights

end

function constrained_OLS(forecaster_preds, y_true)

    F = hcat([forecaster_preds[f] for f in keys(forecaster_preds)]...)
    l = ones(length(keys(forecaster_preds)))
    alpha = (transpose(F) * F) \ transpose(F) * y_true
    lambda = (l' * alpha .- 1) ./ (l' * ((transpose(F) * F) \ l))
    
    optimal_w = alpha .- (lambda * ((transpose(F) * F) \ l))
    return optimal_w

end


function oracle(forecasters_preds, optimal_weights, T)

    oracle_history = zeros(T)
    for t in 1:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(forecasters_preds)]
        agg_quantile_t = sum(forecasters_preds_t .* optimal_weights)
        oracle_history[t] = agg_quantile_t
    end

    return oracle_history
end

function cumulative_regret(agg_quantiles_history, oracle_history)

    regret = cumsum(agg_quantiles_history) .- cumsum(oracle_history)

    return regret
end

n_experiments = 150
T = 10000
q = 0.5
n_forecasters = 3
exp_weights = zeros((n_forecasters, T))
opt_weights = zeros((n_forecasters, 1))
cum_regrets = zeros(T)
case_study = "Gneiting"

for i in 1:n_experiments
    forecaster_weight = initialize_weights(n_forecasters)
    
    invariant_data, forecasters_preds = data_generation(case_study, T, q)
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)
    
    weights_history, agg_quantiles_history = bernstein_online_aggregation(sorted_forecasters, forecaster_weight, invariant_data, T, q)
    if case_study == "Gneiting"
        optimal_w = optimal_weights(sorted_forecasters, invariant_data)
        #oracle_history = oracle(sorted_forecasters, optimal_w, T)
        opt_weights .+= optimal_w
    end
    #regret = cumulative_regret(agg_quantiles_history, oracle_history)
    
    exp_weights .+= weights_history 
    #cum_regrets .+= regret
end

exp_weights = exp_weights ./ n_experiments
#cum_regrets = cum_regrets ./ n_experiments

p = plot(1:T, exp_weights', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], xlabel="Time", ylabel="Weights", title="Weights History Over Time")
if case_study == "Gneiting"
    opt_weights = opt_weights ./ n_experiments
    plot!(p, 1:T, hcat([opt_weights for _ in 1:T]...)', label=["w1" "w2" "w3"])
end
display(p)
#plot(1:T, cum_regrets)