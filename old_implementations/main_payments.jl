using LinearAlgebra
using Plots
using Statistics
using DataStructures
using Base.Threads
using LossFunctions

include("functions/functions.jl")
include("functions/functions_payoff.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/quantile_regression.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
include("payoff/leave_one_out.jl")
include("payoff/shapley_values.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .DataGeneration
using .QuantileRegression
using .LOO
using .Shapley
using .AdaptiveRobustRegression

# Environment Settings
q = 0.9
n_forecasters = 3
algorithms = ["QR", "RQR"]
payoff_functions = "Shapley"
daily_reward = 500
T = 10000
n_experiments = 100
delta = 0.7

# Environment Variables
payoffs = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
rewards = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
weights = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
scores = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])

# Environment generation
## Weights initialization
for algo in algorithms
    weights[algo][:, 1] .= initialize_weights(n_forecasters)
end

#################### Quantile Regression ####################
algo = "QR"

Threads.@threads for exp in 1:n_experiments
    # Experiment Variables
    weights_exp = zeros(n_forecasters, T)
    payoffs_exp = zeros(n_forecasters, T)
    rewards_exp = zeros(n_forecasters, T)
    scores_exp = zeros(n_forecasters, T)

    weights_exp[:, 1] = initialize_weights(n_forecasters)

    # Data generation
    realizations, forecasters_preds, true_weights = generate_time_invariant_data(T, q)
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)

    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = realizations[t]

        weights_exp[:, t], _ = online_quantile_regression_update(forecasters_preds_t, weights_exp[:, t-1], y_true, q, 0.01)
        temp_payoffs = shapley_payoff(forecasters_preds_t, weights_exp[:, t-1], y_true, q)
        temp_scores = 1 .- (QuantileLoss(q).(forecasters_preds_t, y_true) ./ sum(QuantileLoss(q).(forecasters_preds_t, y_true)))
        payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, 0.999)
        scores_exp[:, t] = payoff_update(scores_exp[:, t-1], temp_scores, 0.999)

        rewards_in = delta .* daily_reward .* (max.(0, payoffs_exp[:, t]) ./ sum(max.(0, payoffs_exp[:, t])))
        rewards_out = (1-delta) .* daily_reward .* (temp_scores ./ sum(temp_scores))
        rewards_exp[:, t] = rewards_in .+ rewards_out

    end

    payoffs[algo] += payoffs_exp
    weights[algo] += weights_exp
    rewards[algo] += rewards_exp
    scores[algo] += scores_exp
end

payoffs[algo] = payoffs[algo] ./ n_experiments
weights[algo] = weights[algo] ./ n_experiments
rewards[algo] = rewards[algo] ./ n_experiments
scores[algo] = scores[algo] ./ n_experiments

scores[algo] = scores[algo] ./ sum(scores[algo], dims=1)

plot_scores = plot(size=(1000, 400))
plot!(plot_scores, 1:T, rewards["QR"]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], lw=2)
xlabel!(plot_scores, "Time")
ylabel!(plot_scores, "Reward")
title!(plot_scores, "Rewards for All Forecasters (QR)")
display(plot_scores)


#################### Robust Quantile Regression ####################
algo = "RQR"
missing_forecast = 2

Threads.@threads for exp in 1:n_experiments
    # Experiment Variables
    weights_exp = zeros(n_forecasters, T)
    payoffs_exp = zeros(n_forecasters, T)
    rewards_exp = zeros(n_forecasters, T)

    weights_exp[:, 1] = initialize_weights(n_forecasters)

    # Data generation
    realizations, forecasters_preds, true_weights = generate_time_invariant_data(T, q)
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)

    D_exp = zeros(n_forecasters, n_forecasters)
    alpha = zeros(n_forecasters, T)
    alpha[missing_forecast, :] = Int.(rand(T) .< 0.9)

    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = realizations[t]

        # Learning Phase
        weights_exp[:, t], new_D, _ = online_adaptive_robust_quantile_regression(forecasters_preds_t, y_true, weights_exp[:, t-1], D_exp, alpha[:, t], q)
        prev_D = D_exp
        D_exp = new_D

        # Shapley Calculation
        temp_forecasts_t = [forecasters_preds_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
        temp_weights_t = weights_exp[:, t-1] .+ prev_D * alpha[:, t]
        temp_weights_t = [temp_weights_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]

        temp_payoffs = shapley_payoff(temp_forecasts_t, temp_weights_t, y_true, q)
        temp_scores = 1 .- (QuantileLoss(q).(temp_forecasts_t, y_true) ./ sum(QuantileLoss(q).(temp_forecasts_t, y_true)))
        if length(temp_payoffs) < n_forecasters
            insert!(temp_payoffs, missing_forecast, 0.0)
            insert!(temp_scores, missing_forecast, 0.0)
        end
        payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, 0.999)

        # Reward calculation
        rewards_in = zeros(n_forecasters)
        rewards_out = zeros(n_forecasters)

        payoffs_for_rewards_in = [max(0, payoffs_exp[j, t]) for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
        rewards_in[missing_forecast] = 0.0
        rewards_in[[j for j in 1:n_forecasters if alpha[j, t] == 0]] = delta .* daily_reward .* (payoffs_for_rewards_in ./ sum(payoffs_for_rewards_in))

        scores_for_rewards_out = [temp_scores[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
        rewards_out[missing_forecast] = 0.0
        rewards_out[[j for j in 1:n_forecasters if alpha[j, t] == 0]] = (1-delta) .* daily_reward .* (scores_for_rewards_out / sum(scores_for_rewards_out))

        rewards_exp[:, t] = rewards_in .+ rewards_out
    end

    payoffs[algo] += payoffs_exp
    weights[algo] += weights_exp
    rewards[algo] += rewards_exp
end

payoffs[algo] = payoffs[algo] ./ n_experiments
weights[algo] = weights[algo] ./ n_experiments
rewards[algo] = rewards[algo] ./ n_experiments
display(rewards[algo])

for window in [1000, 9999]
    #plot_rewards = plot(layout=(n_forecasters-1, 1), size=(1000, 500))
    plot_rewards = plot(layout=(n_forecasters, 1), size=(1000, 500))
    for forecaster_idx in 1:n_forecasters
        #if forecaster_idx == missing_forecast
        #    continue
        #end
        #i = forecaster_idx - (forecaster_idx > missing_forecast ? 1 : 0)
        i = forecaster_idx
        plot!(plot_rewards[i], T-window:T, payoffs["QR"][forecaster_idx, end-window:end], label="QR - Forecaster $forecaster_idx", lw=2)
        plot!(plot_rewards[i], T-window:T, payoffs["RQR"][forecaster_idx, end-window:end], label="RQR - Forecaster $forecaster_idx", lw=2)
        xlabel!(plot_rewards[i], "Time")
        ylabel!(plot_rewards[i], "Value")
        title!(plot_rewards[i], "Shapley Forecaster $forecaster_idx (window=$window)")
    end
    display(plot_rewards)
end

for window in [9999]
    #plot_rewards = plot(layout=(n_forecasters-1, 1), size=(1000, 500))
    plot_rewards = plot(layout=(n_forecasters, 1), size=(1000, 500))
    for forecaster_idx in 1:n_forecasters
        #if forecaster_idx == missing_forecast
        #    continue
        #end
        #i = forecaster_idx - (forecaster_idx > missing_forecast ? 1 : 0)
        i = forecaster_idx
        plot!(plot_rewards[i], T-window:T, rewards["QR"][forecaster_idx, end-window:end], label="QR - Forecaster $forecaster_idx", lw=2)
        plot!(plot_rewards[i], T-window:T, rewards["RQR"][forecaster_idx, end-window:end], label="RQR - Forecaster $forecaster_idx", lw=2)
        xlabel!(plot_rewards[i], "Time")
        ylabel!(plot_rewards[i], "£")
        title!(plot_rewards[i], "Rewards Forecaster $forecaster_idx (window=$window)")
    end
    display(plot_rewards)
end

plot_weigths = plot(layout=(length(algorithms), 1), size=(1000, 500))
for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, weights[algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
          xlabel="Time", ylabel="Weights", title="Weights History Over Time - $algo")
end
#display(plot_weigths)