using LinearAlgebra
using Plots
using DataStructures
using ProgressBars
using Base.Threads
using Plots.PlotMeasures
using LossFunctions
using Normalization

include("../functions/functions.jl")
include("../functions/functions_payoff.jl")
include("../online_algorithms/quantile_regression.jl")
include("../online_algorithms/adaptive_robust_quantile_regression.jl")
include("../payoff/shapley_values.jl")
include("data_preprop.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .QuantileRegression
using .Shapley
using .AdaptiveRobustRegression
using .RealWorldtestData


# Environment Settings
n_experiments = 10
T = 31
lead_time = 96
quantiles = [0.1, 0.5, 0.9]
total_reward = 100
delta = 0.7
n_forecasters = 2
algorithms = ["QR", "RQR"]
payoff_function = "Shapley"
path_ecmwf = "real_world_test/data/forecasts_ecmwf_ifs_paper.parquet"
path_noaa = "real_world_test/data/forecasts_noaa_gfs_paper.parquet"
path_elia = "real_world_test/data/historical_load_data_predico_2025_08_15.csv"

# Environment Variables
realizations = Dict([q => [] for q in quantiles])
algo_forecasts = Dict([q => Dict(["ecmwf" => [], "noaa" => []]) for q in quantiles])
elia_forecasts = Dict([q => [] for q in quantiles])
payoffs = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
weights = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_in_sample = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_out_sample = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
losses_rqr = Dict([q => zeros(T) for q in quantiles])
losses_qr = Dict([q => zeros(T) for q in quantiles])

#################### Quantile Regression ####################
algo = "QR"

for q in quantiles
    quantile_step_reward = total_reward / length(quantiles)

    Threads.@threads for exp in ProgressBar(1:n_experiments)
        # Experiment Variables
        weights_exp = zeros(n_forecasters, T)
        payoffs_exp = zeros(n_forecasters, T)
        rewards_exp = zeros(n_forecasters, T)
        rewards_in_exp = zeros(n_forecasters, T)
        rewards_out_exp = zeros(n_forecasters, T)

        weights_exp[:, 1] = initialize_weights(n_forecasters)

        # Data generation
        true_prod, forecasters_preds, forecasts_elia, scaler_ecmwf, scaler_noaa, scaler_target, scaler_elia = preprocessing_forecasts(path_ecmwf, path_noaa, path_elia, q)
        sorted_f = sort(collect(forecasters_preds), by=first)
        sorted_forecasters = OrderedDict(sorted_f)

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
            y_true = true_prod[t]
            if exp == 1
                push!(realizations[q], denormalize(y_true, scaler_target))
                push!(elia_forecasts[q], denormalize(forecasts_elia[t], scaler_elia))

                for f in keys(forecasters_preds)
                    if f == "ecmwf"
                        push!(algo_forecasts[q][f], denormalize(forecasters_preds[f][t], scaler_ecmwf))
                    else
                        push!(algo_forecasts[q][f], denormalize(forecasters_preds[f][t], scaler_noaa))
                    end
                end
            end

            weights_exp[:, t], aggregated_forecast_t = online_quantile_regression_update_multiple_lead_times(forecasters_preds_t, weights_exp[:, t-1], y_true, q, 0.01)
            loss_t = mean(QuantileLoss(q).(denormalize(y_true, scaler_target) .- denormalize(aggregated_forecast_t, scaler_target)))
            losses_qr[q][t] += loss_t

            temp_payoffs = shapley_payoff_multiple_lead_times(forecasters_preds_t, weights_exp[:, t-1], y_true, q)
            forecasters_losses = [mean(QuantileLoss(q).(forecasters_preds_t[i], y_true)) for (i, f) in enumerate(keys(sorted_forecasters))]
            temp_scores = 1 .- (forecasters_losses ./ sum(forecasters_losses))
            payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, 0.999)

            rewards_in = delta .* quantile_step_reward .* (max.(0, payoffs_exp[:, t]) ./ max(sum(max.(0, payoffs_exp[:, t])), eps()))
            rewards_out = (1-delta) .* quantile_step_reward .* (temp_scores ./ sum(temp_scores))

            rewards_in_exp[:, t] = rewards_in
            rewards_out_exp[:, t] = rewards_out
            rewards_exp[:, t] = rewards_in .+ rewards_out
        end

        payoffs[q][algo] += payoffs_exp
        weights[q][algo] += weights_exp
        rewards[q][algo] += rewards_exp
        rewards_in_sample[q][algo] += rewards_in_exp
        rewards_out_sample[q][algo] += rewards_out_exp
    end
end

for q in quantiles
    payoffs[q][algo] = payoffs[q][algo] ./ n_experiments
    weights[q][algo] = weights[q][algo] ./ n_experiments
    rewards[q][algo] = rewards[q][algo] ./ n_experiments
    rewards_in_sample[q][algo] = rewards_in_sample[q][algo] ./ n_experiments
    rewards_out_sample[q][algo] = rewards_out_sample[q][algo] ./ n_experiments
end

#################### Robust Quantile Regression ####################
algo = "RQR"
missing_rate = 0.05

for q in quantiles

    quantile_step_reward = total_reward / length(quantiles)

    Threads.@threads for exp in ProgressBar(1:n_experiments)
        # Experiment Variables
        weights_exp = zeros(n_forecasters, T)
        payoffs_exp = zeros(n_forecasters, T)
        rewards_exp = zeros(n_forecasters, T)
        rewards_in_exp = zeros(n_forecasters, T)
        rewards_out_exp = zeros(n_forecasters, T)

        weights_exp[:, 1] = initialize_weights(n_forecasters)

        # Data generation
        true_prod, forecasters_preds, forecasts_elia, scaler_ecmwf, scaler_noaa, scaler_target, scaler_elia = preprocessing_forecasts(path_ecmwf, path_noaa, path_elia, q)
        sorted_f = sort(collect(forecasters_preds), by=first)
        sorted_forecasters = OrderedDict(sorted_f)

        sorted_f = sort(collect(forecasters_preds), by=first)
        sorted_forecasters = OrderedDict(sorted_f)

        D_exp = zeros(n_forecasters, n_forecasters)
        alpha = Int.(rand(n_forecasters, T) .< missing_rate)
        for t in 1:T
            if sum(alpha[:, t]) == length(alpha[:, t])
                idx = rand(1:length(alpha[:, t]))
                alpha[idx, t] = 0
            end
        end

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
            y_true = true_prod[t]

            # Learning Phase
            weights_exp[:, t], new_D, aggregated_forecast_t = online_adaptive_robust_quantile_regression_multiple_lead_times(forecasters_preds_t, y_true, weights_exp[:, t-1], D_exp, alpha[:, t], q)
            prev_D = D_exp
            D_exp = new_D
            loss_t = mean(QuantileLoss(q).(denormalize(y_true, scaler_target) .- denormalize(aggregated_forecast_t, scaler_target)))
            losses_rqr[q][t] += loss_t

            # Payoff Calculation
            temp_forecasts_t = [forecasters_preds_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
            temp_weights_t = weights_exp[:, t-1] .+ prev_D * alpha[:, t]
            temp_weights_t = [temp_weights_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]

            temp_payoffs = nothing
            forecasters_losses = nothing
            if length(temp_forecasts_t) > 0
                temp_payoffs = shapley_payoff_multiple_lead_times(temp_forecasts_t, temp_weights_t, y_true, q)
                forecasters_losses = [mean(QuantileLoss(q).(temp_forecasts_t[i], y_true)) for i in 1:length(temp_forecasts_t)]
                temp_scores = [max(1 - (forecasters_losses[i] / sum(forecasters_losses)), eps()) for i in 1:length(forecasters_losses)]
            else 
                temp_payoffs = zeros(n_forecasters)
                forecasters_losses = ones(n_forecasters)
            end
            
            if length(temp_payoffs) < n_forecasters
                for j in findall(a -> a == 1, alpha[:, t])
                    insert!(temp_payoffs, j, 0.0)
                    insert!(temp_scores, j, 0.0)
                end
            end
            payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, 0.999)

            # Reward calculation
            rewards_in = zeros(n_forecasters)
            rewards_out = zeros(n_forecasters)

            payoffs_for_rewards_in = [max(0, payoffs_exp[j, t]) for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
            rewards_in[[j for j in 1:n_forecasters if alpha[j, t] == 1]] .= eps()
            rewards_in[[j for j in 1:n_forecasters if alpha[j, t] == 0]] = delta .* quantile_step_reward .* (payoffs_for_rewards_in ./ max(sum(payoffs_for_rewards_in), eps()))

            scores_for_rewards_out = [temp_scores[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
            rewards_out[[j for j in 1:n_forecasters if alpha[j, t] == 1]] .= eps()
            rewards_out[[j for j in 1:n_forecasters if alpha[j, t] == 0]] = (1-delta) .* quantile_step_reward .* (scores_for_rewards_out / sum(scores_for_rewards_out))

            rewards_in_exp[:, t] = rewards_in
            rewards_out_exp[:, t] = rewards_out
            rewards_exp[:, t] = rewards_in .+ rewards_out
        end

        payoffs[q][algo] += payoffs_exp
        weights[q][algo] += weights_exp
        rewards[q][algo] += rewards_exp
        rewards_in_sample[q][algo] += rewards_in_exp
        rewards_out_sample[q][algo] += rewards_out_exp
    end
end

for q in quantiles
    payoffs[q][algo] = payoffs[q][algo] ./ n_experiments
    weights[q][algo] = weights[q][algo] ./ n_experiments
    rewards[q][algo] = rewards[q][algo] ./ n_experiments
    rewards_in_sample[q][algo] = rewards_in_sample[q][algo] ./ n_experiments
    rewards_out_sample[q][algo] = rewards_out_sample[q][algo] ./ n_experiments
end

#################### Calculate Errors ####################
for q in quantiles
    losses_rqr[q] = losses_rqr[q] ./ n_experiments
    losses_qr[q] = losses_qr[q] ./ n_experiments

    losses_aggregated = []
    losses_ecmwf = []
    losses_noaa = []

    for t in 1:T-1
        loss_t_ecmwf = mean(QuantileLoss(q).(realizations[q][t], algo_forecasts[q]["ecmwf"][t]))
        loss_t_noaa = mean(QuantileLoss(q).(realizations[q][t], algo_forecasts[q]["noaa"][t]))
        append!(losses_ecmwf, loss_t_ecmwf)
        append!(losses_noaa, loss_t_noaa)
    end
    println("############ RESULTS $q ############")
    println("Loss aggregated QR: $(mean(losses_qr[q]))")
    println("Loss aggregated RQR: $(mean(losses_rqr[q]))")
    println("Loss ECWMF: $(mean(losses_ecmwf))")
    println("Loss NOAA: $(mean(losses_noaa))")
end

#################### Plot Results ####################
total_rewards_forecasters = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_in_rewards_forecasters = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_out_rewards_forecasters = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
for algo in algorithms
    for i in 1:n_forecasters
        for q in quantiles
            total_rewards_forecasters[algo][i, :] .+= rewards[q][algo][i, :]
            total_in_rewards_forecasters[algo][i, :] .+= rewards_in_sample[q][algo][i, :]
            total_out_rewards_forecasters[algo][i, :] .+= rewards_out_sample[q][algo][i, :]
        end
    end
end

println("############ REWARDS SINGLE QUANTILES ############")
for q in quantiles
    for algo in algorithms
        println("REWARDS $q")
        println("Total reward ECMWF $algo: $(sum(rewards[q][algo][1, :]))")
        println("Total reward NOAA $algo: $(sum(rewards[q][algo][2, :]))")
        println("In-sample reward ECMWF $algo: $(sum(rewards_in_sample[q][algo][1, :]))")
        println("In-sample reward NOAA $algo: $(sum(rewards_in_sample[q][algo][2, :]))")
        println("Out-of-sample reward ECMWF $algo: $(sum(rewards_out_sample[q][algo][1, :]))")
        println("Out-of-sample reward NOAA $algo: $(sum(rewards_out_sample[q][algo][2, :]))")
    end
end

println("############ TOTAL REWARDS ############")
for algo in algorithms
    println("Total reward ECMWF $algo: $(sum(total_rewards_forecasters[algo][1, :]))")
    println("Total reward NOAA $algo: $(sum(total_rewards_forecasters[algo][2, :]))")
    println("In-sample reward ECMWF $algo: $(sum(total_in_rewards_forecasters[algo][1, :]))")
    println("In-sample reward NOAA $algo: $(sum(total_in_rewards_forecasters[algo][2, :]))")
    println("Out-of-sample reward ECMWF $algo: $(sum(total_out_rewards_forecasters[algo][1, :]))")
    println("Out-of-sample reward NOAA $algo: $(sum(total_out_rewards_forecasters[algo][2, :]))")
end