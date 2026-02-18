using LinearAlgebra
using Plots
using DataStructures
using ProgressBars
using Base.Threads
using Plots.PlotMeasures
using LossFunctions
using Normalization
using RollingFunctions

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

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))


# Environment Settings
n_experiments = 100
T = 184
lead_time = 96
quantiles = [0.1, 0.5, 0.9]
total_reward = 100
delta = 0.7
algorithms = ["QR", "RQR"]
payoff_function = "Shapley"

root_dir = @__DIR__
models_paths = Dict(
    "ecmwf_nn" => joinpath(root_dir, "saved_models", "predictions_nn_ecmwf_ifs.parquet"),
    "noaa_nn" => joinpath(root_dir, "saved_models", "predictions_nn_noaa_gfs.parquet"),
    "dwd_nn" => joinpath(root_dir, "saved_models", "predictions_nn_dwd_icon_eu.parquet"),
    "ecmwf_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_ecmwf_ifs.parquet"),
    "noaa_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_noaa_gfs.parquet"),
    "dwd_xgb" => joinpath(root_dir, "saved_models", "predictions_xgb_dwd_icon_eu.parquet"),
    "ecmwf_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_ecmwf_ifs.parquet"),
    "noaa_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_noaa_gfs.parquet"),
    "dwd_qrf" => joinpath(root_dir, "saved_models", "predictions_qrf_dwd_icon_eu.parquet"),
)
model_names = collect(keys(models_paths))
n_forecasters = length(model_names)

# Environment Variables
realizations = Dict([q => Vector{Vector{Float64}}() for q in quantiles])
algo_forecasts = Dict([q => Dict([name => [] for name in model_names]) for q in quantiles])
payoffs = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
weights = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_in_sample = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_out_sample = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
losses_rqr = Dict([q => zeros(T) for q in quantiles])
losses_qr = Dict([q => zeros(T) for q in quantiles])

#################### Quantile Regression ####################
algo = "QR"
data_lock = ReentrantLock()

for q in quantiles
    quantile_step_reward = total_reward / length(quantiles)

    Threads.@threads for exp in ProgressBar(1:n_experiments)
        # Experiment Variables
        weights_exp = zeros(n_forecasters, T)
        payoffs_exp = zeros(n_forecasters, T)
        rewards_exp = zeros(n_forecasters, T)
        rewards_in_exp = zeros(n_forecasters, T)
        rewards_out_exp = zeros(n_forecasters, T)
        losses_qr_exp = zeros(T)

        weights_exp[:, 1] = initialize_weights(n_forecasters)

        # Data generation
        true_prod, forecasters_preds, scalers, scaler_target = preprocessing_forecasts(models_paths, q)
        sorted_f = sort(collect(forecasters_preds), by=first)
        sorted_forecasters = OrderedDict(sorted_f)

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
            y_true = true_prod[t]
            if exp == 1
                lock(data_lock) do
                    if isempty(realizations[q])
                        push!(realizations[q], zeros(length(y_true))) 
                    end
                    push!(realizations[q], copy(y_true))

                    for name in keys(forecasters_preds)
                        if isempty(algo_forecasts[q][name])
                            push!(algo_forecasts[q][name], zeros(length(y_true))) 
                        end
                        push!(algo_forecasts[q][name], denormalize(forecasters_preds[name][t], scalers[name])) 
                    end
                end
            end

            weights_exp[:, t], aggregated_forecast_t = online_quantile_regression_update_multiple_lead_times(forecasters_preds_t, weights_exp[:, t-1], y_true, q, 0.01)
            aggregated_forecast_t = denormalize(aggregated_forecast_t, scaler_target)
            # Loss calculation
            loss_t = mean(QuantileLoss(q).(aggregated_forecast_t .- y_true))
            losses_qr_exp[t] = loss_t

            temp_payoffs = shapley_payoff_multiple_lead_times(forecasters_preds_t, weights_exp[:, t-1], y_true, q)
            forecasters_losses = [mean(QuantileLoss(q).(forecasters_preds_t[i] .- y_true)) for (i, f) in enumerate(keys(sorted_forecasters))]
            temp_scores = 1 .- (forecasters_losses ./ sum(forecasters_losses))
            payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, 0.999)

            rewards_in = delta .* quantile_step_reward .* (max.(0, payoffs_exp[:, t]) ./ max(sum(max.(0, payoffs_exp[:, t])), eps()))
            rewards_out = (1-delta) .* quantile_step_reward .* (temp_scores ./ sum(temp_scores))

            rewards_in_exp[:, t] = rewards_in
            rewards_out_exp[:, t] = rewards_out
            rewards_exp[:, t] = rewards_in .+ rewards_out
        end
        lock(data_lock) do
            payoffs[q][algo] += payoffs_exp
            weights[q][algo] += weights_exp
            rewards[q][algo] += rewards_exp
            rewards_in_sample[q][algo] += rewards_in_exp
            rewards_out_sample[q][algo] += rewards_out_exp
            losses_qr[q] += losses_qr_exp
        end
    end
end

for q in quantiles
    payoffs[q][algo] = payoffs[q][algo] ./ n_experiments
    weights[q][algo] = weights[q][algo] ./ n_experiments
    rewards[q][algo] = rewards[q][algo] ./ n_experiments
    rewards_in_sample[q][algo] = rewards_in_sample[q][algo] ./ n_experiments
    rewards_out_sample[q][algo] = rewards_out_sample[q][algo] ./ n_experiments
    losses_qr[q] = losses_qr[q] ./ n_experiments
end

#################### Robust Quantile Regression ####################
algo = "RQR"
missing_rate = 0.05
data_lock = ReentrantLock()

for q in quantiles

    quantile_step_reward = total_reward / length(quantiles)

    Threads.@threads for exp in ProgressBar(1:n_experiments)
        # Experiment Variables
        weights_exp = zeros(n_forecasters, T)
        payoffs_exp = zeros(n_forecasters, T)
        rewards_exp = zeros(n_forecasters, T)
        rewards_in_exp = zeros(n_forecasters, T)
        rewards_out_exp = zeros(n_forecasters, T)
        losses_rqr_exp = zeros(T)

        weights_exp[:, 1] = initialize_weights(n_forecasters)

        # Data generation
        true_prod, forecasters_preds, scalers, scaler_target = preprocessing_forecasts(models_paths, q)
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
            aggregated_forecast_t = denormalize(aggregated_forecast_t, scaler_target)
            # Loss calculation
            loss_t = mean(QuantileLoss(q).(aggregated_forecast_t .- y_true))
            losses_rqr_exp[t] = loss_t

            # Payoff Calculation
            temp_forecasts_t = [forecasters_preds_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
            temp_weights_t = weights_exp[:, t-1] .+ prev_D * alpha[:, t]
            temp_weights_t = [temp_weights_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]

            temp_payoffs = nothing
            forecasters_losses = nothing
            if length(temp_forecasts_t) > 0
                temp_payoffs = shapley_payoff_multiple_lead_times(temp_forecasts_t, temp_weights_t, y_true, q)
                forecasters_losses = [mean(QuantileLoss(q).(temp_forecasts_t[i] .- y_true)) for i in 1:length(temp_forecasts_t)]
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

        lock(data_lock) do
            payoffs[q][algo] += payoffs_exp
            weights[q][algo] += weights_exp
            rewards[q][algo] += rewards_exp
            rewards_in_sample[q][algo] += rewards_in_exp
            rewards_out_sample[q][algo] += rewards_out_exp
            losses_rqr[q] += losses_rqr_exp
        end
    end
end

for q in quantiles
    payoffs[q][algo] = payoffs[q][algo] ./ n_experiments
    weights[q][algo] = weights[q][algo] ./ n_experiments
    rewards[q][algo] = rewards[q][algo] ./ n_experiments
    rewards_in_sample[q][algo] = rewards_in_sample[q][algo] ./ n_experiments
    rewards_out_sample[q][algo] = rewards_out_sample[q][algo] ./ n_experiments
    losses_rqr[q] = losses_rqr[q] ./ n_experiments
end

#################### Calculate Errors ####################
global_loss_qr = zeros(T-1)
global_loss_rqr = zeros(T-1)
global_loss_models = Dict(name => zeros(T-1) for name in model_names)

for q in quantiles

    global_loss_qr .+= losses_qr[q][2:T]
    global_loss_rqr .+= losses_rqr[q][2:T]

    individual_losses = Dict(name => Float64[] for name in model_names)
    for t in 2:T
        y_true = realizations[q][t]

        for name in model_names
            y_pred = algo_forecasts[q][name][t]
            loss_t = mean(QuantileLoss(q).(y_pred .- y_true))

            global_loss_models[name][t-1] += loss_t
            push!(individual_losses[name], loss_t)
        end
    end
    println("\n############ RESULTS QUANTILE $q ############")
    println("Loss Aggregated QR : $(mean(losses_qr[q][2:T]))")
    println("Loss Aggregated RQR: $(mean(losses_rqr[q][2:T]))")
    
    for name in model_names
        avg_loss = mean(individual_losses[name])
        println("Loss $(uppercase(name)): $avg_loss")
    end
end

#################### PLOTTING CUMULATIVE AVERAGE LOSS ####################
n_q = length(quantiles)
avg_ts_qr = global_loss_qr ./ n_q
avg_ts_rqr = global_loss_rqr ./ n_q
for name in model_names
    global_loss_models[name] ./= n_q
end

cum_avg(v) = cumsum(v) ./ (1:length(v))
moving_avg(v, window) = runmean(v, window)

p1 = plot(
    title = "Cumulative Average Loss (Avg over all Quantiles)",
    xlabel = "",
    ylabel = "Cumulative Mean Loss",
    legend = :bottomright,
    lw = 2
)

# Plot Algorithms - Cumulative
plot!(p1, cum_avg(avg_ts_qr), label="Aggregated QR", color=:blue, lw=2.5)
plot!(p1, cum_avg(avg_ts_rqr), label="Aggregated RQR", color=:red, linestyle=:dash, lw=2.5)

# Plot Individual Models - Cumulative (Dynamic)
colors = [:green, :orange, :purple, :cyan] # Add more if you have many models
for (i, name) in enumerate(model_names)
    c = colors[mod1(i, length(colors))]
    plot!(p1, cum_avg(global_loss_models[name]), label="Model $(uppercase(name))", color=c, alpha=0.7)
end

p2 = plot(
    title = "Moving Average Loss (Avg over all Quantiles)",
    xlabel = "Time Step",
    ylabel = "Moving Average Loss",
    legend = :bottomright,
    lw = 2
)

# Plot Algorithms - Moving Average
window = 5
plot!(p2, moving_avg(avg_ts_qr, window), label="Aggregated QR", color=:blue, lw=2.5)
plot!(p2, moving_avg(avg_ts_rqr, window), label="Aggregated RQR", color=:red, linestyle=:dash, lw=2.5)

# Plot Individual Models - Moving Average (Dynamic)
for (i, name) in enumerate(model_names)
    c = colors[mod1(i, length(colors))]
    plot!(p2, moving_avg(global_loss_models[name], window), label="Model $(uppercase(name))", color=c, alpha=0.7)
end

# Combine plots with shared x-axis
p = plot(p1, p2, layout=(2, 1), size=(1200, 900), plot_title="Loss Analysis", margin=5mm)

# Display the plot
display(p)

#################### PLOTTING WEIGHTS BY QUANTILE ####################
weight_plots = []
for q in quantiles
    p_weights = plot(
        title = "Weights for Quantile $q",
        xlabel = "Time Step",
        ylabel = "Weight",
        legend = :bottomright,
        lw = 2
    )
    for (i, name) in enumerate(model_names)
        plot!(p_weights, weights[q]["QR"][i, :], label="$(uppercase(name))", lw=2)
    end
    
    push!(weight_plots, p_weights)
end

p_weights_combined = plot(weight_plots..., layout=(length(quantiles), 1), size=(1200, 400*length(quantiles)), plot_title="Weights Analysis by Quantile", margin=5mm)
display(p_weights_combined)

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

#################### Print Rewards (Single Quantiles) ####################
println("\n############ REWARDS SINGLE QUANTILES ############")
for q in quantiles
    println("\n--- Quantile $q ---")
    for algo in algorithms
        println("Algorithm: $algo")
        for (i, name) in enumerate(model_names)
            # Total Reward
            total_r = sum(rewards[q][algo][i, :])
            println("  Total Reward $(uppercase(name)): $total_r")
            
            # In-Sample Reward
            in_r = sum(rewards_in_sample[q][algo][i, :])
            println("    -> In-sample: $in_r")
            
            # Out-of-Sample Reward
            out_r = sum(rewards_out_sample[q][algo][i, :])
            println("    -> Out-sample: $out_r")
        end
    end
end

#################### Print Rewards (Total Aggregated) ####################
println("\n############ TOTAL REWARDS (All Quantiles Summed) ############")
for algo in algorithms
    println("\nAlgorithm: $algo")
    for (i, name) in enumerate(model_names)
        # Total
        total_r = sum(total_rewards_forecasters[algo][i, :])
        println("  Total Reward $(uppercase(name)): $total_r")
        
        # In-Sample
        in_r = sum(total_in_rewards_forecasters[algo][i, :])
        println("    -> In-sample: $in_r")
        
        # Out-of-Sample
        out_r = sum(total_out_rewards_forecasters[algo][i, :])
        println("    -> Out-sample: $out_r")
    end
end

