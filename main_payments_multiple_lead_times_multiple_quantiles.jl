using LinearAlgebra
using Plots
using Statistics
using DataStructures
using Base.Threads
using LossFunctions
using ProgressBars
using Plots.PlotMeasures

include("functions/functions.jl")
include("functions/functions_payoff.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/quantile_regression.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
include("payoff/leave_one_out.jl")
include("payoff/new_shapley.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .DataGeneration
using .QuantileRegression
using .LOO
using .Shapley
using .AdaptiveRobustRegression

# Environment Settings
quantiles = [0.1, 0.5, 0.9]
n_forecasters = 3
algorithms = ["QR", "RQR"]
payoff_functions = "Shapley"
total_reward = 100
T = 20000
n_experiments = 200
lead_time = 1
delta = 0.7
environment = "variant"

# Environment Variables
payoffs = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
weights = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_in_sample = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])
rewards_out_sample = Dict([q => Dict([algo => zeros(n_forecasters, T) for algo in algorithms]) for q in quantiles])

# Environment generation
## Weights initialization
for algo in algorithms
    for q in quantiles
        weights[q][algo][:, 1] .= initialize_weights(n_forecasters)
    end
end

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
        if environment == "invariant"
            realizations, forecasters_preds, w = generate_time_invariant_data_multiple_lead_times(T, lead_time, q)
        elseif  environment == "abrupt"
            realizations, forecasters_preds, w = generate_abrupt_data_multiple_lead_times(T, lead_time, q)
        elseif environment == "variant"
            realizations, forecasters_preds, w = generate_dynamic_data_sin_multiple_lead_times(T, lead_time, q)
        else
            ErrorException("The defined environment is not yet implemented")
        end

        sorted_f = sort(collect(forecasters_preds), by=first)
        sorted_forecasters = OrderedDict(sorted_f)

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
            y_true = realizations[t]

            weights_exp[:, t], _ = online_quantile_regression_update_multiple_lead_times(forecasters_preds_t, weights_exp[:, t-1], y_true, q, 0.01)
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
#missing_forecast = 2

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
        if environment == "invariant"
            realizations, forecasters_preds, w = generate_time_invariant_data_multiple_lead_times(T, lead_time, q)
        elseif  environment == "abrupt"
            realizations, forecasters_preds, w = generate_abrupt_data_multiple_lead_times(T, lead_time, q)
        elseif environment == "variant"
            realizations, forecasters_preds, w = generate_dynamic_data_sin_multiple_lead_times(T, lead_time, q)
        else
            ErrorException("The defined environment is not yet implemented")
        end

        sorted_f = sort(collect(forecasters_preds), by=first)
        sorted_forecasters = OrderedDict(sorted_f)

        D_exp = zeros(n_forecasters, n_forecasters)
        #= alpha = zeros(n_forecasters, T)
        alpha[missing_forecast, :] = Int.(rand(T) .< 0.05) =#
        alpha = Int.(rand(n_forecasters, T) .< 0.05)
        for t in 1:T
            if sum(alpha[:, t]) == length(alpha[:, t])
                idx = rand(1:length(alpha[:, t]))
                alpha[idx, t] = 0
            end
        end

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
            y_true = realizations[t]

            # Learning Phase
            weights_exp[:, t], new_D, _ = online_adaptive_robust_quantile_regression_multiple_lead_times(forecasters_preds_t, y_true, weights_exp[:, t-1], D_exp, alpha[:, t], q)
            prev_D = D_exp
            D_exp = new_D

            # Payoff Calculation
            temp_forecasts_t = [forecasters_preds_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
            temp_weights_t = weights_exp[:, t-1] .+ prev_D * alpha[:, t]
            temp_weights_t = [temp_weights_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]

            temp_payoffs = nothing
            forecasters_losses = nothing
            if length(temp_forecasts_t) > 0
                temp_payoffs = shapley_payoff_multiple_lead_times(temp_forecasts_t, temp_weights_t, y_true, q)
                forecasters_losses = [mean(QuantileLoss(q).(temp_forecasts_t[i], y_true)) for i in 1:length(temp_forecasts_t)]
                #temp_scores = 1 .- (forecasters_losses ./ sum(forecasters_losses))
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

# Plot total rewards for each algorithm
#= plot_rewards = plot(layout=(length(algorithms), 1), size=(800, 500))
for (i, algo) in enumerate(algorithms)
    plot!(plot_rewards[i], 1:T, total_rewards_forecasters[algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
    xlabel="Time", 
    ylabel="Total reward (£)",
    legend=false,
    fg_legend=:transparent,
    bg_legend=:transparent,
    ylabelfontsize=14,
    xlabelfontsize=14,
    bottom_margin=2mm,
    left_margin=5mm,
    tickfontsize=12,
    lw=2
    )
end
plot!(plot_rewards, 
      subplot=1,
      legend=(0.2, 1.15),
      legendfont=12,
      legendcolumns=3,
      fg_legend=:transparent,
      bg_legend=:transparent,
      top_margin=10mm)
display(plot_rewards)
savefig(plot_rewards, "plots/rewards/plot_total_rewards_$(lead_time)lt_$environment.pdf")

# Plot total in-sample and out-of-sample rewards
subplot_letters = [('a' + i - 1) for i in 1:length(algorithms)]
plot_in_out_rewards = plot(layout=(2, length(algorithms)), size=(1000, 700))
for (i, algo) in enumerate(algorithms)
    plot!(plot_in_out_rewards[1, i], 1:T, total_in_rewards_forecasters[algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
    xlabel="Time", 
    ylabel="in-sample reward (£)",
    legend=false,
    fg_legend=:transparent,
    bg_legend=:transparent,
    ylabelfontsize=14,
    xlabelfontsize=14,
    bottom_margin=5mm,
    left_margin=5mm,
    tickfontsize=10,
    lw=2,
    rotation=15
    )

    plot!(plot_in_out_rewards[2, i], 1:T, total_out_rewards_forecasters[algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
    xlabel="Time", 
    ylabel="out-of-sample reward (£)",
    legend=false,
    fg_legend=:transparent,
    bg_legend=:transparent,
    ylabelfontsize=14,
    xlabelfontsize=14,
    bottom_margin=5mm,
    left_margin=5mm,
    tickfontsize=10,
    lw=2,
    rotation=15
    )
end
display(plot_in_out_rewards)
savefig(plot_in_out_rewards, "plots/rewards/plot_total_rewards_in_out_$(lead_time)lt_$environment.pdf") =#

# Plot total reward for each quantile
#= plot_reward_quantiles = plot(layout=(length(quantiles), length(algorithms)), size=(1000, 800))
for (i, algo) in enumerate(algorithms)
    for (j, q) in enumerate(quantiles)
        
        plot!(plot_reward_quantiles[j, i], 1:T, rewards[q][algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
        xlabel="Time", 
        ylabel="Total reward (£)",
        legend=false,
        fg_legend=:transparent,
        bg_legend=:transparent,
        ylabelfontsize=14,
        xlabelfontsize=14,
        bottom_margin=5mm,
        left_margin=5mm,
        tickfontsize=10,
        lw=2,
        rotation=15
        )
    end
end
plot!(plot_reward_quantiles[1, 1],
      legend=(0.6, 1.15),
      legendfont=12,
      legendcolumns=3,
      fg_legend=:transparent,
      bg_legend=:transparent,
      top_margin=10mm)
display(plot_reward_quantiles)
savefig(plot_reward_quantiles, "plots/rewards/plot_total_rewards_quantiles_$(lead_time)lt_$environment.pdf") =#

# Plot instantaneous vs cumulative total reward
plot_insta_cum_reward = plot(layout=(2, length(algorithms)), size=(1000, 800)) 
for (i, algo) in enumerate(algorithms)

    plot!(plot_insta_cum_reward[1, i], 1:T, total_rewards_forecasters[algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
        xlabel="Time", 
        ylabel="Instantaneous reward (£)",
        legend=false,
        fg_legend=:transparent,
        bg_legend=:transparent,
        ylabelfontsize=14,
        xlabelfontsize=14,
        bottom_margin=5mm,
        left_margin=5mm,
        tickfontsize=10,
        lw=2,
        rotation=15,
        formatter=:plain
    )
    
    plot!(plot_insta_cum_reward[2, i], 1:T, cumsum(total_rewards_forecasters[algo], dims=2)', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
        xlabel="Time", 
        ylabel="Cumulative reward (£)",
        legend=false,
        fg_legend=:transparent,
        bg_legend=:transparent,
        ylabelfontsize=14,
        xlabelfontsize=14,
        bottom_margin=5mm,
        left_margin=5mm,
        tickfontsize=10,
        lw=2,
        rotation=15,
        formatter=:plain
    )
end
plot!(plot_insta_cum_reward[1, 1],
      legend=(0.6, 1.1),
      legendfont=12,
      legendcolumns=3,
      fg_legend=:transparent,
      bg_legend=:transparent,
      top_margin=10mm)
display(plot_insta_cum_reward)
savefig(plot_insta_cum_reward, "plots/rewards/plot_inst_cum_total_rewards_$(lead_time)lt_$environment.pdf")