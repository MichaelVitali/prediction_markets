using LinearAlgebra
using Plots
using Statistics
using DataStructures
using Base.Threads
using LossFunctions
using Normalization
using ProgressBars
using Plots.PlotMeasures
using Random

include("functions/functions.jl")
include("functions/functions_payoff.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/quantile_regression.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
include("payoff/shapley_values.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .DataGeneration
using .QuantileRegression
using .Shapley
using .AdaptiveRobustRegression

# Environment Settings
quantiles = [0.1, 0.5, 0.9]
n_forecasters = 3
algorithms = ["QR", "RQR"]
total_reward = 100
T = 10000
n_experiments = 1
lead_time = 24
delta = 0.7
environment = "invariant"
lr = 0.1   # shared learning rate for all algorithms (QR and RQR)
batch_percentage = 0.2
missing_rate = 0.05                        # RQR missing-submission rate
seed = 1234
payoff_halflife = 60                       # EMA half-life for payoff smoothing, in steps (days)
lambda_payoff = 0.5^(1 / payoff_halflife)  # -> old payoffs lose half their weight every 60 steps

# Data bounds for normalization. Synthetic data is Gaussian (unbounded), so we impose a
# fixed a-priori box covering the generating distributions (forecaster means ~0..2, sd ~1).
# A single MinMax scaler (as in the real-world data_preprop.jl) maps all data to [0,1],
# which bounds the pinball loss by max(q, 1-q). Same box as main_convergence.jl.
lower_bound = -3.0
upper_bound = 5.0
scaler = MinMax([Float64(lower_bound), Float64(upper_bound)])

Random.seed!(seed)

# Environment Variables
# Results averaged over the Monte-Carlo runs, keyed by (quantile, algorithm) => (n_forecasters, T)
payoffs = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
weights = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards_in_sample = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards_out_sample = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])

#################### Quantile Regression ####################
algo = "QR"
data_lock = ReentrantLock()

for q in quantiles
    quantile_step_reward = total_reward / length(quantiles)
    loss_bound = max(q, 1 - q)   # max pinball loss on the [0,1]-normalized scale

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
            error("The defined environment is not yet implemented")
        end

        # Normalize to [0,1] with the shared MinMax scaler so the pinball loss is bounded
        # by max(q, 1-q). Clamp first to guarantee scores stay in [0,1].
        for tt in eachindex(realizations)
            realizations[tt] = scaler(clamp.(realizations[tt], lower_bound, upper_bound))
        end
        for f in keys(forecasters_preds)
            forecasters_preds[f] = [scaler(clamp.(v, lower_bound, upper_bound)) for v in forecasters_preds[f]]
        end

        forecaster_names = sort(collect(keys(forecasters_preds)))

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecaster_names]
            y_true = realizations[t]

            # Forecast combination (session closes), then update once y_true is observed
            aggregated_forecast_t = qr_aggregate(forecasters_preds_t, weights_exp[:, t-1])
            weights_exp[:, t] = qr_update(forecasters_preds_t, weights_exp[:, t-1], y_true, aggregated_forecast_t, q, lr, batch_percentage)

            # Payoff calculation
            temp_payoffs = shapley_payoff_multiple_lead_times_refit(forecasters_preds_t, weights_exp[:, t-1], y_true, q)
            forecasters_losses = [mean(QuantileLoss(q).(forecasters_preds_t[i] .- y_true)) for i in 1:n_forecasters]
            temp_scores = 1 .- (forecasters_losses ./ loss_bound)
            payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, lambda_payoff)

            # In-sample: delta share of the budget; out-of-sample: leave-one-out wagering on the rest
            all_active = trues(n_forecasters)
            rewards_in = in_sample_rewards(payoffs_exp[:, t], all_active, delta * quantile_step_reward)
            rewards_out = out_of_sample_rewards(temp_scores, all_active, (1 - delta) * quantile_step_reward)

            rewards_in_exp[:, t] = rewards_in
            rewards_out_exp[:, t] = rewards_out
            rewards_exp[:, t] = rewards_in .+ rewards_out
        end

        lock(data_lock) do
            payoffs[(q, algo)] .+= payoffs_exp
            weights[(q, algo)] .+= weights_exp
            rewards[(q, algo)] .+= rewards_exp
            rewards_in_sample[(q, algo)] .+= rewards_in_exp
            rewards_out_sample[(q, algo)] .+= rewards_out_exp
        end
    end
end

for q in quantiles, results in (payoffs, weights, rewards, rewards_in_sample, rewards_out_sample)
    results[(q, algo)] ./= n_experiments
end

#################### Robust Quantile Regression ####################
algo = "RQR"
acc_lock_rqr = ReentrantLock()

for q in quantiles

    quantile_step_reward = total_reward / length(quantiles)
    loss_bound = max(q, 1 - q)   # max pinball loss on the [0,1]-normalized scale

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
            error("The defined environment is not yet implemented")
        end

        # Normalize to [0,1] with the shared MinMax scaler so the pinball loss is bounded
        # by max(q, 1-q). Clamp first to guarantee scores stay in [0,1].
        for tt in eachindex(realizations)
            realizations[tt] = scaler(clamp.(realizations[tt], lower_bound, upper_bound))
        end
        for f in keys(forecasters_preds)
            forecasters_preds[f] = [scaler(clamp.(v, lower_bound, upper_bound)) for v in forecasters_preds[f]]
        end

        forecaster_names = sort(collect(keys(forecasters_preds)))

        D_exp = zeros(n_forecasters, n_forecasters)
        alpha = Int.(rand(n_forecasters, T) .< missing_rate)
        for t in 1:T
            if sum(alpha[:, t]) == length(alpha[:, t])
                idx = rand(1:length(alpha[:, t]))
                alpha[idx, t] = 0
            end
        end

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecaster_names]
            y_true = realizations[t]

            # Forecast combination (session closes), then update once y_true is observed
            aggregated_forecast_t = rqr_aggregate(forecasters_preds_t, weights_exp[:, t-1], D_exp, alpha[:, t])
            weights_exp[:, t], new_D = rqr_update(forecasters_preds_t, y_true, weights_exp[:, t-1], D_exp, alpha[:, t], aggregated_forecast_t, q, lr, batch_percentage)
            prev_D = D_exp
            D_exp = new_D

            # Payoff Calculation
            temp_forecasts_t = [forecasters_preds_t[j] for j in 1:n_forecasters if alpha[j, t] == 0]
            temp_weights_t = weights_exp[:, t-1] .+ prev_D * alpha[:, t]
            temp_weights_t = [temp_weights_t[j] for j in 1:n_forecasters if alpha[j, t] == 0]
            temp_weights_t = project_to_simplex(temp_weights_t)

            # At least one forecaster is always available (enforced when drawing alpha)
            temp_payoffs = shapley_payoff_multiple_lead_times_refit(temp_forecasts_t, temp_weights_t, y_true, q)
            forecasters_losses = [mean(QuantileLoss(q).(temp_forecasts_t[i] .- y_true)) for i in 1:length(temp_forecasts_t)]
            temp_scores = 1 .- (forecasters_losses ./ loss_bound)

            if length(temp_payoffs) < n_forecasters
                for j in findall(a -> a == 1, alpha[:, t])
                    insert!(temp_payoffs, j, 0.0)
                    insert!(temp_scores, j, 0.0)
                end
            end
            payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, lambda_payoff)

            # Reward calculation: only active sellers share the budget; missing ones get zero (zero-element property)
            active = alpha[:, t] .== 0
            rewards_in = in_sample_rewards(payoffs_exp[:, t], active, delta * quantile_step_reward)
            rewards_out = out_of_sample_rewards(temp_scores, active, (1 - delta) * quantile_step_reward)

            rewards_in_exp[:, t] = rewards_in
            rewards_out_exp[:, t] = rewards_out
            rewards_exp[:, t] = rewards_in .+ rewards_out
        end

        lock(acc_lock_rqr) do
            payoffs[(q, algo)] .+= payoffs_exp
            weights[(q, algo)] .+= weights_exp
            rewards[(q, algo)] .+= rewards_exp
            rewards_in_sample[(q, algo)] .+= rewards_in_exp
            rewards_out_sample[(q, algo)] .+= rewards_out_exp
        end
    end
end

for q in quantiles, results in (payoffs, weights, rewards, rewards_in_sample, rewards_out_sample)
    results[(q, algo)] ./= n_experiments
end

#################### Plot Results ####################

total_rewards_forecasters = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_in_rewards_forecasters = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_out_rewards_forecasters = Dict([algo => zeros(n_forecasters, T) for algo in algorithms])
for algo in algorithms
    for i in 1:n_forecasters
        for q in quantiles
            total_rewards_forecasters[algo][i, :] .+= rewards[(q, algo)][i, :]
            total_in_rewards_forecasters[algo][i, :] .+= rewards_in_sample[(q, algo)][i, :]
            total_out_rewards_forecasters[algo][i, :] .+= rewards_out_sample[(q, algo)][i, :]
        end
    end
end

my_tick_formatter(vals) = ["$(Int(round(val/1000)))" for val in vals]
x = 1:5000:T

# Shared, data-driven y-limits so panels of each figure use identical y-ticks.
# Instantaneous and cumulative rewards live on different scales, so they keep separate ranges.
ylims_total = padded_ylims(values(total_rewards_forecasters)...)
ylims_in    = padded_ylims(values(total_in_rewards_forecasters)...)
ylims_out   = padded_ylims(values(total_out_rewards_forecasters)...)
ylims_rq    = padded_ylims([rewards[(q, algo)] for q in quantiles for algo in algorithms]...)
ylims_cum   = padded_ylims([cumsum(total_rewards_forecasters[algo], dims=2) for algo in algorithms]...)

# Plot total rewards for each algorithm
l = @layout [a{0.03w} [b; c]]
plot_rewards = plot(layout=l, size=(800, 500))

plot!(plot_rewards[1],
    framestyle=:none, ticks=nothing, grid=false,
    xlims=(0, 1), ylims=(0, 1),
    annotations=[(0.5, 0.5, text("Total reward [£]", 14, :center, rotation=90))],
    left_margin=0mm, right_margin=0mm,
    bottom_margin=0mm, top_margin=0mm
)

for (i, algo) in enumerate(algorithms)
    plot!(plot_rewards[i+1], 1:T, total_rewards_forecasters[algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
    ylabel="",
    ylims=ylims_total,
    legend=false,
    fg_legend=:transparent,
    bg_legend=:transparent,
    ylabelfontsize=14,
    xlabelfontsize=14,
    bottom_margin=2mm,
    left_margin=5mm,
    tickfontsize=12,
    xticks=(i == length(algorithms)) ? (x, my_tick_formatter(x)) : (x, fill("", length(x))),
    lw=2
    )
end
plot!(plot_rewards[2], 
      legend=(0.2, 1.15),
      legendfont=12,
      legendcolumns=3,
      fg_legend=:transparent,
      bg_legend=:transparent,
      top_margin=10mm)
plot!(
    plot_rewards[3],
    xlabel="Session # [x10\u00b3]",
    xlabelfontsize=14
)
display(plot_rewards)
savefig(plot_rewards, "plots/rewards/plot_total_rewards_$(lead_time)lt_$environment.pdf")

# Plot total in-sample and out-of-sample rewards
subplot_letters = [('a' + i - 1) for i in 1:length(algorithms)]
plot_in_out_rewards = plot(layout=(2, length(algorithms)), size=(1000, 700))
for (i, algo) in enumerate(algorithms)
    plot!(plot_in_out_rewards[1, i], 1:T, total_in_rewards_forecasters[algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
    xlabel="Session #", 
    ylims=ylims_in,
    ylabel="in-sample reward [£]",
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
    xlabel="Session #", 
    ylims=ylims_out,
    ylabel="out-of-sample reward [£]",
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
savefig(plot_in_out_rewards, "plots/rewards/plot_total_rewards_in_out_$(lead_time)lt_$environment.pdf")

# Plot total reward for each quantile
plot_reward_quantiles = plot(layout=(length(quantiles), length(algorithms)), size=(1000, 800))
for (i, algo) in enumerate(algorithms)
    for (j, q) in enumerate(quantiles)
        
        plot!(plot_reward_quantiles[j, i], 1:T, rewards[(q, algo)]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
        xlabel="Session #", 
        ylims=ylims_rq,
        ylabel="Total reward [£]",
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
savefig(plot_reward_quantiles, "plots/rewards/plot_total_rewards_quantiles_$(lead_time)lt_$environment.pdf")

# Plot instantaneous vs cumulative total reward
plot_insta_cum_reward = plot(layout=(2, length(algorithms)), size=(1000, 800)) 
for (i, algo) in enumerate(algorithms)

    plot!(plot_insta_cum_reward[1, i], 1:T, total_rewards_forecasters[algo]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
        xlabel="Session #", 
        ylims=ylims_total,
        ylabel="Instantaneous reward [£]",
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
        xlabel="Session #", 
        ylims=ylims_cum,
        ylabel="Cumulative reward [£]",
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