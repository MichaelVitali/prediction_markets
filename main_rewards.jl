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
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .DataGeneration
using .QuantileRegression
using .AdaptiveRobustRegression

# Environment Settings
quantiles = [0.1, 0.5, 0.9]
n_forecasters = 3
algorithms = ["QR", "RQR"]
total_reward = 100
T = 20000
n_experiments = 1000
lead_time = 1
delta = 0.7
environment = "invariant"
lr = 0.1   # shared learning rate for all algorithms (QR and RQR)
batch_percentage = 0.2
missing_rate = 0.05                        # RQR missing-submission rate
seed = 1234
n_lambda = 1000                              # effective window size of the long-term (in-sample) scores, in steps (days)
lambda_score = 1 - 1 / n_lambda            # forgetting factor, n_lambda = (1 - lambda)^-1
plot_start = 100                           # first session plotted/summarised (session 1 has no reward and the long-term scores start from 0)

# Data bounds for normalization. Synthetic data is Gaussian (unbounded), so we impose a
# fixed a-priori box covering the generating distributions (forecaster means ~0..2, sd ~1).
# A single MinMax scaler (as in the real-world data_preprop.jl) maps all data to [0,1],
# which bounds the pinball loss by max(q, 1-q). Same box as main_convergence.jl.
lower_bound = -3.0
upper_bound = 5.0
scaler = MinMax([Float64(lower_bound), Float64(upper_bound)])

generators = Dict("invariant" => generate_time_invariant_data_multiple_lead_times,
                  "abrupt" => generate_abrupt_data_multiple_lead_times,
                  "variant" => generate_dynamic_data_sin_multiple_lead_times)
haskey(generators, environment) || error("The defined environment is not yet implemented")

# One data draw per experiment, shared by all quantiles and by the QR and RQR sections: every experiment has
# its own seeded rng, so both sections regenerate exactly the same data (and results don't depend on threading).
function generate_experiment(exp)
    rng = Xoshiro(seed + exp)
    realizations, forecasts, _ = generators[environment](T, lead_time, quantiles; rng)

    # Normalize to [0,1] with the shared MinMax scaler so the pinball loss is bounded
    # by max(q, 1-q). Clamp first to guarantee scores stay in [0,1].
    to_unit(v) = scaler(clamp.(v, lower_bound, upper_bound))
    realizations = to_unit.(realizations)
    forecasts = Dict(q => Dict(f => to_unit.(preds) for (f, preds) in forecasts[q]) for q in quantiles)

    # Missing submissions (used by RQR): a missing seller misses every quantile; at least one is always available
    alpha = Int.(rand(rng, n_forecasters, T) .< missing_rate)
    for t in 1:T
        if sum(alpha[:, t]) == n_forecasters
            alpha[rand(rng, 1:n_forecasters), t] = 0
        end
    end
    return realizations, forecasts, alpha
end

# Environment Variables
# Results averaged over the Monte-Carlo runs, keyed by (quantile, algorithm) => (n_forecasters, T)
long_term_scores = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
weights = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards_in_sample = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards_out_sample = Dict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
# Per-run mean total reward per session (summed over quantiles, sessions plot_start:T), keyed by algorithm => (n_forecasters, n_experiments)
run_mean_rewards = Dict([algo => zeros(n_forecasters, n_experiments) for algo in algorithms])

#################### Quantile Regression ####################
algo = "QR"
data_lock = ReentrantLock()

Threads.@threads for exp in ProgressBar(1:n_experiments)
    realizations, forecasts, _ = generate_experiment(exp)

    for q in quantiles
        quantile_step_reward = total_reward / length(quantiles)
        loss_bound = max(q, 1 - q)   # max pinball loss on the [0,1]-normalized scale
        forecasters_preds = forecasts[q]
        forecaster_names = sort(collect(keys(forecasters_preds)))

        # Experiment Variables
        weights_exp = zeros(n_forecasters, T)
        long_term_scores_exp = zeros(n_forecasters, T)   # S_0 = 0: new sellers are not trusted a priori
        rewards_exp = zeros(n_forecasters, T)
        rewards_in_exp = zeros(n_forecasters, T)
        rewards_out_exp = zeros(n_forecasters, T)

        weights_exp[:, 1] = initialize_weights(n_forecasters)

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecaster_names]
            y_true = realizations[t]

            # Forecast combination (session closes), then update once y_true is observed
            aggregated_forecast_t = qr_aggregate(forecasters_preds_t, weights_exp[:, t-1])
            weights_exp[:, t] = qr_update(forecasters_preds_t, weights_exp[:, t-1], y_true, aggregated_forecast_t, q, lr, batch_percentage)

            # Normalized quantile scores in [0,1] and their long-term EMA
            forecasters_losses = [mean(QuantileLoss(q).(forecasters_preds_t[i] .- y_true)) for i in 1:n_forecasters]
            temp_scores = 1 .- (forecasters_losses ./ loss_bound)
            long_term_scores_exp[:, t] = payoff_update(long_term_scores_exp[:, t-1], temp_scores, lambda_score)

            # In-sample: equal-stake rule on the long-term scores (delta share of the budget); out-of-sample: on the session scores
            all_active = trues(n_forecasters)
            rewards_in = in_sample_rewards(long_term_scores_exp[:, t], all_active, delta * quantile_step_reward)
            rewards_out = out_of_sample_rewards(temp_scores, all_active, (1 - delta) * quantile_step_reward)

            rewards_in_exp[:, t] = rewards_in
            rewards_out_exp[:, t] = rewards_out
            rewards_exp[:, t] = rewards_in .+ rewards_out
        end

        lock(data_lock) do
            long_term_scores[(q, algo)] .+= long_term_scores_exp
            weights[(q, algo)] .+= weights_exp
            rewards[(q, algo)] .+= rewards_exp
            rewards_in_sample[(q, algo)] .+= rewards_in_exp
            rewards_out_sample[(q, algo)] .+= rewards_out_exp
            run_mean_rewards[algo][:, exp] .+= vec(mean(rewards_exp[:, plot_start:T], dims=2))
        end
    end
end

for q in quantiles, results in (long_term_scores, weights, rewards, rewards_in_sample, rewards_out_sample)
    results[(q, algo)] ./= n_experiments
end

#################### Robust Quantile Regression ####################
algo = "RQR"
acc_lock_rqr = ReentrantLock()

Threads.@threads for exp in ProgressBar(1:n_experiments)
    realizations, forecasts, alpha = generate_experiment(exp)

    for q in quantiles
        quantile_step_reward = total_reward / length(quantiles)
        loss_bound = max(q, 1 - q)   # max pinball loss on the [0,1]-normalized scale
        forecasters_preds = forecasts[q]
        forecaster_names = sort(collect(keys(forecasters_preds)))

        # Experiment Variables
        weights_exp = zeros(n_forecasters, T)
        long_term_scores_exp = zeros(n_forecasters, T)   # S_0 = 0: new sellers are not trusted a priori
        rewards_exp = zeros(n_forecasters, T)
        rewards_in_exp = zeros(n_forecasters, T)
        rewards_out_exp = zeros(n_forecasters, T)

        weights_exp[:, 1] = initialize_weights(n_forecasters)
        D_exp = zeros(n_forecasters, n_forecasters)

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecaster_names]
            y_true = realizations[t]

            # Forecast combination (session closes), then update once y_true is observed
            aggregated_forecast_t = rqr_aggregate(forecasters_preds_t, weights_exp[:, t-1], D_exp, alpha[:, t])
            weights_exp[:, t], new_D = rqr_update(forecasters_preds_t, y_true, weights_exp[:, t-1], D_exp, alpha[:, t], aggregated_forecast_t, q, lr, batch_percentage)
            D_exp = new_D

            # Normalized quantile scores of the available sellers (at least one, enforced when drawing alpha)
            temp_forecasts_t = [forecasters_preds_t[j] for j in 1:n_forecasters if alpha[j, t] == 0]
            forecasters_losses = [mean(QuantileLoss(q).(temp_forecasts_t[i] .- y_true)) for i in 1:length(temp_forecasts_t)]
            temp_scores = 1 .- (forecasters_losses ./ loss_bound)

            # Missing sellers score 0, so their long-term score decays while they are inactive
            for j in findall(a -> a == 1, alpha[:, t])
                insert!(temp_scores, j, 0.0)
            end
            long_term_scores_exp[:, t] = payoff_update(long_term_scores_exp[:, t-1], temp_scores, lambda_score)

            # Reward calculation: only active sellers share the budget; missing ones get zero (zero-element property)
            active = alpha[:, t] .== 0
            rewards_in = in_sample_rewards(long_term_scores_exp[:, t], active, delta * quantile_step_reward)
            rewards_out = out_of_sample_rewards(temp_scores, active, (1 - delta) * quantile_step_reward)

            rewards_in_exp[:, t] = rewards_in
            rewards_out_exp[:, t] = rewards_out
            rewards_exp[:, t] = rewards_in .+ rewards_out
        end

        lock(acc_lock_rqr) do
            long_term_scores[(q, algo)] .+= long_term_scores_exp
            weights[(q, algo)] .+= weights_exp
            rewards[(q, algo)] .+= rewards_exp
            rewards_in_sample[(q, algo)] .+= rewards_in_exp
            rewards_out_sample[(q, algo)] .+= rewards_out_exp
            run_mean_rewards[algo][:, exp] .+= vec(mean(rewards_exp[:, plot_start:T], dims=2))
        end
    end
end

for q in quantiles, results in (long_term_scores, weights, rewards, rewards_in_sample, rewards_out_sample)
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

# Plotted sessions: skip the first ones (plot_start set in the settings)
ts = plot_start:T
cum_rewards_forecasters = Dict([algo => cumsum(total_rewards_forecasters[algo], dims=2) for algo in algorithms])   # summed from session 1

# Shared, data-driven y-limits so panels of each figure use identical y-ticks.
# Instantaneous and cumulative rewards live on different scales, so they keep separate ranges.
ylims_total = padded_ylims([total_rewards_forecasters[algo][:, ts] for algo in algorithms]...)
ylims_in    = padded_ylims([total_in_rewards_forecasters[algo][:, ts] for algo in algorithms]...)
ylims_out   = padded_ylims([total_out_rewards_forecasters[algo][:, ts] for algo in algorithms]...)
ylims_rq    = padded_ylims([rewards[(q, algo)][:, ts] for q in quantiles for algo in algorithms]...)
ylims_cum   = padded_ylims([cum_rewards_forecasters[algo][:, ts] for algo in algorithms]...)

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
    plot!(plot_rewards[i+1], ts, total_rewards_forecasters[algo][:, ts]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
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

# Mean total reward per session minus the equal share U/N, split into its in-sample and out-of-sample parts.
# Bars: Monte-Carlo means over sessions ts; error bars: ±1 std of the per-run mean across the experiments.
equal_share = total_reward / n_forecasters
excess_in  = Dict([algo => vec(mean(total_in_rewards_forecasters[algo][:, ts], dims=2)) .- delta * equal_share for algo in algorithms])
excess_out = Dict([algo => vec(mean(total_out_rewards_forecasters[algo][:, ts], dims=2)) .- (1 - delta) * equal_share for algo in algorithms])
excess_std = Dict([algo => vec(std(run_mean_rewards[algo], dims=2)) for algo in algorithms])
algo_offsets = Dict("QR" => -0.2, "RQR" => 0.2)
algo_colors = Dict("QR" => "#2a78d6", "RQR" => "#eb6834")
ylims_excess = padded_ylims([0.0], [excess_in[algo] .+ excess_out[algo] .+ s .* excess_std[algo] for algo in algorithms for s in (-1, 1)]...; pad_frac=0.1)

plot_excess_rewards = plot(size=(800, 500), ylims=ylims_excess, legend=:topleft, fg_legend=:transparent, bg_legend=:transparent,
    ylabel="Excess reward per session [£]", xticks=(1:n_forecasters, ["Forecaster $i" for i in 1:n_forecasters]),
    tickfontsize=12, ylabelfontsize=14, legendfontsize=11, left_margin=5mm, grid=:y)
hline!(plot_excess_rewards, [0], color=:black, lw=1, label="")
for algo in algorithms
    xs = (1:n_forecasters) .+ algo_offsets[algo]
    total_excess = excess_in[algo] .+ excess_out[algo]
    bar!(plot_excess_rewards, xs, excess_in[algo], bar_width=0.38, color=algo_colors[algo], linecolor=:white, label="$algo in-sample")
    bar!(plot_excess_rewards, xs, total_excess, fillrange=excess_in[algo], bar_width=0.38, color=algo_colors[algo], fillalpha=0.4,
         linecolor=:white, label="$algo out-of-sample")
    scatter!(plot_excess_rewards, xs, total_excess, yerror=excess_std[algo], markersize=0, color=:black, markerstrokecolor=:black, label="")
end
display(plot_excess_rewards)
savefig(plot_excess_rewards, "plots/rewards/plot_excess_rewards_$(lead_time)lt_$environment.pdf")

# Plot total in-sample and out-of-sample rewards
subplot_letters = [('a' + i - 1) for i in 1:length(algorithms)]
plot_in_out_rewards = plot(layout=(2, length(algorithms)), size=(1000, 700))
for (i, algo) in enumerate(algorithms)
    plot!(plot_in_out_rewards[1, i], ts, total_in_rewards_forecasters[algo][:, ts]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
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

    plot!(plot_in_out_rewards[2, i], ts, total_out_rewards_forecasters[algo][:, ts]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
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
        
        plot!(plot_reward_quantiles[j, i], ts, rewards[(q, algo)][:, ts]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
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

    plot!(plot_insta_cum_reward[1, i], ts, total_rewards_forecasters[algo][:, ts]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
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
    
    plot!(plot_insta_cum_reward[2, i], ts, cum_rewards_forecasters[algo][:, ts]', labels=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
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