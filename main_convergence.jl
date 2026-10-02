using LinearAlgebra
using Plots
using DataStructures
using ProgressBars
using Base.Threads
using Normalization
using Plots.PlotMeasures
using Random

include("functions/functions.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/quantile_regression.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
using .UtilsFunctions
using .DataGeneration
using .QuantileRegression
using .AdaptiveRobustRegression


# Environment Settings
n_experiments = 100
T = 20000
lead_time = 1
quantiles = [0.1, 0.5, 0.9]
n_forecasters = 3
algorithms = ["QR", "RQR"]
environment = "invariant"
lr = 0.1
batch_percentage = 0.2
missing_rate = 0.05         # RQR missing-submission rate
seed = 1234

# Data bounds for normalization (same fixed box as main_rewards.jl). A single MinMax scaler
# maps all data to [0,1] so convergence is checked on the same normalized inputs the reward
# pipeline uses.
lower_bound = -3.0
upper_bound = 5.0
scaler = MinMax([Float64(lower_bound), Float64(upper_bound)])

generators = Dict("invariant" => generate_time_invariant_data_multiple_lead_times,
                  "abrupt" => generate_abrupt_data_multiple_lead_times,
                  "variant" => generate_dynamic_data_sin_multiple_lead_times)
haskey(generators, environment) || error("The defined environment is not yet implemented")

# Environment Variables
# Weights averaged over the Monte-Carlo runs, keyed by (quantile, algorithm) => (n_forecasters, T)
exp_weights = Dict([(q, algo) => zeros((n_forecasters, T)) for q in quantiles for algo in algorithms])
true_weights = Dict()

data_lock = ReentrantLock()
Threads.@threads for i in ProgressBar(1:n_experiments)

    # One data draw per experiment, shared by all quantiles and algorithms (own seed: independent of threading)
    rng = Xoshiro(seed + i)
    realizations, forecasts, w = generators[environment](T, lead_time, quantiles; rng)

    # Normalize to [0,1] with the shared MinMax scaler (same box as main_rewards.jl).
    # Clamp first to keep everything in range.
    realizations = [scaler(clamp.(v, lower_bound, upper_bound)) for v in realizations]
    forecaster_names = sort(collect(keys(forecasts[quantiles[1]])))

    # Missing submissions (RQR), drawn once per experiment: a missing seller misses every quantile
    alpha = Int.(rand(rng, n_forecasters, T) .< missing_rate)
    for t in 1:T
        if sum(alpha[:, t]) == n_forecasters
            alpha[rand(rng, 1:n_forecasters), t] = 0
        end
    end

    for q in quantiles
        forecasters_preds = Dict(f => [scaler(clamp.(v, lower_bound, upper_bound)) for v in forecasts[q][f]] for f in forecaster_names)

        # Initialization
        weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
        for algo in algorithms
            weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
        end
        D_exp = zeros(n_forecasters, n_forecasters)

        # Learning process
        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecaster_names]
            y_true = realizations[t]

            for algo in algorithms
                if algo == "RQR"
                    # Forecast combination (session closes), then update once y_true is observed
                    aggregated_forecast_t = rqr_aggregate(forecasters_preds_t, weights_history[algo][:, t-1], D_exp, alpha[:, t])
                    weights_history[algo][:, t], D_exp = rqr_update(forecasters_preds_t, y_true, weights_history[algo][:, t-1], D_exp, alpha[:, t], aggregated_forecast_t, q, lr, batch_percentage)
                elseif algo == "QR"
                    aggregated_forecast_t = qr_aggregate(forecasters_preds_t, weights_history[algo][:, t-1])
                    weights_history[algo][:, t] = qr_update(forecasters_preds_t, weights_history[algo][:, t-1], y_true, aggregated_forecast_t, q, lr, batch_percentage)
                end
            end
        end

        lock(data_lock) do
            for algo in algorithms
                exp_weights[(q, algo)] .+= weights_history[algo]
            end
            if i == 1
                true_weights[q] = w   # same true weights for every quantile
            end
        end
    end
end

#################### Monte-Carlo average ####################
for key in keys(exp_weights)
    exp_weights[key] ./= n_experiments
end

# Plot weights for all quantiles and algorithms
my_tick_formatter(vals) = ["$(Int(round(val/1000)))" for val in vals]
x = 1:5000:T

# Shared y-limits for every weight panel (all quantiles, algorithms and per-quantile figures)
ylims_w = padded_ylims(
    [exp_weights[(q, algo)] for q in quantiles for algo in algorithms]...,
    [true_weights[q] for q in quantiles]...
)

plot_weigths = plot(layout=(length(quantiles), length(algorithms)), size=(2200, 1700))

for (i, q) in enumerate(quantiles)
    for (j, algo) in enumerate(algorithms)
        plot!(plot_weigths[i, j], 1:T, exp_weights[(q, algo)]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"],
        ylims=ylims_w,
        ylabel = (i == 2) ? "Weights [p.u.]" : "",
        legend=:topright,
        legendfont=:20,
        fg_legend=:transparent,
        bg_legend=:transparent,
        ylabelfontsize=20,
        xlabelfontsize=20,
        bottom_margin=15mm,
        left_margin=15mm,
        tickfontsize=16,
        lw=2,
        framestyle=:arrows,
        xticks = (i == length(quantiles)) ? (x, my_tick_formatter(x)) : (x, fill("", length(x)))
        )
        plot!(plot_weigths[i, j], 1:T, true_weights[q]', label=["" "" ""])
    end
end
plot!(
    plot_weigths[3, 1],
    xlabel="Session # [x10\u00b3]",
    xlabelfontsize=14
)
    plot!(
    plot_weigths[3, 2],
    xlabel="Session # [x10\u00b3]",
    xlabelfontsize=14
)
display(plot_weigths)
savefig(plot_weigths, "plots/convergence/plot_weight_$(environment)_$(lead_time)lt_all_q.pdf")

# Plot weight for each quantile and algorithm
for q in quantiles
    # Thin left strip (subplot 1) acts as a shared y-label centered across both rows
    l = @layout [a{0.03w} [b; c]]
    plot_weights_q = plot(layout=l, size=[830, 500])

    plot!(plot_weights_q[1],
        framestyle=:none, ticks=nothing, grid=false,
        xlims=(0, 1), ylims=(0, 1),
        annotations=[(0.5, 0.5, text("Weights [p.u.]", 14, :center, rotation=90))],
        left_margin=0mm, right_margin=0mm,
        bottom_margin=0mm, top_margin=0mm
    )

    for (j, algo) in enumerate(algorithms)
        plot!(plot_weights_q[j+1], 1:T, exp_weights[(q, algo)]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"],
        ylims=ylims_w,
        ylabel="",
        legend=false,
        legendfont=:12,
        fg_legend=:transparent,
        bg_legend=:transparent,
        ylabelfontsize=14,
        bottom_margin=5mm,
        left_margin=5mm,
        tickfontsize=12,
        lw=2,
        framestyle=:arrows,
        xticks = (j == length(algorithms)) ? (x, my_tick_formatter(x)) : (x, fill("", length(x)))
        )
        plot!(plot_weights_q[j+1], 1:T, true_weights[q]', label=["" "" ""])
    end
    plot!(plot_weights_q[2],
      legend=(0.2, 1.15),
      legendfont=12,
      legendcolumns=3,
      fg_legend=:transparent,
      bg_legend=:transparent,
      top_margin=10mm)
    plot!(
        plot_weights_q[3],
        xlabel="Session # [x10\u00b3]",
        xlabelfontsize=14
    )
    display(plot_weights_q)
    savefig(plot_weights_q, "plots/convergence/plot_weight_$(environment)_$(lead_time)lt_q$(Int(q*100)).pdf")
end