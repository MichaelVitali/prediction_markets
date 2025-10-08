using LinearAlgebra
using Plots
using DataStructures
using ProgressBars
using Base.Threads
using Plots.PlotMeasures

include("functions/functions.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/quantile_regression.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
using .UtilsFunctions
using .DataGeneration
using .QuantileRegression
using .AdaptiveRobustRegression


# Environment Settings
n_experiments = 200
T = 20000
lead_time = 1
quantiles = [0.1, 0.5, 0.9]
n_forecasters = 3
algorithms = ["QR", "RQR"]
environment = "invariant"

# Environment Variables
exp_weights = Dict([q => Dict([algo => zeros((n_forecasters, T)) for algo in algorithms]) for q in quantiles])
true_weights = Dict()

for q in quantiles
    Threads.@threads for i in ProgressBar(1:n_experiments)

        # Initialization
        weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
        for algo in algorithms
            weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
            exp_weights[q][algo][:, 1] .+= weights_history[algo][:, 1]
        end
        
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

        true_weights[q] = w # saving true weights for each forecaster 
        sorted_f = sort(collect(forecasters_preds), by=first)
        sorted_forecasters = OrderedDict(sorted_f)

        # Initialization RQR
        if "RQR" in algorithms
            alpha = Int.(rand(n_forecasters, T) .< 0.05)
            D_exp = zeros(n_forecasters, n_forecasters)
        end

        # Learning process
        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
            y_true = realizations[t]
            
            for algo in algorithms
                # Forecasting combination and weights update
                if algo == "RQR"
                    weights_history[algo][:, t], new_D, _ = online_adaptive_robust_quantile_regression_multiple_lead_times(forecasters_preds_t, y_true, weights_history[algo][:, t-1], D_exp, alpha[:, t], q, 0.01)
                    prev_D = D_exp
                    D_exp = new_D
                    exp_weights[q][algo][:, t] .+= weights_history[algo][:, t]
                elseif algo == "QR"
                    weights_history[algo][:, t], _ = online_quantile_regression_update_multiple_lead_times(forecasters_preds_t, weights_history[algo][:, t-1], y_true, q, 0.01)
                    exp_weights[q][algo][:, t] .+= weights_history[algo][:, t]
                end
            end
        end
    end
end

#################### Post-processing monte- ####################
for algo in algorithms
    for q in quantiles
        exp_weights[q][algo] = exp_weights[q][algo] ./ n_experiments
    end
end

# Plot weights for all quantiles and algorithms
my_tick_formatter(vals) = ["$(Int(round(val/1000)))" for val in vals]
x = 1:5000:T

plot_weigths = plot(layout=(length(quantiles), length(algorithms)), size=(2200, 1700))

for (i, q) in enumerate(quantiles)
    for (j, algo) in enumerate(algorithms)
        plot!(plot_weigths[i, j], 1:T, exp_weights[q][algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"],
        ylabel="Weights",
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
        xticks=(x, my_tick_formatter(x))
        )
        plot!(plot_weigths[i, j], 1:T, true_weights[q]', label=["" "" ""])
    end
end
plot!(
    plot_weigths[3, 1],
    xlabel="Time (10\u00b3)",
    xlabelfontsize=14
)
    plot!(
    plot_weigths[3, 2],
    xlabel="Time (10\u00b3)",
    xlabelfontsize=14
)
display(plot_weigths)
savefig(plot_weigths, "plots/convergence/plot_weight_$(environment)_1lt_all_q.pdf")

# Plot weight for each quantile and algorithm
for q in quantiles
    plot_weights_q = plot(layout=(length(algorithms), 1), size=[800, 500])
    for (j, algo) in enumerate(algorithms)
        plot!(plot_weights_q[j], 1:T, exp_weights[q][algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
        ylabel="Weights",
        legend=false,
        legendfont=:12,
        fg_legend=:transparent,
        bg_legend=:transparent,
        ylabelfontsize=14,
        bottom_margin=5mm,
        left_margin=5mm,
        tickfontsize=12,
        lw=2,
        xticks=(x, my_tick_formatter(x))
        )
        plot!(plot_weights_q[j], 1:T, true_weights[q]', label=["" "" ""])
    end
    plot!(plot_weights_q[1, 1],
      legend=(0.2, 1.15),
      legendfont=12,
      legendcolumns=3,
      fg_legend=:transparent,
      bg_legend=:transparent,
      top_margin=10mm)
    plot!(
        plot_weights_q[2, 1],
        xlabel="Time (10\u00b3)",
        xlabelfontsize=14
    )
    display(plot_weights_q)
    savefig(plot_weights_q, "plots/convergence/plot_weight_$(environment)_1lt_q$(Int(q*100)).pdf")
end