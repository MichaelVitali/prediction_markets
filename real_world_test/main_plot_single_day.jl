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
include("data_preprop.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .QuantileRegression
using .AdaptiveRobustRegression
using .RealWorldtestData


# Environment Settings
n_experiments = 1
T = 31
lead_time = 96
quantiles = [0.1, 0.5, 0.9]
n_forecasters = 2
algorithms = ["QR"]
path_ecmwf = "real_world_test/data/forecasts_ecmwf_ifs_paper.parquet"
path_noaa = "real_world_test/data/forecasts_noaa_gfs_paper.parquet"
path_elia = "real_world_test/data/historical_load_data_predico_2025_08_15.csv"

# Environment Variables
exp_weights = Dict([q => Dict([algo => zeros((n_forecasters, T)) for algo in algorithms]) for q in quantiles])
realizations = Dict([q => [] for q in quantiles])
aggregated_forecasts = Dict([q => Dict([algo => [] for algo in algorithms]) for q in quantiles])
algo_forecasts = Dict([q => Dict(["ecmwf" => [], "noaa" => []]) for q in quantiles])
elia_forecasts = Dict([q => [] for q in quantiles])

###################### QR Model ########################
for q in quantiles

    # Initialization
    weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
    for algo in algorithms
        weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
        exp_weights[q][algo][:, 1] .+= weights_history[algo][:, 1]
    end
    
    # Data generation
    true_prod, forecasters_preds, forecasts_elia, scaler_ecmwf, scaler_noaa, scaler_target, scaler_elia = preprocessing_forecasts(path_ecmwf, path_noaa, path_elia, q)
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)

    # Initialization RQR
    if "QR" in algorithms
        alpha = Int.(rand(n_forecasters, T) .< 0.1)
        D_exp = zeros(n_forecasters, n_forecasters)
    end

    # Learning process
    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = true_prod[t]
        push!(realizations[q], denormalize(y_true, scaler_target))
        push!(elia_forecasts[q], denormalize(forecasts_elia[t], scaler_elia))

        for f in keys(forecasters_preds)
            if f == "ecmwf"
                push!(algo_forecasts[q][f], denormalize(forecasters_preds[f][t], scaler_ecmwf))
            else
                push!(algo_forecasts[q][f], denormalize(forecasters_preds[f][t], scaler_noaa))
            end
        end
        
        for algo in algorithms
            # Forecasting combination and weights update
            if algo == "RQR"
                weights_history[algo][:, t], new_D, aggregated_forecast_t = online_adaptive_robust_quantile_regression_multiple_lead_times(forecasters_preds_t, y_true, weights_history[algo][:, t-1], D_exp, alpha[:, t], q, 0.01)
                prev_D = D_exp
                D_exp = new_D
                exp_weights[q][algo][:, t] .+= weights_history[algo][:, t]
            elseif algo == "QR"
                weights_history[algo][:, t], aggregated_forecast_t = online_quantile_regression_update_multiple_lead_times(forecasters_preds_t, weights_history[algo][:, t-1], y_true, q, 0.01)
                exp_weights[q][algo][:, t] .+= weights_history[algo][:, t]
            end
            push!(aggregated_forecasts[q][algo], denormalize(aggregated_forecast_t, scaler_target))
        end
    end
end

#################### Post-processing monte- ####################
for algo in algorithms
    for q in quantiles
        exp_weights[q][algo] = exp_weights[q][algo] ./ n_experiments
    end
end

# Plot realizations and aggregated forecasts for quantile 0.5
plot_forecast = plot(size=[1200, 500],
    xlabel="Lead Time [15-minutes]",
    ylabel="MW",
    legendfont=12,
    fg_legend=:transparent,
    bg_legend=:transparent,
    ylabelfontsize=14,
    xlabelfontsize=14,
    bottom_margin=10mm,
    left_margin=10mm,
    tickfontsize=12
)
plot!(plot_forecast, realizations[0.5][20], 
    label="True Value", 
    lw=4, 
    color=:black, 
    ls=:solid
)
plot!(plot_forecast, aggregated_forecasts[0.5]["QR"][20], 
    label="Combination", 
    lw=2, 
    color=:blue, 
    ls=:solid
)
plot!(plot_forecast, algo_forecasts[0.5]["ecmwf"][20], 
    label="ECMWF", 
    lw=2, 
    color=:red, 
    ls=:dash
)
plot!(plot_forecast, algo_forecasts[0.5]["noaa"][20], 
    label="NOAA", 
    lw=2, 
    color=:orange,
    ls=:dot
)
display(plot_forecast)
savefig(plot_forecast, "plots/real_test/plot_elia_q50.pdf")

# Calculate Losses
algo = "QR"
for q in quantiles
    losses_aggregated = []
    losses_ecmwf = []
    losses_noaa = []

    for t in 1:T-1
        if q == 0.5
            loss_t = sqrt(mean((realizations[q][t] .- aggregated_forecasts[q][algo][t]).^2))
            loss_t_ecmwf = sqrt(mean((realizations[q][t] .- algo_forecasts[q]["ecmwf"][t]).^2))
            loss_t_noaa = sqrt(mean((realizations[q][t] .- algo_forecasts[q]["noaa"][t]).^2))
        else
            loss_t = mean(QuantileLoss(q).(realizations[q][t], aggregated_forecasts[q][algo][t]))
            loss_t_ecmwf = mean(QuantileLoss(q).(realizations[q][t], algo_forecasts[q]["ecmwf"][t]))
            loss_t_noaa = mean(QuantileLoss(q).(realizations[q][t], algo_forecasts[q]["noaa"][t]))
        end
        append!(losses_aggregated, loss_t)
        append!(losses_ecmwf, loss_t_ecmwf)
        append!(losses_noaa, loss_t_noaa)
    end
    println("RESULTS $q")
    println("Loss aggregated: $(mean(losses_aggregated))")
    println("Loss ECWMF: $(mean(losses_ecmwf))")
    println("Loss NOAA: $(mean(losses_noaa))")
end
    