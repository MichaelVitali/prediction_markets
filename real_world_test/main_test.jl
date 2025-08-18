using LinearAlgebra
using Plots
using DataStructures
using ProgressBars
using Base.Threads
using Plots.PlotMeasures
using LossFunctions

include("../functions/functions.jl")
include("../functions/functions_payoff.jl")
include("../online_algorithms/quantile_regression.jl")
include("../online_algorithms/adaptive_robust_quantile_regression.jl")
include("../payoff/new_shapley.jl")
include("data_preprop.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .QuantileRegression
using .Shapley
using .AdaptiveRobustRegression
using .RealWorldtestData


# Environment Settings
n_experiments = 1
T = 42
lead_time = 96
quantiles = [0.1, 0.5, 0.9]
n_forecasters = 2
algorithms = ["QR", "RQR"]
payoff_function = "Shapley"
path_ecmwf = "real_world_test/results/forecasts_ecmwf_ifs_paper.parquet"
path_noaa = "real_world_test/results/forecasts_noaa_gfs_paper.parquet"

# Environment Variables
exp_weights = Dict([q => Dict([algo => zeros((n_forecasters, T)) for algo in algorithms]) for q in quantiles])
exp_payoffs = Dict([q => Dict([algo => zeros((n_forecasters, T)) for algo in algorithms]) for q in quantiles])
realizations = Dict([q => [] for q in quantiles])
aggregated_forecasts = Dict([q => Dict([algo => [] for algo in algorithms]) for q in quantiles])
algo_forecasts = Dict([q => Dict(["ecmwf" => [], "noaa" => []]) for q in quantiles])

for q in quantiles

    # Initialization
    payoffs_exp = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
    weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
    for algo in algorithms
        weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
        exp_weights[q][algo][:, 1] .+= weights_history[algo][:, 1]
    end
    
    # Data generation
    true_prod, forecasters_preds = preprocessing_forecasts(path_ecmwf, path_noaa, q)
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)

    # Initialization RQR
    if "RQR" in algorithms
        alpha = Int.(rand(n_forecasters, T) .< 0.01)
        D_exp = zeros(n_forecasters, n_forecasters)
    end

    # Learning process
    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = true_prod[t]
        append!(realizations[q], y_true)
        for f in keys(forecasters_preds)
            append!(algo_forecasts[q][f], forecasters_preds[f][t])
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
            append!(aggregated_forecasts[q][algo], aggregated_forecast_t)
            
            # Payoff calculation
            if algo == "RQR"
                temp_forecasts_t = [forecasters_preds_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
                temp_weights_t = weights_history[algo][:, t-1] .+ prev_D * alpha[:, t]
                temp_weights_t = [temp_weights_t[j] for j in 1:length(forecasters_preds_t) if alpha[j, t] == 0]
            else
                temp_forecasts_t = forecasters_preds_t
                temp_weights_t = weights_history[algo][:, t-1]
            end

            ## Payoff Calculation
            temp_payoffs = nothing
            if length(temp_weights_t) > 0
                temp_payoffs = shapley_payoff_multiple_lead_times(temp_forecasts_t, temp_weights_t, y_true, q)
            end
            if algo == "RQR"
                if temp_payoffs === nothing
                    temp_payoffs = zeros(n_forecasters)
                else
                    for j in findall(a -> a == 1, alpha[:, t])
                        insert!(temp_payoffs, j, 0.0)
                    end
                end
            end
            payoffs_exp[algo][:, t] = payoff_update(payoffs_exp[algo][:, t-1], temp_payoffs, 0.999)
        end
    end

    for algo in algorithms
        exp_payoffs[q][algo] .+= payoffs_exp[algo]
    end
end

#################### Post-processing monte- ####################
for algo in algorithms
    for q in quantiles
        exp_weights[q][algo] = exp_weights[q][algo] ./ n_experiments
        exp_payoffs[q][algo] = exp_payoffs[q][algo] ./ n_experiments
    end
end

# Plot realizations and aggregated forecasts for quantile 0.5
plot_forecast = plot(size=[1500, 800])
plot!(plot_forecast, realizations[0.5][end-480:end], label="True Value", lw=2, color=:black)
plot!(plot_forecast, aggregated_forecasts[0.5]["QR"][end-480:end], label="Combination", lw=2, color=:blue)
plot!(plot_forecast, algo_forecasts[0.5]["ecmwf"][end-480:end], label="ECMWF", lw=2, color=:red)
plot!(plot_forecast, algo_forecasts[0.5]["noaa"][end-480:end], label="NOAA", lw=2, color=:orange)

loss_combined = mean(QuantileLoss(0.5).(realizations[0.5],aggregated_forecasts[0.5]["QR"]))
loss_ecmwf = mean(QuantileLoss(0.5).(realizations[0.5], algo_forecasts[0.5]["ecmwf"]))
loss_noaa = mean(QuantileLoss(0.5).(realizations[0.5], algo_forecasts[0.5]["noaa"]))
println(loss_combined)
println(loss_ecmwf)
println(loss_noaa)