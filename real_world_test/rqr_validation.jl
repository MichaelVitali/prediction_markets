using LinearAlgebra
using DataStructures
using Statistics
using Dates
using Normalization
using Random
using ProgressBars

include("../functions/functions.jl")
include("../online_algorithms/adaptive_robust_quantile_regression.jl")
include("data_preprop.jl")

using .UtilsFunctions
using .AdaptiveRobustRegression
using .RealWorldtestData

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Configuration
validation_size = 60 # Validation on the first month (approx 30 days)
quantiles = [0.1, 0.5, 0.9]    # Set the quantile(s) you want to validate
missing_rate = 0.05
n_simulations = 100   # Number of Monte Carlo simulations
burn_in_period = 10  # Number of initial time steps to ignore for loss calculation

# Grid Search Parameters
learning_rates = [0.2, 0.1, 0.05, 0.01, 0.005]
batch_percentages = [0.05, 0.1, 0.2, 0.5, 1.0]

root_dir = @__DIR__
models_paths = OrderedDict(
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

# Generate Fixed Missingness Pattern (Monte Carlo)
# We generate this ONCE to ensure all grid search iterations use the exact same simulation
println("Generating fixed Monte Carlo missingness simulation for validation...")
alpha_simulations = Vector{Matrix{Int}}(undef, n_simulations)
for s in 1:n_simulations
    alpha_s = Int.(rand(n_forecasters, validation_size + 1) .< missing_rate)
    for t in 1:(validation_size + 1)
        if sum(alpha_s[:, t]) == n_forecasters
            # Ensure at least one forecaster is present
            idx = rand(1:n_forecasters)
            alpha_s[idx, t] = 0
        end
    end
    alpha_simulations[s] = alpha_s
end

best_overall_loss = Inf
best_overall_params = (lr=NaN, batch=NaN)

println("Starting Grid Search to find best parameters across all quantiles...")

# Iterate through grid
for lr in learning_rates
    for batch_pct in batch_percentages
        
        total_quantile_loss = 0.0

        for q in quantiles
            # Load Data for the current quantile
            true_prod, forecasters_preds, scalers, scaler_target, _ = preprocessing_forecasts(models_paths, q)
            
            total_loss_sims = 0.0
            # Monte Carlo simulations for the current hyperparameter set and quantile
            for s in 1:n_simulations
                alpha_validation = alpha_simulations[s]

                # Initialize weights and D matrix for this run
                weights = zeros(n_forecasters, validation_size + 1)
                weights[:, 1] .= 1.0 / n_forecasters
                D = zeros(n_forecasters, n_forecasters)
                
                cumulative_loss = 0.0
                count = 0

                # Validation Loop
                for t in 2:validation_size
                    # Prepare inputs
                    forecasters_preds_t = [forecasters_preds[f][t] for f in model_names]
                    y_true = true_prod[t]
                    y_true_sc = scaler_target(y_true)
                    
                    alpha_t = alpha_validation[:, t]

                    # Update Step
                    new_w, new_D, agg_forecast_sc = online_adaptive_robust_quantile_regression_multiple_lead_times(
                        forecasters_preds_t,
                        y_true_sc,
                        weights[:, t-1],
                        D,
                        alpha_t,
                        q,
                        lr,
                        batch_pct
                    )

                    weights[:, t] = new_w
                    D = new_D

                    # Calculate Loss (on original scale)
                    if t > burn_in_period
                        agg_forecast = denormalize(agg_forecast_sc, scaler_target)
                        loss_t = mean(quantile_loss.(y_true, agg_forecast, q))
                        cumulative_loss += loss_t
                        count += 1
                    end
                end
                # Average loss for this simulation
                total_loss_sims += (count > 0 ? cumulative_loss / count : 0)
            end

            # Average loss for this quantile
            avg_loss_for_quantile = total_loss_sims / n_simulations
            total_quantile_loss += avg_loss_for_quantile
        end

        # Average loss across all quantiles for the current (lr, batch_pct)
        overall_avg_loss = total_quantile_loss / length(quantiles)
        println("  LR: $lr | Batch: $batch_pct | Overall Avg Loss: $(round(overall_avg_loss, digits=6))")

        if overall_avg_loss < best_overall_loss
            global best_overall_loss = overall_avg_loss
            global best_overall_params = (lr=lr, batch=batch_pct)
        end
    end
end

println("\n>>> Best Overall Params: LR=$(best_overall_params.lr), Batch=$(best_overall_params.batch) with Loss=$(best_overall_loss)")
