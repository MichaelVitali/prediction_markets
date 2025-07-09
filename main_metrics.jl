using LinearAlgebra
using Plots
using DataStructures
using Statistics
using ProgressBars
using Base.Threads

include("functions/functions.jl")
include("functions/metrics.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/quantile_regression.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
include("online_algorithms/robust_optimization_benchmark.jl")
using .UtilsFunctions
using .DataGeneration
using .Metrics
using .QuantileRegression
using .AdaptiveRobustRegression
using .RobustOptimizationBenchmarks

# Settings Monte-Carlo simulation
n_experiments = 500
T = 20000
q = 0.5
n_forecasters = 3
algorithms = ["RQR"]
show_benchmarks = true

if show_benchmarks
    push!(algorithms, "mean_impute")
    push!(algorithms, "last_impute")
end

exp_weights = Dict([algo => Dict([f => zeros(n_experiments, T) for f in 1:n_forecasters]) for algo in algorithms])
exp_forecasts = Dict([algo => zeros(n_experiments, T) for algo in algorithms])
exp_realizations = zeros(n_experiments, T)
true_weights = nothing


Threads.@threads for i in ProgressBar(1:n_experiments)

    # Weights initialization
    weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
    forecasts_history = Dict([algo => zeros((T)) for algo in algorithms])
    for algo in algorithms
        weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
        for f in 1:n_forecasters
            exp_weights[algo][f][i, 1] = initialize_weights(n_forecasters)[f]
        end
    end
    
    # Data generation
    realizations, forecasters_preds, w = generate_time_invariant_data(T, q)
    global exp_realizations[i, :] .= realizations
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)

    if i == 1
        global true_weights = w'
    end

    if "RQR" in algorithms
        alpha = Int.(rand(n_forecasters, T) .< 0.05)
        D_exp = zeros(n_forecasters, n_forecasters)
    end

    # Learning process
    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = realizations[t]
        
        for algo in algorithms
            if algo == "QR"
                weights_history[algo][:, t], forecasts_history[algo][t] = online_quantile_regression_update(forecasters_preds_t, weights_history[algo][:, t-1], y_true, q)
            elseif algo == "RQR"
                weights_history[algo][:, t], new_D, forecasts_history[algo][t] = online_adaptive_robust_quantile_regression(forecasters_preds_t, y_true, weights_history[algo][:, t-1], D_exp, alpha[:, t], q)
                exp_forecasts[algo][t] += forecasts_history[algo][t]
                D_exp = new_D
            end

            for f in 1:n_forecasters
                exp_weights[algo][f][i, t] = weights_history[algo][f, t]
                exp_forecasts[algo][i, :] .= forecasts_history[algo]
            end
        end
    end

    # Benchmark calculation
    if show_benchmarks
        forecasters_preds = hcat([forecasters_preds[f] for f in keys(sorted_forecasters)]...)
        forecasters_preds = permutedims(forecasters_preds)

        if "mean_impute" in algorithms
            w0 = weights_history["mean_impute"][:, 1]
            results_w, results_f = quantile_regression_mean_imputation(forecasters_preds, realizations, w0, alpha, q)

            for f in 1:n_forecasters
                exp_weights["mean_impute"][f][i, :] .= results_w[f, :]
                exp_forecasts["mean_impute"][i, :] .= results_f
            end
        end
        if "last_impute" in algorithms
            w0 = weights_history["last_impute"][:, 1]
            results_w, results_f = quantile_regression_last_impute(forecasters_preds, realizations, w0, alpha, q)
            
            for f in 1:n_forecasters
                exp_weights["last_impute"][f][i, :] .= results_w[f, :]
                exp_forecasts["last_impute"][i, :] .= results_f
            end
        end
    end
end

# Post-processing monte-carlo

## Calculate tracking errors
errors = Dict([algo => Dict([f => zeros(n_experiments, T) for f in 1:n_forecasters]) for algo in algorithms])
for algo in algorithms
    for f in 1:n_forecasters
        for i in 1:n_experiments
            errors[algo][f][i, :] .= calculate_instantaneous_errors(exp_weights[algo][f][i, :], true_weights[f, :])
        end
    end
end

# Calculate Metrics
weights_mc = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
biasses_mc = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
losses_mc = Dict([algo => zeros(T) for algo in algorithms])

## Calculate Monte-carlo Biasses and Weights
for algo in algorithms
    for f in 1:n_forecasters
        weights_mc[algo][f, :] = mean(exp_weights[algo][f], dims=1)
        biasses_mc[algo][f, :] = mean(errors[algo][f], dims=1)
    end

    temp_losses = Dict([algo => zeros((n_experiments, T)) for algo in algorithms])
    for i in 1:n_experiments
        temp_losses[algo][i, :] .= calculate_instantaneous_quantile_loss(exp_forecasts[algo][i, :], exp_realizations[i, :], q)
    end
    losses_mc[algo] .= vec(mean(temp_losses[algo], dims=1))
end

## Calculate Variances
variances = Dict([algo => Dict([f => zeros(n_experiments, T) for f in 1:n_forecasters]) for algo in algorithms])
variances_mc = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
for algo in algorithms
    for f in 1:n_forecasters
        for i in 1:n_experiments
            variances[algo][f][i, :] .= calculate_instantaneous_variance(errors[algo][f][i, :], biasses_mc[algo][f, :])
        end

        variances_mc[algo][f, :] = sum(variances[algo][f], dims=1) ./ (n_experiments - 1)
    end
end

# Plot weights
plot_weigths = plot(layout=(length(algorithms), 1), size=(1000, 1000), legend=:topright)
for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, weights_mc[algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
          xlabel="Time", ylabel="Weights", title="Weights History Over Time - $algo")
end

for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, true_weights', label=["w1" "w2" "w3"])
end
display(plot_weigths)

# Plot Metrics
cut_start = 5001

plot_biasses = plot(layout=(n_forecasters, 1), size=(1000, 300 * n_forecasters), legend=:topright)
for f in 1:n_forecasters
    for algo in algorithms
        biass_w = biasses_mc[algo][f, cut_start:end]
        plot!(plot_biasses[f], 1:length(biass_w), biass_w, label="$(algo)")
    end
    xlabel!(plot_biasses[f], "Time")
    ylabel!(plot_biasses[f], "Bias")
    title!(plot_biasses[f], "Bias Over Time - w$f")
end
display(plot_biasses)

plot_variances = plot(layout=(n_forecasters, 1), size=(1000, 300 * n_forecasters), legend=:topright)
for f in 1:n_forecasters
    for algo in algorithms
        var_w = variances_mc[algo][f, cut_start:end]
        plot!(plot_variances[f], 1:length(var_w), var_w, label="$(algo)")
    end
    xlabel!(plot_variances[f], "Time")
    ylabel!(plot_variances[f], "Variance")
    title!(plot_variances[f], "Variance Over Time - w$f")
end
display(plot_variances)

# Plot Losses
plot_losses = plot(layout=(length(algorithms) + 1, 1), size=(1000, 400 * (length(algorithms) + 1)), legend=:topright)
for (i, algo) in enumerate(algorithms)
    loss_algo = losses_mc[algo][cut_start:end]
    plot!(plot_losses[i], 1:length(loss_algo), loss_algo, label=algo)
    xlabel!(plot_losses[i], "Time")
    ylabel!(plot_losses[i], "Loss")
    title!(plot_losses[i], "Quantile Loss Over Time - $algo")
end

# Add subplot for running mean of quantile loss
for (i, algo) in enumerate(algorithms)
    running_mean = [mean(losses_mc[algo][cut_start:t]) for t in cut_start:T]
    plot!(plot_losses[length(algorithms) + 1], 1:length(running_mean), running_mean, label=algo)
end
xlabel!(plot_losses[length(algorithms) + 1], "Time")
ylabel!(plot_losses[length(algorithms) + 1], "Running Mean Loss")
title!(plot_losses[length(algorithms) + 1], "Running Mean Quantile Loss")

display(plot_losses)