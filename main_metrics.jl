using LinearAlgebra
using Plots
using DataStructures
using Statistics

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
n_experiments = 100
T = 20000
q = 0.5
n_forecasters = 3
algorithms = ["RQR"]
show_benchmarks = true

if show_benchmarks
    push!(algorithms, "mean_impute")
    push!(algorithms, "last_impute")
    #push!(algorithms, "oracle")
end
exp_weights = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
exp_forecasts = Dict([algo => zeros(T) for algo in algorithms])
exp_realizations = zeros(T)
true_weights = nothing

for i in 1:n_experiments

    # Weights initialization
    weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
    forecasts_history = Dict([algo => zeros((T)) for algo in algorithms])
    for algo in algorithms
        weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
        exp_weights[algo][:, 1] .+= initialize_weights(n_forecasters)
    end
    
    # Data generation
    realizations, forecasters_preds, true_weights = generate_time_invariant_data(T, q)
    global true_weights = true_weights'
    global exp_realizations += realizations
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)

    if "RQR" in algorithms
        alpha = Int.(rand(n_forecasters, T) .< 0.1)
        D_exp = zeros(n_forecasters, n_forecasters)
    end

    # Learning process
    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = realizations[t]
        
        for algo in algorithms
            if algo == "QR"
                weights_history[algo][:, t] = online_quantile_regression_update(forecasters_preds_t, weights_history[algo][:, t-1], y_true, q)
                exp_weights[algo][:, t] .+= weights_history[algo][:, t]
            elseif algo == "RQR"
                weights_history[algo][:, t], new_D, forecasts_history[algo][t] = online_adaptive_robust_quantile_regression(forecasters_preds_t, y_true, weights_history[algo][:, t-1], D_exp, alpha[:, t], q)
                exp_weights[algo][:, t] .+= weights_history[algo][:, t]
                exp_forecasts[algo][t] += forecasts_history[algo][t]
                D_exp = new_D
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
            exp_weights["mean_impute"][:, 2:end] .+= results_w[:, 2:end]
            exp_forecasts["mean_impute"] .+= results_f
        end
        if "last_impute" in algorithms
            w0 = weights_history["last_impute"][:, 1]
            results_w, results_f = quantile_regression_last_impute(forecasters_preds, realizations, w0, alpha, q)
            exp_weights["last_impute"][:, 2:end] .+= results_w[:, 2:end]
            exp_forecasts["last_impute"] .+= results_f
        end
        if "oracle" in algorithms
            w0 = weights_history["oracle"][:, 1]
            results_w, results_f = quantile_regression_oracle(forecasters_preds, realizations, w0, alpha, q)
            exp_weights["oracle"][:, 2:end] .+= results_w[:, 2:end]
            exp_forecasts["oracle"] .+= results_f
        end
    end
end


# Post-processing monte-carlo
biasses = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
variances = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
losses = Dict([algo => zeros((T)) for algo in algorithms])

exp_realizations = exp_realizations ./ n_experiments
for algo in algorithms
    exp_weights[algo] = exp_weights[algo] ./ n_experiments
    exp_forecasts[algo] = exp_forecasts[algo] ./ n_experiments

    ## Metrics calculation
    biasses[algo] = calculate_bias(exp_weights[algo], true_weights)
    variances[algo] = calculate_variance(exp_weights[algo])
    losses[algo] = calculate_mean_quantile_loss(exp_forecasts[algo], exp_realizations, q)
end

# Plot weights
plot_weigths = plot(layout=(length(algorithms), 1), size=(1000, 1000), legend=:topright)
for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, exp_weights[algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
          xlabel="Time", ylabel="Weights", title="Weights History Over Time - $algo")
end

for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, true_weights', label=["w1" "w2" "w3"])
end
display(plot_weigths)

# Plot Forecasts 
plot_forecasts = plot(layout=(length(algorithms), 1), size=(1000, 1000), legend=:topright)
for (i, algo) in enumerate(algorithms)
    plot!(plot_forecasts[i], 1:T, exp_forecasts[algo], label="$q", 
          xlabel="Time", title="Forecasts Over Time - $algo")
    plot!(plot_forecasts[i], 1:T, exp_realizations, label="Realization", color=:black, lw=2, ls=:dash)
end
display(plot_forecasts)

# Plot Metrics
cut_start = 2001

plot_metrics = plot(size=(1000, 800), layout=(3, 1), legend=:topright)
for algo in algorithms

    # Plot Bias Over Time
    mean_bias = mean(biasses[algo], dims=1)[cut_start:end]
    plot!(plot_metrics[1], 1:length(mean_bias), mean_bias, label=algo)
    xlabel!(plot_metrics[1], "Time")
    ylabel!(plot_metrics[1], "Bias")
    title!(plot_metrics[1], "Biasses Over Time")

    # Plot Variance Over Time
    mean_var = mean(variances[algo], dims=1)[cut_start:end]
    plot!(plot_metrics[2], 1:length(mean_var), mean_var, label=algo)
    xlabel!(plot_metrics[2], "Time")
    ylabel!(plot_metrics[2], "Variance")
    title!(plot_metrics[2], "Variances Over Time")

    # Plot Mean Quantile Loss Over Time
    plot!(plot_metrics[3], 1:length(losses[algo][cut_start:end]), losses[algo][cut_start:end], label=algo)
    xlabel!(plot_metrics[3], "Time")
    ylabel!(plot_metrics[3], "Loss")
    title!(plot_metrics[3], "Quantile Loss Over Time")
end
display(plot_metrics)