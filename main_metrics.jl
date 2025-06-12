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

function quantile_regression_mean_imputation(forecasters_preds, y, w, alpha, q, learning_rate=0.01)

    n_forecasters, T = size(forecasters_preds)
    received_f = Dict([i => copy(forecasters_preds[i, 1:100]) for i in 1:n_forecasters])
    weights_history = zeros(n_forecasters, T)
    for t in 1:100
        weights_history[:, t] = w
    end

    for t in 100:T
        forecasters_preds_t = zeros(3)
        prev_w = weights_history[:, t-1]
        y_t = y[t]

        # Mean imputation
        for i in 1:n_forecasters
            if alpha[i, t] == 1
                forecasters_preds_t[i] = mean(received_f[i])
            else
                forecasters_preds_t[i] = forecasters_preds[i, t]
                push!(received_f[i], forecasters_preds[i, t])
            end
        end

        # Combination and gradient step
        combined_quantile = sum(forecasters_preds_t .* prev_w)
        gradient_w = quantile_loss_gradient(y_t, combined_quantile, q) .* forecasters_preds_t

        weights_history[:, t] = prev_w .- learning_rate .* gradient_w
    end

    return weights_history
end


# Settings Monte-Carlo simulation
n_experiments = 100
T = 20000
q = 0.5
n_forecasters = 3
algorithms = ["QR", "RQR"]
show_benchmarks = true

if show_benchmarks
    push!(algorithms, "mean_impute")
    push!(algorithms, "last_impute")
end
exp_weights = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
true_weights = nothing

for i in 1:n_experiments

    # Weights initialization
    weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
    for algo in algorithms
        weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
        exp_weights[algo][:, 1] .+= weights_history[algo][:, 1]
    end
    
    # Data generation
    realizations, forecasters_preds, true_weights = generate_time_invariant_data(T, q)
    global true_weights = true_weights
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)

    if "RQR" in algorithms
            alpha = zeros(Int, n_forecasters, T)
        for t in 1:T
            alpha[:, t] = Int.(rand(n_forecasters) .< 0.1)
        end
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
                weights_history[algo][:, t], new_D = online_adaptive_robust_quantile_regression(forecasters_preds_t, y_true, weights_history[algo][:, t-1], D_exp, alpha[:, t], q)
                exp_weights[algo][:, t] .+= weights_history[algo][:, t]
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
            exp_weights["mean_impute"] .+= quantile_regression_mean_imputation(forecasters_preds, realizations, w0, alpha, q)
        end
        if "last_impute" in algorithms
            w0 = weights_history["last_impute"][:, 1]
            exp_weights["last_impute"] .+= quantile_regression_last_impute(forecasters_preds, realizations, w0, alpha, q)
        end
    end
end


# Post-processing monte-carlo
biasses = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
variances = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
for algo in algorithms
    exp_weights[algo] = exp_weights[algo] ./ n_experiments

    ## Metrics calculation
    biasses[algo] = calculate_bias(exp_weights[algo], true_weights)
    variances[algo] = calculate_variance(exp_weights[algo])
end

# Plot weights
plot_weigths = plot(layout=(length(algorithms), 1), size=(1000, 1000))
for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, exp_weights[algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
          xlabel="Time", ylabel="Weights", title="Weights History Over Time - $algo")
end

for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, true_weights, label=["w1" "w2" "w3"])
end

# Plot payoffs
cut_start = 2001

plot_biasses = plot(size=(1000, 500), xlabel="Time", ylabel="Bias", title="Biasses Over Time")
for algo in algorithms
    mean_bias = mean(biasses[algo], dims=1)[cut_start:end]
    plot!(plot_biasses, 1:length(mean_bias), mean_bias, label=algo)
end

plot_variances = plot(size=(1000, 500), xlabel="Time", ylabel="Variance", title="Variance Over Time")
for algo in algorithms
    mean_var = mean(variances[algo], dims=1)[cut_start:end]
    plot!(plot_variances, 1:length(mean_var), mean_var, label=algo)
end

display(plot_weigths)
display(plot_biasses)
display(plot_variances)