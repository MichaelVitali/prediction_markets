using LinearAlgebra
using Plots
using DataStructures


include("functions/functions.jl")
include("functions/functions_payoff.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/BOA.jl")
include("online_algorithms/quantile_regression.jl")
include("payoff/proportion_variance.jl")
include("payoff/leave_one_out.jl")
include("payoff/new_shapley.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .DataGeneration
using .BOA
using .QuantileRegression
using .ProportionVariance
using .LOO
using .Shapley

# Settings Monte-Carlo simulation
n_experiments = 100
T = 20000
q = 0.9
n_forecasters = 3
algorithms = ["QR"]
payoff_function = "variance"

exp_weights = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
cumulative_payoffs = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
true_weights = nothing

for i in 1:n_experiments

    # Weights initialization
    weights_history = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
    for algo in algorithms
        weights_history[algo][:, 1] .= initialize_weights(n_forecasters)
        exp_weights[algo][:, 1] .+= weights_history[algo][:, 1]
    end
    
    # Data generation
    realizations, forecasters_preds, true_weights = generate_dynamic_data(T, q)
    global true_weights = true_weights
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)
    payoffs_exp = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])

    # Learning process
    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = realizations[t]
        
        for algo in algorithms
            if algo == "BOA"
                weights_history[algo][:, t] = bernstein_online_aggregation_update(forecasters_preds_t, weights_history[algo][:, t-1], y_true, q)
                exp_weights[algo][:, t] .+= weights_history[algo][:, t]
            elseif algo == "QR"
                weights_history[algo][:, t] = online_quantile_regression_update(forecasters_preds_t, weights_history[algo][:, t-1], y_true, q, 0.01)
                exp_weights[algo][:, t] .+= weights_history[algo][:, t]
            end

            # Payoff calculation
            if payoff_function == "variance"
                payoffs_exp[algo][:, t] = proportion_variance_payoff(weights_history[algo][:, t])
            elseif payoff_function == "loo"
                temp_payoffs = leave_one_out_payoff(forecasters_preds_t, weights_history[algo][:, t], y_true, q)
                payoffs_exp[algo][:, t] = payoff_update(payoffs_exp[algo][:, t-1], temp_payoffs, 0.999)
            elseif payoff_function == "shapley"
                temp_payoffs = shapley_payoff(forecasters_preds_t, weights_history[algo][:, t], y_true, q)
                payoffs_exp[algo][:, t] = payoff_update(payoffs_exp[algo][:, t-1], temp_payoffs, 0.999)
            end
        end
    end

    for algo in algorithms
        cumulative_payoffs[algo] .+= payoffs_exp[algo]
    end
    
end

# Post-processing monte-carlo
for algo in algorithms
    exp_weights[algo] = exp_weights[algo] ./ n_experiments
    cumulative_payoffs[algo] = cumulative_payoffs[algo] ./ n_experiments
    display(cumulative_payoffs[algo])
    #cumulative_payoffs[algo] = cumsum(cumulative_payoffs[algo] ./ n_experiments, dims=2)
end

# Plot weights
plot_weigths = plot(layout=(length(algorithms), 1), size=(1000, 500))
for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, exp_weights[algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
          xlabel="Time", ylabel="Weights", title="Weights History Over Time - $algo")
end

for (i, algo) in enumerate(algorithms)
    plot!(plot_weigths[i], 1:T, true_weights, label=["w1" "w2" "w3"])
end

# Plot payoffs
plot_payoffs = plot(layout=(length(algorithms), 1), size=(1000, 500))
for (i, algo) in enumerate(algorithms)
    plot!(plot_payoffs[i], 1:T, cumulative_payoffs[algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
          xlabel="Time", ylabel="Payoff", title="Payoffs Over Time - $algo")
end
display(plot_weigths)
#display(plot_payoffs)