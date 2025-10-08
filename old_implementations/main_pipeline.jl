using LinearAlgebra
using Plots
using DataStructures
using ProgressBars
using Base.Threads

include("functions/functions.jl")
include("functions/functions_payoff.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/quantile_regression.jl")
include("online_algorithms/adaptive_robust_quantile_regression.jl")
include("payoff/proportion_variance.jl")
include("payoff/leave_one_out.jl")
include("payoff/shapley_values.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .DataGeneration
using .QuantileRegression
using .ProportionVariance
using .LOO
using .Shapley
using .AdaptiveRobustRegression


# Settings Monte-Carlo simulation
n_experiments = 200
T = 10000
q = 0.5
n_forecasters = 3
algorithms = ["QR"]
payoff_functions = ["Shapley"]

exp_weights = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
exp_payoffs = Dict([payoff => Dict([algo => zeros((n_forecasters, T)) for algo in algorithms]) for payoff in payoff_functions])
true_weights = nothing
alpha = Int.(rand(n_forecasters, T) .< 0.05)

#for i in ProgressBar(1:n_experiments)
Threads.@threads for i in ProgressBar(1:n_experiments)

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
    payoffs_exp = Dict([payoff => Dict([algo => zeros((n_forecasters, T)) for algo in algorithms]) for payoff in payoff_functions])
     # Initialization RQR
    if "RQR" in algorithms
        # Ensure not all values in any column are 1
        #alpha = Int.(rand(n_forecasters, T) .< 0.05)
        D_exp = zeros(n_forecasters, n_forecasters)
    end

    # Learning process
    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in keys(sorted_forecasters)]
        y_true = realizations[t]
        
        for algo in algorithms
            if algo == "RQR"
                weights_history[algo][:, t], new_D, _ = online_adaptive_robust_quantile_regression(forecasters_preds_t, y_true, weights_history[algo][:, t-1], D_exp, alpha[:, t], q)
                prev_D = D_exp
                D_exp = new_D
                exp_weights[algo][:, t] .+= weights_history[algo][:, t]
            elseif algo == "QR"
                weights_history[algo][:, t], _ = online_quantile_regression_update(forecasters_preds_t, weights_history[algo][:, t-1], y_true, q, 0.01)
                exp_weights[algo][:, t] .+= weights_history[algo][:, t]
            end

            # Payoff calculation
            if "variance" in payoff_functions
                payoffs_exp["variance"][algo][:, t] = proportion_variance_payoff(weights_history[algo][:, t-1])
            end

            for payoff in payoff_functions
                ## This section extract only the available forecasters for the payoff calculus
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
                if payoff == "LOO"
                    if length(temp_weights_t) > 0
                        temp_payoffs = leave_one_out_payoff(temp_forecasts_t, temp_weights_t, y_true, q)
                    end
                elseif payoff == "Shapley"
                    if length(temp_weights_t) > 0
                        temp_payoffs = shapley_payoff(temp_forecasts_t, temp_weights_t, y_true, q)
                    end
                end
                ## If some forecaster is missing their payoff is added as zero
                if algo == "RQR"
                    if temp_payoffs === nothing
                        temp_payoffs = zeros(n_forecasters)
                    else
                        for j in findall(a -> a == 1, alpha[:, t])
                            insert!(temp_payoffs, j, 0.0)
                        end
                    end
                end
                payoffs_exp[payoff][algo][:, t] = payoff_update(payoffs_exp[payoff][algo][:, t-1], temp_payoffs, 0.999)

            end
        end
    end

    for payoff in payoff_functions
        for algo in algorithms
            exp_payoffs[payoff][algo] .+= payoffs_exp[payoff][algo]
        end
    end
    
end

# Post-processing monte-carlo
for algo in algorithms
    exp_weights[algo] = exp_weights[algo] ./ n_experiments
    
    for payoff in payoff_functions
        exp_payoffs[payoff][algo] = exp_payoffs[payoff][algo] ./ n_experiments
        #exp_payoffs[payoff][algo] = exp_payoffs[payoff][algo] ./ sum(exp_payoffs[payoff][algo], dims=1)
    end
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
display(plot_weigths)

# Plot payoffs
plot_payoffs_dict = Dict()
for payoff in payoff_functions
    plot_payoffs = plot(layout=(length(algorithms), 1), size=(1000, 500))
    for (i, algo) in enumerate(algorithms)
        plot!(plot_payoffs[i], 1:T, exp_payoffs[payoff][algo]', 
              label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
              xlabel="Time", ylabel="Payoff", 
              title="$payoff Over Time - $algo")
    end
    display(plot_payoffs)
end