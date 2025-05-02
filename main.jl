using LinearAlgebra
using Plots
using DataStructures

include("functions/functions.jl")
include("data_generation/DataGeneration.jl")
include("online_algorithms/BOA.jl")
include("online_algorithms/squint.jl")
include("online_algorithms/ML_poly.jl")
using .UtilsFunctions
using .DataGeneration
using .BOA
using .Squint
using .ML_Poly


n_experiments = 100
T = 100000
q = 0.5
n_forecasters = 3
case_study = "Gneiting"
algorithms = ["BOA", "Squint", "ML_Poly"]

exp_weights = Dict([algo => zeros((n_forecasters, T)) for algo in algorithms])
true_weights = nothing

for i in 1:n_experiments
    forecaster_weight = initialize_weights(n_forecasters)
    
    invariant_data, forecasters_preds, true_weights = generate_abrupt_data(T, q)
    global true_weights = true_weights
    sorted_f = sort(collect(forecasters_preds), by=first)
    sorted_forecasters = OrderedDict(sorted_f)
    
    for algo in algorithms
        if algo == "BOA"
            weights_history = bernstein_online_aggregation(sorted_forecasters, forecaster_weight, invariant_data, T, q)
        elseif algo == "Squint"
            weights_history = squint_CP(sorted_forecasters, invariant_data, T, q)
        elseif algo == "ML_Poly"
            weights_history = ml_poly(sorted_forecasters, invariant_data, T, q)
        end
        exp_weights[algo] .+= weights_history
    end
end

for algo in algorithms
    exp_weights[algo] = exp_weights[algo] ./ n_experiments
    display(exp_weights[algo])
end

p = plot(layout=(length(algorithms), 1), size=(1000, 1000))
for (i, algo) in enumerate(algorithms)
    plot!(p[i], 1:T, exp_weights[algo]', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], 
          xlabel="Time", ylabel="Weights", title="Weights History Over Time - $algo")
end

for (i, algo) in enumerate(algorithms)
    plot!(p[i], 1:T, true_weights, label=["w1" "w2" "w3"])
end

display(p)