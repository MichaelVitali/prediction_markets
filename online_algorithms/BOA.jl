using Revise
using LinearAlgebra
using Distributions
using Plots
using DataFrames
using LossFunctions
using DataStructures

include("../functions/functions.jl")
include("../data_generation/DataGeneration.jl")
using .UtilsFunctions
using .DataGeneration

function initialize_weights(n_experts::Integer)

    weigths = fill(1 / n_experts, n_experts)

    return weigths

end

function create_forecasters_preds(
    distributions,
    q,
    coefs::Vector{Float64},
)

    forecasters_dict = Dict()

    for (i, coef) in enumerate(coefs)
        forecaster_name = "f" * string(i)

        other_coefs = vcat(coefs[1:i-1], coefs[i+1:end])
        forecaster_mean = distributions["X0"] .+ (coef .* distributions["X$i"])
        forecaster_var = 1 + sum(other_coefs.^2)

        forecaster_dists = Normal.(forecaster_mean, sqrt(forecaster_var))
        forecasters_dict[forecaster_name] = quantile.(forecaster_dists, q)
    end

    return forecasters_dict

end

function bernstein_online_aggregation(forecasters_preds, forecaster_weights, y_true, T, q)

    forecasters_names = collect(keys(forecasters_preds))
    learning_rate = 0.001
    weights_history = Matrix{Float64}(undef, length(forecaster_weights), T)
    weights_history[:, 1] = forecaster_weights

    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in forecasters_names]
        println(forecasters_preds_t)
        agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
        #agg_loss = QuantileLoss(q)(agg_quantile_t, y_true[t])
        agg_loss = sqrt(sum((agg_quantile_t .- y_true[t]).^2) / length(agg_quantile_t))

        #losses = [QuantileLoss(q)(forecasters_preds[f][t], y_true[t]) for f in forecasters_names]
        losses = [sqrt(sum((forecasters_preds[f][t] .- y_true[t]).^2) / length(forecasters_preds[f][t])) for f in forecasters_names]
        lks = losses .- agg_loss
        lw = sum(weights_history[:, t-1] .* lks)

        numerators = -learning_rate .* lks .* (1 .+ (learning_rate .* lks))
        denoms = -learning_rate * lw * (1 + (learning_rate * lw))
        #println(denoms)
        weights_history[:, t] = weights_history[:, t-1] .* exp.(numerators) ./ exp(denoms)

    end

    return weights_history
end

n_experiments = 100
T = 11500
experts_coefs = [1.0, 1.0, 0.5]
q = 0.5
exp_weights = zeros((3, T))

for i in 1:n_experiments
    forecaster_weight = initialize_weights(3)
    #forecaster_weight = [0.2, 0.2, 0.6]
    invariant_data, distributions = generate_time_invariant_data(T, experts_coefs)
    distributions = OrderedDict(sort(collect(distributions), by=first))
    
    forecasters_preds = create_forecasters_preds(distributions, q, experts_coefs)
    weights_history = bernstein_online_aggregation(forecasters_preds, forecaster_weight, invariant_data, T, q)
    
    exp_weights .+= weights_history 
end

exp_weights = exp_weights ./ n_experiments
display(exp_weights)
plot(1:T, exp_weights', label=["Forecaster 1" "Forecaster 2" "Forecaster 3"], xlabel="Time", ylabel="Weights", title="Weights History Over Time")

