module QuantileRegression

    using LinearAlgebra

    include("../functions/functions.jl")
    include("../data_generation/DataGeneration.jl")
    using .UtilsFunctions
    using .DataGeneration

export online_quantile_regression

    function online_quantile_regression(forecasters_preds, forecaster_weights, y_true, T, q)

        forecasters_names = collect(keys(forecasters_preds))
        weights_history = Matrix{Float64}(undef, length(forecaster_weights), T)
        weights_history[:, 1] = forecaster_weights
        learning_rate = 0.01

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecasters_names]
            agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
            lks = quantile_loss_gradient(y_true[t], agg_quantile_t, q) .* forecasters_preds_t

            weights_history[:, t] = weights_history[:, t-1] .- learning_rate .* lks
            weights_history[:, t] = project_to_simplex(weights_history[:, t])
            end

            return weights_history
    end
end
