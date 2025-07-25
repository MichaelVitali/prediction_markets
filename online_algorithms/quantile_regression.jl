module QuantileRegression

    using LinearAlgebra
    using Statistics

    include("../functions/functions.jl")
    include("../data_generation/DataGeneration.jl")
    using .UtilsFunctions
    using .DataGeneration

export online_quantile_regression, online_quantile_regression_update, online_quantile_regression_update_multiple_lead_times

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

    function online_quantile_regression_update(forecasters_preds, prev_forecaster_weights, y_true, q, learning_rate=0.01)

        agg_quantile_t = sum(forecasters_preds .* prev_forecaster_weights)
        lks = quantile_loss_gradient(y_true, agg_quantile_t, q) .* forecasters_preds

        new_weights = prev_forecaster_weights .- learning_rate .* lks
        new_weights = project_to_simplex(new_weights)

        return new_weights, agg_quantile_t

    end

    function online_quantile_regression_update_multiple_lead_times(forecasters_preds, prev_forecaster_weights, y_true, q, learning_rate=0.01)

        agg_quantile_t = sum(forecasters_preds .* prev_forecaster_weights, dims=1)[1]
        gradient_loss = quantile_loss_gradient.(y_true, agg_quantile_t, q)
        lks = [mean(row .* gradient_loss) for row in forecasters_preds]

        new_weights = prev_forecaster_weights .- learning_rate .* lks
        new_weights = project_to_simplex(new_weights)

        return new_weights, agg_quantile_t

    end
end
