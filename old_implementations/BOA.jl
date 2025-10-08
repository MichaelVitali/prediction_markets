module BOA

    using LinearAlgebra

    include("../functions/functions.jl")
    using .UtilsFunctions

export bernstein_online_aggregation, bernstein_online_aggregation_update

    function bernstein_online_aggregation(forecasters_preds, forecaster_weights, y_true, T, q)

        forecasters_names = collect(keys(forecasters_preds))
        weights_history = Matrix{Float64}(undef, length(forecaster_weights), T)
        weights_history[:, 1] = forecaster_weights
        avg_quantiles_history = zeros(T-1)
        learning_rate = 0.01

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in forecasters_names]
            agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
            lks = quantile_loss_gradient(y_true[t], agg_quantile_t, q) .* (forecasters_preds_t .- agg_quantile_t)

            numerators = weights_history[:, t-1] .* exp.(-learning_rate .* lks .* (1 .- (learning_rate .* lks)))
            weights_history[:, t] = numerators ./ sum(numerators)
            avg_quantiles_history[t-1] = agg_quantile_t
        end

        return weights_history
    end

    function bernstein_online_aggregation_update(forecasters_preds, prev_forecaster_weights, y_true, q, learning_rate=0.01)

        agg_quantile_t = sum(forecasters_preds .* prev_forecaster_weights)
        lks = quantile_loss_gradient(y_true, agg_quantile_t, q) .* (forecasters_preds .- agg_quantile_t)

        numerators = prev_forecaster_weights .* exp.(-learning_rate .* lks .* (1 .- (learning_rate .* lks)))
        new_weights = numerators ./ sum(numerators)

        return new_weights
    end

end