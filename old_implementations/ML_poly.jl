module ML_Poly

using LinearAlgebra

include("../functions/functions.jl")
using .UtilsFunctions

export ml_poly

    function ml_poly(forecasters_preds, y_true, T, q)

        n_forecasters = length(keys(forecasters_preds))
        regrets = zeros(n_forecasters)
        cum_squared_loss = zeros(n_forecasters)
        weights_history = Matrix{Float64}(undef, n_forecasters, T)
        weights_history[:, 1] = initialize_weights(n_forecasters)

        for t in 2:T
            forecasters_preds_t = [forecasters_preds[f][t] for f in keys(forecasters_preds)]
            agg_quantile_t = sum(forecasters_preds_t .* weights_history[:, t-1])
            lks = quantile_loss_gradient(y_true[t], agg_quantile_t, q) .* forecasters_preds_t
            cum_squared_loss = cum_squared_loss .+ (sum(weights_history[:, t-1] .* lks) .- lks).^2

            regrets = regrets .+ sum(weights_history[:, t-1] .* lks) .- lks
            learning_rates = 1 ./ (1 .+ cum_squared_loss)

            weights_history[:, t] = (learning_rates .* max.(0, regrets)) ./ sum(learning_rates .* max.(0, regrets))        

        end

        return weights_history

    end

end