module Shapley

    using LinearAlgebra
    using Combinatorics
    using LossFunctions

    include("../functions/functions_payoff.jl")
    using .UtilsFunctionsPayoff

export shapley_payoff

    function shapley_payoff(forecasters_preds, weights_combination, y_true, q)

        n_forecasters = length(forecasters_preds)
        shapley_values = zeros(n_forecasters)

        # Calculate Shapley value for each forecaster
        for f in 1:n_forecasters
            value = 0.0
            subsets = get_subsets_excluding_players(forecasters_preds, f)   #Get subset of forecasters excluding f
            subsets_weights = get_subsets_excluding_players(weights_combination, f) #Get subset of forecasters weights excluding the one of f

            # Calculate relative value for each subset
            for (i, sub) in enumerate(subsets)
                w = length(sub)
                factor = factorial(w) * factorial((n_forecasters - w - 1)) / factorial(n_forecasters)
                sub_weights = subsets_weights[i]

                # Calculate loss of subset combined forecast
                subset_forecast = sum(sub .* sub_weights)
                loss_subset = QuantileLoss(q).(subset_forecast, y_true)

                # Calculate loss of subset combined forecast including forecaster f
                expanded_subset = push!(sub, forecasters_preds[f])
                expanded_weights = push!(sub_weights, weights_combination[f])
                expanded_forecast = sum(expanded_subset .* expanded_weights)
                loss_expanded = QuantileLoss(q).(expanded_forecast, y_true)

                # Update shapley value
                value += factor * (loss_subset - loss_expanded)
            end

            ## Calculate value for empty set
            w = 0
            factor = factorial(w) * factorial((n_forecasters - w - 1)) / factorial(n_forecasters)

            subset_forecast = 0.0
            loss_subset = QuantileLoss(q).(subset_forecast, y_true)
            expanded_forecast = forecasters_preds[f] * weights_combination[f]
            loss_expanded = QuantileLoss(q).(expanded_forecast, y_true)
            value += factor * (loss_subset - loss_expanded)

            shapley_values[f] = value
        end

        return shapley_values
    end

end