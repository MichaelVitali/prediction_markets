module Shapley

    using LinearAlgebra
    using Combinatorics
    using LossFunctions
    using Statistics

    include("../functions/functions_payoff.jl")
    using .UtilsFunctionsPayoff
    include("../functions/functions.jl")
    using .UtilsFunctions        # project_to_simplex, quantile_loss, quantile_loss_gradient

export shapley_payoff, shapley_payoff_multiple_lead_times, shapley_payoff_multiple_lead_times_refit

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
                loss_subset = QuantileLoss(q).(subset_forecast .- y_true)

                # Calculate loss of subset combined forecast including forecaster f
                expanded_subset = push!(sub, forecasters_preds[f])
                expanded_weights = push!(sub_weights, weights_combination[f])
                expanded_forecast = sum(expanded_subset .* expanded_weights)
                loss_expanded = QuantileLoss(q).(expanded_forecast .- y_true)

                # Update shapley value
                value += factor * (loss_subset - loss_expanded)
            end

            ## Calculate value for empty set
            w = 0
            factor = factorial(w) * factorial((n_forecasters - w - 1)) / factorial(n_forecasters)

            subset_forecast = 0.0
            loss_subset = QuantileLoss(q).(subset_forecast .- y_true)
            expanded_forecast = forecasters_preds[f] * weights_combination[f]
            loss_expanded = QuantileLoss(q).(expanded_forecast .- y_true)
            value += factor * (loss_subset - loss_expanded)

            shapley_values[f] = value
        end

        return shapley_values
    end

    function shapley_payoff_multiple_lead_times(forecasters_preds, weights_combination, y_true, q)

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
                loss_subset = mean(QuantileLoss(q).(subset_forecast .- y_true))

                # Calculate loss of subset combined forecast including forecaster f
                expanded_subset = push!(sub, forecasters_preds[f])
                expanded_weights = push!(sub_weights, weights_combination[f])
                expanded_forecast = sum(expanded_subset .* expanded_weights)
                loss_expanded = mean(QuantileLoss(q).(expanded_forecast .- y_true))

                # Update shapley value
                value += factor * (loss_subset - loss_expanded)
            end

            ## Calculate value for empty set
            w = 0
            factor = factorial(w) * factorial((n_forecasters - w - 1)) / factorial(n_forecasters)

            subset_forecast = 0.0
            loss_subset = mean(QuantileLoss(q).(subset_forecast .- y_true))
            expanded_forecast = forecasters_preds[f] * weights_combination[f]
            loss_expanded = mean(QuantileLoss(q).(expanded_forecast .- y_true))
            value += factor * (loss_subset - loss_expanded)

            shapley_values[f] = value
        end

        return shapley_values
    end

    """
        coalition_min_loss(subset_preds, y_true, q; lr, iters, tol)

    Value v(S) of a coalition: the *minimum* mean pinball loss its members can reach,
    re-optimizing the combination weights on the simplex (projected sub-gradient
    descent — the same update the online QR uses). This is what makes the Shapley
    payoff incentive-compatible: a coalition is worth its best-fit loss, not the loss
    of some frozen weight vector.

    - empty coalition -> no forecast -> predict 0
    - single member   -> weight must be 1 on the simplex -> its own forecast
    - >= 2 members     -> minimize the loss of the convex combination
    """
    function coalition_min_loss(subset_preds, y_true, q; lr=0.2, iters=500, tol=1e-9)
        k = length(subset_preds)
        k == 0 && return mean(quantile_loss.(y_true, 0.0, q))
        k == 1 && return mean(quantile_loss.(y_true, subset_preds[1], q))

        w = fill(1.0 / k, k)                      # start from the uniform ensemble
        prev = Inf
        agg = sum(w[i] .* subset_preds[i] for i in 1:k)
        for _ in 1:iters
            g = quantile_loss_gradient.(y_true, agg, q)        # dLoss/dagg (per lead time)
            grad = [mean(subset_preds[i] .* g) for i in 1:k]   # dLoss/dw_i
            w = project_to_simplex(w .- lr .* grad)            # gradient step + simplex projection
            agg = sum(w[i] .* subset_preds[i] for i in 1:k)
            loss = mean(quantile_loss.(y_true, agg, q))
            abs(prev - loss) < tol && break                    # early stop once it plateaus
            prev = loss
        end
        return mean(quantile_loss.(y_true, agg, q))
    end

    """
        shapley_payoff_multiple_lead_times_refit(forecasters_preds, weights_combination, y_true, q)

    Drop-in replacement for `shapley_payoff_multiple_lead_times` that values every
    coalition with `coalition_min_loss` (weights re-optimized per coalition) instead of
    the frozen `weights_combination`. `weights_combination` is kept in the signature only
    for interchangeability with the original — it is not used here.
    """
    function shapley_payoff_multiple_lead_times_refit(forecasters_preds, weights_combination, y_true, q)

        n_forecasters = length(forecasters_preds)

        # Memoize coalition values so every subset is re-optimized only once.
        cache = Dict{Vector{Int}, Float64}()
        vloss(idxs) = get!(cache, sort(idxs)) do
            coalition_min_loss([forecasters_preds[i] for i in sort(idxs)], y_true, q)
        end

        shapley_values = zeros(n_forecasters)
        for f in 1:n_forecasters
            others = [j for j in 1:n_forecasters if j != f]
            value = 0.0
            for sub in powerset(others)          # every subset of the others, incl. []
                w = length(sub)
                factor = factorial(w) * factorial(n_forecasters - w - 1) / factorial(n_forecasters)
                value += factor * (vloss(sub) - vloss(vcat(sub, f)))   # marginal loss drop from adding f
            end
            shapley_values[f] = value
        end

        return shapley_values
    end

end