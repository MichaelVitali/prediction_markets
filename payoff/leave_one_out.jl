module LOO

using LinearAlgebra
using LossFunctions

export leave_one_out_payoff

    function leave_one_out_payoff(forecasters_preds, weights_combination, y_true, q)

        n_forecasters = length(forecasters_preds)
        combined_forecast = sum(forecasters_preds .* weights_combination)

        combined_loss = QuantileLoss(q).(combined_forecast, y_true)
        forecasters_losses = zeros(n_forecasters)

        for i in 1:n_forecasters
            subset_forecasters = [forecasters_preds[j] for j in 1:n_forecasters if i != j]
            subset_weights = [weights_combination[j] for j in 1:n_forecasters if i != j]

            subset_forecast = sum(subset_forecasters .* subset_weights)
            forecasters_losses[i] = QuantileLoss(q).(subset_forecast, y_true)
        end

        phis = forecasters_losses .- combined_loss
        return phis
    end

end