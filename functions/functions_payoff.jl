module UtilsFunctionsPayoff

using LinearAlgebra
using Combinatorics

export payoff_update, get_subsets, get_subsets_excluding_players, mean_squared_error, in_sample_rewards, out_of_sample_rewards

    function payoff_update(prev_payoffs, new_payoffs, lambda)

        payoffs = lambda .* prev_payoffs .+ (1-lambda) .* new_payoffs
        return payoffs
    end

    function get_subsets(coalition)

        subsets = combinations(coalition)
        return subsets

    end

    function get_subsets_excluding_players(coalition, idx_player)

        coalition_no_player = [p for (i, p) in enumerate(coalition) if i != idx_player]
        subsets = collect(get_subsets(coalition_no_player))

        return subsets
    end

    function mean_squared_error(y_hat, y_true)
        return (y_hat .- y_true).^2
    end

    """
        in_sample_rewards(payoffs, active, budget)

    Split `budget` among the active sellers (`active[i] == true`) proportionally to the positive part of
    their (smoothed) payoffs. If no active seller has a positive payoff, the budget is split uniformly
    among them. Missing sellers get zero (zero-element property).
    """
    function in_sample_rewards(payoffs, active, budget)
        rewards = zeros(length(payoffs))
        positive = max.(0, payoffs[active])
        if sum(positive) > 0
            rewards[active] = budget .* (positive ./ sum(positive))
        else
            rewards[active] .= budget / count(active)
        end
        return rewards
    end

    """
        out_of_sample_rewards(scores, active, budget)

    Leave-one-out wagering mechanism (Lambert et al., Thm 2) with equal wagers `budget / N` over the N active
    sellers: seller i gets `(budget / N) * (1 + s_i - mean_{j != i} s_j)`. Truthful, budget-balanced over the
    active set and non-negative for scores in [0, 1]. A single active seller gets the whole budget; missing
    sellers get zero (zero-element property).
    """
    function out_of_sample_rewards(scores, active, budget)
        rewards = zeros(length(scores))
        s = scores[active]
        n_active = length(s)
        if n_active >= 2
            loo_mean = (sum(s) .- s) ./ (n_active - 1)   # mean score of the other active sellers
            rewards[active] = (budget / n_active) .* (1 .+ s .- loo_mean)
        elseif n_active == 1
            rewards[active] .= budget
        end
        return rewards
    end
end