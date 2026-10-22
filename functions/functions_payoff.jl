module UtilsFunctionsPayoff

using LinearAlgebra
using Combinatorics

export payoff_update, get_subsets, get_subsets_excluding_players, mean_squared_error, in_sample_rewards, out_of_sample_rewards

    # Exponentially weighted average, also used for the long-term scores S_t = lambda * S_{t-1} + (1 - lambda) * sc_t
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
        in_sample_rewards(long_term_scores, active, budget)

    Equal-stake competitive scoring rule (Lambert et al., Lemma 4) applied to the long-term scores
    (EMA of the normalized quantile scores, in [0, 1]): seller i gets `(budget / N) * (1 + S_i - mean_{j != i} S_j)`
    over the N active sellers. Same rule as `out_of_sample_rewards`, so it is budget-balanced and non-negative;
    a single active seller gets the whole budget and missing sellers get zero.
    """
    in_sample_rewards(long_term_scores, active, budget) = out_of_sample_rewards(long_term_scores, active, budget)

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