module UtilsFunctionsPayoff

using LinearAlgebra
using Combinatorics

export payoff_update, get_subsets, get_subsets_excluding_players, mean_squared_error

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
end