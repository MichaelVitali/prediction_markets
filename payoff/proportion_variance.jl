module ProportionVariance

    using LinearAlgebra

export proportion_variance_payoff

    function proportion_variance_payoff(weights)

        phis = weights ./ sum(weights)
        return phis
    end

end