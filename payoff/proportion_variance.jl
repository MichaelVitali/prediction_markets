module ProportionVariance

    using LinearAlgebra

export proportion_variance_payoff

    function proportion_variance_payoff(weights)

        phis = weights.^2 ./ sum(weights.^2)
        return phis
    end

end