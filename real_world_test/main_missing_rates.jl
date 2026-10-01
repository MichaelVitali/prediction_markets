using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using DataStructures
using ProgressBars
using Base.Threads
using Normalization
using Statistics
using Random
using Dates
using DataFrames
using CSV
using RollingFunctions
using PlotlyJS

include("../functions/functions.jl")
include("../online_algorithms/quantile_regression.jl")
include("../online_algorithms/adaptive_robust_quantile_regression.jl")
include("data_preprop.jl")
using .UtilsFunctions
using .QuantileRegression
using .AdaptiveRobustRegression
using .RealWorldtestData

# Aggregation-only run: quantile loss of QR and RQR for different missing rates (no payoffs/rewards).
# QR sees every forecaster, so it is run once per quantile as the complete-data reference.

# Environment Settings (shared ones in settings.jl)
include("settings.jl")
include("plot_utils.jl")
n_experiments = 100
#missing_rates = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
missing_rates = [0.2]
plot_missing_rate = 0.05    # missing rate whose RQR weights are plotted

# Session losses [MW], keyed by (quantile, algorithm, missing_rate) => (n_runs, T) matrix
losses = OrderedDict{Tuple{Float64, String, Float64}, Matrix{Float64}}()
# RQR base weights averaged over the Monte-Carlo runs, keyed by (quantile, missing_rate) => (n_forecasters, T)
weights_rqr = OrderedDict{Tuple{Float64, Float64}, Matrix{Float64}}()
dates = Date[]

Random.seed!(seed)

for q in quantiles
    true_prod, forecasters_preds, scaler, q_dates = preprocessing_forecasts(models_paths, q, lower_bound_mw, upper_bound_mw)
    if isempty(dates)
        global dates = q_dates
    end

    #################### Individual models ####################
    for name in model_names
        losses_model = zeros(1, T)
        for t in 2:T
            losses_model[1, t] = mean(quantile_loss.(true_prod[t], denormalize(forecasters_preds[name][t], scaler), q))
        end
        losses[(q, uppercase(name), 0.0)] = losses_model
    end

    #################### Quantile Regression ####################
    losses_qr = zeros(1, T)
    weights_qr = initialize_weights(n_forecasters)
    for t in 2:T
        forecasters_preds_t = [forecasters_preds[f][t] for f in model_names]
        y_true = true_prod[t]

        aggregated_forecast_t = qr_aggregate(forecasters_preds_t, weights_qr)
        weights_qr = qr_update(forecasters_preds_t, weights_qr, scaler(y_true), aggregated_forecast_t, q, lr, batch_percentage)
        losses_qr[1, t] = mean(quantile_loss.(y_true, denormalize(aggregated_forecast_t, scaler), q))
    end
    losses[(q, "QR", 0.0)] = losses_qr

    #################### Robust Quantile Regression ####################
    for missing_rate in missing_rates
        losses_rqr = zeros(n_experiments, T)
        weights_mc = zeros(n_forecasters, T)
        data_lock = ReentrantLock()

        Threads.@threads for exp in ProgressBar(1:n_experiments)
            # Missingness pattern: at least one forecaster is always available
            alpha = Int.(rand(n_forecasters, T) .< missing_rate)
            for t in 1:T
                if sum(alpha[:, t]) == length(alpha[:, t])
                    idx = rand(1:length(alpha[:, t]))
                    alpha[idx, t] = 0
                end
            end

            weights_exp = zeros(n_forecasters, T)
            weights_exp[:, 1] = initialize_weights(n_forecasters)
            D_exp = zeros(n_forecasters, n_forecasters)

            for t in 2:T
                forecasters_preds_t = [forecasters_preds[f][t] for f in model_names]
                y_true = true_prod[t]

                aggregated_forecast_t = rqr_aggregate(forecasters_preds_t, weights_exp[:, t-1], D_exp, alpha[:, t])
                weights_exp[:, t], D_exp = rqr_update(forecasters_preds_t, scaler(y_true), weights_exp[:, t-1], D_exp, alpha[:, t], aggregated_forecast_t, q, lr, batch_percentage)
                losses_rqr[exp, t] = mean(quantile_loss.(y_true, denormalize(aggregated_forecast_t, scaler), q))
            end

            lock(data_lock) do
                weights_mc .+= weights_exp
            end
        end
        losses[(q, "RQR", missing_rate)] = losses_rqr
        weights_rqr[(q, missing_rate)] = weights_mc ./ n_experiments
    end
end

#################### Save results ####################
eval_range = (burn_in_period + 1):T

# Summary: mean session loss after burn-in; std across Monte-Carlo runs of the per-run mean loss
summary = DataFrame(quantile=Float64[], algorithm=String[], missing_rate=Float64[], mean_loss=Float64[], std_loss=Float64[])
for ((q, algo, missing_rate), l) in losses
    run_means = vec(mean(l[:, eval_range], dims=2))
    push!(summary, (q, algo, missing_rate, mean(run_means), length(run_means) > 1 ? std(run_means) : 0.0))
end

# Per-session losses averaged over the Monte-Carlo runs
sessions = DataFrame(date=Date[], session=Int[], quantile=Float64[], algorithm=String[], missing_rate=Float64[], mean_loss=Float64[], std_loss=Float64[])
for ((q, algo, missing_rate), l) in losses
    for t in 2:T
        push!(sessions, (dates[t], t, q, algo, missing_rate, mean(l[:, t]), size(l, 1) > 1 ? std(l[:, t]) : 0.0))
    end
end

CSV.write(joinpath(results_dir, "missing_rates_summary.csv"), summary)
CSV.write(joinpath(results_dir, "missing_rates_sessions.csv"), sessions)

#################### Print results ####################
mean_loss(q, algo, missing_rate) = only(summary[(summary.quantile .== q) .& (summary.algorithm .== algo) .& (summary.missing_rate .== missing_rate), :mean_loss])

println("\n############ AVG QUANTILE LOSS [MW] (after burn-in) ############")
println(rpad("", 14), join([lpad("q=$q", 10) for q in quantiles]), lpad("avg", 10))
function print_row(label, values)
    println(rpad(label, 14), join([lpad(string(round(v, digits=2)), 10) for v in values]), lpad(string(round(mean(values), digits=2)), 10))
end
for name in model_names
    print_row(uppercase(name), [mean_loss(q, uppercase(name), 0.0) for q in quantiles])
end
print_row("QR", [mean_loss(q, "QR", 0.0) for q in quantiles])
for missing_rate in missing_rates
    print_row("RQR $(Int(missing_rate * 100))%", [mean_loss(q, "RQR", missing_rate) for q in quantiles])
end

#################### PLOTTING RQR WEIGHTS BY QUANTILE ####################
# Same figure as in main_rewards.jl, for the RQR weights at `plot_missing_rate` (helpers in plot_utils.jl)

n_rows = length(quantiles)
p_weights_combined = make_subplots(
    rows = n_rows,
    cols = 1,
    subplot_titles = reshape(["Weights for Quantile $q" for q in quantiles], n_rows, 1),
    vertical_spacing = 0.10,
    shared_xaxes = true
)

for (r, q) in enumerate(quantiles)
    for (i, name) in enumerate(model_names)
        trace = scatter(
            x = dates[(burn_in_period+1):T],
            y = moving_avg(weights_rqr[(q, plot_missing_rate)][i, (burn_in_period+1):T], window),
            name = uppercase(name),
            mode = "lines",
            line = attr(width=1.5, color=get(model_colors_dict, name, "gray"), dash=get_dash(name)),
            legendgroup = name,
            showlegend = (r == 1)
        )
        add_trace!(p_weights_combined, trace, row=r, col=1)
    end
end

# Shared y-limits across all quantile panels (same y-ticks for every subplot)
ylims_weights = collect(padded_ylims(
    [moving_avg(weights_rqr[(q, plot_missing_rate)][i, (burn_in_period+1):T], window) for q in quantiles for i in 1:n_forecasters]...
))

layout_updates = Dict{Symbol, Any}(
    :height => 500 * n_rows,
    :width => 1200,
    :paper_bgcolor => "white",
    :plot_bgcolor => "white",
    :xaxis_gridcolor => "lightgray", :xaxis_linecolor => "black",
    :xaxis2_gridcolor => "lightgray", :xaxis2_linecolor => "black",
    :xaxis3_gridcolor => "lightgray", :xaxis3_linecolor => "black",
    :yaxis_gridcolor => "lightgray", :yaxis_linecolor => "black",
    :yaxis2_gridcolor => "lightgray", :yaxis2_linecolor => "black",
    :yaxis3_gridcolor => "lightgray", :yaxis3_linecolor => "black",
    :legend => attr(
        orientation = "h",
        yanchor = "top",
        y = -0.2,
        xanchor = "center",
        x = 0.5,
        bgcolor = "rgba(0,0,0,0)",
        font = attr(size=14)
    ),
    :margin => attr(l=55, r=10, t=40, b=110)
)

last_xaxis_name = (n_rows == 1) ? "xaxis" : "xaxis$(n_rows)"
layout_updates[Symbol("$(last_xaxis_name)_tickformat")] = "%b"
layout_updates[Symbol("$(last_xaxis_name)_title")] = attr(text="Time [15min]", font=attr(size=18))
layout_updates[Symbol("$(last_xaxis_name)_title_standoff")] = 20
layout_updates[Symbol("$(last_xaxis_name)_tickfont")] = attr(size=16)
layout_updates[:yaxis_tickfont]  = attr(size=16)
layout_updates[:yaxis2_tickfont] = attr(size=16)
layout_updates[:yaxis3_tickfont] = attr(size=16)
for r in 1:n_rows
    ax = r == 1 ? "yaxis" : "yaxis$(r)"
    layout_updates[Symbol("$(ax)_range")] = ylims_weights
end
relayout!(p_weights_combined, layout_updates)

existing_annotations = p_weights_combined.plot.layout[:annotations]
ylabel_annotation = attr(
    text = "Weights [p.u.]",
    x = -0.07,
    xref = "paper",
    y = 0.5,
    yref = "paper",
    showarrow = false,
    textangle = -90,
    font_size = 18,
    xanchor = "center",
    yanchor = "middle"
)
relayout!(p_weights_combined, annotations = vcat(existing_annotations, [ylabel_annotation]))

savefig(p_weights_combined, joinpath(plot_dir, "weights_distribution_miss$(Int(plot_missing_rate * 100)).pdf"))
display(p_weights_combined)
println("\nResults saved to $results_dir")