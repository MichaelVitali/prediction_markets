using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using DataStructures
using ProgressBars
using Base.Threads
using Plots.PlotMeasures
using Normalization
using RollingFunctions
using PlotlyJS
using Statistics
using Dates
using Random
using DataFrames
using CSV

include("../functions/functions.jl")
include("../functions/functions_payoff.jl")
include("../online_algorithms/quantile_regression.jl")
include("../online_algorithms/adaptive_robust_quantile_regression.jl")
include("../payoff/shapley_values.jl")
include("data_preprop.jl")
using .UtilsFunctions
using .UtilsFunctionsPayoff
using .QuantileRegression
using .Shapley
using .AdaptiveRobustRegression
using .RealWorldtestData

# Environment Settings (shared ones in settings.jl)
include("settings.jl")
include("plot_utils.jl")
n_experiments = 5
total_reward = 100
delta = 0.7
algorithms = ["QR", "RQR"]
missing_rate = 0.05         # RQR missing-submission rate
lambda_payoff = 0.999       # EMA factor of the in-sample (Shapley) payoffs

# Environment Variables
realizations = OrderedDict([q => Vector{Vector{Float64}}() for q in quantiles])
algo_forecasts = OrderedDict([q => OrderedDict([name => [] for name in model_names]) for q in quantiles])
# Results averaged over the Monte-Carlo runs, keyed by (quantile, algorithm) => (n_forecasters, T), losses => (T,)
payoffs = OrderedDict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards = OrderedDict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
weights = OrderedDict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards_in_sample = OrderedDict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
rewards_out_sample = OrderedDict([(q, algo) => zeros(n_forecasters, T) for q in quantiles for algo in algorithms])
losses = OrderedDict([(q, algo) => zeros(T) for q in quantiles for algo in algorithms])

# Wall-clock time [ns] of each market stage per session, keyed by (quantile, algorithm, stage) => (n_experiments, T)
#   aggregation: forecast combination    update: weights (and D) update
#   payoff: in-sample Shapley payoffs, out-of-sample scores and final reward allocation
stages = ["aggregation", "update", "payoff"]
timings = OrderedDict([(q, algo, s) => zeros(n_experiments, T) for q in quantiles for algo in algorithms for s in stages])

dates = Date[]

#################### Loading data, saving predictions and realizations ####################
# Data loaded once per quantile and shared (read-only) by every experiment: quantile => (true_prod, forecasters_preds, scaler)
market_data = OrderedDict{Float64, Any}()
for q in quantiles
    true_prod, forecasters_preds, scaler, q_dates = preprocessing_forecasts(models_paths, q, lower_bound_mw, upper_bound_mw)
    market_data[q] = (true_prod, forecasters_preds, scaler)
    if isempty(dates)
        global dates = q_dates
    end

    for t in ProgressBar(2:T)
        y_true = true_prod[t]

        if isempty(realizations[q])
            push!(realizations[q], zeros(length(y_true)))
        end
        push!(realizations[q], copy(y_true))

        for name in model_names
            if isempty(algo_forecasts[q][name])
                push!(algo_forecasts[q][name], zeros(length(y_true)))
            end
            push!(algo_forecasts[q][name], denormalize(forecasters_preds[name][t], scaler))
        end
    end
end

#################### Quantile Regression ####################
algo = "QR"
data_lock = ReentrantLock()

if algo in algorithms
    for q in quantiles
        quantile_step_reward = total_reward / length(quantiles)
        loss_bound = max(q, 1 - q)

        Threads.@threads for exp in ProgressBar(1:n_experiments)
            # Experiment Variables
            weights_exp = zeros(n_forecasters, T)
            payoffs_exp = zeros(n_forecasters, T)
            rewards_exp = zeros(n_forecasters, T)
            rewards_in_exp = zeros(n_forecasters, T)
            rewards_out_exp = zeros(n_forecasters, T)
            losses_qr_exp = zeros(T)
            timings_exp = OrderedDict([s => zeros(T) for s in stages])

            weights_exp[:, 1] = initialize_weights(n_forecasters)

            true_prod, forecasters_preds, scaler = market_data[q]

            for t in 2:T
                forecasters_preds_t = [forecasters_preds[f][t] for f in model_names]
                y_true = true_prod[t]

                # Forecast combinatotion (session closes), then update once y_true is observed
                t0 = time_ns()
                aggregated_forecast_t = qr_aggregate(forecasters_preds_t, weights_exp[:, t-1])
                timings_exp["aggregation"][t] = time_ns() - t0
                y_true_sc = scaler(y_true)
                t0 = time_ns()
                weights_exp[:, t] = qr_update(forecasters_preds_t, weights_exp[:, t-1], y_true_sc, aggregated_forecast_t, q, lr, batch_percentage)
                timings_exp["update"][t] = time_ns() - t0
                aggregated_forecast_t = denormalize(aggregated_forecast_t, scaler)
                # Loss calculation
                loss_t = mean(quantile_loss.(y_true, aggregated_forecast_t, q))
                losses_qr_exp[t] = loss_t

                t0 = time_ns()
                temp_payoffs = shapley_payoff_multiple_lead_times_refit(forecasters_preds_t, weights_exp[:, t-1], y_true_sc, q)
                forecasters_losses = [mean(quantile_loss.(y_true_sc, forecasters_preds_t[i], q)) for i in 1:n_forecasters]
                temp_scores = 1 .- (forecasters_losses ./ loss_bound)
                payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, lambda_payoff)

                if t > burn_in_period
                    # In-sample: delta share of the budget; out-of-sample: leave-one-out wagering on the rest
                    all_active = trues(n_forecasters)
                    rewards_in = in_sample_rewards(payoffs_exp[:, t], all_active, delta * quantile_step_reward)
                    rewards_out = out_of_sample_rewards(temp_scores, all_active, (1 - delta) * quantile_step_reward)

                    rewards_in_exp[:, t] = rewards_in
                    rewards_out_exp[:, t] = rewards_out
                    rewards_exp[:, t] = rewards_in .+ rewards_out
                end
                timings_exp["payoff"][t] = time_ns() - t0
            end
            lock(data_lock) do
                for s in stages
                    timings[(q, algo, s)][exp, :] = timings_exp[s]
                end
                payoffs[(q, algo)] += payoffs_exp
                weights[(q, algo)] += weights_exp
                rewards[(q, algo)] += rewards_exp
                rewards_in_sample[(q, algo)] += rewards_in_exp
                rewards_out_sample[(q, algo)] += rewards_out_exp
                losses[(q, algo)] += losses_qr_exp
            end
        end
    end

    for q in quantiles, results in (payoffs, weights, rewards, rewards_in_sample, rewards_out_sample, losses)
        results[(q, algo)] ./= n_experiments
    end
end

#################### Robust Quantile Regression ####################
algo = "RQR"
data_lock = ReentrantLock()

# Missing submissions, drawn once per experiment (own seed: independent of threading) and shared by all
# quantiles: a missing seller misses the whole session. At least one seller is always available.
alphas = map(1:n_experiments) do exp
    rng = Xoshiro(seed + exp)
    alpha = Int.(rand(rng, n_forecasters, T) .< missing_rate)
    for t in 1:T
        if sum(alpha[:, t]) == n_forecasters
            alpha[rand(rng, 1:n_forecasters), t] = 0
        end
    end
    alpha
end

if algo in algorithms
    for q in quantiles

        quantile_step_reward = total_reward / length(quantiles)
        loss_bound = max(q, 1 - q)

        Threads.@threads for exp in ProgressBar(1:n_experiments)
            # Experiment Variables
            weights_exp = zeros(n_forecasters, T)
            payoffs_exp = zeros(n_forecasters, T)
            rewards_exp = zeros(n_forecasters, T)
            rewards_in_exp = zeros(n_forecasters, T)
            rewards_out_exp = zeros(n_forecasters, T)
            losses_rqr_exp = zeros(T)
            timings_exp = OrderedDict([s => zeros(T) for s in stages])

            weights_exp[:, 1] = initialize_weights(n_forecasters)

            true_prod, forecasters_preds, scaler = market_data[q]

            D_exp = zeros(n_forecasters, n_forecasters)
            alpha = alphas[exp]

            for t in 2:T
                forecasters_preds_t = [forecasters_preds[f][t] for f in model_names]
                y_true = true_prod[t]

                # Forecast combination (session closes), then update once y_true is observed
                t0 = time_ns()
                aggregated_forecast_t = rqr_aggregate(forecasters_preds_t, weights_exp[:, t-1], D_exp, alpha[:, t])
                timings_exp["aggregation"][t] = time_ns() - t0
                y_true_sc = scaler(y_true)
                t0 = time_ns()
                weights_exp[:, t], new_D = rqr_update(forecasters_preds_t, y_true_sc, weights_exp[:, t-1], D_exp, alpha[:, t], aggregated_forecast_t, q, lr, batch_percentage)
                timings_exp["update"][t] = time_ns() - t0
                prev_D = D_exp
                D_exp = new_D
                aggregated_forecast_t = denormalize(aggregated_forecast_t, scaler)
                # Loss calculation
                loss_t = mean(quantile_loss.(y_true, aggregated_forecast_t, q))
                losses_rqr_exp[t] = loss_t

                # Payoff Calculation
                t0 = time_ns()
                temp_forecasts_t = [forecasters_preds_t[j] for j in 1:n_forecasters if alpha[j, t] == 0]
                temp_weights_t = weights_exp[:, t-1] .+ prev_D * alpha[:, t]
                temp_weights_t = [temp_weights_t[j] for j in 1:n_forecasters if alpha[j, t] == 0]
                temp_weights_t = project_to_simplex(temp_weights_t)

                # At least one forecaster is always available (enforced when drawing alpha)
                temp_payoffs = shapley_payoff_multiple_lead_times_refit(temp_forecasts_t, temp_weights_t, y_true_sc, q)
                forecasters_losses = [mean(quantile_loss.(y_true_sc, temp_forecasts_t[i], q)) for i in 1:length(temp_forecasts_t)]
                temp_scores = 1 .- (forecasters_losses ./ loss_bound)

                if length(temp_payoffs) < n_forecasters
                    for j in findall(a -> a == 1, alpha[:, t])
                        insert!(temp_payoffs, j, 0.0)
                        insert!(temp_scores, j, 0.0)
                    end
                end
                payoffs_exp[:, t] = payoff_update(payoffs_exp[:, t-1], temp_payoffs, lambda_payoff)

                # Reward calculation
                if t > burn_in_period
                    # Only active sellers share the budget; missing ones get zero (zero-element property)
                    active = alpha[:, t] .== 0
                    rewards_in = in_sample_rewards(payoffs_exp[:, t], active, delta * quantile_step_reward)
                    rewards_out = out_of_sample_rewards(temp_scores, active, (1 - delta) * quantile_step_reward)

                    rewards_in_exp[:, t] = rewards_in
                    rewards_out_exp[:, t] = rewards_out
                    rewards_exp[:, t] = rewards_in .+ rewards_out
                end
                timings_exp["payoff"][t] = time_ns() - t0
            end

            lock(data_lock) do
                for s in stages
                    timings[(q, algo, s)][exp, :] = timings_exp[s]
                end
                payoffs[(q, algo)] += payoffs_exp
                weights[(q, algo)] += weights_exp
                rewards[(q, algo)] += rewards_exp
                rewards_in_sample[(q, algo)] += rewards_in_exp
                rewards_out_sample[(q, algo)] += rewards_out_exp
                losses[(q, algo)] += losses_rqr_exp
            end
        end
    end

    for q in quantiles, results in (payoffs, weights, rewards, rewards_in_sample, rewards_out_sample, losses)
        results[(q, algo)] ./= n_experiments
    end
end
#################### Wall-clock timings ####################
# Session 2 is excluded everywhere: it is the first timed call of every experiment and includes JIT compilation.
# Per-session stats [us] over the sessions with rewards; per-experiment totals [s] over sessions 3:T.
function timing_stats(M, session_range, total_range)
    per_session = vec(M[:, session_range]) ./ 1e3
    totals = vec(sum(M[:, total_range], dims=2)) ./ 1e9
    return median(per_session), quantile(per_session, 0.95), mean(totals), length(totals) > 1 ? std(totals) : 0.0
end

timing_summary = DataFrame(quantile=String[], algorithm=String[], stage=String[], median_us=Float64[], p95_us=Float64[], mean_total_s=Float64[], std_total_s=Float64[])
for algo in algorithms, s in stages
    for q in quantiles
        push!(timing_summary, (string(q), algo, s, timing_stats(timings[(q, algo, s)], (burn_in_period+1):T, 3:T)...))
    end
    # Full session: all quantiles together
    push!(timing_summary, ("all", algo, s, timing_stats(sum(timings[(q, algo, s)] for q in quantiles), (burn_in_period+1):T, 3:T)...))
end
CSV.write(joinpath(results_dir, "timings_summary.csv"), timing_summary)

println("\n############ WALL-CLOCK TIMINGS ############")
println("Julia $(VERSION), $(Threads.nthreads()) thread(s), CPU: $(Sys.cpu_info()[1].model), $n_experiments experiment(s)")
show(transform(timing_summary, [:median_us, :p95_us, :mean_total_s, :std_total_s] .=> (c -> round.(c, sigdigits=4)), renamecols=false), allrows=true, allcols=true)
println()

    #################### Calculate Errors ####################
global_loss_qr = zeros(T-1)
global_loss_rqr = zeros(T-1)
global_loss_models = OrderedDict(name => zeros(T-1) for name in model_names)
individual_losses_by_quantile = OrderedDict(q => OrderedDict(name => Float64[] for name in model_names) for q in quantiles)

for q in quantiles

    global_loss_qr .+= losses[(q, "QR")][2:T]
    global_loss_rqr .+= losses[(q, "RQR")][2:T]

    individual_losses = OrderedDict(name => Float64[] for name in model_names)
    for t in 2:T
        y_true = realizations[q][t]

        for name in model_names
            y_pred = algo_forecasts[q][name][t]
            loss_t = mean(quantile_loss.(y_true, y_pred, q))

            global_loss_models[name][t-1] += loss_t
            push!(individual_losses[name], loss_t)
            push!(individual_losses_by_quantile[q][name], loss_t)
        end
    end
    println("\n############ RESULTS QUANTILE $q ############")
    println("Loss Aggregated QR : $(mean(losses[(q, "QR")][(burn_in_period+1):T]))")
    println("Loss Aggregated RQR: $(mean(losses[(q, "RQR")][(burn_in_period+1):T]))")
    
    for name in model_names
        avg_loss = mean(individual_losses[name][burn_in_period:end])
        println("Loss $(uppercase(name)): $avg_loss")
    end
end

########### Plots Settings ##############
# Colors, line styles and moving average in plot_utils.jl
model_name = "ecmwf_xgb"

#################### PLOTTING AVERAGE LOSSES ####################
n_q = length(quantiles)
avg_ts_qr = global_loss_qr ./ n_q
avg_ts_rqr = global_loss_rqr ./ n_q
for name in model_names
    global_loss_models[name] ./= n_q
end

function make_loss_plot(avg_ts_qr, avg_ts_rqr, global_loss_models, model_name, window, dates)
    p = make_subplots(
        rows = 2,
        cols = 1,
        subplot_titles = reshape([
            "Moving Average Loss",
            "Instantaneous Loss",
        ], 2, 1),
        vertical_spacing = 0.08,
        shared_xaxes = true
    )

    # Helper to add all traces for a specific row and transformation
    function add_loss_traces!(p, transform_fn, row; show_legend=(row==1))
        # Preserve original line widths (thinner lines for row 3 instantaneous loss)
        w_agg = row == 2 ? 1.5 : 2.5
        w_mod = row == 2 ? 1.5 : 2.0

        # QR
        add_trace!(p, scatter(
            x = dates,
            y = transform_fn(avg_ts_qr),
            name = "QR",
            line = attr(color="blue", width=w_agg),
            legendgroup = "QR",
            showlegend = show_legend
        ), row=row, col=1)

        # RQR
        add_trace!(p, scatter(
            x = dates,
            y = transform_fn(avg_ts_rqr),
            name = "RQR",
            line = attr(color="red", width=w_agg),
            legendgroup = "RQR",
            showlegend = show_legend
        ), row=row, col=1)

        # Single Model 
        if haskey(global_loss_models, model_name)
            add_trace!(p, scatter(
                x = dates,
                y = transform_fn(global_loss_models[model_name]),
                name = "$(uppercase(model_name))",
                line = attr(color="black", width=w_mod),
                opacity = 0.8,
                legendgroup = model_name,
                showlegend = show_legend
            ), row=row, col=1)
        elseif row == 1 # Print warning only once to avoid spamming the console
            println("Warning: Model $model_name not found in global_loss_models")
        end
    end

    # Add Moving Average (Row 1)
    add_loss_traces!(p, v -> moving_avg(v, window), 1)
    # Add Instantaneous (Row 2)
    add_loss_traces!(p, identity, 2)

    # Shared y-limits so both panels use identical y-ticks (instantaneous drives the span)
    model_series = haskey(global_loss_models, model_name) ? global_loss_models[model_name] : Float64[]
    yl = collect(padded_ylims(avg_ts_qr, avg_ts_rqr, model_series))

    # Layout matched exactly to the reward plot
    relayout!(p,
        height = 1200,
        width  = 1800,
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        xaxis_gridcolor = "lightgray", xaxis_linecolor = "black",
        xaxis2_gridcolor = "lightgray", xaxis2_linecolor = "black",
        yaxis_gridcolor = "lightgray", yaxis_linecolor = "black",
        yaxis2_gridcolor = "lightgray", yaxis2_linecolor = "black",
        yaxis_range = yl, yaxis2_range = yl,
        xaxis2_tickformat = "%b",
        xaxis2_tickfont = attr(size=16),
        xaxis_tickfont = attr(size=16),
        yaxis_tickfont = attr(size=16),
        yaxis2_tickfont = attr(size=16),
        xaxis2_title = attr(text="Time [15min]", font=attr(size=18)),
        hovermode = "x unified",
        legend = attr(
            orientation = "h",
            yanchor = "top",
            y = -0.20,
            xanchor = "center",
            x = 0.5,
            font = attr(size = 16)
        ),
        margin = attr(l=75, r=10, t=40, b=120)
    )

    # Adding shared ylabel annotation
    existing_annotations = p.plot.layout[:annotations]
    ylabel_annotation = attr(
        text = "Avg Quantile Loss [MW]",
        x = -0.10,
        xref = "paper",
        y = 0.5,
        yref = "paper",
        showarrow = false,
        textangle = -90,
        font_size = 18,
        xanchor = "center",
        yanchor = "middle"
    )
    relayout!(p, annotations = vcat(existing_annotations, [ylabel_annotation]))

    return p
end

# Generate, save, and display
p_loss = make_loss_plot(avg_ts_qr[burn_in_period:end], avg_ts_rqr[burn_in_period:end], 
                        OrderedDict(k => v[burn_in_period:end] for (k,v) in global_loss_models), 
                        model_name, window, dates[(burn_in_period+1):T])

savefig(p_loss, joinpath(plot_dir, "loss_analysis.pdf"))
display(p_loss)

#################### PLOTTING WEIGHTS BY QUANTILE ####################
n_rows = length(quantiles)

# 1. Initialize Subplots
p_weights_combined = make_subplots(
    rows = n_rows, 
    cols = 1,
    subplot_titles = reshape(["Weights for Quantile $q" for q in quantiles], n_rows, 1),
    vertical_spacing = 0.10, 
    shared_xaxes = true
)

# 2. Iterate through quantiles and models
for (r, q) in enumerate(quantiles)
    for (i, name) in enumerate(model_names)
        show_leg = (r == 1)
        color = get(model_colors_dict, name, "gray")
        trace = scatter(
            #y = weights[(q, "RQR")][i, :],
            x = dates[(burn_in_period+1):T],
            y = moving_avg(weights[(q, "RQR")][i, (burn_in_period+1):T], window),
            name = uppercase(name),
            mode = "lines",
            line = attr(width=1.5, color=color, dash=get_dash(name)), # Slightly thicker for visibility
            legendgroup = name,
            showlegend = show_leg
        )
        
        add_trace!(p_weights_combined, trace, row=r, col=1)
    end
end

# Shared y-limits across all quantile panels (same y-ticks for every subplot)
ylims_weights = collect(padded_ylims(
    [moving_avg(weights[(q, "RQR")][i, (burn_in_period+1):T], window) for q in quantiles for i in 1:n_forecasters]...
))

# 3. Configure the Global Layout and Legend
total_height = 500 * n_rows
layout_updates = Dict{Symbol, Any}(
    :height => total_height,
    :width => 1200,
    :paper_bgcolor => "white",
    :plot_bgcolor => "white",
    :xaxis_gridcolor => "lightgray", :xaxis_linecolor => "black",
    :xaxis2_gridcolor => "lightgray", :xaxis2_linecolor => "black",
    :xaxis3_gridcolor => "lightgray", :xaxis3_linecolor => "black",
    :yaxis_gridcolor => "lightgray", :yaxis_linecolor => "black",
    :yaxis2_gridcolor => "lightgray", :yaxis2_linecolor => "black",
    :yaxis3_gridcolor => "lightgray", :yaxis3_linecolor => "black",

    # LEGEND CONFIGURATION
    :legend => attr(
        orientation = "h",      
        yanchor = "top",       
        y = -0.2,  # Pushes legend down below the x-axis label
        xanchor = "center",     
        x = 0.5,
        bgcolor = "rgba(0,0,0,0)", # Transparent background
        font = attr(size=14)
    ),
    
    # MARGINS
    :margin => attr(l=55, r=10, t=40, b=110)
)

# Set X-axis label on the very last subplot
last_xaxis_name = (n_rows == 1) ? "xaxis" : "xaxis$(n_rows)"
layout_updates[Symbol("$(last_xaxis_name)_tickformat")] = "%b"
layout_updates[Symbol("$(last_xaxis_name)_title")] = attr(text="Time [15min]", font=attr(size=18))
layout_updates[Symbol("$(last_xaxis_name)_title_standoff")] = 20
layout_updates[Symbol("$(last_xaxis_name)_tickfont")] = attr(size=16)
layout_updates[:yaxis_tickfont]  = attr(size=16)
layout_updates[:yaxis2_tickfont] = attr(size=16)
layout_updates[:yaxis3_tickfont] = attr(size=16)

# Apply the shared y-range to every quantile panel
for r in 1:n_rows
    ax = r == 1 ? "yaxis" : "yaxis$(r)"
    layout_updates[Symbol("$(ax)_range")] = ylims_weights
end

relayout!(p_weights_combined, layout_updates)

# Add shared y-label annotation separately — preserve existing subplot annotations
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

relayout!(p_weights_combined, 
    annotations = vcat(existing_annotations, [ylabel_annotation])
)

# 4. Save and Display
savefig(p_weights_combined, joinpath(plot_dir, "weights_distribution.pdf"))
display(p_weights_combined)

#################### Plot Results ####################
total_rewards_forecasters = OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_in_rewards_forecasters = OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms])
total_out_rewards_forecasters = OrderedDict([algo => zeros(n_forecasters, T) for algo in algorithms])
for algo in algorithms
    for i in 1:n_forecasters
        for q in quantiles
            total_rewards_forecasters[algo][i, :] .+= rewards[(q, algo)][i, :]
            total_in_rewards_forecasters[algo][i, :] .+= rewards_in_sample[(q, algo)][i, :]
            total_out_rewards_forecasters[algo][i, :] .+= rewards_out_sample[(q, algo)][i, :]
        end
    end
end

#################### PLOTTING TOTAL REWARDS ####################

# Per-model time series for QR and RQR
model_rewards_qr  = OrderedDict(model_names[i] => total_rewards_forecasters["QR"][i,  (burn_in_period+1):T] for i in 1:n_forecasters)
model_rewards_rqr = OrderedDict(model_names[i] => total_rewards_forecasters["RQR"][i, (burn_in_period+1):T] for i in 1:n_forecasters)

function make_reward_plot(model_rewards, window, dates, y_range)
    p = make_subplots(
        rows = 2,
        cols = 1,
        subplot_titles = reshape([
            "Moving Average Reward",
            "Instantaneous Reward",
        ], 2, 1),
        vertical_spacing = 0.08,
        shared_xaxes = true
    )

    # Helper to add all traces to a given row
    function add_all_traces!(p, transform_fn, row; show_legend=(row==1))

        # Per-model
        for name in model_names
            color = get(model_colors_dict, name, "gray")
            add_trace!(p, scatter(
                x = dates,
                y = transform_fn(model_rewards[name]),
                name = uppercase(name),
                line = attr(color=color, width=1.5, dash=get_dash(name)),
                opacity = 0.8,
                legendgroup = name,
                showlegend = show_legend
            ), row=row, col=1)
        end
    end

    add_all_traces!(p, v -> moving_avg(v, window), 1)
    add_all_traces!(p, identity, 2)

    relayout!(p,
        height = 1200,
        width  = 1800,
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        xaxis_gridcolor = "lightgray", xaxis_linecolor = "black",
        xaxis2_gridcolor = "lightgray", xaxis2_linecolor = "black",
        yaxis_gridcolor = "lightgray", yaxis_linecolor = "black",
        yaxis2_gridcolor = "lightgray", yaxis2_linecolor = "black",
        yaxis_range = y_range, yaxis2_range = y_range,
        xaxis2_tickformat = "%b",
        xaxis2_tickfont = attr(size=16),
        xaxis_tickfont = attr(size=16),
        yaxis_tickfont = attr(size=16),
        yaxis2_tickfont = attr(size=16),
        xaxis2_title = attr(text="Time [15min]", font=attr(size=18)),
        hovermode = "x unified",
        legend = attr(
            orientation = "h",
            yanchor = "top",
            y = -0.20,
            xanchor = "center",
            x = 0.5,
            font = attr(size = 14)
        ),
        margin = attr(l=55, r=10, t=40, b=120)
    )

    # Adding shared ylabel
    existing_annotations = p.plot.layout[:annotations]
    ylabel_annotation = attr(
        text = "Rewards [£]",
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

    relayout!(p, annotations = vcat(existing_annotations, [ylabel_annotation]))

    return p
end

# Shared y-range across both reward figures (QR and RQR) so they are directly comparable
ylims_reward = collect(padded_ylims(values(model_rewards_qr)..., values(model_rewards_rqr)...))

p_reward_qr  = make_reward_plot(model_rewards_qr,  window, dates[(burn_in_period+1):T], ylims_reward)
p_reward_rqr = make_reward_plot(model_rewards_rqr, window, dates[(burn_in_period+1):T], ylims_reward)

savefig(p_reward_qr,  joinpath(plot_dir, "reward_analysis_qr.pdf"))
savefig(p_reward_rqr, joinpath(plot_dir, "reward_analysis_rqr.pdf"))
display(p_reward_qr)
display(p_reward_rqr)

#################### Print Monthly Rewards ####################
println("\n############ MONTHLY REWARDS (Excluding Burn-in) ############")

for algo in algorithms
    println("\nAlgorithm: $algo")
    
    # Aggregate rewards by month
    monthly_rewards = OrderedDict{Tuple{Int, Int}, Vector{Float64}}()
    
    for t in (burn_in_period + 1):T
        d = dates[t]
        ym = (year(d), month(d))
        
        if !haskey(monthly_rewards, ym)
            monthly_rewards[ym] = zeros(n_forecasters)
        end
        
        monthly_rewards[ym] .+= total_rewards_forecasters[algo][:, t]
    end
    
    # Print results
    for (ym, rewards_vec) in monthly_rewards
        m_name = monthname(ym[2])
        y_val = ym[1]
        println("  Period: $m_name $y_val")
        for (i, name) in enumerate(model_names)
            println("    $(uppercase(name)): $(round(rewards_vec[i], digits=2))")
        end
    end
end
