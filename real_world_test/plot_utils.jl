# Plotting helpers shared by the real-world test drivers (main_rewards.jl, main_missing_rates.jl).
using RollingFunctions

# Colors kept per model; ordering matches Table II (ECMWF, NOAA, DWD -> QRF, MLP, XGB).
model_colors_dict = Dict(
    "ecmwf_qrf" => "lime",
    "ecmwf_mlp" => "green",
    "ecmwf_xgb" => "cyan",
    "noaa_qrf" => "navy",
    "noaa_mlp" => "orange",
    "noaa_xgb" => "magenta",
    "dwd_qrf" => "brown",
    "dwd_mlp" => "purple",
    "dwd_xgb" => "gold"
)

# Line style distinguishes provider without relying only on color.
model_dash_dict = Dict(
    "ecmwf" => "solid",
    "noaa" => "dash",
    "dwd" => "dot"
)
get_dash(name) = get(model_dash_dict, String(split(name, "_")[1]), "solid")

moving_avg(v, window) = runmean(v, window)
