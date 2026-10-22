# Plotting helpers shared by the real-world test drivers (main_rewards.jl, main_missing_rates.jl).
using RollingFunctions

# Color encodes the weather provider and line style the ML model, so the 9 sellers ({ECMWF, NOAA, DWD} x {QRF, MLP, XGB},
# ordering as in Table II) read as 3 x 3. The three colors pass colorblind-separation checks for every pair.
provider_colors = Dict(
    "ecmwf" => "#2a78d6",   # blue
    "noaa" => "#eb6834",    # orange
    "dwd" => "#1baf7a"      # green
)
model_colors_dict = Dict(
    "ecmwf_qrf" => provider_colors["ecmwf"],
    "ecmwf_mlp" => provider_colors["ecmwf"],
    "ecmwf_xgb" => provider_colors["ecmwf"],
    "noaa_qrf" => provider_colors["noaa"],
    "noaa_mlp" => provider_colors["noaa"],
    "noaa_xgb" => provider_colors["noaa"],
    "dwd_qrf" => provider_colors["dwd"],
    "dwd_mlp" => provider_colors["dwd"],
    "dwd_xgb" => provider_colors["dwd"]
)

# Line style distinguishes the ML model within a provider.
model_dash_dict = Dict(
    "qrf" => "solid",
    "mlp" => "dash",
    "xgb" => "dot"
)
get_dash(name) = get(model_dash_dict, String(split(name, "_")[2]), "solid")

moving_avg(v, window) = runmean(v, window)
