module RealWorldtestData

using DataFrames
using Parquet2
using Dates
using Normalization
using CSV

export preprocessing_forecasts

    function preprocessing_forecasts(path_ecmwf, path_noaa, path_elia, q)

        forecasters_dict = Dict("ecmwf" => [], "noaa" => [])
        true_values = []
        forecast_elia = []

        ######## Extract ECMWF IFS FORECASTS ########
        df_ecmwf = DataFrame(Parquet2.Dataset(path_ecmwf))
        df_ecmwf = dropmissing(df_ecmwf)
        rename!(df_ecmwf, "__index_level_0__" => "datetime")

        first_date = Date(first(df_ecmwf.datetime))
        last_date = Date(last(df_ecmwf.datetime)) - Day(1)
        column_name = Symbol("q$(Int(q * 100))")
        scaler_ecmwf = MinMax(df_ecmwf[!, column_name])

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_ecmwf)
            daily_data = Float64.(daily_data[!, column_name])
            daily_data = scaler_ecmwf(daily_data)
            push!(forecasters_dict["ecmwf"], daily_data)
        end

        ######## Extract NOAA GFS FORECASTS ########
        df_noaa = DataFrame(Parquet2.Dataset(path_noaa))
        df_noaa = dropmissing(df_noaa)
        rename!(df_noaa, "__index_level_0__" => "datetime")

        first_date = Date(first(df_noaa.datetime))
        last_date = Date(last(df_noaa.datetime)) - Day(1)
        column_name = Symbol("q$(Int(q * 100))")
        scaler_noaa = MinMax(df_noaa[!, column_name])

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_noaa)
            daily_data = Float64.(daily_data[!, column_name])
            daily_data = scaler_noaa(daily_data)
            push!(forecasters_dict["noaa"], daily_data)
        end

        ############ Extract True Values ############
        df_ecmwf = DataFrame(Parquet2.Dataset(path_ecmwf))
        df_ecmwf = dropmissing(df_ecmwf)
        rename!(df_ecmwf, "__index_level_0__" => "datetime")

        first_date = Date(first(df_ecmwf.datetime))
        last_date = Date(last(df_ecmwf.datetime)) - Day(1)
        scaler_target = MinMax(df_ecmwf[!, :measured])

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_ecmwf)
            daily_data = daily_data[!, :measured]
            daily_data = scaler_target(daily_data)
            push!(true_values, daily_data)
        end

        ############ Extract Elia Forecasts ############
        df_elia = DataFrame(CSV.File(path_elia))
        df_elia = dropmissing(df_elia)
        transform!(df_elia, :datetime => ByRow(s -> DateTime(replace(s, r"\+00:00$" => ""), "yyyy-mm-ddTHH:MM:SS")) => :datetime)
        sort!(df_elia, :datetime)
        
        if q == 0.5
            column_name = Symbol("dayaheadforecast")
        elseif q == 0.1
            column_name = Symbol("dayaheadconfidence10")
        elseif q == 0.9
            column_name = Symbol("dayaheadconfidence90")
        end
        scaler_elia = MinMax(df_elia[!, column_name])

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_elia)
            daily_data = daily_data[!, column_name]
            daily_data = scaler_elia(daily_data)
            push!(forecast_elia, daily_data)
        end

        return true_values, forecasters_dict, forecast_elia, scaler_ecmwf, scaler_noaa, scaler_target, scaler_elia
    end

end