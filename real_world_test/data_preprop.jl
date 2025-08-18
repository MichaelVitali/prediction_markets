module RealWorldtestData

using DataFrames
using Parquet2
using Dates

export preprocessing_forecasts

    function preprocessing_forecasts(path_ecmwf, path_noaa, q)

        forecasters_dict = Dict("ecmwf" => [], "noaa" => [])
        true_values = []

        ######## Extract ECMWF IFS FORECASTS ########
        df_ecmwf = DataFrame(Parquet2.Dataset(path_ecmwf))
        df_ecmwf = dropmissing(df_ecmwf)
        rename!(df_ecmwf, "__index_level_0__" => "datetime")

        first_date = Date(first(df_ecmwf.datetime))
        last_date = Date(last(df_ecmwf.datetime)) - Day(1)
        column_name = Symbol("q$(Int(q * 100))")

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_ecmwf)
            daily_data = Float64.(daily_data[!, column_name])
            push!(forecasters_dict["ecmwf"], daily_data)
        end

        ######## Extract NOAA GFS FORECASTS ########
        df_noaa = DataFrame(Parquet2.Dataset(path_noaa))
        df_noaa = dropmissing(df_noaa)
        rename!(df_noaa, "__index_level_0__" => "datetime")

        first_date = Date(first(df_noaa.datetime))
        last_date = Date(last(df_noaa.datetime)) - Day(1)
        column_name = Symbol("q$(Int(q * 100))")

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_noaa)
            daily_data = Float64.(daily_data[!, column_name])
            push!(forecasters_dict["noaa"], daily_data)
        end

        ############ Extract True Values ############
        df_ecmwf = DataFrame(Parquet2.Dataset(path_ecmwf))
        df_ecmwf = dropmissing(df_ecmwf)
        rename!(df_ecmwf, "__index_level_0__" => "datetime")

        first_date = Date(first(df_ecmwf.datetime))
        last_date = Date(last(df_ecmwf.datetime)) - Day(1)

        for day in first_date:Day(1):last_date
            start_time = DateTime(day) + Hour(22)
            end_time = DateTime(day) + Day(1) + Minute(45) + Hour(21)

            daily_data = filter(row -> start_time <= row.datetime <= end_time, df_ecmwf)
            daily_data = daily_data[!, :measured]
            push!(true_values, daily_data)
        end

        return true_values, forecasters_dict
    end

end