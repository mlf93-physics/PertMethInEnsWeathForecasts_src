# High res model (https://www.ecmwf.int/en/forecasts/documentation-and-support)
resolution = 0.1  # Degrees
n_latitudes = 2560
n_longitudes = resolution * 360
n_vertical_layers = 137

n_degrees_freedom = n_latitudes * n_longitudes * n_vertical_layers

n_variables = 5

print("n_degrees_freedom", n_degrees_freedom * n_variables)
