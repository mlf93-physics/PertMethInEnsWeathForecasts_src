import numpy as np

ks = np.array([2 ** n for n in range(1, 31)], np.int64)

wavelength = 2 * np.pi / ks
norm_wavelength = wavelength / wavelength[0]

dearth = 40.075e6

spatial_wavelength = norm_wavelength * dearth

print("spatial_wavelength", spatial_wavelength)
