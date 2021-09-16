import numpy as np

# Parameters for the Lorentz63 model
sample_rate = 1  # sample rate of saved data
dt = 0.01  # the timestep
sdim = 3  # Dimension of dynamical system

# Define some conversion shortcuts
tts = sample_rate / dt
stt = dt / sample_rate


# Arrays
dx_array = np.zeros(sdim, dtype=np.float64)
derivMatrix = np.zeros((sdim, sdim))
