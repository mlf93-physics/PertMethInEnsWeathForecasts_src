import numpy as np

# Parameters for the Lorentz63 model
sample_rate = 1  # sample rate of saved data
dt = 0.01  # the timestep
bd_size = 0

# Parameters for perturbations
# For the Lorentz63 model, some perturbations can settle down to reference if
# norm is ~ 1e-14
seeked_error_norm = 1e-2

# Define some conversion shortcuts
tts = sample_rate / dt
stt = dt / sample_rate

sdim = 3
# Arrays
du_array = np.zeros(sdim, dtype=np.float64)
lorentz_matrix = np.zeros((sdim, sdim))
