import numpy as np

# Parameters for the Lorentz63 model
sample_rate = 1  # sample rate of saved data
dt = 0.01  # the timestep
sdim = 3  # Dimension of dynamical system
bd_size = 0

# Define some conversion shortcuts
tts = sample_rate / dt
stt = dt / sample_rate

# Define types
dtype = np.float64

# Define u_init_slice
u_init_slice = np.s_[0:-1:1]

# Arrays
du_array = np.zeros(sdim, dtype=np.float64)
derivMatrix = np.zeros((sdim, sdim))
