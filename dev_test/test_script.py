import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 10)
# t0 = 10
# times = t0*1/np.exp(10)*np.exp(x)

# plt.plot(x, times)
# plt.show()
s = Path('./data/ny2')
print(s.name)