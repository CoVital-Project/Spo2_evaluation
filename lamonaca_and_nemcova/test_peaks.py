import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

x = np.sin(2*np.pi*(2**np.linspace(2,10,1000))*np.arange(1000)/48000) + np.random.normal(0, 1, 1000) * 0.15

print(x)
peaks, _ = find_peaks(x, distance=20)
peaks2, _ = find_peaks(x, prominence=1)      # BEST!
peaks3, paek3_property = find_peaks(x, height=0, prominence=1)
peaks4, _ = find_peaks(x, threshold=0.4)     # Required vertical distance to its direct neighbouring samples, pretty useless

print(paek3_property["peak_heights"])
print(paek3_property["prominences"])
print(paek3_property["left_bases"])

left_bases = paek3_property["left_bases"]
bottom_values = x[left_bases]
print("peaks", x[peaks3])
print("bottom val", bottom_values)
peak_heights = x[peaks3] - bottom_values
print("peak hieight", peak_heights, " mean ", peak_heights.mean())



plt.subplot(2, 2, 1)
plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['distance'])
plt.subplot(2, 2, 2)
plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
plt.subplot(2, 2, 3)
plt.plot(peaks3, x[peaks3], "vg"); plt.plot(x); plt.legend(['height and prom'])
plt.subplot(2, 2, 4)
plt.plot(peaks4, x[peaks4], "xk"); plt.plot(x); plt.legend(['threshold'])
plt.show()
