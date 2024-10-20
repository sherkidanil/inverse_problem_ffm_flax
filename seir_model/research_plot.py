import numpy as np
import matplotlib.pyplot as plt

loss_with_noise = np.load('models/seir_4p_ds_with_noise/losses.npy')
loss_without_noise = np.load('models/seir_4p_ds_without_noise/losses.npy')

plt.figure(figsize=(12,6), dpi=300)
plt.plot(loss_with_noise, label=fr'$x_t$ with noise')
plt.plot(loss_without_noise, label=fr'$x_t$ without noise')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.yscale('log')
plt.savefig('imgs/non_smooth.png')
plt.show()

from scipy.ndimage import gaussian_filter1d

plt.figure(figsize=(12,6), dpi=300)
# , err: $2.2\% \pm 0.15\%$
plt.plot(gaussian_filter1d(loss_with_noise, 1000), label=fr'$x_t$ with noise')
plt.plot(gaussian_filter1d(loss_without_noise, 1000), label=fr'$x_t$ without noise')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.yscale('log')
# plt.grid(True)
plt.savefig('imgs/smooth.png')
plt.show()

plt.figure(figsize=(12,6), dpi=300)
# , err: $2.2\% \pm 0.15\%$
plt.plot(gaussian_filter1d(loss_with_noise, 1000), label=fr'$x_t$ with noise')
plt.plot(gaussian_filter1d(loss_without_noise, 1000), label=fr'$x_t$ without noise')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.yscale('log')
# plt.grid(True)
plt.savefig('imgs/smooth_log.png')
plt.show()