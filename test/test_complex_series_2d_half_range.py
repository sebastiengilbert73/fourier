import logging
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/fourier")
import complex_series_2d as cs2d

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info(f"test_complex_series_2d_half_range.main()")

    Lx = 1.0
    Ly = 1.0
    reconstruction_shapeHW = (31, 31)
    reconstructed_delta_x = Lx / (reconstruction_shapeHW[1] - 1)
    reconstructed_xs = np.arange(0, Lx + reconstructed_delta_x, reconstructed_delta_x)
    reconstructed_delta_y = Ly / (reconstruction_shapeHW[0] - 1)
    reconstructed_ys = np.arange(0, Ly + reconstructed_delta_y, reconstructed_delta_y)

    """signal_HW = (11, 15)
    signal = np.zeros(signal_HW, dtype=float)
    for y in range(signal_HW[0]):
        for x in range(signal_HW[1]):
            r2_1 = (x - 7.0)**2 + (y - 4.0)**2
            signal[y, x] += 2.0 * np.exp((-r2_1/5**2))
            r2_2 = (x - 10)**2 + (y - 13)**2
            signal[y, x] += 1.0 * np.exp((-r2_2 / 6 ** 2))
    """
    signal_HW = (11, 31)
    signal = np.zeros(signal_HW, dtype=float)
    for y_ndx in range(signal_HW[0]):
        y = y_ndx / (signal_HW[0] - 1) * Ly
        for x_ndx in range(signal_HW[1]):
            x = x_ndx/(signal_HW[1] - 1) * Lx
            #signal[y_ndx, x_ndx] = np.sin(6 * np.pi * x/Lx)# * np.cos(2 * np.pi * y/Ly)
            r2_1 = (x - 0.7) ** 2 + (y - 0.4) ** 2
            signal[y_ndx, x_ndx] += 2.0 * np.exp((-r2_1 / 0.7 ** 2))
            r2_2 = (x - 0.3) ** 2 + (y - 0.8) ** 2
            signal[y_ndx, x_ndx] -= 1.0 * np.exp((-r2_2 / 0.6 ** 2))
            signal[y_ndx, x_ndx] += 0.2 * np.cos(8 * np.pi * (x + 2 * y)/Lx)

    delta_x = Lx / (signal_HW[1] - 1)
    xs = np.arange(0, Lx + delta_x / 2, delta_x)
    delta_y = Ly / (signal_HW[0] - 1)
    ys = np.arange(0, Ly + delta_y / 2, delta_y)

    # Duplicate half-range expansion
    duplicate_expander = cs2d.Expander(Lx, Ly, expansion_type='duplicate')
    c_m_n_duplicate = duplicate_expander.coefficients(signal, maximum_m=100, maximum_n=100)
    duplicate_reconstructed_signal = duplicate_expander.reconstruct(c_m_n_duplicate, reconstruction_shapeHW)

    # Odd half-range expansion
    odd_expander = cs2d.Expander(Lx, Ly, expansion_type='odd')
    c_m_n_odd = odd_expander.coefficients(signal, maximum_m=100, maximum_n=100)
    odd_reconstructed_signal = odd_expander.reconstruct(c_m_n_odd, reconstruction_shapeHW)

    # Even half-range expansion
    even_expander = cs2d.Expander(Lx, Ly, expansion_type='even')
    c_m_n_even = even_expander.coefficients(signal, maximum_m=100, maximum_n=100)
    even_reconstructed_signal = even_expander.reconstruct(c_m_n_even, reconstruction_shapeHW)

    fig, axs = plt.subplots(3, 3, figsize=(15, 7))
    left_texts = ["Original signal", "Reconstruction\n(real part)", "c m,n\n(magnitude)"]
    for ax, text in zip(axs[:, 0], left_texts):
        ax.set_ylabel(text, rotation=0, size='large')

    axs[0, 0].set_title("Duplicate half-range")
    im00 = axs[0, 0].imshow(signal, label="Original signal", cmap="viridis")
    plt.colorbar(im00, ax=axs[0, 0])
    im10 = axs[1, 0].imshow(duplicate_reconstructed_signal.real, label="Reconstruction, real part", cmap="viridis")
    plt.colorbar(im10, ax=axs[1, 0])
    im20 = axs[2, 0].imshow(np.absolute(c_m_n_duplicate.T))
    plt.colorbar(im20, ax=axs[2, 0])

    axs[0, 1].set_title("Odd half-range")
    im01 = axs[0, 1].imshow(signal, label="Original signal", cmap="viridis")
    plt.colorbar(im01, ax=axs[0, 1])
    im11 = axs[1, 1].imshow(odd_reconstructed_signal.real, label="Reconstruction, real part", cmap="viridis")
    plt.colorbar(im11, ax=axs[1, 1])
    im21 = axs[2, 1].imshow(np.absolute(c_m_n_odd.T))
    plt.colorbar(im21, ax=axs[2, 1])

    axs[0, 2].set_title("Even half-range")
    im02 = axs[0, 2].imshow(signal, label="Original signal", cmap="viridis")
    plt.colorbar(im02, ax=axs[0, 2])
    im12 = axs[1, 2].imshow(even_reconstructed_signal.real, label="Reconstruction, real part", cmap="viridis")
    plt.colorbar(im12, ax=axs[1, 2])
    im22 = axs[2, 2].imshow(np.absolute(c_m_n_even.T))
    plt.colorbar(im22, ax=axs[2, 2])

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()