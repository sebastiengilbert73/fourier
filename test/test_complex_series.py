import logging
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/fourier")
import complex_series as cs

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info(f"test_complex_series.main()")

    length = 2.0
    reconstruction_length = 101
    reconstructed_delta_x = length / (reconstruction_length - 1)
    reconstructed_xs = np.arange(0, length + reconstructed_delta_x, reconstructed_delta_x)

    signal = np.array([1.0, 0.25, 0, 0.5, 1.5, 2.1, 2.0, 1.3, 1.0, 0.9, 0.7, 0.2, -0.5, -1.6])
    delta_x = length/(len(signal) - 1)
    xs = np.arange(0, length + delta_x/2, delta_x)

    logging.info(f"signal:\n{signal}")

    # Duplicate half-range expansion
    duplicate_expander = cs.Expander(length=length, expansion_type='duplicate')
    c_n_duplicate = duplicate_expander.coefficients(signal, 100)
    ns = np.arange(-(len(c_n_duplicate)//2), len(c_n_duplicate)//2 + 0.5, 1)
    duplicate_reconstructed_signal = duplicate_expander.reconstruct(c_n_duplicate, reconstruction_length)

    # Odd half-range expansion
    odd_expander = cs.Expander(length=length, expansion_type='odd')
    c_n_odd = odd_expander.coefficients(signal, 100)
    odd_reconstructed_signal = odd_expander.reconstruct(c_n_odd, reconstruction_length)

    # Even half-range expansion
    even_expander = cs.Expander(length=length, expansion_type='even')
    c_n_even = even_expander.coefficients(signal, 100)
    even_reconstructed_signal = even_expander.reconstruct(c_n_even, reconstruction_length)

    # Odd quarter-range expansion
    quarter_odd_expander = cs.Expander(length=length, expansion_type='quarter_odd')
    c_n_quarter_odd = quarter_odd_expander.coefficients(signal, 100)
    quarter_odd_reconstructed_signal = quarter_odd_expander.reconstruct(c_n_quarter_odd, reconstruction_length)
    quarter_odd_ns = np.arange(-(len(c_n_quarter_odd)//2), len(c_n_quarter_odd)//2 + 0.5, 1)

    # Even quarter-range expansion
    quarter_even_expander = cs.Expander(length=length, expansion_type='quarter_even')
    c_n_quarter_even = quarter_even_expander.coefficients(signal, 100)
    quarter_even_reconstructed_signal = quarter_even_expander.reconstruct(c_n_quarter_even, reconstruction_length)
    quarter_even_ns = np.arange(-(len(c_n_quarter_even) // 2), len(c_n_quarter_even) // 2 + 0.5, 1)

    fig, axs = plt.subplots(4, 3, figsize=(10, 7))
    axs[0, 0].set_title("Duplicate half-range")
    axs[0, 0].plot(xs, signal, label="Original signal")
    axs[0, 0].plot(reconstructed_xs, duplicate_reconstructed_signal.real, label="Reconstruction, real part")
    axs[0, 0].plot(reconstructed_xs, duplicate_reconstructed_signal.imag, label="Reconstruction, imaginary part")
    axs[0, 0].legend()
    axs[0, 0].grid()
    axs[1, 0].bar(ns, c_n_duplicate.real, label="c_n, real part")
    axs[1, 0].bar(ns, c_n_duplicate.imag, label="c_n, imaginary part", width=0.5)
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[0, 1].set_title("Odd half-range")
    axs[0, 1].plot(xs, signal, label="Original signal")
    axs[0, 1].plot(reconstructed_xs, odd_reconstructed_signal.real, label="Reconstruction, real part")
    axs[0, 1].plot(reconstructed_xs, odd_reconstructed_signal.imag, label="Reconstruction, imaginary part")
    axs[0, 1].legend()
    axs[0, 1].grid()
    axs[1, 1].bar(ns, c_n_odd.real, label="c_n, real part")
    axs[1, 1].bar(ns, c_n_odd.imag, label="c_n, imaginary part", width=0.5)
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[0, 2].set_title("Even half-range")
    axs[0, 2].plot(xs, signal, label="Original signal")
    axs[0, 2].plot(reconstructed_xs, even_reconstructed_signal.real, label="Reconstruction, real part")
    axs[0, 2].plot(reconstructed_xs, even_reconstructed_signal.imag, label="Reconstruction, imaginary part")
    axs[0, 2].legend()
    axs[0, 2].grid()
    axs[1, 2].bar(ns, c_n_even.real, label="c_n, real part")
    axs[1, 2].bar(ns, c_n_even.imag, label="c_n, imaginary part", width=0.5)
    axs[1, 2].legend()
    axs[1, 2].grid()

    axs[2, 1].set_title("Odd quarter-range")
    axs[2, 1].plot(xs, signal, label="Original signal")
    axs[2, 1].plot(reconstructed_xs, quarter_odd_reconstructed_signal.real, label="Reconstruction, real part")
    axs[2, 1].plot(reconstructed_xs, quarter_odd_reconstructed_signal.imag, label="Reconstruction, imaginary part")
    axs[2, 1].legend()
    axs[2, 1].grid()
    axs[3, 1].bar(quarter_odd_ns, c_n_quarter_odd.real, label="c_n, real part")
    axs[3, 1].bar(quarter_odd_ns, c_n_quarter_odd.imag, label="c_n, imaginary part", width=0.5)
    axs[3, 1].legend()
    axs[3, 1].grid()

    axs[2, 2].set_title("Even quarter-range")
    axs[2, 2].plot(xs, signal, label="Original signal")
    axs[2, 2].plot(reconstructed_xs, quarter_even_reconstructed_signal.real, label="Reconstruction, real part")
    axs[2, 2].plot(reconstructed_xs, quarter_even_reconstructed_signal.imag, label="Reconstruction, imaginary part")
    axs[2, 2].legend()
    axs[2, 2].grid()
    axs[3, 2].bar(quarter_even_ns, c_n_quarter_even.real, label="c_n, real part")
    axs[3, 2].bar(quarter_even_ns, c_n_quarter_even.imag, label="c_n, imaginary part", width=0.5)
    axs[3, 2].legend()
    axs[3, 2].grid()


    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()