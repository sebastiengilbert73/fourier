import logging
import sys
sys.path.append("../src/fourier")
import series as fourier_series
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("expand_signal.main()")

    L = 0.3
    signal_length = 301
    expander_odd = fourier_series.Expander(L, 'odd')
    expander_even = fourier_series.Expander(L, 'even')
    expander_quarter_odd = fourier_series.Expander(L, 'quarter_odd')
    expander_quarter_even = fourier_series.Expander(L, 'quarter_even')

    signal = np.zeros((signal_length), dtype=float)
    for k in range(len(signal)):
        if k < 50:
            signal[k] = 5.0
        if k > 100 and k < 150:
            signal[k] = k/10.
        if k > 200 and k < 280:
            signal[k] = 25.0
        if k > 280:
            signal[k] = 17.0 - k/28.

    a_n_odd, b_n_odd = expander_odd.coefficients(signal)
    a_n_even, b_n_even = expander_even.coefficients(signal)
    a_n_quarter_odd, b_n_quarter_odd = expander_quarter_odd.coefficients(signal)
    a_n_quarter_even, b_n_quarter_even = expander_quarter_even.coefficients(signal)

    n_max = 50 #len(a_n_odd) - 1
    delta_x = L/(signal_length - 1)
    xs = np.arange(0, L + delta_x/2, delta_x)

    reconstruction_odd = expander_odd.reconstruct(a_n_odd[0: n_max + 1], b_n_odd[0: n_max + 1], len(xs))
    reconstruction_even = expander_even.reconstruct(a_n_even[0: n_max + 1], b_n_even[0: n_max + 1], len(xs))
    reconstruction_quarter_odd = expander_quarter_odd.reconstruct(a_n_quarter_odd[0: n_max + 1], b_n_quarter_odd[0: n_max + 1], len(xs))
    reconstruction_quarter_even = expander_quarter_even.reconstruct(a_n_quarter_even[0: n_max + 1], b_n_quarter_even[0: n_max + 1], len(xs))

    fig, ax = plt.subplots()
    ax.plot(xs, signal, label='Original signal', linewidth=2)
    ax.plot(xs, reconstruction_odd, label=f'Reconstruction, odd half-range (n_max = {n_max})', linewidth=2)
    ax.plot(xs, reconstruction_even, label=f'Reconstruction, even half-range (n_max = {n_max})', linewidth=2)
    ax.plot(xs, reconstruction_quarter_odd, label=f'Reconstruction, odd quarter-range (n_max = {n_max})', linewidth=2)
    ax.plot(xs, reconstruction_quarter_even, label=f'Reconstruction, even quarter-range (n_max = {n_max})', linewidth=2)
    ax.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    main()