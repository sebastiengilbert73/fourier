import logging
import sys
sys.path.append("../src/fourier")
import series as fourier_series
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("reconstruct.main()")

    xs = np.linspace(0, 1, 101)
    signal = xs**2 * np.sin(xs * np.pi)
    expander = fourier_series.Expander(length=1.0, expansion_type='odd')
    a, b = expander.coefficients(signal, maximum_n=1000)
    logging.info(f"a = {a}\n\tb = {b}")

    reconstruction = expander.reconstruct(a, b, signal_length=101)
    logging.info(f"reconstruction = {reconstruction}")

    plt.plot(xs, signal, label='original', lw=2)
    plt.plot(xs, reconstruction, label='reconstruction')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()