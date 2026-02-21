import logging
import sys
sys.path.append("../src/fourier")
import series as fourier_series
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("reconstruct.main()")

    signal = np.array([0, 1, 0])
    expander = fourier_series.Expander(length=1.0, expansion_type='odd')
    a, b = expander.coefficients(signal, maximum_n=1000)
    logging.info(f"a = {a}\n\tb = {b}")

    reconstruction = expander.reconstruct(a, b, signal_length=len(signal))
    logging.info(f"reconstruction = {reconstruction}")


if __name__ == '__main__':
    main()