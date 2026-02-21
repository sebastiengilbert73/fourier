import logging
import sys
sys.path.append("../src/fourier")
import series
import math
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def signal(x):
    return math.sin(2 * math.pi * (x/1.0)**2) + 2*(x - 0.7)**2 + 0.5

def main():
    logging.info("duplicate.main()")

    xs = np.arange(0.0, 1.005, 0.01)
    ys = np.array([signal(x) for x in xs])

    expander = series.Expander(1.0, 'duplicate')
    a, b = expander.coefficients(ys, maximum_n=100)
    reconstruction = expander.reconstruct(a, b, len(xs))

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label='Original signal', lw=3)
    ax.plot(xs, reconstruction, label='Reconstruction')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()