import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def f(x):
    return 10 * x + 50 * (x-0.15)**2 - 100 * x**3

def g(y, xxs, L):
    g_k = np.zeros_like(xxs)
    f_L = y[-1] + (y[-1] - y[-2])
    N = len(xxs)//4
    for k in range(len(xxs)):
        x = xxs[k]
        if x < -L:
            i = k
            g_k[k] = 2 * f_L - y[i]
        elif x < 0:
            i = 2 * N - k - 1
            g_k[k] = y[i]
        elif x < L:
            i = k - 2 * N
            g_k[k] = y[i]
        else:
            i = 4 * N - k - 1
            g_k[k] = 2 * f_L - y[i]
    return g_k

def main():
    logging.info(f"quarter_range_even.main()")

    signal_length = 100
    L = 0.3
    delta_x = L/signal_length
    xs = np.arange(0, L, delta_x)
    ys = f(xs)
    xxs = np.arange(-2 * L, 2 * L, delta_x)
    gs = g(ys, xxs, L)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label='f(x)', linewidth=4)
    ax.plot(xxs, gs, label='g(x)', linewidth=2)

    ax.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()