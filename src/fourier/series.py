import numpy as np
import math
from scipy import integrate

def cos_vectorize(s):
    return np.vectorize(math.cos)(s)

def sin_vectorize(s):
    return np.vectorize(math.sin)(s)

class Expander:
    def __init__(self, length):
        self.L = length

    def coefficients(self, signal, expansion_type='odd', maximum_n=1000):
        if not type(signal) == np.ndarray:
            raise ValueError(f"Expander.coefficients(): type(signal) = {type(signal)}. We expect np.ndarray.")
        maximum_n = min(maximum_n, len(signal)//2)
        if expansion_type == 'odd':
            return self._half_range_odd(signal, maximum_n)
        elif expansion_type == 'even':
            return self._half_range_even(signal, maximum_n)
        elif expansion_type == 'quarter_odd':
            return self._quarter_range_odd(signal, maximum_n)
        elif expansion_type == 'quarter_even':
            return self._quarter_range_even(signal, maximum_n)
        else:
            raise NotImplementedError(f"Expander.coefficients(): Not implemented expansion type '{expansion_type}'")


    def evaluate(self, a_n, b_n, x, expansion_type='odd'):
        if expansion_type == 'odd':
            s = 0
            for n in range(len(b_n)):
                s += b_n[n] * math.sin(n * math.pi * x/self.L)
            return s
        elif expansion_type == 'even':
            s = a_n[0]
            for n in range(1, len(a_n)):
                s += a_n[n] * math.cos(n * math.pi * x/self.L)
            return s
        elif expansion_type == 'quarter_odd':
            pass
        elif expansion_type == 'quarter_even':
            s = a_n[0]
            for n in range(1, len(a_n)):
                s += a_n[n] * math.cos(n * math.pi * x/(2.0 * self.L))
            return s
        else:
            raise NotImplementedError(f"Expander.evaluate(): Not implemented expansion type '{expansion_type}'")

    def reconstruct(self, a_n, b_n, expansion_type, signal_length):
        delta_x = self.L/signal_length
        xs = np.arange(0, self.L, delta_x)
        reconstruction = np.zeros((signal_length), dtype=float)
        for k in range(signal_length):
            x = xs[k]
            reconstruction[k] = self.evaluate(a_n, b_n, x, expansion_type)
        return reconstruction

    def _half_range_odd(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L/len(signal)
        xs = np.arange(0, self.L + delta_x, delta_x)  # [0, dx, ..., L]
        next_signal_value = signal[-1] + (signal[-1] - signal[-2])
        extended_signal = np.concatenate((signal, np.array([next_signal_value])), axis=0)
        for n in range(1, maximum_n + 1):
            sinnpix_L = sin_vectorize(n * math.pi * xs/self.L)
            b = 2.0 / self.L * integrate.simpson(y=(extended_signal * sinnpix_L), x=xs)
            b_n[n] = b
        return a_n, b_n

    def _half_range_even(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L / len(signal)
        xs = np.arange(0, self.L + delta_x, delta_x)  # [0, dx, ..., L]
        next_signal_value = signal[-1] + (signal[-1] - signal[-2])
        extended_signal = np.concatenate((signal, np.array([next_signal_value])), axis=0)
        a_n[0] = 1.0/self.L * integrate.simpson(y=(extended_signal), x=xs)
        for n in range(1, maximum_n + 1):
            cosnpix_L = cos_vectorize(n * math.pi * xs/self.L)
            a = 2.0/self.L * integrate.simpson(y=(extended_signal * cosnpix_L), x=xs)
            a_n[n] = a
        return a_n, b_n

    def _quarter_range_even(self, signal, maximum_n):
        a_n = np.zeros((maximum_n + 1), dtype=float)
        b_n = np.zeros((maximum_n + 1), dtype=float)
        delta_x = self.L / len(signal)
        xs = np.arange(0, self.L + delta_x, delta_x)  # [0, dx, ..., L]
        next_signal_value = signal[-1] + (signal[-1] - signal[-2])
        extended_signal = np.concatenate((signal, np.array([next_signal_value])), axis=0)
        f_L = next_signal_value
        a_n[0] = f_L
        for n in range(1, maximum_n + 1):
            sinnpi_2 = sin_vectorize(n * math.pi/2)
            cosnpix_2L = cos_vectorize(n * math.pi * xs/(2 * self.L))
            #a = -4.0 * f_L/(n * math.pi) * sinnpi_2 + 1.0/self.L * integrate.simpson(y=(extended_signal * cosnpix_2L), x=xs)
            t1 = -8.0 * self.L * f_L/(n * math.pi) * sinnpi_2
            t2 = 2.0 * (1 - (-1)**n) * integrate.simpson(y=(extended_signal * cosnpix_2L), x=xs)
            a = 1.0/(2 * self.L) * (t1 + t2)
            a_n[n] = a
        return a_n, b_n