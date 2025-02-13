import numpy as np
from scipy import integrate

"""
Computation of the complex Fourier series for 1D signals defined over a 1D array, with equally spaced x's
over [0, L]. The periodic signal is built over [-L, L] for half-range expansions, and over [-2L, 2L] for
quarter-range expansions.

Half-range expansions:
    c_n = 1/(2L) Integ_{-L, L} f(x) exp(-i * n * pi * x/L) dx
    f(x) = Sum_{n: -inf, inf} c_n exp(i * n * pi * x/L)

Quarter-range expansions:
    c_n = 1/(4L) Integ_{-2L, 2L} f(x) exp( -i * n * pi * x/(2L) ) dx
    f(x) = Sum_{n: -inf, inf} c_n exp( i * n * pi * x/(2L) )
"""
def exp_vectorize(s):
    return np.vectorize(np.exp)(s)

class Expander:
    def __init__(self, length, expansion_type='duplicate'):
        self.expansion_types = ['odd', 'even', 'quarter_odd', 'quarter_even', 'duplicate', 'odd_quarter', 'even_quarter']
        if not expansion_type in self.expansion_types:
            raise NotImplementedError(f"Expander.__init__(): Not implemented expansion type {expansion_type}. The valid expansion types are {self.expansion_types}.")
        self.L = length
        self.expansion_type = expansion_type

    def coefficients(self, signal, maximum_n):
        if not type(signal) == np.ndarray:
            raise ValueError(f"Expander.coefficients(): type(signal) = {type(signal)}. We expect np.ndarray.")
        if self.expansion_type in ['odd', 'even', 'duplicate']:
            maximum_n = min(maximum_n, len(signal)//2)
        elif self.expansion_type in ['quarter_odd', 'quarter_even', 'odd_quarter', 'even_quarter']:
            maximum_n = min(maximum_n, len(signal))
        else:
            raise NotImplementedError(f"Expander.coefficients(): Not implemented expansion type '{self.expansion_type}'")

        if self.expansion_type == 'duplicate':
            return self._half_range_duplicate(signal, maximum_n)
        elif self.expansion_type == 'odd':
            return self._half_range_odd(signal, maximum_n)
        elif self.expansion_type == 'even':
            return self._half_range_even(signal, maximum_n)
        elif self.expansion_type == 'quarter_odd' or self.expansion_type == 'odd_quarter':
            return self._quarter_range_odd(signal, maximum_n)
        elif self.expansion_type == 'quarter_even' or self.expansion_type == 'even_quarter':
            return self._quarter_range_even(signal, maximum_n)
        else:
            raise NotImplementedError(f"Expander.coefficients(): self.expansion_type ('{self.expansion_type}') is not implemented.")

    def evaluate(self, c_n, x):
        if len(c_n) % 2 == 0:
            raise ValueError(f"Expander.evaluate(): len(c_n) ({len(c_n)}) is even")
        n_max = len(c_n)//2
        if self.expansion_type == 'duplicate' or self.expansion_type == 'odd' or self.expansion_type == 'even':
            s = 0
            for idx in range(len(c_n)):
                n = idx - n_max
                c = c_n[idx]
                s += c * np.exp(1j * n * np.pi * x / self.L)
            return s
        elif self.expansion_type == 'quarter_odd' or self.expansion_type == 'odd_quarter' or \
                self.expansion_type == 'quarter_even' or self.expansion_type == 'even_quarter':
            s = 0
            for idx in range(len(c_n)):
                n = idx - n_max
                c = c_n[idx]
                s += c * np.exp(1j * n * np.pi * x /(2 * self.L))
            return s
        else:
            raise NotImplementedError(
                f"Expander.evaluate(): self.expansion_type ('{self.expansion_type}') is not implemented.")

    def reconstruct(self, c_n, signal_length):
        delta_x = self.L/(signal_length - 1)
        xs = np.arange(0, self.L + delta_x/2, delta_x)
        reconstruction = np.zeros((signal_length), dtype=complex)
        for k in range(signal_length):
            x = xs[k]
            reconstruction[k] = self.evaluate(c_n, x)
        return reconstruction

    def _half_range_duplicate(self, signal, maximum_n):
        # c_n = 1/(2*L) [ ((-1)**n + 1) Int_{0, L} f(x) exp(-i*n*pi*x/L) dx ]
        c_n = np.zeros((2 * maximum_n + 1), dtype=complex)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x / 2, delta_x)  # [0, dx, ..., L]
        for n in range(-maximum_n, maximum_n + 1):
            cplx_exp = exp_vectorize(-1j * n * np.pi * xs/self.L)
            c = 0.5 / self.L * ((-1)**n + 1) * integrate.simpson(y=(signal * cplx_exp), x=xs)
            c_n[n + maximum_n] = c
        return c_n

    def _half_range_odd(self, signal, maximum_n):
        # c_n = 1/(2*L) [ -Integ_{0, L} f(x) exp(i*n*pi*x/L) dx + Integ_{0, L} f(x) exp(-i*n*pi*x/L) dx ]
        c_n = np.zeros((2 * maximum_n + 1), dtype=complex)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x / 2, delta_x)  # [0, dx, ..., L]
        for n in range(-maximum_n, maximum_n + 1):
            cplx_exp_plus = exp_vectorize(1j * n * np.pi * xs/self.L)
            cplx_exp_minus = exp_vectorize(-1j * n * np.pi * xs/self.L)
            c = 0.5 / self.L * (-1.0 * integrate.simpson(y=(signal * cplx_exp_plus), x=xs) + integrate.simpson(y=(signal * cplx_exp_minus), x=xs) )
            c_n[n + maximum_n] = c
        return c_n

    def _half_range_even(self, signal, maximum_n):
        # c_n = 1/(2*L) [ Integ_{0, L} f(x) exp(i*n*pi*x/L) dx + Integ_{0, L} f(x) exp(-i*n*pi*x/L) dx ]
        c_n = np.zeros((2 * maximum_n + 1), dtype=complex)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x / 2, delta_x)  # [0, dx, ..., L]
        for n in range(-maximum_n, maximum_n + 1):
            cplx_exp_plus = exp_vectorize(1j * n * np.pi * xs/self.L)
            cplx_exp_minus = exp_vectorize(-1j * n * np.pi * xs/self.L)
            c = 0.5 / self.L * (integrate.simpson(y=(signal * cplx_exp_plus), x=xs) + integrate.simpson(y=(signal * cplx_exp_minus), x=xs) )
            c_n[n + maximum_n] = c
        return c_n

    def _quarter_range_odd(self, signal, maximum_n):
        # c_n = 1/(4*L) [ (1 - (-1)**n) Integ_{0, L} f(x) exp(-i*n*pi*x/(2L)) dx + ((-1)**n - 1) Integ_{0, L} f(x) exp(i*n*pi*x/(2L)) dx ]
        c_n = np.zeros((2 * maximum_n + 1), dtype=complex)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x / 2, delta_x)  # [0, dx, ..., L]
        for n in range(-maximum_n, maximum_n + 1):
            cplx_exp_plus = exp_vectorize(1j * n * np.pi * xs/(2 * self.L))
            cplx_exp_minus = exp_vectorize(-1j * n * np.pi * xs/(2 * self.L))
            c = 0.25 / self.L * ( (1 - (-1)**n) * integrate.simpson(y=(signal * cplx_exp_minus), x=xs) + ((-1)**n - 1) * integrate.simpson(y=(signal * cplx_exp_plus), x=xs) )
            c_n[n + maximum_n] = c
        return c_n

    def _quarter_range_even(self, signal, maximum_n):
        # c_n = 1/(4*L) [ (1 - (-1)**n) Integ_{0, L} f(x) exp(-i*n*pi*x/(2L)) dx + (1 - (-1)**n) Integ_{0, L} f(x) exp(i*n*pi*x/(2L)) dx ]
        c_n = np.zeros((2 * maximum_n + 1), dtype=complex)
        delta_x = self.L / (len(signal) - 1)
        xs = np.arange(0, self.L + delta_x / 2, delta_x)  # [0, dx, ..., L]
        for n in range(-maximum_n, maximum_n + 1):
            cplx_exp_plus = exp_vectorize(1j * n * np.pi * xs/(2 * self.L))
            cplx_exp_minus = exp_vectorize(-1j * n * np.pi * xs/(2 * self.L))
            c = 0.25 / self.L * ( (1 - (-1)**n) * integrate.simpson(y=(signal * cplx_exp_minus), x=xs) + (1 - (-1)**n) * integrate.simpson(y=(signal * cplx_exp_plus), x=xs) )
            c_n[n + maximum_n] = c
        return c_n