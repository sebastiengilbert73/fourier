import numpy as np
from scipy import integrate
import itertools
import copy

"""
Computation of the complex Fourier series for 2D signals defined over a 2D array, with equally spaced x's
over [0, Lx], and equally spaced y's over [0, Ly]. The periodic signal is built over [(-Lx, -Ly), (Lx, Ly)] for half-range 
expansions, and over [(-2Lx, -2Ly), (2Lx, 2Ly)] for quarter-range expansions.

Half-range expansions:
    c_m_n = 1/(4LxLy) Integ_{-Ly, Ly} Integ_{-Lx, Lx} f(x, y) exp(-i pi * (m x/Lx + n y/Ly) ) dx dy
    f(x, y) = Sum_{m: -inf, inf} Sum_{n: -inf, inf} c_m_n exp(i pi * (m x/Lx + n y/Ly))

Quarter-range expansions:
    +++ TBD +++
"""
def exp_vectorize(xs, ys):
    combinations = list(itertools.product(ys, xs))
    values = np.vectorize(np.exp)([x + y for (y, x) in combinations])
    arr = np.array(values).reshape((len(ys), len(xs)))
    return arr

"""def sum_vectorize(xs, ys):
    sums = np.zeros((len(ys), len(xs)), dtype=complex)
    for y_ndx in range(len(ys)):
        y = ys[y_ndx]
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            sums[y_ndx, x_ndx] = x + y
    return sums

def exp_correct(xs, ys):
    exps = np.zeros((len(ys), len(xs)), dtype=complex)
    for y_ndx in range(len(ys)):
        y = ys[y_ndx]
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            e = np.exp(x + y)
            exps[y_ndx, x_ndx] = e
    return exps
"""

class Expander:
    def __init__(self, length_x, length_y, expansion_type='odd'):
        self.expansion_types = ['odd', 'even', 'quarter_odd', 'quarter_even', 'duplicate', 'odd_quarter', 'even_quarter']
        if not expansion_type in self.expansion_types:
            raise NotImplementedError(f"Expander.__init__(): Not implemented expansion type {expansion_type}. The valid expansion types are {self.expansion_types}.")
        self.Lx = length_x
        self.Ly = length_y
        self.expansion_type = expansion_type

    def coefficients(self, signal, maximum_m, maximum_n):
        if not type(signal) == np.ndarray:
            raise ValueError(f"Expander.coefficients(): type(signal) = {type(signal)}. We expect np.ndarray.")
        signal_shapeHW = signal.shape
        if len(signal_shapeHW) != 2:
            raise ValueError(f"Expander.coefficients(): len(signal_shapeHW) ({len(signal_shapeHW)}) != 2")
        if self.expansion_type in ['odd', 'even', 'duplicate']:
            maximum_m = min(maximum_m, signal_shapeHW[1] // 2)
            maximum_n = min(maximum_n, signal_shapeHW[0]//2)
        elif self.expansion_type in ['quarter_odd', 'quarter_even', 'odd_quarter', 'even_quarter']:
            maximum_m = min(maximum_m, signal_shapeHW[1])
            maximum_n = min(maximum_n, signal_shapeHW[0])
        else:
            raise NotImplementedError(f"Expander.coefficients(): Not implemented expansion type '{self.expansion_type}'")

        if self.expansion_type == 'duplicate':
            return self._half_range_duplicate(signal, maximum_m, maximum_n)
        elif self.expansion_type == 'odd':
            return self._half_range_odd(signal, maximum_m, maximum_n)
        else:
            raise NotImplementedError(f"Expander.coefficients(): self.expansion_type ('{self.expansion_type}') is not implemented.")
        """elif self.expansion_type == 'odd':
            return self._half_range_odd(signal, maximum_n)
        elif self.expansion_type == 'even':
            return self._half_range_even(signal, maximum_n)
        elif self.expansion_type == 'quarter_odd' or self.expansion_type == 'odd_quarter':
            return self._quarter_range_odd(signal, maximum_n)
        elif self.expansion_type == 'quarter_even' or self.expansion_type == 'even_quarter':
            return self._quarter_range_even(signal, maximum_n)
        """

    def evaluate(self, c_m_n, x, y):
        if c_m_n.shape[0] % 2 == 0 or c_m_n.shape[1] % 2 == 0:
            raise ValueError(f"Expander.evaluate(): One of the dimensions of c_m_n ({c_m_n.shape}) is even")
        m_max = c_m_n.shape[0]//2
        n_max = c_m_n.shape[1]//2
        if self.expansion_type == 'duplicate' or self.expansion_type == 'odd' or self.expansion_type == 'even':
            s = 0
            for idx_m in range(c_m_n.shape[0]):
                m = idx_m - m_max
                for idx_n in range(c_m_n.shape[1]):
                    n = idx_n - n_max
                    c = c_m_n[idx_m, idx_n]
                    s += c * np.exp(1j * np.pi * ( m * x / self.Lx + n * y/self.Ly))
            return s
        else:
            raise NotImplementedError(
                f"Expander.evaluate(): self.expansion_type ('{self.expansion_type}') is not implemented.")

    def reconstruct(self, c_m_n, signal_shapeHW):
        delta_x = self.Lx/(signal_shapeHW[1] - 1)
        xs = np.arange(0, self.Lx + delta_x/2, delta_x)
        delta_y = self.Ly / (signal_shapeHW[0] - 1)
        ys = np.arange(0, self.Ly + delta_y / 2, delta_y)
        reconstruction = np.zeros((signal_shapeHW), dtype=complex)
        for ky in range(len(ys)):
            y = ys[ky]
            for kx in range(len(xs)):
                x = xs[kx]
                reconstruction[ky, kx] = self.evaluate(c_m_n, x, y)
        return reconstruction

    def _under_weight_boundaries(self, signal):
        # Weighted signal: under-weight the boundaries: 0.5 for the edges; 0.25 for the corners
        weighted_snl = copy.deepcopy(signal)
        weighted_snl[:, 0] = 0.5 * weighted_snl[:, 0]
        weighted_snl[0, :] = 0.5 * weighted_snl[0, :]
        weighted_snl[:, -1] = 0.5 * weighted_snl[:, -1]
        weighted_snl[-1, :] = 0.5 * weighted_snl[-1, :]
        return weighted_snl

    def _half_range_duplicate(self, signal, maximum_m, maximum_n):  # signal.shape = (H, W)
        # c_m_n = 1/(4LxLy) [ (1 + (-1)**m + (-1)**n + (-1)**(m+n)) Integ_{0, Ly} Integ_{0, Lx} f(x, y) exp(-i pi (m x/Lx + n y/Ly)) dx dy ]
        c_m_n = np.zeros((2 * maximum_m + 1, 2 * maximum_n + 1), dtype=complex)
        HW = signal.shape
        delta_x = self.Lx / (HW[1] - 1)
        delta_y = self.Ly / (HW[0] - 1)
        xs = np.arange(0, self.Lx + delta_x / 2, delta_x)  # [0, dx, ..., Lx]
        ys = np.arange(0, self.Ly + delta_y / 2, delta_y)  # [0, dy, ..., Ly]
        weighted_snl = self._under_weight_boundaries(signal)

        for m in range(-maximum_m, maximum_m + 1):
            for n in range(-maximum_n, maximum_n + 1):
                cplx_exp = exp_vectorize(-1j * np.pi * m * xs / self.Lx, -1j * np.pi * n * ys/self.Ly)  # (H, W)
                c = 0.25 / (self.Lx * self.Ly) * (1 + (-1)**m + (-1)**n + (-1)**(m+n)) * np.sum(cplx_exp * weighted_snl) * delta_x * delta_y
                c_m_n[m + maximum_m, n + maximum_n] = c
        return c_m_n

    def _half_range_odd(self, signal, maximum_m, maximum_n):  # signal.shape = (H, W)
        # c_m_n = 1/(4 Lx Ly) [ - Integ_{0, Ly} Integ_{0, Lx} f(x, y) exp(-i pi (-m x/Lx + n y/Ly) ) dx dy
        #   + Integ_{0, Ly} Integ_{0, Lx} f(x, y) exp(-i pi (m x/Lx + n y/Ly) ) dx dy
        #   - Integ_{0, Ly} Integ_{0, Lx} f(x, y) exp(-i pi (m x/Lx - n y/Ly) ) dx dy
        #   + Integ_{0, Ly} Integ_{0, Lx} f(x, y) exp(-i pi (-m x/Lx - n y/Ly) ) dx dy ]
        c_m_n = np.zeros((2 * maximum_m + 1, 2 * maximum_n + 1), dtype=complex)
        HW = signal.shape
        delta_x = self.Lx / (HW[1] - 1)
        delta_y = self.Ly / (HW[0] - 1)
        xs = np.arange(0, self.Lx + delta_x / 2, delta_x)  # [0, dx, ..., Lx]
        ys = np.arange(0, self.Ly + delta_y / 2, delta_y)  # [0, dy, ..., Ly]
        weighted_snl = self._under_weight_boundaries(signal)

        for m in range(-maximum_m, maximum_m + 1):
            for n in range(-maximum_n, maximum_n + 1):
                exp_m_p = exp_vectorize(-1j * np.pi * (-m * xs/self.Lx), -1j * np.pi * (n * ys/self.Ly) )
                exp_p_p = exp_vectorize(-1j * np.pi * (m * xs/self.Lx), -1j * np.pi * (n * ys/self.Ly) )
                exp_p_m = exp_vectorize(-1j * np.pi * (m * xs/self.Lx), -1j * np.pi * (-n * ys/self.Ly) )
                exp_m_m = exp_vectorize(-1j * np.pi * (-m * xs/self.Lx), -1j * np.pi * (-n * ys/self.Ly) )
                c = 0.25 / (self.Lx * self.Ly) * (
                    -np.sum(exp_m_p * weighted_snl) * delta_x * delta_y \
                    + np.sum(exp_p_p * weighted_snl) * delta_x * delta_y \
                    - np.sum(exp_p_m * weighted_snl) * delta_x * delta_y \
                    + np.sum(exp_m_m * weighted_snl) * delta_x * delta_y
                )
                c_m_n[m + maximum_m, n + maximum_n] = c
        return c_m_n