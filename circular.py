from numpy import arange, r_, asarray, zeros
from scipy.interpolate import splev

def circular_convolve(signal, ker):
    from numpy import convolve
    N_sig, N_ker = signal.shape[0], ker.shape[0]
    pad = np.tile(signal, np.ceil(N_ker/N_sig))
    extended = np.concatenate([pad, signal, pad])
    return convolve(extended, ker, mode='same')[pad.shape[0]:pad.shape[0]+signal.shape[0]]

def get_cuts(x, N_x, dx_max):
    cuts = np.where(np.abs(np.diff(x)) >= N_x - dx_max)[0] + 1
    cuts = np.concatenate([[0], cuts, [x.size]])
    return cuts

def circular_plot(x, N_x, dx_max, t=None):
    if t is None:
        t = np.arange(x.shape[0])
    cuts = get_cuts(x, N_x, dx_max)
    for i, j in zip(cuts[:-1], cuts[1:]):
        plt.plot(t[np.arange(i, j)], x[i:j], c=sns.color_palette()[0])
    plt.ylim([0, N_x-1])

def get_winding_nums(x, cuts):
    winding_nums = [0]
    for i in cuts[1:-1]:
        dw = -1 if x[i] > x[i-1] else 1
        winding_nums.append(winding_nums[-1] + dw)
    return winding_nums

def unwind(x, N_x, dx_max):
    cuts = get_cuts(x, N_x, dx_max)
    winding_nums = get_winding_nums(x, cuts)
    unwound = x.copy()
    for w, i, j in zip(winding_nums[1:], cuts[1:-1], cuts[2:]):
        unwound[i:j] += w*N_x
    return unwound

def wind(x, N_x):
    i = 0
    wound = x.copy()
    while True:
        cut = np.where(np.logical_or(wound<0, wound>=N_x))[0]
        if cut.size == 0:
            break
        else:
            cut = cut[0]
        sign = 1 if wound[cut] < 0 else -1
        wound[cut:] += sign * N_x
        i = cut
    return wound

def circular_vel(x, N_x, dx_max):
    from numpy import gradient
    unwound = unwind(x, N_x, dx_max)
    return gradient(unwound)

class CircularSpline:

    def __init__(self, k, n, domain=[0, 1], w=None):
        self.k = k
        self.n = n
        self.domain = domain

        if w is None:
            w = zeros(n)
        else:
            w = asarray(w)
            if w.shape[0] != n:
                raise ValueError('number of weights must equal number of basis functions')

        L = domain[1] - domain[0]
        d = 1.0*L/ n

        self.knots = d*arange(-k, n+k+1)
        self.weights = self._expand_weights(w)

    def set_weights(self, w):
        self.weights = self._expand_weights(w)

    def _expand_weights(self, w):
        '''
        expands weights to account for:
        (a) circular boundary conditions
        (b) boundary behavior of SciPy spline functions
        '''
        return r_[0, w, w[:self.k-1], 0]

    @property
    def w(self):
        return self.weights[1:self.n+1]

    def __call__(self, x):
        return splev(x, (self.knots, self.weights, self.k-1), ext=1)

    def eval_basis(self, i, x):
        w = zeros(self.n)
        w[i] = 1
        return splev(x, (self.knots, self._expand_weights(w), self.k-1), ext=1)
