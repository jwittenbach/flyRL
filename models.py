import numpy as np

class DecisionModel(object):

    def __init__(self, nStates, nParams, params=None):
        '''
        Initialize model
        '''
        self.nStates = nStates
        self.nParams = nParams
        self.params = params

    def _checkParams(self):
        '''
        Used to check if parameters are set before doing any simulation
        '''
        if self.params is None:
            raise ValueError('must initialize parameters')

    def _pNext(self, state, params):
        '''
        Given the model parameters and the current state, returns the
        probability distribution over next states.

        This is what defines the model and must be overridden by any child
        class to implement the model behavior.
        '''
        raise NotImplementedError

    def _logL(self, states, params):
        '''
        Given model parameters and a sequence of states, returns the
        log-likelihood of the model producing the data.
        '''
        logL = 0
        for i in range(len(states) - 1):
            s1, s2 = states[i], states[i+1]
            probNext = self._pNext(s1, params)[s2]
            logL += np.log(probNext)
        return logL

    def pNext(self, state):
        '''
        Given the current state, returns the probability distribution over next
        states.
        '''
        self._checkParams()
        return self._pNext(state, self.params)

    def logL(self, states):
        '''
        Given a sequence of states, returns the log-likelihood of the model
        producing the data.

        Assumes that prior distribution over the initial state has all mass at
        the observed intial state (i.e. P[states[0]] = 1; any initial state is
        just as good)
        '''
        self._checkParams()
        return self._logL(states, self.params)

    def simulate(self, nSteps, start=0):
        '''
        Forward simulation of the model from an initial state.
        '''
        self._checkParams()
        states = np.zeros(nSteps+1, dtype='int')
        states[0] = start
        for i in range(nSteps):
            #TODO: this could be sped up with memoization
            p = self.pNext(states[i])
            states[i+1] = np.random.choice(self.nStates, p=p)
        return states

    def _fit(self, states, init, bounds=None, constraints=None):
        '''
        Fit models parameters given data consisiting of a sequence of states.

        General function that allows for specification of initial values,
        bounds, and constraints on parameters.
        '''
        from scipy.optimize import minimize
        f = lambda params, states: -self._logL(states, params)
        result = minimize(f, init, args=(states,), bounds=bounds,
                          constraints=constraints)
        return result

    def fit(self, states):
        '''
        Fit model parameters given data consisting of a sequence of states.

        Subclasses should override this method and use it to define reasonable
        initial parameter estimates as well as any bounds or constraints on the
        parameters. Those can then be passes to _fit for the actual fitting.
        '''
        raise NotImplementedError

class TestModel(DecisionModel):
    '''
    A simple test model where, on each time step, states are drawn
    independently from a fixed distribution that constitutes the parameters.
    '''

    def __init__(self, p):
        params = p
        super(TestModel, self).__init__(nStates=len(p), nParams=len(p), params=params)

    def _pNext(self, state, params):
        return params

    def fit(self, states):
        n = self.nParams
        bounds = n*((0, 1),)
        constraint = {'type': 'eq',
                      'fun': lambda params: np.sum(params) - 1}
        init = np.random.uniform(size=n)
        return self._fit(states, init, bounds, constraint)

class ValueModel(DecisionModel):
    '''
    Model of decisions based on a value function + a penalty for large steps
    '''

    def __init__(self, vSpline, nStates):
        self.vSpline = vSpline
        dom = vSpline.domain
        self.locs = np.linspace(dom[0], dom[1], nStates+1)[:-1]
        params = vSpline.w
        super(ValueModel, self).__init__(nStates=nStates, nParams=len(params),
                                         params=params)

    def _pNext(self, state, params):
        w = params[:]
        self.vSpline.set_weights(w)
        vals = self.vSpline(self.locs)
        exp = np.exp(vals)
        return exp/np.sum(exp)
