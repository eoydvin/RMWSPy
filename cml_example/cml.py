import os
import sys
import datetime
import logging
import numpy as np
import scipy.stats as st

fpth = os.path.abspath(os.path.join("..", "rmwspy"))
sys.path.append(fpth)
from random_mixing_whittaker_shannon import *
from concurrent.futures import ThreadPoolExecutor


class CMLModel(NonLinearProblemTemplate):
    def __init__(self, data, marginal, cmllinks):
        self.data = data
        self.marginal = marginal
        self.cmllinks = cmllinks
        self.nthreads = 4

    def objective_function(self, prediction):
        return np.mean((self.data - prediction) ** 2, axis=1) ** 0.5

    def allforwards(self, fields):
        out = np.empty((fields.shape[0], self.data.shape[0]))
        out_ = self.threading_forward(fields)
        for i, o in enumerate(out_):
            out[i] = o
        #         for k in range(fields.shape[0]):
        #             out[k] = self.forward(k, fields[k])
        return out

    def threading_forward(self, fields):
        i = np.arange(fields.shape[0])
        with ThreadPoolExecutor(max_workers=self.nthreads) as executor:
            results = executor.map(self.forward, i, fields)
        return results

    def forward(self, i, field):
        rain = st.norm.cdf(field)
        mp0 = rain <= self.marginal["p0"]
        rain[mp0] = 0.0
        rain[~mp0] = (rain[~mp0] - self.marginal["p0"]) / (1.0 - self.marginal["p0"])
        rain[~mp0] = self.marginal["invcdf"](rain[~mp0])
        rain[~mp0] = np.exp(rain[~mp0]) / 10.0

        # get CML integrals from simulated rain field
        CML = []
        for link in range(self.data.shape[0]):
            nlvals_at_x = self.get_cml_on_path(rain, self.cmllinks[link])
            CML.append(np.mean(nlvals_at_x))
        CML = np.array(CML)
        return CML

    def get_cml_on_path(self, data, cp):
        assert cp.ndim > 1
        dimensions = list(map(lambda x: cp[:, x], range(cp.ndim)))
        fullslice = [slice(None, None)]
        if data.ndim > cp.ndim:
            return data[tuple(fullslice + dimensions)].T
        else:
            return data[tuple(dimensions)].T
