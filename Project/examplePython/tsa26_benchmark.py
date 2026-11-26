"""Runs your submission like the practice benchmark. Set SUBMISSION and run this file,
or in a terminal:  python tsa26_benchmark.py example_submission.py"""

import os
import runpy
import sys
from datetime import date

import numpy as np
from scipy.io import loadmat

SUBMISSION = 'example_submission.py'  # your file, in the same folder as this one and projectData26.mat
PERIODS = ['1992-02-13', '1992-07-01', '1993-03-04']  # first day of each test period
PERIOD_DAYS = 14  # days in each test period
K = [15, 30]
PREDICTS = {'solutionA': ['waterflow'],
            'solutionB': ['waterflow', 'temp', 'nbd'],
            'solutionC': ['waterflow', 'temp', 'nbd']}
HERE = os.path.dirname(os.path.abspath(__file__))


def run(path):
    solutions = runpy.run_path(path)  
    mat = loadmat(os.path.join(HERE, 'projectData26.mat'))
    data = {'waterflow': mat['waterflowData'].ravel(), 'temp': mat['tempData'].ravel(), 'nbd': mat['nbdData'].ravel()}
    starts = [(date.fromisoformat(d) - date(1988, 1, 1)).days + 1 for d in PERIODS]  # day 1 = 1 Jan 1988

    # every test day t and every k, in time order of t-k 
    calls = []
    for start in starts:
        for k in K:
            for t in range(start, start + PERIOD_DAYS):
                calls.append((t - k, t, k))
    calls.sort()

    predictions = {}
    for last, t, k in calls:
        p = {'waterflow': data['waterflow'][:last].copy(), 'temp': data['temp'][:last].copy(),
             'nbd': data['nbd'][:last].copy(), 'k': k, 't': t}
        for name, signals in PREDICTS.items():
            values = np.ravel(solutions[name](p))
            assert len(values) == len(signals) and np.all(np.isfinite(values)), f'{name} returned {values}'
            predictions[name, k, t] = values

    # each prediction: MSE / MSE of the naive predictor y(t-k), per period, averaged; < 1: 1 point, <= 0.5: 2 points
    total = 0
    for name, signals in PREDICTS.items():
        for i, signal in enumerate(signals):
            y = data[signal]
            for k in K:
                ratios = []
                for start in starts:
                    days = range(start, start + PERIOD_DAYS)
                    mse = np.mean([(y[t - 1] - predictions[name, k, t][i]) ** 2 for t in days])
                    naive_mse = np.mean([(y[t - 1] - y[t - k - 1]) ** 2 for t in days])
                    ratios.append(mse / naive_mse)
                ratio = np.mean(ratios)
                points = 2 if ratio <= 0.5 else 1 if ratio < 1 - 1e-9 else 0
                total += points
                print(f'{name}_{signal}_k{k}: {ratio:.4f} {np.round(ratios, 4)} {points} point(s)')
    print(f'Total {total}/28 points')


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, SUBMISSION))
