def solutionA(p):
    return p['waterflow'][-1]


def solutionB(p):
    return [p['waterflow'][-1], p['temp'][-1], p['nbd'][-1]]


state = {'days': 0}  # variables kept between days (calls)


def solutionC(p):
    for d in range(state['days'], len(p['waterflow'])):  # only the days that are new since the last call
        state['x'] = [p['waterflow'][d], p['temp'][d], p['nbd'][d]]  # e.g. a Kalman filter update
    state['days'] = len(p['waterflow'])
    return state['x']
