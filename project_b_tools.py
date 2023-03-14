from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import datetime

iterations = 0
max_iterations = 500


def to_time(sec):
    return str(datetime.timedelta(seconds=int(sec)))


def to_sec(time):
    pt = datetime.datetime.strptime(time, '%H:%M:%S')
    return pt.second + pt.minute * 60 + pt.hour * 3600


def myprint(array, T=False):
    print(*[[(to_time(e) if T else int(e)) for e in elem] for elem in array.tolist()], sep="\n")


def print_result(g):
    sol = g.sol
    Tsol, Lsol, Psol = g.extract(sol.x)
    print("\nsuccess: ", sol.success, sol.message, "iterations:", sol.nit)
    print("Depart times")
    myprint(Tsol, T=False)
    print("\nLoad at depart")
    myprint(Lsol)
    print("\nnumber of passengers on platform before depart")
    myprint(Psol)


def random(avg, var, length, width=-1):
    # random number between avg * [(1-var) , (1+var)]
    if width == -1:
        return avg * ((np.random.rand(length) * 2 * var) + (1 - var))
    else:
        return avg * ((np.random.rand(length, width) * 2 * var) + (1 - var))


class States:

    def __init__(self):
        self.WAITING_FOR_FIRST_DEPART = 0
        self.UNLOADING = 1
        self.LOADING = 2
        self.MOVING = 3
        self.FINISHED = 4

class Units:

    def __init__(self):
        self.hour = 3600
        self.minute = 60
        self.second = 1



states = States()
units = Units()


class TrainState:

    def __init__(self):
        self.state = states.WAITING_FOR_FIRST_DEPART
        self.station = 0  # station marks the last station if between station and current station if in station
