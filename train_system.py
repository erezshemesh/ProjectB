import numpy as np
from gym.spaces.space import Space
from generator import *
import random


class TrainSystem:

    def __init__(self, T, L, P, gen: Generator):
        self.T = T
        self.L = L
        self.P = P
        self.gen = gen
        self.time = 21600  # 6:00AM
        self.location = np.zeros(gen.trains)
        self.states = []
        for _ in range(self.gen.trains):
            self.states += [TrainState()]
        self.load = np.zeros(gen.trains)
        self.load_before_alight = np.zeros(gen.trains)
        self.platform = np.zeros(gen.stations)
        self.agent_speed = np.zeros(gen.trains)
        self.start_time = [T[train, 0] - L[train, 0] * self.gen.beta[0] for train in range(self.gen.trains)]


    def estimated_T_diff(self, train, station):
        # we only take into account trains that hasn't departed yet from 'station' , and this is the next station to depart.
        # if train already departed or train has a previous station to depart from , then estimated_T diff is not important  and accounted as ZERO.
        estimated_T = self.time
        if self.location[train] > station * self.gen.km_between_stations:
            # CASE 1 - train already departed station
            return 0
        elif self.location[train] <= (station - 1) * self.gen.km_between_stations:
            # CASE 2 - train has a previous station to depart from
            return 0
        elif self.location[train] == station * self.gen.km_between_stations:
            # CASE 3 - train is in station
            if self.states[train].state == states.WAITING_FOR_FIRST_DEPART:
                estimated_T = self.start_time[train]
                estimated_T += min(((self.platform[station]) * self.gen.lambda_[station]),(self.gen.lmax - self.load[train])) * self.gen.beta[station]
            if self.states[train].state == states.UNLOADING:
                estimated_T += (self.load[train] - (self.load_before_alight[train] * self.gen.eta[train, station])) * self.gen.alpha[station]
                estimated_T += min((self.platform[station] + (estimated_T - self.time) * self.gen.lambda_[station]), (self.gen.lmax - self.load[train])) * self.gen.beta[station]
            if self.states[train].state == states.LOADING:
                estimated_T += min((self.platform[station] * self.gen.lambda_[station]), (self.gen.lmax - self.load[train])) * self.gen.beta[station]
        elif self.location[train] < station * self.gen.km_between_stations:
            # CASE 4 - train is moving to station
            estimated_T += (station * self.gen.km_between_stations - self.location[train]) / (self.gen.speed_kmh/3600 + self.agent_speed[train])
            estimated_T += (self.load[train] - (self.load_before_alight[train] * self.gen.eta[train, station])) * self.gen.alpha[station]
            estimated_T += min((self.platform[station] + (estimated_T - self.time) * self.gen.lambda_[station]), (self.gen.lmax - self.load[train])) * self.gen.beta[station]
        return abs(estimated_T - self.T[train, station])
    def reward(self):
        diff = 0
        for train in range(self.gen.trains):
            for station in range(self.gen.stations):
                diff += self.estimated_T_diff(train, station)
        return diff
    def new_state_reward(self):
        reward = self.reward()
        done = (self.states[-1].state == states.FINISHED)
        info = {}
        return self.get_obs(), reward, done, info

    def reset(self):
        self.time = to_sec('06:00:00')
        self.location = np.zeros(self.gen.trains)
        self.states = []
        for _ in range(self.gen.trains):
            self.states += [TrainState()]
        self.load = np.zeros(self.gen.trains)
        self.load_before_alight = np.zeros(self.gen.trains)
        self.platform = np.zeros(self.gen.stations)
        self.agent_speed = np.zeros(self.gen.trains)
        return self.get_obs()

    def Wait(self, train, epoch):
        max_wait = self.start_time[train] - self.time
        if epoch > max_wait:
            self.Load(train, epoch - max_wait)

    def Load(self, train, effective_epoch):
        self.states[train].state = states.LOADING
        if effective_epoch > 0:
            station = self.states[train].station
            potential_load = min(effective_epoch / self.gen.beta[station], self.gen.lmax - self.load[train])
            self.load[train] += min(potential_load, self.platform[station])
            if potential_load < self.platform[station]:
                self.platform[station] -= potential_load
                if self.load[train] == self.gen.lmax:
                    loading_time = (potential_load * self.gen.beta[station])
                    self.Move(train, effective_epoch - loading_time)
            else:
                loading_time = (self.platform[station] * self.gen.beta[station])
                self.platform[station] = 0
                self.Move(train, effective_epoch - loading_time)

    def Unload(self, train, effective_epoch):
        self.states[train].state = states.UNLOADING  # maybe it should be outside, think about it later
        if effective_epoch > 0:
            station = self.states[train].station
            potential_unload = effective_epoch / self.gen.alpha[station]
            max_unload = self.load[train] - self.load_before_alight[train] * (1 - self.gen.eta[train, station])
            self.load[train] -= min(potential_unload, max_unload)
            if potential_unload >= max_unload:
                self.Load(train, effective_epoch - max_unload * self.gen.alpha[station])

    def Move(self, train, effective_epoch):
        self.states[train].state = states.MOVING
        speed = (self.gen.speed_kmh / units.hour) + self.agent_speed[train]
        if (self.states[train].station == self.gen.stations - 1):
            self.states[train].state = states.FINISHED
        else:
            if effective_epoch > 0:
                potential_move = effective_epoch * speed
                max_move = (10 - (self.location[train]) % 10)
                moving_distance = min(potential_move, max_move)
                moving_time = moving_distance / speed
                self.location[train] += moving_distance
                if potential_move >= max_move:
                    self.states[train].station += 1
                    self.load_before_alight[train] = self.load[train]
                    self.Unload(train, effective_epoch - moving_time)

    def step(self, epoch=60, noise=0):
        self.time = self.time + epoch
        for i in range(self.gen.stations):
            if self.gen.open_time[i] <= self.time <= self.gen.close_time[i]:
                self.platform[i] = self.platform[i] + (self.gen.lambda_[i] + noise * random.uniform(-0.3, 1.2)) * epoch
        for train in range(self.gen.trains):
            # CASE 0 - Finished
            if self.states[train].state == states.MOVING and self.states[train].station == self.gen.stations - 1:
                self.states[train].state = states.FINISHED
            elif self.states[train].state == states.WAITING_FOR_FIRST_DEPART:
                self.Wait(train, epoch)
            # CASE 2 - loading
            elif self.states[train].state == states.LOADING:
                self.Load(train, epoch)
            # CASE 3 - Unloading
            elif self.states[train].state == states.UNLOADING:
                self.Unload(train, epoch)
            # CASE 4 - Moving
            elif self.states[train].state == states.MOVING:
                self.Move(train, epoch)
        return self.new_state_reward()

    def get_obs(self):
        obs = np.concatenate((self.load, self.location, self.platform, np.array([self.time])), axis=0)
        return obs


class GymTrainSystem(gym.Env):
    def __init__(self, T, L, P, g):
        super().__init__()
        self.sys = TrainSystem(T, L, P, g)
        self.action_space = gym.spaces.Box(
            low=np.full(self.sys.gen.trains, -1, dtype=np.float32),
            high=np.full(self.sys.gen.trains, 1, dtype=np.float32),
            dtype=np.float32
        )
        min_load, max_load = 0, self.sys.gen.lmax
        min_location, max_location = 0, (self.sys.gen.stations - 1) * self.sys.gen.km_between_stations
        min_platform, max_platform = 0, np.inf
        min_time, max_time = to_sec('06:00:00'), to_sec('23:59:59')
        obs_low = np.concatenate((np.full(self.sys.gen.trains, min_location, dtype=np.float32),
                                  np.full(self.sys.gen.trains, min_load, dtype=np.float32),
                                  np.full(self.sys.gen.stations, min_platform, dtype=np.float32),
                                  np.array([min_time], dtype=np.float32)
                                  ), axis=0)
        obs_high = np.concatenate((np.full(self.sys.gen.trains, max_location, dtype=np.float32),
                                   np.full(self.sys.gen.trains, max_load, dtype=np.float32),
                                   np.full(self.sys.gen.stations, max_platform, dtype=np.float32),
                                   np.array([max_time], dtype=np.float32)
                                   ), axis=0)
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

    def reset(self):
        return self.sys.reset()

    def step(self, action):
        self.sys.agent_speed = action
        return self.sys.step()

    def render(self, mode='human'):
        pass
