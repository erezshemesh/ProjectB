import numpy as np
from gym.spaces.space import Space
from generator import *
import random
import gym
from sys import exit


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

    def debug_print(self, train, station, est_depart_time):
        ratio = est_depart_time / self.T[train, station]
        if self.states[train].state != states.FINISHED:
            print("Time:", self.time, "\tTrain:", train, "\tState:", self.states[train].state, "\tActual Station:",
                  station, "\tEstimated Time:", est_depart_time, "\t PDT Time[Train,Actual Station]:",
                  self.T[train, station])

    # Returns the abs difference between estimated departure time of train from station and the time at schedule (T):
    def calc_est_time_diff(self, train):
        station = self.get_current_station(train)
        est_depart_time = self.calc_est_depart_time(train)
        return abs(est_depart_time - self.T[train, station])

    # This function returns the station that train is currently at.
    # We defined that if train is moving, its current station would be the next one.
    def get_current_station(self, train):
        return self.states[train].station + (self.states[train].state == states.MOVING)

    # This function returns potential time that would take train to arrive at station
    def calc_arriving_time(self, train, station):
        if self.states[train].state == states.MOVING:
            train_speed = self.gen.speed_kmh / 3600 + self.agent_speed[train]
            distance_to_next_station = station * self.gen.km_between_stations - self.location[train]
            return distance_to_next_station / train_speed
        return 0

    # Returns potential time that would take alighting passengers on train to alight at station
    def calc_alighting_time(self, train, station):
        if self.states[train].state == states.LOADING:
            return 0
        else:
            # Calculates how many passengers will alight:
            if self.states[train].state == states.UNLOADING:
                load_before_alight = self.load_before_alight[train]
            else:
                load_before_alight = self.load[train]
            staying_passengers = (1 - self.gen.eta[train, station]) * load_before_alight
            alighting_passengers = self.load[train] - staying_passengers
            return alighting_passengers * self.gen.alpha[station]

    # Returns the period until train arrives at station and starts boarding passengers
    def calc_period_till_boarding(self, train, station):
        return self.calc_waiting_time(train)+self.calc_arriving_time(train, station)\
            + self.calc_alighting_time(train, station)

    # Returns true if prev_train is at station and haven't moved to FINISHED yet.
    def is_prev_train_at_station (self, prev_train, station):
        if self.get_current_station(prev_train) == station and self.states[prev_train].state != states.FINISHED:
            return True
        return False

    # This function calculates the potential time that would take train to load passengers at station
    def calc_boarding_time(self, train, station, period_till_boarding):
        # boarded_prev = amount of passengers that would board the previous train. Note that it can be equal to zero if:
        # A. There isn't a previous train. B. Previous train have already passed station.
        boarded_prev = 0
        for prev_train in range(0, train):
            if self.is_prev_train_at_station(prev_train, station):
                prev_train_period_till_boarding = self.calc_period_till_boarding(prev_train, station)
                boarded_prev += self.calc_boarding_time(prev_train, station, prev_train_period_till_boarding) / self.gen.beta[station]
        # TODO: tried to wrap it in a function, didn't work because of recursion I assume, think how to simplify later

        # calculating potential load when the train arrives at the station:
        # potential_addition = passengers that would be added to station's load when train will start loading.
        potential_addition = period_till_boarding * self.gen.lambda_[station] - boarded_prev
        factor = 1 - self.gen.lambda_[station] * self.gen.beta[station] # TODO: rename factor
        pot_station_load = (self.platform[station] + potential_addition) / factor
        # When train arrives at the station, max_load will increase upon boarding because of alighting passengers:
        alighting_passengers = self.calc_alighting_time(train, station) / self.gen.alpha[station]
        pot_max_load = self.gen.lmax - self.load[train] + alighting_passengers
        pot_boarding_passengers = min(pot_max_load, pot_station_load)
        return self.gen.beta[station] * pot_boarding_passengers

    # Calculates the time till train starts loading passengers from the first station:
    def calc_waiting_time(self, train):
        if self.states[train].state == states.WAITING_FOR_FIRST_DEPART:
            return self.start_time[train] - self.time
        return 0

    # Returns estimated departure time from next station
    def calc_est_depart_time(self, train):
        station = self.get_current_station(train)
        # calculating time until train is start boarding passengers:
        period_till_boarding = self.calc_period_till_boarding(train, station)
        boarding_time = self.calc_boarding_time(train, station, period_till_boarding)
        # calculating estimated departure time from station:
        est_depart_time = self.time + period_till_boarding + boarding_time
        # debug print: #TODO: remove and use step printing instead.
        self.debug_print(train, station, est_depart_time) #TODO: implement here step printing
        return est_depart_time

    # This function is called every step and calculates reward to the agent.
    # At this moment the reward is the sum of all differences at some step, divided by 10.
    # TODO: Ask Erez, we might need to change it later somehow I assume, based on learning rate, etc...
    def reward(self):
        diff = 0
        for train in range(self.gen.trains):
            if self.states[train].state != states.FINISHED:
                diff += self.calc_est_time_diff(train)
        return -diff / 10

    def new_state_reward(self):
        reward = self.reward()
        done = (self.states[-1].state == states.FINISHED)
        info = {}
        return self.get_obs(), self.reward(), (self.states[-1].state == states.FINISHED), {}

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
                self.load_before_alight[train] = self.load[train]
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
        # The train is already at the last station:
        if self.states[train].station == self.gen.stations - 1:
            self.states[train].state = states.FINISHED
        else:
            if effective_epoch > 0:
                # distance covered in effective epoch:
                eff_epoch_dist = effective_epoch * speed
                next_station_dist = (self.gen.km_between_stations - (self.location[train]) %
                                      self.gen.km_between_stations)
                moving_dist = min(eff_epoch_dist, next_station_dist)
                moving_time = moving_dist / speed
                self.location[train] += moving_dist
                if eff_epoch_dist >= next_station_dist:
                    self.states[train].station += 1
                    # updates the load on the train right after we reach next station and begins unloading:
                    self.load_before_alight[train] = self.load[train]
                    self.Unload(train, effective_epoch - moving_time)

    def step(self, epoch=300, noise=0):
        self.time = self.time + epoch
        if self.time == 40200:
            pass
        for i in range(self.gen.stations):
            if self.gen.open_time[i] <= self.time <= self.gen.close_time[i]:
                self.platform[i] = self.platform[i] + (
                            self.gen.lambda_[i] + noise * random.uniform(-0.027, 0.05)) * epoch

        reward = self.reward()

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

        done = (self.states[-1].state == states.FINISHED)

        return self.get_obs(), reward, done, {}

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
        # TDOO: INSIDE SETP WE ARE USING EPOCH BUT DON'T WE CALL STEP WHEN CALCULATING THE REWARDS?

        # self.sys.agent_speed = action
        # dummy agent:
        self.sys.agent_speed = np.zeros(self.sys.gen.trains)
        # for debugging:
        return self.sys.step()

    def render(self, mode='human'):
        pass
