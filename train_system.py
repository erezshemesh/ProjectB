import numpy as np
from gym.spaces.space import Space
from generator import *
import random
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
 
    def estimated_T_diff(self, train):
        est_depart_time, station = self.time_to_next_depart(train)
        return abs(est_depart_time - self.T[train, station])
 
    def get_next_station(self, train):
        return self.states[train].station + (self.states[train].state == states.MOVING)
 
    def time_to_reach_station(self, train, station):
        if self.states[train].state == states.MOVING:
            train_speed = self.gen.speed_kmh / 3600 + self.agent_speed[train]
            distance_to_next_station = station * self.gen.km_between_stations - self.location[train]
            return distance_to_next_station / train_speed
        return 0
 
    def time_to_alight(self, train, station):
        if self.states[train].state != states.LOADING:
            load_before_alight = self.load_before_alight[train] if self.states[train].state == states.UNLOADING else \
            self.load[train]
            return (self.load[train] - (1 - self.gen.eta[train, station]) * load_before_alight) * \
                   self.gen.alpha[station]
        return 0
 
    def time_to_board(self, train, station, tau):
        max_load = self.gen.lmax - self.load[train]
        steal = 0
        # TODO:add condition - do not for the first one
        for train_ahead in range(0, train):
            if self.get_next_station(train_ahead) == station and self.states[train_ahead].state != states.FINISHED:
                tau2 = self.time_to_wait(train_ahead) +\
                       self.time_to_reach_station(train_ahead, station) +\
                       self.time_to_alight(train_ahead, station)
                steal += self.time_to_board(train_ahead, station, tau2) / self.gen.beta[station]
            a=(self.platform[station] - steal + tau * self.gen.lambda_[station]) / (1 - self.gen.lambda_[station] * self.gen.beta[station])
            b= max_load + self.time_to_alight(train,station) / self.gen.alpha[station]
        return self.gen.beta[station] * min(a,b)
                    
 
    def time_to_wait(self, train):
        if self.states[train].state == states.WAITING_FOR_FIRST_DEPART:
            return self.start_time[train] - self.time
        return 0
 
    def time_to_next_depart(self, train):
        station = self.get_next_station(train)
        waiting_time = self.time_to_wait(train)
        arriving_time = self.time_to_reach_station(train, station)
        alighting_time = self.time_to_alight(train, station)
        tau = arriving_time + alighting_time + waiting_time
        
        pdt_time=self.T[train, station]
        
        boarding_time = self.time_to_board(train, station, tau)
        
        
        est_time = self.time + waiting_time + arriving_time + alighting_time + boarding_time
        ratio=est_time / self.T[train, station]
        
        
        if self.states[train].state != states.FINISHED:
            #print(ratio)
            #if(ratio <0.999): #ratio>1.0001
            print("Time:", self.time, "\tTrain:", train, "\tState:", self.states[train].state, "\tActual Station:",
                station, "\tEstimated Time:", est_time, "\t PDT Time[Train,Actual Station]:", self.T[train, station])
        return est_time, station
 
    def reward(self):
        diff = 0
        for train in range(self.gen.trains):
            if self.states[train].state != states.FINISHED:
                diff += self.estimated_T_diff(train)
        return -diff/10
 
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
        if self.states[train].station == self.gen.stations - 1:
            self.states[train].state = states.FINISHED
        else:
            if effective_epoch > 0:
                potential_move = effective_epoch * speed
                max_move = (self.gen.km_between_stations - (self.location[train]) % self.gen.km_between_stations)
                moving_distance = min(potential_move, max_move)
                moving_time = moving_distance / speed
                self.location[train] += moving_distance
                if potential_move >= max_move:
                    self.states[train].station += 1
                    self.load_before_alight[train] = self.load[train]
                    self.Unload(train, effective_epoch - moving_time)
 
    def step(self, epoch=300, noise=0):
        self.time = self.time + epoch
        if self.time == 40200:
            pass
        for i in range(self.gen.stations):
            if self.gen.open_time[i] <= self.time <= self.gen.close_time[i]:
                self.platform[i] = self.platform[i] + (self.gen.lambda_[i] + noise * random.uniform(-0.027, 0.05)) * epoch
 
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
 
    def debug_print(self, train):
        est_time, next_station = self.time_to_next_depart(train)
        PDT_time = self.T[train, next_station]
        if self.states[train].state != states.FINISHED:
            if 0.99 > (est_time / PDT_time) or (est_time / PDT_time) > 1.01:
                print("Time:", self.time, "\tTrain:", train, "\tState:", self.states[train].state, "\tActual Station:",
                    next_station, "\tEstimated Time:", est_time, "\t PDT Time[Train,Actual Station]:", PDT_time)
 
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
 
        #self.sys.agent_speed = action
        # dummy agent:
        self.sys.agent_speed = np.zeros(self.sys.gen.trains)
        # for debugging:
        return self.sys.step()
 
    def render(self, mode='human'):
        pass
