from project_b_tools import *


class Generator:

    def __init__(self,
                 trains,
                 stations,
                 t_alight_per_person=3,
                 t_board_per_person=4,
                 platform_arrivals_per_t=0.3,
                 alight_fraction=0.4,
                 number_of_carts=10,
                 km_between_stations=30,
                 speed_kmh=100,
                 stop_t=0,
                 tmin=180,
                 train_capacity=10000,
                 platform_capacity=100000,
                 var=0
                 ):
        run_t = 3600 * km_between_stations / speed_kmh
        self.km_between_stations = km_between_stations
        self.speed_kmh = speed_kmh
        self.run_t = run_t
        self.trains = trains
        self.stations = stations
        self.shape = (trains, stations)
        self.alpha = random((t_alight_per_person / number_of_carts), var, stations)
        self.beta = random((t_board_per_person / number_of_carts), var, stations)
        self.lambda_ = random(platform_arrivals_per_t, var, stations)
        self.eta = random(alight_fraction, var, trains, stations)
        self.R = random(run_t, var, trains, stations)
        self.S = random(stop_t, var, trains, stations)
        self.tmin = tmin
        self.lmax = train_capacity
        self.pmax = platform_capacity
        self.T = np.full(self.shape, 100)
        self.L = np.full(self.shape, 100)
        self.P = np.random.rand(trains, stations) * 100
        self.V = np.concatenate(
            (
                self.T.reshape(self.T.size),
                self.L.reshape(self.L.size),
                self.P.reshape(self.P.size)
            )
        )
        self.open_time = np.array([to_sec('06:00:00') + run_t * j for j in range(self.stations)])
        self.close_time = np.array(
            [to_sec('23:59:59') - run_t * (self.stations - 1 - j) for j in range(self.stations)])
        self.sol = None

    def extract(self, V, name='all'):
        assert (V is not None and (name == 'T' or name == 'L' or name == 'P' or name == 'all'))
        if name == 'all':
            return self.extract(V, 'T'), self.extract(V, 'L'), self.extract(V, 'P')
        size = int(self.trains * self.stations)
        (start, end) = (0, size) if name == 'T' else ((size, 2 * size) if name == 'L' else (2 * size, 3 * size))
        return V[start:end].reshape(self.trains, self.stations)

    # an objective that doesn't depend on V
    def objective_no_obj(self, V):
        return 1

    # blocked does not mean  blocked 1 time .. it means it arrived sometime not to late ,
    # and didn't depart from the last station
    def objective_min_blocked(self, V):
        T, L, P = self.extract(V)
        total_blocked = 0
        for j in range(self.stations):
            total_blocked = self.block_amount(L, P, self.trains - 1, j)
        return total_blocked

    def objective_max_board(self, V):
        T, L, P = self.extract(V)
        total_boarded = 0
        for i in range(self.trains):
            for j in range(self.stations):
                total_boarded = total_boarded + self.board_amount(L, P, i, j)
        return -total_boarded

    def perfect_objective(self, V):
        return 1
        # this objective should include all of these demands:
        # 1. max board
        # 2. min block (final)
        # 3. min late
        # 4. min wait
        # 5. min temp blocks
        # 6. evenly spaced trains
        # 7. no overload in the far stations for the first train

    def callback(self, V):
        global iterations, max_iterations
        if iterations == 0:
            print(f"starting optimization (max iterations : {max_iterations})")
        print(iterations, end=' ')
        iterations = iterations + 1

    def load_before_alight(self, L, i, j):
        return 0 if j == 0 else L[i, j - 1]

    def alight_amount(self, L, i, j):
        return self.eta[i, j] * self.load_before_alight(L, i, j)

    def load_after_alight(self, L, i, j):
        return self.load_before_alight(L, i, j) - self.alight_amount(L, i, j)

    def free_space_before_board(self, L, i, j):
        return self.lmax - self.load_after_alight(L, i, j)

    def t_alight(self, L, i, j):
        return self.alpha[j] * self.alight_amount(L, i, j)

    def board_amount(self, L, P, i, j):
        return min(
            self.free_space_before_board(L, i, j),
            P[i, j]
        )

    def block_amount(self, L, P, i, j):
        return P[i, j] - self.board_amount(L, P, i, j)

    def t_board(self, L, P, i, j):
        return self.beta[j] * self.board_amount(L, P, i, j)

    def t_arrive_to_station(self, T, i, j):
        return self.open_time[j] if j == 0 else T[i, j - 1] + self.R[i, j - 1]

    def arrive_amount(self, T, i, j):
        start_arriving_time = self.open_time[j] if i == 0 else T[i - 1, j]
        end_arriving_time = T[i, j]
        arriving_period = end_arriving_time - start_arriving_time
        return self.lambda_[j] * arriving_period

    def total_arrivals(self):
        total_arrivals = []
        for j in range(self.stations):
            total_arrivals = total_arrivals + [self.lambda_[j] * (self.close_time[j] - self.open_time[j])]
        return total_arrivals

    def total_not_to_late_arrivals(self, T):
        total_arrivals = []
        for j in range(self.stations):
            total_arrivals = total_arrivals + [self.lambda_[j] * (T[self.trains - 1, j] - self.open_time[j])]
        return total_arrivals

    def total_to_late_arrivals(self, T):
        total_arrivals = []
        for j in range(self.stations):
            total_arrivals = total_arrivals + [self.lambda_[j] * (self.close_time[j] - T[self.trains - 1, j])]
        return total_arrivals

    # constraint (14) in article
    # type: INEQ
    def train_dt(self, V, train, station):
        if station == self.stations - 1:
            return 0
        T = self.extract(V, 'T')
        t = T[train, station + 1] - T[train, station]
        return t - self.tmin

    # doesn't appear in article , trains shouldn't get around prior trains ,
    # there is always at least a tmin time interval between them
    # type: INEQ
    def station_dt(self, V, train, station):
        if train == self.trains - 1:
            return 0
        T = self.extract(V, 'T')
        t = T[train + 1, station] - T[train, station]
        return t - self.tmin

    def min_valid_L(self, V, train, station):
        L = self.extract(V, 'L')
        return L[train, station]

    def max_valid_L(self, V, train, station):
        L = self.extract(V, 'L')
        return self.lmax - L[train, station]

    def min_valid_P(self, V, train, station):
        P = self.extract(V, 'P')
        return P[train, station]

    def max_valid_P(self, V, train, station):
        P = self.extract(V, 'P')
        return self.pmax - P[train, station]

    def min_valid_T(self, V, train, station):
        T = self.extract(V, 'T')
        return T[train, station] - self.open_time[station]

    def max_valid_T(self, V, train, station):
        T = self.extract(V, 'T')
        return self.close_time[station] - T[train, station]

    def first_T(self, V, station):
        T = self.extract(V, 'T')
        return T[0, station] - self.open_time[station]

    def Last_T(self, V, station):
        T = self.extract(V, 'T')
        return self.close_time[station] - T[self.trains - 1, station]

    # rule (17) in article
    # type-  EQ (usually), and INEQ for the first station
    def T_rule(self, V, i, j):
        T, L, P = self.extract(V)
        t_stay = self.S[i, j]
        return T[i, j] - (
                self.t_arrive_to_station(T, i, j) + t_stay + self.t_alight(L, i, j) + self.t_board(L, P, i, j))

    # rule (18) in article
    # type: EQ
    def L_rule(self, V, i, j):
        T, L, P = self.extract(V)
        return L[i, j] - (
                self.load_before_alight(L, i, j) - self.alight_amount(L, i, j) + self.board_amount(L, P, i, j))

    def P_rule(self, V, i, j):
        T, L, P = self.extract(V)
        was_before = 0 if i == 0 else self.block_amount(L, P, i - 1, j)
        return P[i, j] - (self.arrive_amount(T, i, j) + was_before)

    # creates a list of constraints
    @property
    def make_all_constraints(self):
        train_dt = [
            [{'type': 'ineq', 'fun': (lambda x, train=train, station=station: self.train_dt(x, train, station))}
             for train in range(self.trains)]
            for station in range(self.stations)]
        train_dt = [item for sublist in train_dt for item in sublist]

        station_dt = [
            [{'type': 'ineq', 'fun': lambda x, train=train, station=station: self.station_dt(x, train, station)}
             for train in range(self.trains)]
            for station in range(self.stations)]
        station_dt = [item for sublist in station_dt for item in sublist]

        T_rule = [
            [{'type': 'eq', 'fun': lambda x, train=train, station=station: self.T_rule(x, train, station)}
             for train in range(0, self.trains)]
            for station in range(1, self.stations)]
        T_rule = [item for sublist in T_rule for item in sublist]
        T_rule_first_station = [{'type': 'ineq', 'fun': lambda x, train=train: self.T_rule(x, train, 0)} for
                                train in
                                range(0, self.trains)]
        T_rule = T_rule + T_rule_first_station

        L_rule = [
            [{'type': 'eq', 'fun': lambda x, train=train, station=station: self.L_rule(x, train, station)}
             for train in range(0, self.trains)]
            for station in range(0, self.stations)]
        L_rule = [item for sublist in L_rule for item in sublist]

        P_rule = [
            [{'type': 'eq', 'fun': lambda x, train=train, station=station: self.P_rule(x, train, station)}
             for train in range(0, self.trains)]
            for station in range(0, self.stations)]
        P_rule = [item for sublist in P_rule for item in sublist]

        first_T = [{'type': 'ineq', 'fun': lambda x, station=station: self.first_T(x, station)} for station in
                   range(self.stations)]
        last_T = [{'type': 'ineq', 'fun': lambda x, station=station: self.Last_T(x, station)} for station in
                  range(self.stations)]

        return first_T + train_dt + station_dt + T_rule + L_rule + P_rule + last_T

    def assert_results(self, Vsol, min_error=0):
        Tsol, Lsol, Psol = self.extract(Vsol)
        for i in range(self.trains):
            for j in range(self.stations):
                if not (self.min_valid_L(Vsol, i, j) >= -min_error and self.max_valid_L(Vsol, i, j) >= -min_error):
                    print(f"L[{i},{j}]={Lsol[i, j]}")
                if not (self.min_valid_T(Vsol, i, j) >= -min_error and self.max_valid_T(Vsol, i, j) >= -min_error):
                    print(f"T[{i},{j}]={(Tsol[i, j])}   instead of [{self.open_time[j]}, {self.close_time[j]}]")
                if not (self.min_valid_P(Vsol, i, j) >= -min_error and self.max_valid_P(Vsol, i, j) >= -min_error):
                    print(f"P[{i},{j}]={Psol[i, j]}")
                if not (self.train_dt(Vsol, i, j) >= -min_error):
                    print(f"T[{i},{j + 1}] - T[{i}, {j}] = {Tsol[i, j + 1] - Tsol[i, j]} instead of {self.tmin}")
                if not (self.train_dt(Vsol, i, j) >= -min_error):
                    print(f"T[{i + 1},{j}] - T[{i}, {j}] = {Tsol[i + 1] - Tsol[i, j]} instead of {self.tmin}")
                if not (abs(self.T_rule(Vsol, i, j)) <= min_error):
                    if j == 0:
                        if not (self.T_rule(Vsol, i, j) >= -min_error):
                            print(
                                f"T[{i},{j}] = {Tsol[i, j]} instead of at least {Tsol[i, j] - self.T_rule(Vsol, i, j)}")
                    else:
                        print(f"T[{i},{j}] = {Tsol[i, j]} instead of {Tsol[i, j] - self.T_rule(Vsol, i, j)}")
                if not (abs(self.L_rule(Vsol, i, j)) <= min_error):
                    print(f"L[{i},{j}] = {Lsol[i, j]} instead of {Lsol[i, j] - self.L_rule(Vsol, i, j)}")
                if not (abs(self.P_rule(Vsol, i, j)) <= min_error):
                    print(f"P[{i},{j}] = {Psol[i, j]} instead of {Psol[i, j] - self.P_rule(Vsol, i, j)}")

    def assert_results2(self, Vsol):
        Tsol, Lsol, Psol = self.extract(Vsol)
        for i in range(self.trains):
            for j in range(self.stations):
                assert (self.min_valid_L(Vsol, i, j) >= 0 and self.max_valid_L(Vsol, i, j) >= 0)
                assert (self.min_valid_T(Vsol, i, j) >= 0 and self.max_valid_T(Vsol, i, j) >= 0)
                assert (self.min_valid_P(Vsol, i, j) >= 0 and self.max_valid_P(Vsol, i, j) >= 0)
