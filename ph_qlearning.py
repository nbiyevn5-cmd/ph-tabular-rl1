"""
Ödev 11: Endüstriyel atık su nötralizasyon tankında doğrusal olmayan pH dozajlama kontrolü
Yöntem: Tablolu Q-Learning (klasik pekiştirmeli öğrenme, derin ağ YOK)
"""
import sys
import json
import time
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KW = 1e-14
KA = 10 ** -4.76
CT = 2e-3                 
KA2 = 10 ** -7.2
CT2 = 1e-3

PH_GRID = np.linspace(0, 14, 14001)
_H = 10.0 ** -PH_GRID
Z_GRID = (_H - KW / _H - CT * KA / (KA + _H) - CT2 * KA2 / (KA2 + _H))

def z_of_ph(ph):
    return float(np.interp(ph, PH_GRID, Z_GRID))

def ph_of_z(z):
    return float(np.interp(z, Z_GRID[::-1], PH_GRID[::-1]))

DT = 1.0                   
QV = 0.10                  
DEAD = 3
EP_LEN = 240               
BAND = (6.8, 7.2)          
SENSOR_STD = 0.02          

LEVELS = np.array([0, 1e-5, 2.5e-5, 5e-5, 8e-5, 1.2e-4, 1.7e-4, 2.3e-4, 3e-4])
DOSES = np.concatenate([-LEVELS[:0:-1], LEVELS])   
N_SET = len(DOSES)
OFF = len(LEVELS) - 1
MAXLVL = len(LEVELS) - 1
DELTAS = np.array([-4, -2, -1, 0, 1, 2, 4])
N_ACTIONS = len(DELTAS)

PH_EDGES = [3, 4, 5, 5.5, 6, 6.4, 6.6, 6.8, 6.9, 7.0, 7.1, 7.2, 7.4, 7.6, 8, 8.5, 9, 10]
TREND_EDGES = [-0.3, -0.08, 0.08, 0.3]
N_PH, N_TREND = len(PH_EDGES) + 1, len(TREND_EDGES) + 1
N_DIR = 3
N_STATES = N_PH * N_TREND * N_SET * N_DIR

class PHTank:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.z = z_of_ph(self.rng.uniform(4, 10))
        self.queue = deque([0.0] * DEAD)
        self.setting = OFF
        self.set_hist = deque([OFF] * (DEAD + 1), maxlen=DEAD + 1)
        self._new_inlet()
        self.next_event = int(self.rng.integers(30, 70))
        self.t = 0
        self.ph = ph_of_z(self.z)
        self.meas = self.ph + self.rng.normal(0, SENSOR_STD)
        self.prev_meas = self.meas
        return self.obs()

    def _new_inlet(self):
        self.ph_in = self.rng.uniform(3.5, 10.5)
        self.z_in = z_of_ph(self.ph_in)
        self.flow = self.rng.uniform(0.8, 1.2)

    def obs(self):
        ph_bin = int(np.digitize(self.meas, PH_EDGES))
        trend = int(np.digitize(self.meas - self.prev_meas, TREND_EDGES))
        direction = int(np.sign(self.setting - self.set_hist[0])) + 1
        return ((ph_bin * N_TREND + trend) * N_SET + self.setting) * N_DIR + direction

    def step(self, a):
        self.setting = int(a)
        self.set_hist.append(self.setting)
        dose = DOSES[self.setting]
        self.queue.append(dose)
        applied = self.queue.popleft()
        self.z += DT * (QV * self.flow * (self.z_in - self.z) + applied)
        self.t += 1
        if self.t >= self.next_event:
            self._new_inlet()
            self.next_event = self.t + int(self.rng.integers(30, 70))
        self.ph = ph_of_z(self.z)
        self.prev_meas = self.meas
        self.meas = self.ph + self.rng.normal(0, SENSOR_STD)

        in_band = BAND[0] <= self.ph <= BAND[1]
        if in_band:
            r = 1.0
        else:
            dist = min(abs(self.ph - 7.0) - 0.2, 3.0)
            r = -(0.5 + dist)
        r -= 0.1 * abs(self.setting - OFF) / MAXLVL
        return r, in_band

class QPolicy:
    def __init__(self, Q): self.Q = Q
    def reset(self): pass
    def act(self, env):
        k = int(np.argmax(self.Q[env.obs()]))
        return int(np.clip(env.setting + DELTAS[k], 0, N_SET - 1))
