from __future__ import annotations
import math
import numpy as np


class TwoStateWeatherMarkov:
    def __init__(self, p_clear_to_rain=None, p_rain_to_clear=None, dt_seconds=10.0, seed=None):
        base_p = 1.0 - math.exp(-dt_seconds / 3600.0)
        self.p_clear_to_rain = base_p if p_clear_to_rain is None else p_clear_to_rain
        self.p_rain_to_clear = base_p if p_rain_to_clear is None else p_rain_to_clear
        self.state = "clear"
        self.rng = np.random.default_rng(seed)
    
    def step(self):
        u = self.rng.random()
        if self.state == "clear" and u < self.p_clear_to_rain:
            self.state = "rainy"
        elif self.state == "rainy" and u < self.p_rain_to_clear:
            self.state = "clear"
        
        return self.state

    def atmospheric_loss_db(self):
        return 2.0 if self.state == "clear" else 4.0
