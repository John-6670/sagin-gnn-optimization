import numpy as np

TABLE_III = {
    0: {"urban": 0.5, "rural": 0.2, "hotspot": 1.2},
    1: {"urban": 0.8, "rural": 0.3, "hotspot": 1.4},
    2: {"urban": 1.2, "rural": 0.5, "hotspot": 1.8},
    3: {"urban": 1.0, "rural": 0.4, "hotspot": 1.5},
    4: {"urban": 0.6, "rural": 0.25, "hotspot": 1.25},
}


def generate_traffic(num_clients, mean=1.0):
    return np.random.exponential(mean, size=num_clients)


def _period_from_hour(hour):
    if 0 <= hour < 5: return 0
    if 5 <= hour < 9: return 1
    if 9 <= hour < 17: return 2
    if 17 <= hour < 22: return 3
    return 4


def generate_spatiotemporal_traffic(clients, t_now, area_size_km=2000.0, hotspot_xy_km=(0.0, 0.0)):
    hour = t_now.utc_datetime().hour if t_now is not None else 12
    row = TABLE_III[_period_from_hour(hour)]
    urban_radius = 0.30 * (area_size_km / 2.0)
    rates = {}
    for c in clients:
        x, y = float(c.position[0]), float(c.position[1])
        r = np.sqrt(x*x + y*y)
        base = row["urban"] if r <= urban_radius else row["rural"]
        d_hotspot_m = np.sqrt((x-hotspot_xy_km[0])**2 + (y-hotspot_xy_km[1])**2) * 1000.0
        mult = row["hotspot"] if d_hotspot_m <= 500.0 else 1.0
        rates[c.id] = float(base * mult)
    return rates