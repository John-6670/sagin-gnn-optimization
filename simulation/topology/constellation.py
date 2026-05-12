from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

MU_EARTH_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137


@dataclass
class WalkerConstellation:
    num_planes: int
    sats_per_plane: int
    phasing: int = 1
    inclination: float = 53.0
    altitude_km: float = 550.0

    def _mean_motion_rev_per_day(self) -> float:
        a = EARTH_RADIUS_KM + self.altitude_km
        n_rad_s = np.sqrt(MU_EARTH_KM3_S2 / (a ** 3))
        return n_rad_s * 86400.0 / (2.0 * np.pi)

    def _format_epoch(self, t0) -> str:
        dt = t0.utc_datetime() if t0 is not None else datetime.now(timezone.utc)
        yy = dt.year % 100
        day = dt.timetuple().tm_yday + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
        return f"{yy:02d}{day:012.8f}"

    def generate_tle_list(self, t0=None) -> List[Tuple[str, str]]:
        epoch = self._format_epoch(t0)
        n = self._mean_motion_rev_per_day()
        tles = []
        satnum = 10000
        for p in range(self.num_planes):
            raan = (360.0 / self.num_planes) * p
            for s in range(self.sats_per_plane):
                satnum += 1
                ma = (360.0 / self.sats_per_plane) * s + self.phasing * 360.0 / (self.num_planes * self.sats_per_plane) * p
                line1 = f"1 {satnum:05d}U 24001A   {epoch}  .00000000  00000-0  00000-0 0  9991"
                line2 = (
                    f"2 {satnum:05d} {self.inclination:8.4f} {raan:8.4f} 0001000"
                    f" 000.0000 {ma % 360.0:8.4f} {n:11.8f}00001"
                )
                tles.append((line1, line2))
        return tles
