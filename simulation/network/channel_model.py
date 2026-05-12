from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

import numpy as np


C_MPS = 299_792_458.0
EARTH_ROT_RATE_RAD_S = 7.2921159e-5


class LinkType(Enum):
    LEO_GROUND = "leo_ground"
    HAP_GROUND = "hap_ground"
    GROUND_GROUND = "ground_ground"
    LEO_HAP = "leo_hap"


@dataclass(frozen=True)
class LinkParams:
    k_db: float
    coherence_time_s: float


class SaginChannelModel:
    """Realistic SAGIN channel model with pathloss, fading, Doppler, and temporal correlation."""

    def __init__(self, carrier_freq_hz: float = 2.4e9, bandwidth_hz: float = 20e6, seed: int | None = None):
        self.fc = carrier_freq_hz
        self.bandwidth_hz = bandwidth_hz
        self.rng = np.random.default_rng(seed)
        self.link_params: Dict[LinkType, LinkParams] = {
            LinkType.LEO_GROUND: LinkParams(k_db=10.0, coherence_time_s=25.0),
            LinkType.HAP_GROUND: LinkParams(k_db=6.0, coherence_time_s=10.0),
            LinkType.GROUND_GROUND: LinkParams(k_db=-np.inf, coherence_time_s=2.0),
            LinkType.LEO_HAP: LinkParams(k_db=8.0, coherence_time_s=12.0),
        }
        self._weather_state = "clear"
        self._snr_state: Dict[Tuple[int, int], float] = {}
        self._last_t: Dict[Tuple[int, int], float] = {}

    def classify_link(self, node1, node2) -> LinkType:
        types = {node1.type.value, node2.type.value}
        if "sat" in types and ("ground" in types or "client" in types):
            return LinkType.LEO_GROUND
        if "uav" in types and ("ground" in types or "client" in types):
            return LinkType.HAP_GROUND
        if "sat" in types and "uav" in types:
            return LinkType.LEO_HAP
        return LinkType.GROUND_GROUND

    def _update_weather_state(self):
        # two-state Markov: clear/rain
        if self._weather_state == "clear":
            if self.rng.random() < 0.08:
                self._weather_state = "rain"
        else:
            if self.rng.random() < 0.25:
                self._weather_state = "clear"

    def _atmospheric_loss_db(self) -> float:
        if self._weather_state == "clear":
            return 2.0
        return float(self.rng.choice([3.0, 5.0], p=[0.7, 0.3]))

    def _distance_km(self, node1, node2, t_now):
        p1 = node1.get_position_at(t_now)
        p2 = node2.get_position_at(t_now)
        return max(float(np.linalg.norm(p1 - p2)), 1e-6)

    def _pathloss_linear(self, node1, node2, link_type: LinkType, t_now) -> float:
        d_km = self._distance_km(node1, node2, t_now)
        d_m = d_km * 1000.0
        if link_type in (LinkType.LEO_GROUND, LinkType.LEO_HAP):
            alpha = 1.0
            rho = (C_MPS / (4.0 * np.pi * self.fc * d_m)) ** alpha
            if link_type == LinkType.LEO_GROUND:
                self._update_weather_state()
                loss_lin = 10 ** (-self._atmospheric_loss_db() / 10.0)
                rho *= loss_lin
            return rho
        if link_type == LinkType.HAP_GROUND:
            pl_db = 128.1 + 37.6 * np.log10(max(d_km, 1e-6))
        else:
            pl_db = 46.3 + 33.9 * np.log10(self.fc / 1e6) - 13.82 * np.log10(30.0) + (44.9 - 6.55 * np.log10(30.0)) * np.log10(max(d_km, 1e-6))
        return 10 ** (-pl_db / 10.0)

    def _rician_or_rayleigh(self, link_type: LinkType) -> complex:
        k_db = self.link_params[link_type].k_db
        if np.isneginf(k_db):
            return (self.rng.normal() + 1j * self.rng.normal()) / np.sqrt(2.0)
        k_lin = 10 ** (k_db / 10.0)
        phi = self.rng.uniform(0, 2 * np.pi)
        los = np.sqrt(k_lin / (k_lin + 1.0)) * np.exp(1j * phi)
        nlos = np.sqrt(1.0 / (k_lin + 1.0)) * ((self.rng.normal() + 1j * self.rng.normal()) / np.sqrt(2.0))
        return los + nlos

    def _doppler_hz(self, node1, node2, t_now) -> float:
        vel1 = node1.get_velocity_at(t_now)
        vel2 = node2.get_velocity_at(t_now)
        rel_v = vel1 - vel2
        p1 = node1.get_position_at(t_now)
        p2 = node2.get_position_at(t_now)
        los = p1 - p2
        los_hat = los / (np.linalg.norm(los) + 1e-12)
        radial = float(np.dot(rel_v, los_hat)) * 1000.0
        omega = np.array([0.0, 0.0, EARTH_ROT_RATE_RAD_S])
        v_earth = np.cross(omega, (p2 * 1000.0))
        radial -= float(np.dot(v_earth, los_hat))
        horiz = np.array([los_hat[0], los_hat[1], 0.0])
        cos_azim = np.linalg.norm(horiz)
        cos_elev = np.sqrt(max(0.0, 1.0 - los_hat[2] ** 2))
        return (self.fc / C_MPS) * radial * cos_elev * max(cos_azim, 1e-6)

    def channel_coefficient(self, node1, node2, t_now) -> complex:
        link_type = self.classify_link(node1, node2)
        rho = self._pathloss_linear(node1, node2, link_type, t_now)
        fading = self._rician_or_rayleigh(link_type)
        f_d = self._doppler_hz(node1, node2, t_now)
        t_s = float(getattr(t_now, "tt", 0.0) * 86400.0) if t_now is not None else 0.0
        phase = np.exp(1j * 2.0 * np.pi * f_d * t_s)
        return rho * phase * fading

    def snr(self, node1, node2, t_now=None) -> float:
        h = self.channel_coefficient(node1, node2, t_now)
        raw_snr = node1.power * (np.abs(h) ** 2) / max(node1.noise_variance, 1e-15)

        key = tuple(sorted((node1.id, node2.id)))
        if t_now is None:
            self._snr_state[key] = float(raw_snr)
            return float(raw_snr)
        t_s = float(getattr(t_now, "tt", 0.0) * 86400.0)
        prev = self._snr_state.get(key, float(raw_snr))
        dt = max(0.0, t_s - self._last_t.get(key, t_s))
        tc = self.link_params[self.classify_link(node1, node2)].coherence_time_s
        decay = np.exp(-dt / max(tc, 1e-6))
        eta = self.rng.normal(0.0, 1.0)
        correlated = prev * decay + np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * dt / max(tc, 1e-6)))) * eta
        snr_val = max(1e-12, 0.6 * raw_snr + 0.4 * correlated)
        self._snr_state[key] = float(snr_val)
        self._last_t[key] = t_s
        return float(snr_val)
