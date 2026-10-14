"""Map one phone take onto one LIBERO scene: a 20 Hz grip-site path plus open/close commands. Pure numpy.

Units: the take is in the real line frame, cm (takes.py); the scene and the path are sim world frame, metres.

Bowl, object-relative. Progress s(t) = (bowl_x - start_x) / (final_x - start_x) along the real line, and
lift(t) = bowl base height above its rest height. The sim bowl centre follows
    C(t) = c0 + s * (plate - c0) in xy,   c0_z + lift - s * (real_place_dz - place_dz) in z,
so the bowl starts at the sim bowl, ends on the sim plate's centre, and keeps the real lift in absolute cm.
place_dz is SIM_PLACE_DZ for a bowl that starts on the table; for a raised start (tasks 3, 5, 7, 9) it is the
placed height relative to the start (SimScene.place_z), so the path ends at plate level.

Enclosed start (task 4, bowl in an open drawer): the takes lift and carry at the same time, which drags the bowl
into the drawer's front wall. So s is gated by the lift reached so far (min-jerk ramp from clear_lift -
CLEAR_RAMP to clear_lift), and the bowl stays at least clear_lift above its start until s reaches CLEAR_S
(past the drawer front).

Grasp: while held, the grip site sits at a fixed world offset from the bowl centre (SimScene.grasp_offset).
Before close and after release it follows the hand: the pinch point's motion relative to its position at close
(or release), the along-line part scaled by K_ALONG and mapped onto the sim bowl -> plate direction, height as is.

Timing: a min-jerk blend from the robot's start pose (T_BLEND) into the take's last APPROACH_WINDOW seconds
before close; from there the take plays at TIME_SCALE, one factor for every phase and take, so the variation
between takes survives. The path ends RETREAT_S after release (or when the hand leaves the frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .takes import Take

HZ = 20  # control rate of LIBERO and of the 5-shot demos
TIME_SCALE = 0.6  # session medians x 0.6: close 1.17 -> 0.70 s, carry 4.53 -> 2.72 s, release 0.70 -> 0.42 s
APPROACH_WINDOW = 1.0  # real s before close (x 0.6 = 0.6 s): mostly the vertical descent (median 9 cm down,
# 1 cm along); 1.5 s would include the hand's sideways entry from the plate side
T_BLEND = 1.7  # s; total approach 1.7 + 0.6 = 2.3 s, as in the LIBERO demos
RETREAT_S = 1.0  # real s after release
K_ALONG = 11.15 / 16.5  # sim bowl diameter / real bowl diameter, for the hand's along-line motion
SIM_PLACE_DZ = 0.010  # m: sim bowl centre on the plate, above its table rest height
CLEAR_RAMP = 0.08  # m: enclosed start, the lift over which the motion toward the plate fades in
CLEAR_S = 0.45  # enclosed start: progress at which the bowl is past the drawer front (~16 cm of ~36)


@dataclass
class SimScene:
    bowl_c0: np.ndarray  # (3,) target bowl mesh centre at rest
    plate_c: np.ndarray  # (3,) plate mesh centre
    eef0: np.ndarray  # (3,) grip site at the start
    grasp_offset: np.ndarray  # (3,) grip site minus bowl centre while held
    place_z: float | None = None  # bowl centre z when placed; None: c0_z + SIM_PLACE_DZ (start on the table)
    clear_lift: float = 0.0  # enclosed start: lift needed before the bowl moves toward the plate (0: no gate)


@dataclass
class Plan:
    take: str
    pos: np.ndarray  # (n, 3) grip-site target per control step
    grip: np.ndarray  # (n,) -1 open, +1 close
    steps: dict[str, int] = field(default_factory=dict)  # control step of close / lift / placed / release
    grasp_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # pos - bowl centre while held


def _min_jerk(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)[:, None]
    return a + (b - a) * (10 * x**3 - 15 * x**4 + 6 * x**5)


def _at(path: np.ndarray, t: np.ndarray, t0: float) -> np.ndarray:
    return np.array([np.interp(t0, t, path[:, i]) for i in range(3)])


def take_path(take: Take, scene: SimScene, t: np.ndarray) -> np.ndarray:
    """Grip-site target at real take times t (increasing), before the start blend."""
    ev = take.events
    hx, hz = (np.interp(t, take.t, take.hand[:, i]) for i in range(2))
    bx = np.interp(t, take.t, take.bowl_x)
    bz = np.interp(t, take.t, take.bowl_base_z)

    d = scene.plate_c[:2] - scene.bowl_c0[:2]
    u = d / np.linalg.norm(d)
    s = np.clip((bx - take.start_bowl_x) / (take.final_bowl_x - take.start_bowl_x), 0.0, 1.0)
    lift = np.maximum(bz - take.rest_base_z, 0.0) / 100
    if scene.clear_lift > 0:
        x = np.clip((np.maximum.accumulate(lift) - scene.clear_lift + CLEAR_RAMP) / CLEAR_RAMP, 0.0, 1.0)
        s = s * (10 * x**3 - 15 * x**4 + 6 * x**5)
    real_place_dz = (take.final_base_z - take.rest_base_z) / 100
    place_dz = SIM_PLACE_DZ if scene.place_z is None else scene.place_z - scene.bowl_c0[2]
    centre = np.empty((len(t), 3))
    centre[:, :2] = scene.bowl_c0[:2] + s[:, None] * d
    centre[:, 2] = scene.bowl_c0[2] + lift - s * (real_place_dz - place_dz)
    if scene.clear_lift > 0:
        floor = scene.bowl_c0[2] + scene.clear_lift * (
            1 - _min_jerk(np.zeros(1), np.ones(1), (s - CLEAR_S) / 0.25)[:, 0]
        )
        centre[:, 2] = np.maximum(centre[:, 2], np.where(s > 0, floor, -np.inf))
    held = centre + scene.grasp_offset

    def hand_rel(t_ref: float) -> np.ndarray:
        dx = (hx - np.interp(t_ref, take.t, take.hand[:, 0])) / 100
        dz = (hz - np.interp(t_ref, take.t, take.hand[:, 1])) / 100
        return np.concatenate([K_ALONG * dx[:, None] * u, dz[:, None]], 1)

    t_close, t_rel = ev["close"], ev["release"]
    pos = held.copy()
    before, after = t < t_close, t > t_rel
    g_close, g_rel = _at(held, t, t_close), _at(held, t, t_rel)
    pos[before] = g_close + hand_rel(t_close)[before]
    pos[before, 2] = np.maximum(pos[before, 2], g_close[2])  # never below the grasp height on the way in
    pos[after] = g_rel + hand_rel(t_rel)[after]
    return pos


def make_plan(take: Take, scene: SimScene) -> Plan:
    ev = take.events
    t_a = max(ev["close"] - APPROACH_WINDOW, take.t[0])
    t_end = min(ev["release"] + RETREAT_S, ev.get("hand_leaves") or take.t[-1], take.t[-1])
    n = int(np.floor((T_BLEND + TIME_SCALE * (t_end - t_a)) * HZ)) + 1
    tau = np.arange(n) / HZ
    t_real = t_a + np.maximum(tau - T_BLEND, 0) / TIME_SCALE
    pos = take_path(take, scene, t_real)
    blend = tau < T_BLEND
    pos[blend] = _min_jerk(scene.eef0, pos[np.argmax(~blend)], tau[blend] / T_BLEND)
    grip = np.where((t_real >= ev["close"]) & (t_real < ev["release"]) & ~blend, 1.0, -1.0)
    steps = {k: int(np.argmax(t_real >= ev[k])) for k in ("close", "lift", "placed", "release")}
    return Plan(take=take.name, pos=pos, grip=grip, steps=steps, grasp_offset=scene.grasp_offset)
