import numpy as np
import pandas as pd
import warnings

from src_model.pipeline.model.constants import *
from src_model.pipeline.model.match_input import MatchInput

ROLLING_COLUMNS = [
    "ball_seen",
    "ball_speed",
    "ball_dist_touchline",
    "all_speed_mean",
    "home_nearest_dist",
    "away_nearest_dist",
    "home_minus_away_nearest_dist",
    "home_speed_mean",
    "away_speed_mean",
    "home_centroid_fwd",
    "away_centroid_fwd",
    "all_centroid_fwd",
    "home_in_def_third",
    "home_in_mid_third",
    "home_in_att_third",
    "away_in_def_third",
    "away_in_mid_third",
    "away_in_att_third"
]
SHIFT_COLUMNS = [
    "ball_seen",
    "home_minus_away_nearest_dist"
]
MOTION_FRAMES = 5            # 0.2 s: spans two SkillCorner snapshots, so it measures real movement
MIN_TURN_MOVE_M = 0.2        # ball must move this far before and after for its turn angle to count


def build_features(m: MatchInput) -> pd.DataFrame:
    """Per-frame features of one match (float32, one row per frame)."""
    length, width = m.pitch_length, m.pitch_width
    ball_x = m.tracking["ball_x"].to_numpy(dtype=float, na_value=np.nan) * length
    ball_y = m.tracking["ball_y"].to_numpy(dtype=float, na_value=np.nan) * width
    player_x = slot_array(m.tracking, "x") * length
    player_y = slot_array(m.tracking, "y") * width
    seen = np.isfinite(ball_x) & np.isfinite(ball_y)

    f = {}
    # Ball
    f["ball_seen"] = seen.astype(float)
    f["ball_fwd"] = towards_away_goal(ball_x / length, m.home_attacks_right)
    f["ball_dist_touchline"] = np.minimum(ball_y, width - ball_y)
    f["ball_dist_goalline"] = np.minimum(ball_x, length - ball_x)
    outside = (ball_x < 0) | (ball_x > length) | (ball_y < 0) | (ball_y > width)
    f["ball_out"] = np.where(seen, outside, np.nan)
    f["ball_speed"] = displacement_speed(ball_x, ball_y, seconds=0.2)
    f["secs_since_ball_seen"], f["secs_until_ball_seen"] = seconds_to_ball(seen)

    # Calculate Velocity Vectors for the new kinematic features
    ball_vx, ball_vy = displacement_velocity(ball_x, ball_y, seconds=0.2)
    player_vx, player_vy = displacement_velocity(player_x, player_y, seconds=0.4)
    frame_idx = np.arange(len(m.tracking))

    # Each team
    player_speed = displacement_speed(player_x, player_y, seconds=0.4)
    dist = np.hypot(player_x - ball_x[:, None], player_y - ball_y[:, None])
    player_fwd = towards_away_goal(player_x / length, m.home_attacks_right[:, None])   # 0 = home goal, 1 = away goal
    
    for name, cols in (("home", m.team == 0), ("away", m.team == 1)):
        d = dist[:, cols]
        f[f"{name}_nearest_dist"] = row_min(d)
        f[f"{name}_within_5m"] = np.where(seen, (d <= 5).sum(axis=1), np.nan)
        f[f"{name}_within_10m"] = np.where(seen, (d <= 10).sum(axis=1), np.nan)
        f[f"{name}_speed_mean"] = row_mean(player_speed[:, cols])
        f[f"{name}_centroid_fwd"] = towards_away_goal(row_mean(player_x[:, cols]) / length, m.home_attacks_right)
        f[f"{name}_n_tracked"] = np.isfinite(player_x[:, cols]).sum(axis=1)

        # Players in each third, measured from the team's own goal (players off the pitch count in the end thirds)
        own_fwd = player_fwd[:, cols] if name == "home" else 1 - player_fwd[:, cols]
        anyone = np.isfinite(own_fwd).any(axis=1)
        in_def, in_att = own_fwd < 1 / 3, own_fwd >= 2 / 3
        in_mid = np.isfinite(own_fwd) & ~in_def & ~in_att
        for third, inside in (("def", in_def), ("mid", in_mid), ("att", in_att)):
            f[f"{name}_in_{third}_third"] = np.where(anyone, inside.sum(axis=1), np.nan)

    # Both teams
    f["home_minus_away_nearest_dist"] = f["home_nearest_dist"] - f["away_nearest_dist"]
    f["all_speed_mean"] = row_mean(player_speed)
    f["all_centroid_fwd"] = towards_away_goal(row_mean(player_x) / length, m.home_attacks_right)
    all_centroid_y = row_mean(player_y)
    f["all_centroid_dist_touchline"] = np.minimum(all_centroid_y, width - all_centroid_y)

    # Ball motion: displacement over the 0.2 s before (in) and after (out) each frame
    in_x, in_y, out_x, out_y = in_out_displacement(ball_x, ball_y, MOTION_FRAMES)
    step = MOTION_FRAMES / FPS
    f["ball_accel"] = np.hypot(out_x - in_x, out_y - in_y) / step**2
    f["ball_turn_deg"] = turn_angle(in_x, in_y, out_x, out_y, MIN_TURN_MOVE_M)
    ball_vx, ball_vy = (in_x + out_x) / (2 * step), (in_y + out_y) / (2 * step)

    # Nearest player of either team
    rows = np.arange(len(dist))
    filled = np.where(np.isnan(dist), np.inf, dist)
    nearest = filled.argmin(axis=1)
    has_nearest = np.isfinite(filled[rows, nearest])
    second = np.partition(filled, 1, axis=1)[:, 1]
    f["all_second_nearest_dist"] = np.where(np.isfinite(second), second, np.nan)
    near_vx, near_vy = slot_velocity(player_x, player_y, nearest, MOTION_FRAMES)
    near_vx[~has_nearest] = np.nan
    near_vy[~has_nearest] = np.nan
    f["nearest_speed"] = np.hypot(near_vx, near_vy)
    f["nearest_rel_speed"] = np.hypot(ball_vx - near_vx, ball_vy - near_vy)
    f["nearest_alignment"] = alignment(ball_vx, ball_vy, near_vx, near_vy)
    all_nearest = np.where(has_nearest, filled[rows, nearest], np.nan)
    f["nearest_dist_change"] = np.r_[np.full(MOTION_FRAMES, np.nan),
                                     all_nearest[MOTION_FRAMES:] - all_nearest[:-MOTION_FRAMES]] / step
    f["nearest_persistence_s"] = run_seconds(np.where(has_nearest, nearest, -1), has_nearest)

    # --- Extra features (see EXTRA FEATURES below) ------------------------------
    ball_speed_out = np.hypot(out_x, out_y) / step
    f.update(ball_block(ball_x, ball_y, seen, f["ball_speed"], ball_speed_out, length, width))
    f.update(gap_block(seen, f, ball_x, ball_y, ball_speed_out))
    for name, cols in (("home", m.team == 0), ("away", m.team == 1)):
        attacks_right = m.home_attacks_right if name == "home" else ~m.home_attacks_right
        f.update(team_block(
            name, cols, ~cols, attacks_right, player_x, player_y, player_speed, player_vx,
            dist, ball_x, ball_y, ball_vx, seen, length, width,
        ))
    f.update(touch_block(f, seen))
    f.update(both_teams_block(f, player_speed, player_x, player_y, m.team))

    df = add_time_context(pd.DataFrame(f))
    return add_past_future_context(df).astype("float32")


# =============================================================================
# EXTRA FEATURES
#
# Naming follows what swap_teams relies on, so every new feature stays correct when home and
# away are exchanged:
#   home_* / away_*      the same quantity for each team, measured from that team's point of
#                        view (its own goal, its attacking direction)
#   home_minus_away_*    home value minus away value (sign flips on a swap)
#   anything else        team-neutral (does not change on a swap)
# No new feature uses "_fwd" in its name, since swap_teams maps those to 1 - x.
# =============================================================================
TOUCH_RADIUS_M = 2.0           # a team player this close to the ball can have touched it
TOUCH_TURN_DEG = 20.0          # ball direction change that counts as a touch
TOUCH_SPEED_CHANGE = 2.0       # ball speed change (m/s) over 0.2 s that counts as a touch
ON_BALL_RADIUS_M = 2.0         # nearest player this close "is on the ball"
PRESSURE_RADIUS_M = 3.0        # opponents this close to a team's ball-nearest player press it
STILL_SPEED = 1.0              # m/s, a ball slower than this is "still"
SLOW_PLAYER, FAST_PLAYER = 1.0, 4.0   # m/s
PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH = 16.5, 20.15
TIME_CAP_S = 30.0
GAP_CAP_S = 60.0
CONTEXT_SECONDS = (2, 6)       # past-only and future-only means over these windows
CONTEXT_COLUMNS = [
    "ball_seen", "ball_speed", "ball_out_margin", "all_slow_share", "all_speed_mean",
    "home_minus_away_nearest_dist", "home_on_ball", "away_on_ball", "home_touch", "away_touch",
    "home_vel_attack", "away_vel_attack", "home_pressure", "away_pressure",
]


def ball_block(ball_x, ball_y, seen, speed_in, speed_out, length, width) -> dict:
    """Team-neutral ball features."""
    f = {}
    f["ball_speed_out"] = speed_out                                      # over the next 0.2 s
    f["ball_speed_change"] = speed_out - speed_in
    f["ball_out_margin"] = np.where(seen, np.fmax(np.fmax(-ball_x, ball_x - length),
                                                  np.fmax(-ball_y, ball_y - width)), np.nan)
    goal_dist = np.fmin(np.hypot(ball_x, ball_y - width / 2), np.hypot(ball_x - length, ball_y - width / 2))
    f["ball_dist_nearest_goal"] = goal_dist
    in_box = ((ball_x <= PENALTY_BOX_DEPTH) | (ball_x >= length - PENALTY_BOX_DEPTH)) & \
             (np.abs(ball_y - width / 2) <= PENALTY_BOX_HALF_WIDTH)
    f["ball_in_box"] = np.where(seen, in_box, np.nan)
    for seconds in (1, 3):
        k = seconds * FPS
        moved = np.full(len(ball_x), np.nan)
        moved[k:] = np.hypot(ball_x[k:] - ball_x[:-k], ball_y[k:] - ball_y[:-k])
        f[f"ball_moved_{seconds}s"] = moved
    still = seen & (speed_in < STILL_SPEED)
    f["ball_still_s"] = np.where(seen, np.nan_to_num(run_seconds(still.astype(int), still, TIME_CAP_S)), np.nan)
    return f


def gap_block(seen, f, ball_x, ball_y, speed_out) -> dict:
    """Ball visibility gaps: their length, and the ball state on each side of the gap."""
    n = len(seen)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    has_last, has_next = last >= 0, nxt < n
    li, ni = np.maximum(last, 0), np.minimum(nxt, n - 1)
    g = {}
    g["gap_len_s"] = np.where(seen, 0.0, np.minimum((nxt - last) / FPS, GAP_CAP_S))
    g["gap_position"] = np.where(seen | ~has_last | ~has_next, np.nan, (idx - last) / np.maximum(nxt - last, 1))
    g["gap_jump"] = np.where(~seen & has_last & has_next,
                             np.hypot(ball_x[ni] - ball_x[li], ball_y[ni] - ball_y[li]), np.nan)
    for name in ("ball_out_margin", "ball_dist_touchline", "ball_dist_goalline"):
        short = name.removeprefix("ball_")
        g[f"gap_prev_{short}"] = np.where(~seen & has_last, f[name][li], np.nan)
        g[f"gap_next_{short}"] = np.where(~seen & has_next, f[name][ni], np.nan)
    g["gap_prev_speed"] = np.where(~seen & has_last, f["ball_speed"][li], np.nan)
    g["gap_next_speed"] = np.where(~seen & has_next, speed_out[ni], np.nan)
    diff = f["home_minus_away_nearest_dist"]
    g["home_minus_away_gap_prev_nearest"] = np.where(~seen & has_last, diff[li], np.nan)
    g["home_minus_away_gap_next_nearest"] = np.where(~seen & has_next, diff[ni], np.nan)
    return g


def team_block(name, cols, opp_cols, attacks_right, player_x, player_y, player_speed, player_vx,
               dist, ball_x, ball_y, ball_vx, seen, length, width) -> dict:
    """Features of one team, measured from its own point of view (own goal at x = 0)."""
    f = {}
    n = len(ball_x)
    rows = np.arange(n)
    sign = np.where(attacks_right, 1.0, -1.0)                 # +1: this team attacks towards x = length
    px, py = player_x[:, cols], player_y[:, cols]
    own_x = np.where(attacks_right[:, None], px, length - px)  # metres from the team's own goal line
    ball_own_x = np.where(attacks_right, ball_x, length - ball_x)

    # Ball relative to this team
    f[f"{name}_ball_vel_attack"] = ball_vx * sign             # ball moving towards the goal this team attacks
    f[f"{name}_ball_dist_own_goal"] = np.hypot(ball_own_x, ball_y - width / 2)
    in_own_box = (ball_own_x <= PENALTY_BOX_DEPTH) & (np.abs(ball_y - width / 2) <= PENALTY_BOX_HALF_WIDTH)
    f[f"{name}_ball_in_own_box"] = np.where(seen, in_own_box, np.nan)

    # Ball-to-player distances
    d = np.where(np.isnan(dist[:, cols]), np.inf, dist[:, cols])
    order = np.argsort(d, axis=1)
    ds = np.take_along_axis(d, order, axis=1)
    ds = np.where(np.isfinite(ds), ds, np.nan)
    f[f"{name}_second_nearest_dist"] = ds[:, 1] if ds.shape[1] > 1 else np.full(n, np.nan)
    f[f"{name}_mean3_dist"] = row_mean(ds[:, :3])
    for r in (2, 3, 20):
        f[f"{name}_within_{r}m"] = np.where(seen, (d <= r).sum(axis=1), np.nan)
    near_slot = order[:, 0]
    has_near = np.isfinite(ds[:, 0])
    near_speed = player_speed[:, cols][rows, near_slot]
    f[f"{name}_nearest_player_speed"] = np.where(has_near, near_speed, np.nan)
    k = MOTION_FRAMES
    nd = ds[:, 0]
    change = np.full(n, np.nan)
    change[k:] = (nd[k:] - nd[:-k]) / (k / FPS)
    f[f"{name}_nearest_dist_change"] = change                  # < 0: the team is closing in on the ball

    # Pressure: opponents within PRESSURE_RADIUS_M of this team's ball-nearest player
    nx, ny = px[rows, near_slot], py[rows, near_slot]
    od = np.hypot(player_x[:, opp_cols] - nx[:, None], player_y[:, opp_cols] - ny[:, None])
    f[f"{name}_pressure"] = np.where(has_near & seen, (od <= PRESSURE_RADIUS_M).sum(axis=1), np.nan)

    # Team shape (players only: available whether or not the ball is seen)
    count = np.isfinite(own_x).sum(axis=1)
    sorted_x = np.sort(np.where(np.isfinite(own_x), own_x, np.inf), axis=1)      # missing last
    rows_ok = count >= 1
    top = sorted_x[rows, np.maximum(count - 1, 0)]                                 # most advanced player
    f[f"{name}_topline"] = np.where(rows_ok, top, np.nan)
    defline = sorted_x[:, 1] if sorted_x.shape[1] > 1 else np.full(n, np.inf)    # deepest outfield player
    f[f"{name}_defline"] = np.where(count >= 2, defline, np.nan)
    sorted_y = np.sort(np.where(np.isfinite(py), py, np.inf), axis=1)
    with np.errstate(invalid="ignore"):          # inf - inf on frames with no player of this team
        f[f"{name}_length"] = np.where(rows_ok, top - sorted_x[:, 0], np.nan)
        f[f"{name}_width"] = np.where(rows_ok, sorted_y[rows, np.maximum(count - 1, 0)] - sorted_y[:, 0], np.nan)
    cx, cy = row_mean(px), row_mean(py)
    f[f"{name}_compactness"] = row_mean(np.hypot(px - cx[:, None], py - cy[:, None]))
    f[f"{name}_vel_attack"] = row_mean(player_vx[:, cols] * sign[:, None])   # team moving forward
    f[f"{name}_slow_share"] = row_mean(np.where(np.isfinite(player_speed[:, cols]),
                                                player_speed[:, cols] < SLOW_PLAYER, np.nan))
    off = (px < 0) | (px > length) | (py < 0) | (py > width)
    f[f"{name}_n_off_pitch"] = off.sum(axis=1)
    in_band = np.abs(py - width / 2) <= PENALTY_BOX_HALF_WIDTH
    f[f"{name}_n_in_own_box"] = ((own_x <= PENALTY_BOX_DEPTH) & in_band).sum(axis=1)
    f[f"{name}_n_in_opp_box"] = ((own_x >= length - PENALTY_BOX_DEPTH) & in_band).sum(axis=1)
    f[f"{name}_ball_ahead_of_defline"] = ball_own_x - f[f"{name}_defline"]
    f[f"{name}_n_behind_ball"] = np.where(seen, (own_x < ball_own_x[:, None]).sum(axis=1), np.nan)
    return f


def touch_block(f: dict, seen: np.ndarray) -> dict:
    """
    Who is on the ball and who touched it, and how long ago / until when.

    on_ball: the nearest player overall is within ON_BALL_RADIUS_M and belongs to the team.
    touch:   the ball turns or changes speed while the team's nearest player is within
             TOUCH_RADIUS_M and at least as close as the other team's.
    """
    t = {}
    turn = np.nan_to_num(f["ball_turn_deg"], nan=0.0)
    change = np.abs(np.nan_to_num(f["ball_speed_change"], nan=0.0))
    event = seen & ((turn >= TOUCH_TURN_DEG) | (change >= TOUCH_SPEED_CHANGE))
    dh, da = f["home_nearest_dist"], f["away_nearest_dist"]
    with np.errstate(invalid="ignore"):
        on = {"home": seen & (dh <= ON_BALL_RADIUS_M) & (dh <= da),
              "away": seen & (da <= ON_BALL_RADIUS_M) & (da < dh)}
        touch = {"home": event & (dh <= TOUCH_RADIUS_M) & (dh <= da),
                 "away": event & (da <= TOUCH_RADIUS_M) & (da < dh)}
    for name in ("home", "away"):
        t[f"{name}_on_ball"] = np.where(seen, on[name], np.nan)
        t[f"{name}_touch"] = touch[name].astype(float)
        t[f"{name}_since_on_ball"] = seconds_since(on[name])
        t[f"{name}_until_on_ball"] = seconds_until(on[name])
        t[f"{name}_since_touch"] = seconds_since(touch[name])
        t[f"{name}_until_touch"] = seconds_until(touch[name])
        for seconds in (3,):
            w = seconds * FPS
            tf = pd.Series(touch[name].astype(float))
            t[f"{name}_touches_past{seconds}s"] = tf.rolling(w, min_periods=1).sum().to_numpy()
            t[f"{name}_touches_next{seconds}s"] = tf[::-1].rolling(w, min_periods=1).sum().to_numpy()[::-1]
    for key in ("since_on_ball", "until_on_ball", "since_touch", "until_touch"):
        t[f"home_minus_away_{key}"] = t[f"home_{key}"] - t[f"away_{key}"]
    return t


def both_teams_block(f: dict, player_speed: np.ndarray, player_x, player_y, team) -> dict:
    """Team-neutral features of all players together."""
    b = {}
    q20, q80 = row_quantiles(player_speed, (0.2, 0.8))
    b["all_speed_q20"], b["all_speed_q80"] = q20, q80
    tracked = np.isfinite(player_speed)
    b["all_slow_share"] = row_mean(np.where(tracked, player_speed < SLOW_PLAYER, np.nan))
    b["all_fast_share"] = row_mean(np.where(tracked, player_speed > FAST_PLAYER, np.nan))
    b["all_n_tracked"] = np.isfinite(player_x).sum(axis=1)
    hx, hy = row_mean(player_x[:, team == 0]), row_mean(player_y[:, team == 0])
    ax, ay = row_mean(player_x[:, team == 1]), row_mean(player_y[:, team == 1])
    b["centroid_gap"] = np.hypot(hx - ax, hy - ay)
    for key in ("vel_attack", "defline", "topline", "compactness", "length", "width", "pressure",
                "n_behind_ball", "mean3_dist", "slow_share"):
        b[f"home_minus_away_{key}"] = f[f"home_{key}"] - f[f"away_{key}"]
    return b


def add_past_future_context(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only and future-only means of CONTEXT_COLUMNS, and their difference (a change across the frame)."""
    parts = [df]
    for seconds in CONTEXT_SECONDS:
        w = int(seconds * FPS)
        past = df[CONTEXT_COLUMNS].rolling(w, min_periods=1).mean()
        future = df[CONTEXT_COLUMNS][::-1].rolling(w, min_periods=1).mean()[::-1]
        parts.append(past.add_suffix(f"_past{seconds}s"))
        parts.append(future.add_suffix(f"_next{seconds}s"))
        parts.append((future - past).add_suffix(f"_change{seconds}s"))
    return pd.concat(parts, axis=1)


def seconds_since(flag: np.ndarray, cap: float = TIME_CAP_S) -> np.ndarray:
    """Seconds since `flag` was last True (0 on True frames), capped."""
    idx = np.arange(len(flag))
    last = np.maximum.accumulate(np.where(flag, idx, -1))
    return np.where(last >= 0, np.minimum((idx - last) / FPS, cap), cap)


def seconds_until(flag: np.ndarray, cap: float = TIME_CAP_S) -> np.ndarray:
    """Seconds until `flag` is next True (0 on True frames), capped."""
    return seconds_since(flag[::-1], cap)[::-1]


def row_quantiles(a: np.ndarray, qs: tuple) -> list[np.ndarray]:
    """Quantiles of each row ignoring NaN (nearest rank, one sort for all of them)."""
    s = np.sort(a, axis=1)                                   # NaN sort last
    count = np.isfinite(s).sum(axis=1)
    rows = np.arange(len(s))
    out = []
    for q in qs:
        idx = np.clip(np.rint(q * (count - 1)).astype(int), 0, s.shape[1] - 1)
        out.append(np.where(count > 0, s[rows, idx], np.nan))
    return out


def slot_array(tracking: pd.DataFrame, suffix: str) -> np.ndarray:
    """(n_frames, N_SLOTS) array of one coordinate for every slot."""
    cols = [f"{s}_{suffix}" for s in range(1, N_SLOTS + 1)]
    return tracking[cols].to_numpy(dtype=float, na_value=np.nan)


def towards_away_goal(x_norm: np.ndarray, home_attacks_right: np.ndarray) -> np.ndarray:
    """Position along the pitch from the home goal (0) to the away goal (1)."""
    return np.where(home_attacks_right, x_norm, 1 - x_norm)


def displacement_speed(x: np.ndarray, y: np.ndarray, seconds: float) -> np.ndarray:
    """Speed (m/s) from the displacement over the last `seconds`. NaN if either end is missing."""
    k = int(round(seconds * FPS))
    speed = np.full(x.shape, np.nan)
    speed[k:] = np.hypot(x[k:] - x[:-k], y[k:] - y[:-k]) / (k / FPS)
    return speed


def in_out_displacement(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
    """Displacement over the k frames before each frame and over the k frames after it (NaN at the ends)."""
    in_x, in_y, out_x, out_y = (np.full(x.shape, np.nan) for _ in range(4))
    in_x[k:], in_y[k:] = x[k:] - x[:-k], y[k:] - y[:-k]
    out_x[:-k], out_y[:-k] = in_x[k:], in_y[k:]
    return in_x, in_y, out_x, out_y


def turn_angle(in_x, in_y, out_x, out_y, min_move: float) -> np.ndarray:
    """Angle (degrees, 0 to 180) between the incoming and outgoing movement. NaN if either is too short."""
    in_len, out_len = np.hypot(in_x, in_y), np.hypot(out_x, out_y)
    ok = (in_len >= min_move) & (out_len >= min_move)
    cos = (in_x * out_x + in_y * out_y) / np.where(ok, in_len * out_len, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def slot_velocity(x: np.ndarray, y: np.ndarray, slot: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Velocity (m/s) of the player in `slot[i]` at each frame i, centred over +-k frames."""
    rows = np.arange(len(slot))
    before, after = rows - k, rows + k
    inside = (before >= 0) & (after < len(slot))
    before, after = np.clip(before, 0, len(slot) - 1), np.clip(after, 0, len(slot) - 1)
    dt = 2 * k / FPS
    vx = np.where(inside, (x[after, slot] - x[before, slot]) / dt, np.nan)
    vy = np.where(inside, (y[after, slot] - y[before, slot]) / dt, np.nan)
    return vx, vy


def alignment(ax, ay, bx, by, min_speed: float = 0.5) -> np.ndarray:
    """Cosine of the angle between two velocities: 1 = same direction, -1 = opposite. NaN if either is slow."""
    a, b = np.hypot(ax, ay), np.hypot(bx, by)
    ok = (a >= min_speed) & (b >= min_speed)
    return np.clip((ax * bx + ay * by) / np.where(ok, a * b, np.nan), -1, 1)


def run_seconds(values: np.ndarray, valid: np.ndarray, cap: float = 30.0) -> np.ndarray:
    """Seconds the value has been unchanged up to each frame (capped). NaN where not valid."""
    starts = np.r_[True, values[1:] != values[:-1]]
    run_start = np.flatnonzero(starts)[np.cumsum(starts) - 1]
    seconds = (np.arange(len(values)) - run_start + 1) / FPS
    return np.where(valid, np.minimum(seconds, cap), np.nan)


def displacement_velocity(x: np.ndarray, y: np.ndarray, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity vectors (vx, vy in m/s) from the displacement over the last `seconds`."""
    k = int(round(seconds * FPS))
    vx, vy = np.full(x.shape, np.nan), np.full(y.shape, np.nan)
    dt = k / FPS
    vx[k:] = (x[k:] - x[:-k]) / dt
    vy[k:] = (y[k:] - y[:-k]) / dt
    return vx, vy


def seconds_to_ball(seen: np.ndarray, cap: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the ball was last seen and until it is next seen (0 when seen, capped)."""
    idx = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, len(seen))[::-1])[::-1]
    since = np.where(last >= 0, (idx - last) / FPS, cap)
    until = np.where(nxt < len(seen), (nxt - idx) / FPS, cap)
    return np.minimum(since, cap), np.minimum(until, cap)


def row_min(a: np.ndarray) -> np.ndarray:
    """Minimum of each row ignoring NaN; NaN for all-NaN rows."""
    m = np.min(np.where(np.isnan(a), np.inf, a), axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def row_mean(a: np.ndarray) -> np.ndarray:
    """Mean of each row ignoring NaN; NaN for all-NaN rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # "mean of empty slice"
        return np.nanmean(a, axis=1)


def add_time_context(df: pd.DataFrame) -> pd.DataFrame:
    """Append rolling means and before / after values of the columns that change over time."""
    parts = [df]
    for seconds in ROLLING_SECONDS:
        rolled = df[ROLLING_COLUMNS].rolling(int(seconds * FPS), center=True, min_periods=1).mean()
        parts.append(rolled.add_suffix(f"_r{seconds}s"))
    for seconds in SHIFT_SECONDS:
        k = int(seconds * FPS)
        parts.append(df[SHIFT_COLUMNS].shift(k).add_suffix(f"_m{seconds}s"))    # value `seconds` ago
        parts.append(df[SHIFT_COLUMNS].shift(-k).add_suffix(f"_p{seconds}s"))   # value `seconds` later
    return pd.concat(parts, axis=1)


def swap_teams(features: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """The same frames seen with home and away exchanged."""
    columns = list(features.columns)
    rename = {}
    for c in columns:
        if c.startswith("home_minus_away"):
            continue
        if c.startswith("home_"):
            rename[c] = "away_" + c[len("home_"):]
        elif c.startswith("away_"):
            rename[c] = "home_" + c[len("away_"):]
    swapped = features.rename(columns=rename)[columns].copy()

    differences = [c for c in columns if c.startswith("home_minus_away")]
    swapped[differences] = -swapped[differences]
    forward = [c for c in columns if "_fwd" in c]
    swapped[forward] = 1 - swapped[forward]

    swapped_labels = np.array([{HOME: AWAY, AWAY: HOME}.get(l, l) for l in labels], dtype=object)
    return swapped, swapped_labels