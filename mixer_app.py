#!/usr/bin/env python
"""Live gesture-controlled stem mixer.

Mode select (hold pose ~1s to confirm, either hand):
  one (index finger)   -> Music Player Controls  [not implemented yet]
  two (peace sign)     -> Vocals & Instrumental   [pinch distance sets gain]
  three (3 fingers)    -> Bass & Treble           [not implemented yet]

In Vocals & Instrumental mode: thumb-index pinch distance directly and
continuously sets gain (touching = silent, spread apart = full volume).
Left hand -> vocals, right hand -> instrumental. No classifier involved in
this control -- it's a plain geometric ratio, so there's no "wrong
direction" and no left/right asymmetry to fight.
"""
import copy
import threading
import time
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
import mediapipe as mp

from app import calc_landmark_list, pre_process_landmark, calc_bounding_rect, draw_landmarks, draw_bounding_rect
from model import KeyPointClassifier
from audio_engine import AudioEngine
from utils import CvFpsCalc
import song_loader

MODE_HOLD_SECONDS = 1.0
# ponytail: calibration knobs, not learned -- these are ratios of
# (thumb-index distance) / (wrist-to-middle-knuckle distance), i.e. "how
# open is the pinch relative to this hand's own size". Tune to taste:
# raise PINCH_CLOSED_RATIO if silence isn't fully reached; lower
# PINCH_OPEN_RATIO if full volume needs an uncomfortably wide stretch.
PINCH_CLOSED_RATIO = 0.3
PINCH_OPEN_RATIO = 1.8

# ponytail: geometric (untrained) gesture thresholds -- both are one-off
# universal triggers, not per-hand poses, so a trained class isn't worth the
# extra data collection. Tune the numeric thresholds to taste.
PALM_HOLD_FRAMES = 55        # ~2.3-2.5s at typical ~22fps webcam rate
HEART_HOLD_FRAMES = 25       # ~1-1.2s
TRIGGER_COOLDOWN = 2.0       # seconds after a universal-gesture trigger fires
                             # before it can fire again, regardless of pose
# "Absolute cinema" pose (arms stretched wide framing a rectangle) instead of
# wrists-together -- measured live at wrist_ratio > 6 for this pose, easier
# to hold reliably than bringing wrists close together.
ADD_SONG_WRIST_RATIO = 6.0

# Skip/back (Player mode): gaze-based. Iris position relative to eye
# corners -> horizontal look-ratio, 0.5=center. Held past the threshold for
# GAZE_HOLD_FRAMES (majority vote, same pattern as other hold-gestures)
# triggers a skip.
GAZE_HOLD_FRAMES = 25          # ~1-1.2s
GAZE_LOOK_THRESHOLD = 0.15     # deadzone around 0.5 before counting as a look
GAZE_SKIP_BACK_SECONDS = 10
GAZE_SKIP_FORWARD_SECONDS = 5
GAZE_COOLDOWN = 1.5

# Bass/treble: wrist tilt angle mapped to -1..+1 (0 = hand upright = flat EQ).
TILT_MAX_RADIANS = np.radians(60)  # 60 degrees of tilt reaches full -5/+5 --
                                    # 90 was anatomically uncomfortable to
                                    # sustain (measured max reach ~50-70deg)
EQ_SCALE = 5.0                # bass/treble value range: -5 (cut) .. +5 (boost)

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    KEYPOINT_LABELS = [row.strip() for row in f if row.strip()]
STOP_PLAY_ID = KEYPOINT_LABELS.index('stop_play')
PINCH_OPEN_ID = KEYPOINT_LABELS.index('pinch_open')
PINCH_CLOSED_ID = KEYPOINT_LABELS.index('pinch_closed')
NEUTRAL_ID = KEYPOINT_LABELS.index('neutral')
MODE_NONE = 'NONE'
MODE_PLAYER = 'PLAYER'
MODE_VOCALS_INST = 'VOCALS & INSTRUMENTAL'
MODE_BASS_TREBLE = 'BASS & TREBLE'
SIGN_TO_MODE = {3: MODE_PLAYER, 4: MODE_VOCALS_INST, 5: MODE_BASS_TREBLE}


def hand_scale_of(landmark_list):
    wrist = np.array(landmark_list[0], dtype=np.float32)
    middle_mcp = np.array(landmark_list[9], dtype=np.float32)
    return np.linalg.norm(wrist - middle_mcp)


HAND_MATCH_RATIO = 2.5     # max wrist movement between frames (in hand-scale
                           # units) to count as "the same hand" as last frame
HAND_VACANCY_FRAMES = 10   # ~0.4-0.5s a slot's hand must be absent before a
                           # different hand is allowed to claim that slot --
                           # this is what makes a 3rd/stray hand get ignored
                           # instead of bumping one of your tracked hands out


def track_hand_labels(landmark_lists, active_wrist, vacant_frames, frame_width):
    """Assigns Left/Right by matching each detection to whichever known hand
    (by last-seen wrist position) it's closest to, instead of a stateless
    per-frame x-sort. A hand that doesn't match either known slot (e.g. a
    stray third hand while MediaPipe is capped at reporting 2) is left
    unassigned/ignored until the slot it could fill has been confirmed empty
    for HAND_VACANCY_FRAMES -- not just absent for one flickered frame."""
    n = len(landmark_lists)
    wrists = [np.array(ll[0], dtype=np.float32) for ll in landmark_lists]
    scales = [max(hand_scale_of(ll), 1e-6) for ll in landmark_lists]
    labels_by_index = {}
    used = set()

    for slot in ('Left', 'Right'):
        if active_wrist[slot] is None:
            continue
        best_idx, best_dist = None, None
        for i in range(n):
            if i in used:
                continue
            dist = np.linalg.norm(wrists[i] - active_wrist[slot]) / scales[i]
            if dist < HAND_MATCH_RATIO and (best_dist is None or dist < best_dist):
                best_idx, best_dist = i, dist
        if best_idx is not None:
            labels_by_index[best_idx] = slot
            used.add(best_idx)
            active_wrist[slot] = wrists[best_idx]
            vacant_frames[slot] = 0
        else:
            vacant_frames[slot] += 1
            if vacant_frames[slot] >= HAND_VACANCY_FRAMES:
                active_wrist[slot] = None  # confirmed gone -- slot now free

    free_slots = [s for s in ('Left', 'Right') if active_wrist[s] is None]
    unclaimed = [i for i in range(n) if i not in used]
    if free_slots and unclaimed:
        if len(free_slots) == 2 and len(unclaimed) >= 2:
            # Post-flip (mirror view), smaller x is the user's actual left
            # hand: the earlier "swapped" report traced back to the
            # single-hand MediaPipe-handedness fallback below, not this
            # x-sort -- confirmed by testing, so this stays unswapped.
            order = sorted(unclaimed, key=lambda i: wrists[i][0])
            for slot, idx in (('Left', order[0]), ('Right', order[-1])):
                labels_by_index[idx] = slot
                active_wrist[slot] = wrists[idx]
                vacant_frames[slot] = 0
        elif len(free_slots) == 2 and len(unclaimed) == 1:
            # Only one hand on screen and neither slot has history -- fresh
            # start, no "other hand" to compare position against. Use which
            # half of the (already mirror-flipped) frame it's in, same
            # convention as everywhere else -- NOT MediaPipe's own per-frame
            # handedness label, which can flicker frame to frame for a
            # single hand.
            idx = unclaimed[0]
            slot = 'Left' if wrists[idx][0] < frame_width / 2 else 'Right'
            labels_by_index[idx] = slot
            active_wrist[slot] = wrists[idx]
            vacant_frames[slot] = 0
        elif len(free_slots) == 1:
            slot = free_slots[0]
            idx = min(unclaimed, key=lambda i: wrists[i][0]) if slot == 'Left' else max(unclaimed, key=lambda i: wrists[i][0])
            labels_by_index[idx] = slot
            active_wrist[slot] = wrists[idx]
            vacant_frames[slot] = 0

    return labels_by_index


def pinch_gain(landmark_list):
    thumb = np.array(landmark_list[4], dtype=np.float32)
    index = np.array(landmark_list[8], dtype=np.float32)
    hand_scale = hand_scale_of(landmark_list)
    if hand_scale < 1e-6:
        return None, 0.0
    ratio = np.linalg.norm(thumb - index) / hand_scale
    gain = (ratio - PINCH_CLOSED_RATIO) / (PINCH_OPEN_RATIO - PINCH_CLOSED_RATIO)
    return float(np.clip(gain, 0.0, 1.0)), ratio


def wrist_tilt(landmark_list):
    """Angle of the wrist -> thumb-tip vector versus straight-up, in
    [-EQ_SCALE, +EQ_SCALE] (90 degree tilt = full range). Left = negative,
    right = positive, thumb pointing up = 0. Thumb tip is far from the
    wrist, so this vector stays long and well-conditioned through the whole
    rotation (unlike wrist-to-knuckle, which foreshortens badly when the
    hand's edge faces the camera during a thumbs-up roll -- that caused the
    noisy/random readings). A single-frame angle, not a motion trajectory,
    so no fragile frame-to-frame tracking is involved either."""
    wrist = np.array(landmark_list[0], dtype=np.float32)
    thumb_tip = np.array(landmark_list[4], dtype=np.float32)
    dx, dy = thumb_tip - wrist
    angle = np.arctan2(dx, -dy)  # 0 = straight up, +right, -left
    value = float(np.clip(angle / TILT_MAX_RADIANS, -1.0, 1.0)) * EQ_SCALE
    return value, np.degrees(angle)


# MediaPipe FaceMesh (refine_landmarks=True) indices: iris centers + the
# outer/inner corner of each eye.
_EYE_LANDMARKS = [(468, 33, 133), (473, 263, 362)]  # (iris_center, corner_a, corner_b)


def gaze_ratio(face_landmarks, image_w, image_h):
    """Average horizontal look-ratio across both eyes: 0=fully left corner,
    1=fully right corner, 0.5=centered. Single-frame geometry, same idea as
    pinch/tilt -- no training, just iris position relative to eye corners."""
    ratios = []
    for iris_idx, a_idx, b_idx in _EYE_LANDMARKS:
        iris = face_landmarks.landmark[iris_idx]
        a = face_landmarks.landmark[a_idx]
        b = face_landmarks.landmark[b_idx]
        x_left, x_right = sorted([a.x, b.x])
        span = (x_right - x_left) * image_w
        if span < 1e-3:
            continue
        r = ((iris.x - x_left) * image_w) / (span if span else 1.0)
        ratios.append(float(np.clip(r, 0.0, 1.0)))
    return sum(ratios) / len(ratios) if ratios else None


def wrist_ratio(landmark_lists):
    """Returns None if fewer than 2 hands, else wrist-to-wrist distance
    scaled by average hand size -- smaller means wrists are closer together."""
    if len(landmark_lists) != 2:
        return None
    scales = [hand_scale_of(ll) for ll in landmark_lists]
    avg_scale = sum(scales) / 2
    if avg_scale < 1e-6:
        return None
    w0 = np.array(landmark_lists[0][0], dtype=np.float32)
    w1 = np.array(landmark_lists[1][0], dtype=np.float32)
    return float(np.linalg.norm(w0 - w1) / avg_scale)


# ---------------------------------------------------------------- HUD ----

def draw_translucent_bar(image, y0, y1, alpha=0.45):
    overlay = image.copy()
    cv.rectangle(overlay, (0, y0), (image.shape[1], y1), (0, 0, 0), -1)
    cv.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_gain_bar(image, x, y, w, h, value, label, color):
    cv.rectangle(image, (x, y), (x + w, y + h), (90, 90, 90), 1)
    fill_w = int(w * float(np.clip(value, 0.0, 1.0)))
    if fill_w > 0:
        cv.rectangle(image, (x, y), (x + fill_w, y + h), color, -1)
    cv.putText(image, f"{label} {value:.2f}", (x, y - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


def draw_centered_bar(image, x, y, w, h, value, label, color, scale=1.0):
    """Bar for a -scale..+scale value: fills from the center line outward,
    right for positive (boost), left for negative (cut)."""
    cv.rectangle(image, (x, y), (x + w, y + h), (90, 90, 90), 1)
    mid = x + w // 2
    cv.line(image, (mid, y), (mid, y + h), (150, 150, 150), 1)
    half_fill = int((w / 2) * float(np.clip(abs(value) / scale, 0.0, 1.0)))
    if half_fill > 0:
        if value > 0:
            cv.rectangle(image, (mid, y), (mid + half_fill, y + h), color, -1)
        else:
            cv.rectangle(image, (mid - half_fill, y), (mid, y + h), color, -1)
    cv.putText(image, f"{label} {value:+.2f}", (x, y - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


def draw_centered_text(image, text, cx, y, color, scale=0.45, thickness=1):
    """Text horizontally centered on cx, with a black outline so it stays
    readable over bright/busy backgrounds."""
    if not text:
        return
    (tw, _), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(cx - tw / 2)
    cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale,
               (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, scale,
               color, thickness, cv.LINE_AA)


RING_RADIUS = 42
RING_MARGIN = 70  # distance from the right edge to the ring's center


def draw_hold_ring(image, progress, label):
    """Zelda-style charge wheel: solid dark disc, bright arc filling
    clockwise from the top as a hold-to-confirm gesture progresses.
    Only drawn while a hold is actually in progress (progress > 0)."""
    if progress <= 0:
        return
    h, w = image.shape[:2]
    cx, cy = w - RING_MARGIN, h // 2

    cv.circle(image, (cx, cy), RING_RADIUS, (15, 15, 15), -1)
    cv.circle(image, (cx, cy), RING_RADIUS, (70, 70, 70), 2)

    end_angle = -90 + 360 * float(np.clip(progress, 0.0, 1.0))
    cv.ellipse(image, (cx, cy), (RING_RADIUS - 6, RING_RADIUS - 6), 0,
               -90, end_angle, (0, 255, 255), 6, cv.LINE_AA)

    draw_centered_text(image, f"{int(progress * 100)}%", cx, cy + 5, (255, 255, 255), scale=0.45)
    draw_centered_text(image, label, cx, cy + RING_RADIUS + 20, (0, 255, 255), scale=0.45)


def format_time(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def draw_hud(image, fps, current_mode, v_gain, inst_gain, bass, treble, current_song,
             is_playing, position_seconds, duration_seconds, song_status,
             debug_readout, show_debug, crossfade_seconds, hold_progress, hold_label,
             gaze_enabled):
    h, w = image.shape[:2]
    draw_hold_ring(image, hold_progress, hold_label)

    # Top bar: song + play state + time + mode
    draw_translucent_bar(image, 0, 40)
    play_state = 'PLAYING' if is_playing else 'PAUSED'
    time_str = f"{format_time(position_seconds)}/{format_time(duration_seconds)}"
    top_line = f"{Path(current_song).stem}  [{play_state}]  {time_str}   MODE: {current_mode}"
    if song_status['phase'] == 'separating':
        top_line = "Separating new song's stems... this takes a few minutes, please wait"
    cv.putText(image, top_line, (12, 26), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (255, 255, 255), 1, cv.LINE_AA)

    if current_mode == MODE_PLAYER:
        gaze_msg = "GAZE ACTIVE" if gaze_enabled else "GAZE BLOCKED (Space to unblock)"
        gaze_color = (0, 255, 255) if gaze_enabled else (100, 100, 255)
        draw_centered_text(image, gaze_msg, w // 2, 60, gaze_color, scale=0.7, thickness=2)

    # Gain bars, only meaningful in Vocals & Instrumental mode -- shown
    # dimmed otherwise as a hint of what's available.
    active = current_mode == MODE_VOCALS_INST
    bar_color_v = (80, 200, 255) if active else (70, 70, 70)
    bar_color_i = (120, 255, 150) if active else (70, 70, 70)
    draw_gain_bar(image, 12, 70, 220, 18, v_gain, 'vocals', bar_color_v)
    draw_gain_bar(image, w - 232, 70, 220, 18, inst_gain, 'instrumental', bar_color_i)

    # Bass/treble bars, centered at zero -- same dim-when-inactive treatment.
    bt_active = current_mode == MODE_BASS_TREBLE
    bar_color_bass = (255, 140, 80) if bt_active else (70, 70, 70)
    bar_color_treble = (200, 120, 255) if bt_active else (70, 70, 70)
    draw_centered_bar(image, 12, 105, 220, 18, bass, 'bass', bar_color_bass, scale=EQ_SCALE)
    draw_centered_bar(image, w - 232, 105, 220, 18, treble, 'treble', bar_color_treble, scale=EQ_SCALE)

    # Playback progress: thin green line, sits just above the bottom hint bar.
    draw_translucent_bar(image, h - 42, h - 34)
    if duration_seconds > 0:
        fill_w = int((w - 24) * float(np.clip(position_seconds / duration_seconds, 0.0, 1.0)))
        cv.rectangle(image, (12, h - 40), (12 + fill_w, h - 36), (0, 220, 0), -1)
    cv.rectangle(image, (12, h - 40), (w - 12, h - 36), (90, 90, 90), 1)

    # Debug readout: grey, sits directly above the playback bar at the
    # bottom of the screen instead of up near the meters.
    if show_debug:
        left_cx = 12 + 220 // 2
        right_cx = (w - 232) + 220 // 2
        draw_translucent_bar(image, h - 100, h - 42, alpha=0.75)
        draw_centered_text(image, debug_readout.get('Left', ''), left_cx, h - 84, (170, 170, 170))
        draw_centered_text(image, debug_readout.get('Right', ''), right_cx, h - 84, (170, 170, 170))
        draw_centered_text(image, f"FPS:{fps}   {debug_readout.get('heart', '')}", w // 2, h - 66, (170, 170, 170))
        if current_mode == MODE_PLAYER:
            draw_centered_text(image, debug_readout.get('gaze', ''), w // 2, h - 48, (170, 170, 170))

    # Bottom bar: universal gesture hints
    draw_translucent_bar(image, h - 34, h)
    hint = f"stop-sign = play/pause   Absolute Cinema = add song   [ ] = crossfade {crossfade_seconds:.1f}s   ESC = quit   d = debug"
    if current_mode == MODE_PLAYER:
        hint = (f"look left/right and hold: back {GAZE_SKIP_BACK_SECONDS}s / "
                f"forward {GAZE_SKIP_FORWARD_SECONDS}s   Space=block/unblock   " + hint)
    cv.putText(image, hint, (12, h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.45,
               (200, 200, 200), 1, cv.LINE_AA)


def main():
    print("Choose a song to start...")
    current_song = song_loader.pick_song_file()
    if not current_song:
        print("No song selected, exiting.")
        return
    current_song = song_loader.ensure_in_songs_folder(current_song)
    if not song_loader.is_cached(current_song):
        print(f"Separating stems for {Path(current_song).name} -- this can take a few minutes...")
        song_loader.separate(current_song)

    vocals_path, inst_path = song_loader.stems_for(current_song)
    engine = AudioEngine(str(vocals_path), str(inst_path))
    v_gain, inst_gain = 0.8, 0.8
    bass, treble = 0.0, 0.0
    engine.set_gains(v_gain, inst_gain)
    engine.start()

    song_status = {'phase': 'idle', 'path': None}  # phase: idle|separating|ready

    def separate_worker(path):
        song_loader.separate(path)
        song_status['path'] = path
        song_status['phase'] = 'ready'

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 30)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    # Experimental gaze-skip (Player mode, toggle with Space) -- separate
    # pretrained model, only run when actually needed to save FPS.
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    keypoint_classifier = KeyPointClassifier()

    current_mode = MODE_NONE
    # Rolling window of recent per-frame mode-select signs (None = no
    # mode-select sign that frame). A mode wins by majority vote once the
    # window covers ~MODE_HOLD_SECONDS -- tolerates the odd misclassified
    # frame instead of resetting on any single flicker.
    # ~1s of frames at a typical 20-25fps webcam rate; majority vote inside
    # this window decides the mode, tolerating an occasional flickered frame.
    mode_vote_window = deque(maxlen=20)
    debug_readout = {'Left': 'no hand', 'Right': 'no hand'}
    fps_calc = CvFpsCalc(buffer_len=10)

    # Majority-vote windows (same tolerance trick as mode-select) so one
    # flickered frame doesn't reset the whole hold back to zero.
    palm_vote_window = deque(maxlen=PALM_HOLD_FRAMES)
    heart_vote_window = deque(maxlen=HEART_HOLD_FRAMES)
    # Wall-clock cooldowns, not frame-based flags -- a trigger firing doesn't
    # immediately re-arm just because the pose momentarily looks different
    # for one frame while you relax your hand out of it.
    last_palm_trigger, last_heart_trigger = -999.0, -999.0
    show_debug = False
    crossfade_seconds = 2.0
    consecutive_read_failures = 0
    pending_swap = None  # {'v_path','i_path','song','at_time'} while fading out before a swap
    gaze_enabled = False  # off at launch; Space activates/deactivates it
    gaze_look_left_window = deque(maxlen=GAZE_HOLD_FRAMES)
    gaze_look_right_window = deque(maxlen=GAZE_HOLD_FRAMES)
    last_gaze_trigger = -999.0
    active_wrist = {'Left': None, 'Right': None}  # last-seen wrist position per tracked hand
    vacant_frames = {'Left': 0, 'Right': 0}

    try:
        while True:
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break
            if key == ord('d'):
                show_debug = not show_debug
            if key == ord('['):
                crossfade_seconds = max(0.0, crossfade_seconds - 0.5)
            if key == ord(']'):
                crossfade_seconds = min(5.0, crossfade_seconds + 0.5)
            if key == 32:  # Space: block/unblock gaze-skip if it's misbehaving
                gaze_enabled = not gaze_enabled
                gaze_look_left_window.clear()
                gaze_look_right_window.clear()
            fps = fps_calc.get()

            ret, image = cap.read()
            if not ret:
                # A dropped frame shouldn't kill the whole session (and the
                # audio stream with it) -- only bail if the camera is
                # consistently failing, not on one bad read.
                consecutive_read_failures += 1
                if consecutive_read_failures > 60:
                    break
                continue
            consecutive_read_failures = 0
            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
            h, w = image.shape[:2]

            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            # Only run FaceMesh when it could actually matter -- saves FPS
            # the rest of the time, since it's a second full model.
            face_results = face_mesh.process(rgb) if (gaze_enabled and current_mode == MODE_PLAYER) else None
            rgb.flags.writeable = True

            frame_mode_candidate = None
            palm_frame_true = False
            hold_progress, hold_label = 0.0, ''

            # Gaze-skip: look left/right and hold ~1-1.2s.
            if gaze_enabled and current_mode == MODE_PLAYER:
                look_left = look_right = False
                if face_results and face_results.multi_face_landmarks:
                    g = gaze_ratio(face_results.multi_face_landmarks[0], w, h)
                    if g is not None:
                        debug_readout['gaze'] = f"gaze_ratio={g:.2f} (look <{0.5-GAZE_LOOK_THRESHOLD:.2f}=left, >{0.5+GAZE_LOOK_THRESHOLD:.2f}=right)"
                        look_left = g < 0.5 - GAZE_LOOK_THRESHOLD
                        look_right = g > 0.5 + GAZE_LOOK_THRESHOLD
                else:
                    debug_readout['gaze'] = "gaze: no face"
                gaze_look_left_window.append(look_left)
                gaze_look_right_window.append(look_right)
                gaze_now = time.time()
                for wnd, seek_seconds in ((gaze_look_left_window, -GAZE_SKIP_BACK_SECONDS),
                                           (gaze_look_right_window, GAZE_SKIP_FORWARD_SECONDS)):
                    if (len(wnd) == wnd.maxlen and sum(wnd) / len(wnd) >= 0.7
                            and gaze_now - last_gaze_trigger >= GAZE_COOLDOWN):
                        engine.seek(seek_seconds)
                        last_gaze_trigger = gaze_now
                        gaze_look_left_window.clear()
                        gaze_look_right_window.clear()
                    p = sum(wnd) / len(wnd) if wnd else 0
                    if p > hold_progress:
                        hold_progress, hold_label = p, 'gaze skip'

            if results.multi_hand_landmarks is not None:
                # Slice to 2 defensively -- MediaPipe can occasionally report
                # more than max_num_hands during tracker hand-off glitches.
                hands_this_frame = list(zip(results.multi_hand_landmarks, results.multi_handedness))[:2]
                landmark_lists = [calc_landmark_list(debug_image, hl) for hl, _ in hands_this_frame]
                labels_by_index = track_hand_labels(landmark_lists, active_wrist, vacant_frames, image.shape[1])

                seen_hands = set()
                hand_sign_ids = []
                for idx, (hand_landmarks, _handedness) in enumerate(hands_this_frame):
                    if idx not in labels_by_index:
                        continue  # ignored stray/unconfirmed hand -- see track_hand_labels
                    hand_label = labels_by_index[idx]
                    seen_hands.add(hand_label)

                    landmark_list = landmark_lists[idx]
                    pre_processed = pre_process_landmark(landmark_list)
                    hand_sign_id = keypoint_classifier(pre_processed)
                    hand_sign_ids.append(hand_sign_id)

                    if hand_sign_id in SIGN_TO_MODE and frame_mode_candidate is None:
                        frame_mode_candidate = SIGN_TO_MODE[hand_sign_id]

                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    cv.putText(debug_image, f"{hand_label}:{KEYPOINT_LABELS[hand_sign_id]}",
                               (brect[0], brect[1] - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

                    gain, ratio = pinch_gain(landmark_list)
                    debug_readout[hand_label] = f"{hand_label} pinch_ratio={ratio:.2f} gain={gain:.2f}" if gain is not None else f"{hand_label} pinch: n/a"

                    # Only treat this as an active pinch adjustment when the
                    # hand is actually in a pinch/neutral pose -- if it's
                    # showing one/two/three/stop_play (switching modes or
                    # doing a universal gesture), leave the gain alone.
                    if (current_mode == MODE_VOCALS_INST and gain is not None
                            and hand_sign_id in (PINCH_OPEN_ID, PINCH_CLOSED_ID, NEUTRAL_ID)):
                        if hand_label == 'Left':
                            v_gain = gain
                        else:
                            inst_gain = gain
                        engine.set_gains(v_gain, inst_gain)

                    # Exclusive to the neutral/thumbs-up pose -- other finger
                    # configurations (mode-select signs, pinch, stop_play)
                    # must not move bass/treble.
                    if current_mode == MODE_BASS_TREBLE and hand_sign_id == NEUTRAL_ID:
                        tilt, tilt_deg = wrist_tilt(landmark_list)
                        debug_readout[hand_label] = f"{hand_label} tilt={tilt_deg:.1f}deg value={tilt:+.2f}"
                        if hand_label == 'Left':
                            bass = tilt
                        else:
                            treble = tilt
                        engine.set_eq(bass, treble)

                    if hand_sign_id == STOP_PLAY_ID:
                        palm_frame_true = True

                for missing_hand in ({'Left', 'Right'} - seen_hands):
                    debug_readout[missing_hand] = f"{missing_hand} no hand"

                now = time.time()

                # Universal add-song: "absolute cinema" pose -- wrists far
                # apart AND both hands reading as neutral or stop_play (either
                # one, in any combination -- that's just "hand relaxed/open",
                # not a spread-fingers shape specifically).
                wrist_r = wrist_ratio(landmark_lists)
                both_hands_raised = wrist_r is not None and wrist_r > ADD_SONG_WRIST_RATIO
                both_cinema_shape = len(hand_sign_ids) == 2 and all(
                    sid in (NEUTRAL_ID, STOP_PLAY_ID) for sid in hand_sign_ids)
                # Suppressed while already in Bass & Treble mode -- neutral
                # is the exact pose used for tilting there, so accepting it
                # for cinema too would fire accidentally on a wide two-hand
                # tilt. Outside that mode, neutral/stop_play both still count.
                is_cinema_pose = both_hands_raised and both_cinema_shape and current_mode != MODE_BASS_TREBLE
                if wrist_r is not None:
                    debug_readout['heart'] = f"wrist_ratio={wrist_r:.2f} (need > {ADD_SONG_WRIST_RATIO}) cinema_shape={both_cinema_shape}"
                else:
                    debug_readout['heart'] = "heart: need 2 hands"

                # Universal play/pause: trained stop_play pose, majority vote
                # over the hold window so one flickered frame (more likely
                # with only one hand raised) doesn't reset progress to zero.
                # A wall-clock cooldown (not a pose-based flag) prevents an
                # immediate re-trigger while you relax out of the pose.
                # Suppressed entirely while both hands are raised wide (an
                # Absolute Cinema attempt) so it can't fire just because one
                # hand's shape happens to read as stop_play in that pose.
                if both_hands_raised:
                    palm_frame_true = False
                palm_vote_window.append(palm_frame_true)
                if (len(palm_vote_window) == palm_vote_window.maxlen
                        and sum(palm_vote_window) / len(palm_vote_window) >= 0.7
                        and now - last_palm_trigger >= TRIGGER_COOLDOWN):
                    engine.toggle_pause()
                    last_palm_trigger = now
                    palm_vote_window.clear()
                if palm_frame_true and len(palm_vote_window) > 0:
                    p = sum(palm_vote_window) / len(palm_vote_window)
                    if p > hold_progress:
                        hold_progress, hold_label = p, 'stop/play'
                heart_vote_window.append(is_cinema_pose)
                if (len(heart_vote_window) == heart_vote_window.maxlen
                        and sum(heart_vote_window) / len(heart_vote_window) >= 0.7
                        and now - last_heart_trigger >= TRIGGER_COOLDOWN
                        and song_status['phase'] == 'idle' and pending_swap is None):
                    last_heart_trigger = now
                    heart_vote_window.clear()
                    picked = song_loader.pick_song_file()
                    if picked:
                        picked = song_loader.ensure_in_songs_folder(picked)
                        if song_loader.is_cached(picked):
                            v_path, i_path = song_loader.stems_for(picked)
                            engine.fade_to(0.0, crossfade_seconds / 2)
                            pending_swap = {'v_path': v_path, 'i_path': i_path, 'song': picked,
                                            'at_time': now + crossfade_seconds / 2}
                        else:
                            song_status['phase'] = 'separating'
                            threading.Thread(target=separate_worker, args=(picked,), daemon=True).start()
                if is_cinema_pose and len(heart_vote_window) > 0:
                    p = sum(heart_vote_window) / len(heart_vote_window)
                    if p > hold_progress:
                        hold_progress, hold_label = p, 'add song'
            else:
                debug_readout['Left'] = 'no hand'
                debug_readout['Right'] = 'no hand'
                palm_vote_window.clear()
                heart_vote_window.clear()

            if song_status['phase'] == 'ready' and pending_swap is None:
                v_path, i_path = song_loader.stems_for(song_status['path'])
                engine.fade_to(0.0, crossfade_seconds / 2)
                pending_swap = {'v_path': v_path, 'i_path': i_path, 'song': song_status['path'],
                                'at_time': time.time() + crossfade_seconds / 2}
                song_status['phase'] = 'idle'

            if pending_swap is not None and time.time() >= pending_swap['at_time']:
                engine.stop()
                engine = AudioEngine(str(pending_swap['v_path']), str(pending_swap['i_path']), initial_fade=0.0)
                v_gain, inst_gain = 0.8, 0.8
                bass, treble = 0.0, 0.0
                engine.set_gains(v_gain, inst_gain)
                engine.start()
                engine.fade_to(1.0, crossfade_seconds / 2)
                current_song = pending_swap['song']
                pending_swap = None


            # Mode-select: majority vote over the last ~1s of frames, so one
            # misclassified frame doesn't reset progress back to zero.
            mode_vote_window.append(frame_mode_candidate)
            votes = [m for m in mode_vote_window if m is not None]
            if mode_vote_window.maxlen == len(mode_vote_window) and votes:
                winner = max(set(votes), key=votes.count)
                if votes.count(winner) / len(mode_vote_window) >= 0.6:
                    current_mode = winner
            if frame_mode_candidate is not None and frame_mode_candidate != current_mode:
                p = votes.count(frame_mode_candidate) / mode_vote_window.maxlen
                if p > hold_progress:
                    hold_progress, hold_label = p, f'-> {frame_mode_candidate}'

            draw_hud(debug_image, fps, current_mode, v_gain, inst_gain, bass, treble, current_song,
                     engine.is_playing, engine.position_seconds, engine.duration_seconds,
                     song_status, debug_readout, show_debug, crossfade_seconds,
                     hold_progress, hold_label, gaze_enabled)

            cv.imshow('Gesture Mixer', debug_image)
    finally:
        engine.stop()
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
