#!/usr/bin/env python
"""Real-time stem mixer: mixes a vocals.wav and no_vocals.wav pair with
independently controllable, smoothed gains. Run standalone for a hardcoded-gain
sanity check; import AudioEngine to drive gains live from the gesture loop."""
import sys
import threading

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf

BLOCK_SIZE = 1024
SMOOTHING = 0.15  # new_gain*this + old_gain*(1-this), applied per audio block

BASS_SHELF_HZ = 150     # typical "bass" control point
TREBLE_SHELF_HZ = 4000  # typical "treble" control point
EQ_SCALE = 5.0          # incoming bass/treble values range -5..+5
EQ_MAX_DB = 12          # -5..+5 maps to -12dB..+12dB


def shelf_coeffs(kind, freq_hz, gain_db, samplerate):
    """RBJ Audio EQ Cookbook low-shelf/high-shelf biquad coefficients."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / samplerate
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) + 2)  # slope=1 (gentle shelf)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if kind == 'low':
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    else:
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


class AudioEngine:
    def __init__(self, vocals_path, instrumental_path, initial_fade=1.0):
        vocals, sr1 = sf.read(vocals_path, dtype='float32', always_2d=True)
        instrumental, sr2 = sf.read(instrumental_path, dtype='float32', always_2d=True)
        assert sr1 == sr2, "vocals/instrumental sample rates differ"
        n = min(len(vocals), len(instrumental))
        self.vocals = vocals[:n]
        self.instrumental = instrumental[:n]
        self.samplerate = sr1
        self.channels = self.vocals.shape[1]

        self._pos = 0
        self._lock = threading.Lock()
        # Targets set from outside (gesture thread); actual gains smoothed each block.
        self.target_v_gain = 1.0
        self.target_inst_gain = 1.0
        self._v_gain = 1.0
        self._inst_gain = 1.0
        self._stream = None

        # Master fade envelope (0..1), used for crossfading between songs.
        # Linear, sample-accurate ramp over an explicit duration -- unlike
        # the exponential SMOOTHING above, this needs a precise, adjustable
        # duration since the user controls it directly.
        self._fade_value = float(initial_fade)
        self._fade_start_value = float(initial_fade)
        self._fade_target = float(initial_fade)
        self._fade_duration_samples = 1
        self._fade_progress = 1

        # Bass/treble: -1..+1, 0 = flat/no change. Smoothed like the gains;
        # filter state (zi) carried across blocks per channel to avoid clicks.
        self.target_bass = 0.0
        self.target_treble = 0.0
        self._bass_db = 0.0
        self._treble_db = 0.0
        self._zi_bass = [np.zeros(2) for _ in range(self.channels)]
        self._zi_treble = [np.zeros(2) for _ in range(self.channels)]

    def set_gains(self, v_gain, inst_gain):
        with self._lock:
            self.target_v_gain = float(np.clip(v_gain, 0.0, 1.5))
            self.target_inst_gain = float(np.clip(inst_gain, 0.0, 1.5))

    def set_eq(self, bass, treble):
        with self._lock:
            self.target_bass = float(np.clip(bass, -EQ_SCALE, EQ_SCALE))
            self.target_treble = float(np.clip(treble, -EQ_SCALE, EQ_SCALE))

    def fade_to(self, target, duration_seconds):
        """Linearly ramp the master output envelope to `target` (0..1) over
        `duration_seconds`. duration_seconds=0 jumps immediately."""
        with self._lock:
            self._fade_start_value = self._fade_value
            self._fade_target = float(target)
            self._fade_duration_samples = max(1, int(duration_seconds * self.samplerate))
            self._fade_progress = 0

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        with self._lock:
            tv, ti = self.target_v_gain, self.target_inst_gain
            t_bass, t_treble = self.target_bass, self.target_treble
            start = self._pos

        end = start + frames
        if start >= len(self.vocals):
            outdata.fill(0)
            raise sd.CallbackStop
        v_chunk = self.vocals[start:end]
        i_chunk = self.instrumental[start:end]
        got = len(v_chunk)

        # Smooth gain toward target once per block (good enough to avoid clicks).
        self._v_gain = SMOOTHING * tv + (1 - SMOOTHING) * self._v_gain
        self._inst_gain = SMOOTHING * ti + (1 - SMOOTHING) * self._inst_gain
        self._bass_db = SMOOTHING * (t_bass / EQ_SCALE * EQ_MAX_DB) + (1 - SMOOTHING) * self._bass_db
        self._treble_db = SMOOTHING * (t_treble / EQ_SCALE * EQ_MAX_DB) + (1 - SMOOTHING) * self._treble_db

        mixed = self._v_gain * v_chunk + self._inst_gain * i_chunk

        # Recompute shelf filters from the smoothed dB value each block and
        # carry filter state (zi) across blocks per channel for click-free EQ.
        if abs(self._bass_db) > 0.05:
            b, a = shelf_coeffs('low', BASS_SHELF_HZ, self._bass_db, self.samplerate)
            for c in range(self.channels):
                mixed[:, c], self._zi_bass[c] = scipy.signal.lfilter(b, a, mixed[:, c], zi=self._zi_bass[c])
        if abs(self._treble_db) > 0.05:
            b, a = shelf_coeffs('high', TREBLE_SHELF_HZ, self._treble_db, self.samplerate)
            for c in range(self.channels):
                mixed[:, c], self._zi_treble[c] = scipy.signal.lfilter(b, a, mixed[:, c], zi=self._zi_treble[c])

        # Master fade envelope (crossfading between songs): sample-accurate
        # linear ramp, computed fresh each block from shared progress state.
        with self._lock:
            fs, ft, fd, fp = (self._fade_start_value, self._fade_target,
                              self._fade_duration_samples, self._fade_progress)
        p0, p1 = fp, min(fp + got, fd)
        n_ramp = max(0, p1 - p0)
        if fd <= 0 or p0 >= fd:
            envelope = np.full(got, ft, dtype=np.float32)
        else:
            t0, t1 = p0 / fd, p1 / fd
            ramp = np.linspace(fs + (ft - fs) * t0, fs + (ft - fs) * t1, n_ramp, dtype=np.float32)
            envelope = np.concatenate([ramp, np.full(got - n_ramp, ft, dtype=np.float32)]) if n_ramp < got else ramp
        mixed *= envelope[:, None]
        with self._lock:
            self._fade_progress = min(fp + got, fd)
            self._fade_value = float(envelope[-1])

        if got < frames:
            outdata[:got] = mixed
            outdata[got:] = 0
            raise sd.CallbackStop
        outdata[:] = mixed
        with self._lock:
            self._pos = end

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def toggle_pause(self):
        """Pause/resume in place -- playback position (self._pos) is untouched
        since the callback simply isn't invoked while the stream is stopped."""
        if self._stream is None:
            return
        if self._stream.active:
            self._stream.stop()
        else:
            self._stream.start()

    @property
    def is_playing(self):
        return self._stream is not None and self._stream.active

    def seek(self, delta_seconds):
        with self._lock:
            new_pos = self._pos + int(delta_seconds * self.samplerate)
            self._pos = int(np.clip(new_pos, 0, len(self.vocals) - 1))

    @property
    def position_seconds(self):
        return self._pos / self.samplerate

    @property
    def duration_seconds(self):
        return len(self.vocals) / self.samplerate


if __name__ == '__main__':
    import time

    song_dir = 'separated/htdemucs/Harleys In Hawaii'
    engine = AudioEngine(f'{song_dir}/vocals.wav', f'{song_dir}/no_vocals.wav')
    print(f"Playing {engine.samplerate}Hz, {len(engine.vocals)/engine.samplerate:.1f}s. Ctrl+C to stop.")
    engine.start()
    try:
        segments = [
            ("INSTRUMENTAL ONLY", 0.0, 1.0),
            ("VOCALS ONLY", 1.0, 0.0),
            ("BOTH", 0.9, 0.9),
        ]
        for label, v, i in segments:
            print(f"--- {label} ---")
            engine.set_gains(v, i)
            time.sleep(10)
            if not engine._stream.active:
                break
    except KeyboardInterrupt:
        pass
    engine.stop()
    print("Done.")
