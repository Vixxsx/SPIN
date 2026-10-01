"""Song stem loading with an on-disk separation cache. Demucs separation is
slow (~2x song length on CPU -- measured ~7min for a 3min song), so a song is
only ever separated once; subsequent loads of the same file are instant."""
import shutil
import subprocess
import sys
from pathlib import Path

SEPARATED_ROOT = Path('separated/htdemucs')
SONGS_ROOT = Path('Songs')


def stems_for(song_path):
    name = Path(song_path).stem
    song_dir = SEPARATED_ROOT / name
    return song_dir / 'vocals.wav', song_dir / 'no_vocals.wav'


def is_cached(song_path):
    vocals, instrumental = stems_for(song_path)
    return vocals.exists() and instrumental.exists()


def separate(song_path):
    """Blocking -- run from a background thread, not the UI/gesture loop."""
    subprocess.run(
        [sys.executable, '-m', 'demucs', '--two-stems=vocals', '-o', 'separated', str(song_path)],
        check=True,
    )


def ensure_in_songs_folder(song_path):
    """Copies the picked file into Songs/ if it isn't already there, so every
    song used ever after lives in one place and shows up next time you browse."""
    song_path = Path(song_path)
    SONGS_ROOT.mkdir(exist_ok=True)
    if song_path.resolve().parent == SONGS_ROOT.resolve():
        return str(song_path)
    dest = SONGS_ROOT / song_path.name
    if not dest.exists():
        shutil.copy(song_path, dest)
    return str(dest)


def pick_song_file():
    import tkinter as tk
    from tkinter import filedialog
    SONGS_ROOT.mkdir(exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title='Choose a song',
        initialdir=str(SONGS_ROOT.resolve()),
        filetypes=[('Audio files', '*.mp3 *.wav'), ('All files', '*.*')],
    )
    root.destroy()
    return path or None
