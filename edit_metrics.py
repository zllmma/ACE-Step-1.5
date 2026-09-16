"""Metrics for the repaint editing benchmark (dual-axis protocol).

Axes follow the music-editing literature (AUDIT / MusicMagus / SteerMusic /
Melodia): a consistency axis (out-of-region mel fidelity, full-track chroma
similarity) paired with a fidelity axis (in-region CLAP adherence to the
edit caption), combined into a Melodia-style ASB harmonic mean so that
neither axis can be gamed alone.
"""

import numpy as np
import soundfile as sf

SR = 48000


def load_mono(path: str, sr: int = SR) -> np.ndarray:
    """Load an audio file as mono float32 at the requested sample rate."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def load_log_mel(path: str) -> np.ndarray:
    """Load an audio file as a log-mel spectrogram [n_mels, frames]."""
    import torch
    import torchaudio

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).T
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, n_mels=128
    )(wav)
    return (mel + 1e-6).log().squeeze(0).numpy()


def region_mel_l1(src: str, edited: str, start: float, end: float) -> float:
    """Mean log-mel L1 OUTSIDE [start, end) between source and edit.

    Lower is better: the untouched region should be preserved bit-for-bit
    up to VAE round-trip noise.
    """
    mel_a, mel_b = load_log_mel(src), load_log_mel(edited)
    frames = min(mel_a.shape[1], mel_b.shape[1])
    mel_a, mel_b = mel_a[:, :frames], mel_b[:, :frames]
    keep = np.ones(frames, dtype=bool)
    lo, hi = int(start * SR / 512), int(end * SR / 512)
    keep[lo:hi] = False
    if keep.sum() == 0:
        return float("nan")
    return float(np.abs(mel_a[:, keep] - mel_b[:, keep]).mean())


def chroma_similarity(src: str, edited: str) -> float:
    """Cosine similarity between mean chroma vectors (MusicMagus protocol).

    Higher is better: chroma captures pitch/harmony content, so this measures
    how much musical material survives the edit overall.
    """
    import librosa

    vecs = []
    for path in (src, edited):
        y = load_mono(path, sr=22050)
        chroma = librosa.feature.chroma_stft(y=y, sr=22050)
        vecs.append(chroma.mean(axis=1))
    v1, v2 = np.asarray(vecs[0]), np.asarray(vecs[1])
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(v1 @ v2 / denom) if denom > 0 else float("nan")


def cqt_pcc(src: str, edited: str) -> float:
    """CQT1-PCC melody consistency (SteerMusic protocol).

    Extracts the dominant CQT channel (max per frame) from each clip and
    computes the Pearson correlation between the two contours. Higher is
    better: it tracks whether the lead melodic line survives the edit.
    """
    import librosa

    contours = []
    for path in (src, edited):
        y = load_mono(path, sr=22050)
        cqt = np.abs(librosa.cqt(y, sr=22050, fmin=librosa.note_to_hz("C2"),
                                 n_bins=60, bins_per_octave=12))
        contours.append(cqt.max(axis=0))
    f = min(len(contours[0]), len(contours[1]))
    a, b = contours[0][:f], contours[1][:f]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class ClapScorer:
    """Thin wrapper over laion_clap with the music checkpoint.

    The default 630k-audioset-best.pt on HF links a stale projection shape
    ([512, 768]) incompatible with laion_clap 1.1.4's HTSAT-base config
    ([512, 1024]); the music checkpoint is the maintained one.
    """

    def __init__(self, ckpt: str, device: str = "cpu"):
        from laion_clap import CLAP_Module

        self.model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        self.model.load_ckpt(ckpt=ckpt, verbose=False)

    def audio_embedding(self, samples: np.ndarray) -> np.ndarray:
        """Embed a mono float32 [-1, 1] waveform segment ([T] or [T, ch])."""
        if samples.ndim == 1:
            samples = samples[None, :]
        return self.model.get_audio_embedding_from_data(
            x=samples.astype(np.float32), use_tensor=False
        )[0]

    def text_embedding(self, captions: list[str]) -> np.ndarray:
        """Embed caption strings, one row per caption."""
        return self.model.get_text_embedding(captions, use_tensor=False)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / denom) if denom > 0 else float("nan")


def _norm(values: list[float]) -> np.ndarray:
    """Z-score normalize then min-max scale into [0, 1]."""
    arr = np.asarray(values, dtype=np.float64)
    mean, std = arr.mean(), arr.std()
    z = (arr - mean) / (std if std > 1e-12 else 1.0)
    lo, hi = z.min(), z.max()
    return (z - lo) / (hi - lo) if hi - lo > 1e-12 else np.full_like(z, 0.5)


def asb_scores(clap_scores: list[float], preserve_l1: list[float]) -> list[float]:
    """Adherence-Structure Balance (Melodia-style, repaint-adapted).

    Harmonic mean of normalized CLAP adherence and normalized out-of-region
    preservation (inverted: lower mel L1 means better preservation). Either
    axis collapsing drives the score to ~0, mirroring F1 behaviour.
    """
    adherence = _norm(clap_scores)
    structure = _norm([-v for v in preserve_l1])
    out = []
    for a, s in zip(adherence, structure):
        out.append(0.0 if (a + s) == 0 else float(2 * a * s / (a + s)))
    return out
