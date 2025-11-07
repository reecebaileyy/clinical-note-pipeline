"""Generate synthetic audio and transcript samples for local demos."""

from __future__ import annotations

import math
import wave
from pathlib import Path


def sine_wave(output_path: Path, frequency: float, seconds: float, sample_rate: int = 16000) -> None:
    amplitude = 16000
    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for index in range(int(seconds * sample_rate)):
            value = int(amplitude * math.sin(2 * math.pi * frequency * (index / sample_rate)))
            wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


def main() -> None:
    samples_dir = Path(__file__).parent
    samples_dir.mkdir(parents=True, exist_ok=True)

    sine_wave(samples_dir / "sample_consult.wav", frequency=220.0, seconds=3.0)
    sine_wave(samples_dir / "sample_followup.wav", frequency=261.6, seconds=4.0)


if __name__ == "__main__":
    main()


