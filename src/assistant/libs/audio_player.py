import wave
from typing import Iterator, IO
import pyaudio
from pydub import AudioSegment


class _FrameProvider:
    def has_frames(self):
        pass

    def frames(self) -> Iterator[bytes]:
        pass

    def close(self):
        pass


class WavFrameProvider(_FrameProvider):
    def __init__(self, wave_read: wave.Wave_read):
        self._wave_read = wave_read

    def frames(self):
        chunk = 512
        for frame in self._wave_read.readframes(chunk):
            yield frame

    def close(self):
        self._wave_read.close()


class OpusFrameProvider(_FrameProvider):
    def __init__(self, audio_data: IO[bytes], channels, rate):
        import opuslib
        self._audio_data = audio_data
        self.decoder = opuslib.Decoder(rate, channels)

    def frames(self) -> Iterator[bytes]:
        return [self.decoder.decode(self._audio_data.read(), frame_size=960)]


class Mp3FrameProvider(_FrameProvider):
    def __init__(self, audio_seg: AudioSegment):
        self._audio_seg = audio_seg

    def frames(self) -> Iterator[bytes]:
        return [self._audio_seg.raw_data]


def play_audio(audio_bytes: IO[bytes], audio_ext: str = "mp3"):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    if audio_ext == "opus":
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 48000
        frame_provider = OpusFrameProvider(audio_bytes, channels, rate)
    elif audio_ext == "mp3":
        audio_seg = AudioSegment.from_mp3(audio_bytes)
        audio_format = p.get_format_from_width(audio_seg.sample_width)
        channels = audio_seg.channels
        rate = audio_seg.frame_rate
        frame_provider = Mp3FrameProvider(audio_seg)
    else:
        wf = wave.open(audio_bytes, 'rb')
        audio_format = p.get_format_from_width(wf.getsampwidth())
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frame_provider = WavFrameProvider(wf)

    # Open a stream
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=rate,
                    output=True)

    for frame in frame_provider.frames():
        stream.write(frame)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()

    # Close the WAV file
    frame_provider.close()
