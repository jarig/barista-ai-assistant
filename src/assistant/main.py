import math
import os
import random
import struct
import wave
from io import BytesIO
import pvporcupine
import pyaudio
from pvrecorder import PvRecorder
from src.aiclient.client import OpenAiClient
# Initialize PyAudio
from libs.audio_player import play_audio

audio = pyaudio.PyAudio()

currentDirPath = os.path.dirname(os.path.realpath(__file__))
OPENAI_KEY = "<KEY>"  # get from https://platform.openai.com/api-keys
PROCUPINE_KEY = "<KEY>"  # register and get from the https://console.picovoice.ai/
RECORD_MAX_SECONDS = 20
silence_threshold = 100  # Define your silence threshold
MAX_SILENCE_DURATION = 2.0    # Duration in seconds to consider as silence
INITIAL_PROMPT = "You are home assistant that helps with everyday duties"


porcupine = pvporcupine.create(
  access_key=PROCUPINE_KEY,
  keywords=['hey barista', 'grapefruit'],
)
ai_client = OpenAiClient(key=OPENAI_KEY, initial_prompt=INITIAL_PROMPT)

devices = PvRecorder.get_available_devices()
for i, device in enumerate(devices):
    print('Device %d: %s' % (i, device))

# select microphone device you need to use
pv_device_index = 0
print(f"Using device: {devices[pv_device_index]}")
keyword_recorder = PvRecorder(
    frame_length=porcupine.frame_length,
    device_index=pv_device_index)

keyword_recorder.start()


def is_silent(frame):
    """
    Calculate the root mean square (RMS) amplitude of each frame. RMS is a common way to measure the amplitude.
    """
    count = len(frame)
    sum_squares = sum([sample**2 for sample in frame])
    rms = math.sqrt(sum_squares / count)
    return rms < silence_threshold


while True:
    # Read audio frame
    if not keyword_recorder.is_recording:
        print("Starting recording")
        keyword_recorder.start()
    pcm = keyword_recorder.read()
    keyword_index = porcupine.process(pcm)
    if keyword_index == 0:
        print("Recording prompt...")
        all_data_files = os.listdir(os.path.join(currentDirPath, "data"))
        listen_start_speech = list(filter(lambda df: df.startswith("listen_"), all_data_files))
        rand_welcome_prompt = random.randint(1, len(listen_start_speech))
        # you can record some samples that will be played on the keyword detection
        # use ai_client.convert_to_speech method for record any arbitrary words and serialize to mp3 file
        with open(f"data/listen_{rand_welcome_prompt}.mp3", "rb") as f:
            play_audio(f, audio_ext="mp3")
        keyword_recorder.stop()

        frame_length = 512
        audio_buffer = BytesIO()

        # Initialize the recorder
        prompt_recorder = PvRecorder(device_index=pv_device_index, frame_length=frame_length)
        prompt_recorder.start()

        # save recording into the .wav format
        pvWavfile = wave.open(audio_buffer, "w")
        pvWavfile.setnchannels(1)
        pvWavfile.setsampwidth(2)
        pvWavfile.setframerate(prompt_recorder.sample_rate)
        pvWavfile.setnframes(prompt_recorder.frame_length)
        sample_in_frame = prompt_recorder.sample_rate / prompt_recorder.frame_length
        iteration_count = int(sample_in_frame * RECORD_MAX_SECONDS)
        seconds_per_iteration = float(RECORD_MAX_SECONDS) / float(iteration_count)
        print(f"Seconds per iteration: {seconds_per_iteration}")
        silent_iter_count = MAX_SILENCE_DURATION / seconds_per_iteration
        silent_count = 0
        for _ in range(iteration_count):
            pcm = prompt_recorder.read()
            if is_silent(pcm):
                silent_count += 1
            else:
                silent_count = 0
            if silent_count > silent_iter_count:
                print(f"Silent for {MAX_SILENCE_DURATION} seconds.")
                break
            pvWavfile.writeframes(struct.pack("h" * len(pcm), *pcm))
        print("Finished recording.")
        prompt_recorder.stop()
        pvWavfile.close()
        audio_buffer.seek(0)
        prompt = ai_client.speech_to_text(audio_buffer)
        print(f"Prompt: {prompt}")
        response = ai_client.text_prompt(prompt)
        if response:
            print(f"Response({response.state}): {response.message}")
            audio_data = ai_client.convert_to_speech(response.message)
            play_audio(audio_data, audio_ext="mp3")
    elif keyword_index == 1:
        break


keyword_recorder.stop()
porcupine.delete()
