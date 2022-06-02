#!/usr/local/bin/python3
import keyboard
import time
import datetime
import wave
import threading
import pyaudio

import json
from os import path, makedirs

recording = True
DATASET_DIR = path.join('datasets', f'{datetime.datetime.now():%y-%m-%d_%H-%M}')
makedirs(DATASET_DIR, exist_ok=True)


def exit_check(event):
    global recording
    if event.name == 'esc':
        recording = False
        return True
    return False

def key_logging():
    keys = {}
    table = open(path.join(DATASET_DIR, 'inputs.csv'), 'w')

    table.write('key\tstart\tend\n')

    print("Start key logging...")
    while True:
        event = keyboard.read_event()
        if exit_check(event): break

        if event.name not in keys:
            keys[event.name] = event

        if event.event_type == keyboard.KEY_UP and keys[event.name].event_type == keyboard.KEY_DOWN:
            table.write(f'{event.name}\t{keys[event.name].time}\t{event.time}\n')
        keys[event.name] = event if event.event_type != keys[event.name].event_type else keys[event.name]
    print("Finish key logging")

def audio_dumping():
    global recording
    filename = path.join(DATASET_DIR, "keyboard_audio.wav")
    chunk = 4096
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 41000
    # record_seconds = 5

    p = pyaudio.PyAudio()
 
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []

    print("Recording...")
    start_time = time.time() - 0.2
    while recording:
        data = stream.read(chunk)
        # если вы хотите слышать свой голос во время записи
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")

    metadata = {'start_time' : start_time}
    json.dump(metadata, open(path.join(DATASET_DIR, 'audio_metadata.json'), 'w'))

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, "wb")

    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()


def main():
    key_logging_thread = threading.Thread(target=key_logging)
    key_logging_thread.start()
    # key_logging()

    audio_dumping_thread = threading.Thread(target=audio_dumping)
    audio_dumping_thread.start()
    

if __name__ == "__main__":
    main()
