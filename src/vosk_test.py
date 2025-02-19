#!/usr/bin/env python3

import argparse
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Queue to hold audio data
q = queue.Queue()

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Default microphone settings
DEVICE_ID = 13  # Your microphone device ID
SAMPLE_RATE = 16000  # Your microphone sample rate

try:
    # Load Vosk model (default: English)
    model = Model(lang="en-us")

    # Open microphone stream
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=DEVICE_ID,
                           dtype="int16", channels=1, callback=callback):
        print("#" * 80)
        print("Recording... Press Ctrl+C to stop.")
        print("#" * 80)

        rec = KaldiRecognizer(model, SAMPLE_RATE)

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                print(rec.Result())  # Final result
            else:
                print(rec.PartialResult())  # Intermediate results

except KeyboardInterrupt:
    print("\nRecording stopped.")
except Exception as e:
    print("Error:", str(e))
