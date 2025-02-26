import argparse
import queue
import sys
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class STT_vosk:
    def __init__(self):
        self.q = queue.Queue()
        # for headset BT on Orin
        self.samplerate = 16000
        # self.samplerate = cfg.bt_set_samplerate
        self.device = 13
        self.blocksize = 8000
        # for windows
        # self.samplerate = int(sd.query_devices(None, "input")["default_samplerate"])
        # self.device = None
        self.model = Model(lang="en-us")
 
    def int_or_str(self, text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text
 
    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))
 
    def speech_to_text_vosk(self):
        print(self.samplerate,self.blocksize,self.device)
        with sd.RawInputStream(samplerate=self.samplerate, blocksize=self.blocksize, device=self.device,
                               dtype="int16", channels=1, callback=self.callback):
            rec = KaldiRecognizer(self.model, self.samplerate)

            stt_results = ""
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    stt_results = rec.Result().split('"')[1::2][1]
                    if (stt_results != "") and (stt_results != 'huh'):
                        break
 
            return stt_results
        
#stt_node = STT_vosk()
#
#text = stt_node.speech_to_text_vosk()
#
#print(text)
