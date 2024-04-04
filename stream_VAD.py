import collections
import contextlib
import wave
import webrtcvad
import pyaudio
import os
from models.nllb import nllb_translate
from models.TTS_utils import tts_mutli_process
from models.parakeet import parakeet_ctc_process
from concurrent.futures import ThreadPoolExecutor
import time
from models.noise_red import noise_reduction
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_audio(stream, frame_duration_ms, rate):
    """Generates audio frames from the input stream."""
    frames_per_buffer = int(rate * frame_duration_ms / 1000)
    while True:
        yield stream.read(frames_per_buffer)

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames."""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(f for f, speech in ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
                triggered = False
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])




# ...

def process_segment(parakeet, model_nllb, tokinizer_nllb, tts_model, path_segements, path_results):
    print("Processing segment...")
    start_time1 = time.time()

    # separate_audio(noise_model, path_segements, path_segements)
    noise_reduction(path_segements, path_segements)
    end_time1 = time.time()
    print("Noise removal time:", end_time1 - start_time1)
    print("Noise removed.")
    start_time2 = time.time()
    transcription = parakeet_ctc_process(parakeet, path_segements)
    print("Transcription:", transcription[0])
    end_time2 = time.time()
    print("Transcription time:", end_time2 - start_time2)
    if transcription == " " or transcription == "":
        print("No speech detected.")
        return
    else:
        start_time3 = time.time()
        translation = nllb_translate(model_nllb, tokinizer_nllb, transcription[0], "spanish")
        print("Translation:", translation)
        end_time3 = time.time()
        print("Translation time:", end_time3 - start_time3)
        print("Processing TTS...")
        start_time4 = time.time()
        tts_mutli_process(tts_model, translation, path_segements, "es", path_results)
        print("TTS done.")
        end_time4 = time.time()
        print("TTS time:", end_time4 - start_time4)

    

# ...





def stream(parakeet, model_nllb, tokinizer_nllb, tts_model):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
    PADDING_DURATION_MS = 300
    vad = webrtcvad.Vad(1)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=160)
    frames = read_audio(stream, CHUNK_DURATION_MS, RATE)
    frames = (Frame(f, None, None) for f in frames)

    segments_dir = "audio_segments"
    result_dir = "results"
    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    executor = ThreadPoolExecutor(max_workers=2)  # Adjust the number of workers as per your requirement

    for i, segment in enumerate(vad_collector(RATE, CHUNK_DURATION_MS, PADDING_DURATION_MS, vad, frames)):
        path_segements = os.path.join(segments_dir, f"segment_{i}.wav")
        path_results = os.path.join(result_dir, f"result_{i}.wav")
        print(f"Writing {path_segements}...")
        with contextlib.closing(wave.open(path_segements, 'wb')) as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(segment)
        
        executor.submit(process_segment, parakeet, model_nllb, tokinizer_nllb, tts_model, path_segements,path_results)

    stream.stop_stream()
    stream.close()
    audio.terminate()


