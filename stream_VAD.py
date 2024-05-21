import collections
import contextlib
import wave
import webrtcvad
import pyaudio
import os
import librosa
import numpy as np
from models.nllb import nllb_translate
from models.TTS_utils import append_text_order
from models.parakeet import parakeet_ctc_process
from models.es_fastconformer import stt_es_process
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


def is_segment_empty(file_path):
    audio, _ = librosa.load(file_path)
    rms = librosa.feature.rms(y=audio)  # Pass the audio data as an argument
    rms_mean = np.mean(rms)
    print(rms_mean)
    
    if rms_mean < 0.015:
        return True
    else:
        return False

# ...

def process_segment(asr_model, model_nllb, tokenizer_nllb, path_segments, path_results, target_lang, order, json_path_temp, json_path_record):
    print("Processing segment...")
    if is_segment_empty(path_segments):
        print("No speech detected.")
        # remove the empty segment
        os.remove(path_segments)
        return
    # Noise Reduction
    start_time = time.time()
    noise_reduction(path_segments, path_segments)
    print("Noise removed. Time:", time.time() - start_time)
    
    
    # Transcription
    transcription = transcribe(asr_model, path_segments, target_lang)
    #if not transcription.strip():
    #    print("No speech detected.")
    #    return
    
    # Translation
    print("Translating...")
    translation = translate(model_nllb, tokenizer_nllb, transcription, target_lang)
    
    # Text-to-Speech
    # process_tts(tts_model, translation, path_segments, target_lang, path_results)
    append_text_order(json_path_temp,translation, order, path_segments, path_results, "es" if target_lang == "spanish" else "en")
    append_text_order(json_path_record,translation, order, path_segments, path_results, "es" if target_lang == "spanish" else "en", transcription)
def transcribe(asr_model, path_segments, target_lang):
    start_time = time.time()
    transcription_func = {
        "spanish": parakeet_ctc_process,
        "english": stt_es_process
    }[target_lang]
    transcription = transcription_func(asr_model, path_segments)
    print("Transcription:", transcription[0])
    print("Transcription time:", time.time() - start_time)
    return transcription[0]

def translate(model_nllb, tokenizer_nllb, text, target_lang):
    print("Processing translation...")
    start_time = time.time()
    translation = nllb_translate(model_nllb, tokenizer_nllb, text, target_lang)
    print("Translation:", translation)
    print("Translation time:", time.time() - start_time)
    return translation

"""
def process_tts(tts_model, text, source_path, target_lang, result_path):
    print("Processing TTS...")
    start_time = time.time()
    lang_code = {"spanish": "es", "english": "en"}[target_lang]
    tts_mutli_process(tts_model, text, source_path, lang_code, result_path)
    print("TTS done. Time:", time.time() - start_time)
"""


    

# ...





def stream(asr_model, model_nllb, tokinizer_nllb, source_lang, target_lang, json_file_temp, json_file_record,result_dir = "results",segments_dir = "audio_segments"):
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

    #segments_dir = "audio_segments"
    #result_dir = "results"
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
        
        executor.submit(process_segment, asr_model, model_nllb, tokinizer_nllb, path_segements,path_results, target_lang, i, json_file_temp, json_file_record)

    stream.stop_stream()
    stream.close()
    audio.terminate()


