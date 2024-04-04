import torch
import torchaudio
from torchaudio.transforms import Resample
from datasets import load_dataset
from transformers import MCTCTForCTC, MCTCTProcessor

def get_mctct_processor():
    model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large")
    processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large")
    return model, processor

def transcribe_audio(model, processor, audio_path):
    # Read the local audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample the waveform to 16000 Hz
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

    # Ensure waveform has only one dimension
    waveform = waveform.squeeze()

    # Feature extraction
    input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features

    # Retrieve logits
    with torch.no_grad():
        logits = model(input_features).logits

    # Take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription
