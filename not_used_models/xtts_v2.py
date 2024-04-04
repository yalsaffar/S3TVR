import torch
from TTS.api import TTS
def xtts_v2():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    # print(TTS().list_models())

    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


    return tts



def xtts_v2_process(tts, text, speaker_wav, language, file_path):
    # Run TTS
    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    #wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
    # Text to speech to a file
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
    return True