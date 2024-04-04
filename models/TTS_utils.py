import torch
from TTS.api import TTS

"""
'tts_models/en/ek1/tacotron2',
 'tts_models/en/ljspeech/tacotron2-DDC',
 'tts_models/en/ljspeech/tacotron2-DDC_ph',
 'tts_models/en/ljspeech/glow-tts',
 'tts_models/en/ljspeech/speedy-speech',
 'tts_models/en/ljspeech/tacotron2-DCA',
 'tts_models/en/ljspeech/vits',
 'tts_models/en/ljspeech/vits--neon',
 'tts_models/en/ljspeech/fast_pitch',
 'tts_models/en/ljspeech/overflow',
 'tts_models/en/ljspeech/neural_hmm',
 'tts_models/en/vctk/vits',
 'tts_models/en/vctk/fast_pitch',
 'tts_models/en/sam/tacotron-DDC',
 'tts_models/en/blizzard2013/capacitron-t2-c50',
 'tts_models/en/blizzard2013/capacitron-t2-c150_v2',
 'tts_models/en/multi-dataset/tortoise-v2',
 """
def load_tts_model(model_name):
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS with the tacotron2-DDC model
    tts = TTS(model_name).to(device)

    return tts
def your_tts():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS with the tacotron2-DDC model
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)

    return tts

def tts_es_vits():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS with the vits model
    tts = TTS('tts_models/es/css10/vits').to(device)

    return tts

def tts_es_tacotron2():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS with the tacotron2 model
    tts = TTS('tts_models/es/mai/tacotron2-DDC').to(device)

    return tts
def tts_bark():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS with the bark model
    tts = TTS( 'tts_models/multilingual/multi-dataset/bark').to(device)

    return tts

def xtts_v2():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    # print(TTS().list_models())

    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


    return tts



# ------------------------------- TTS Functions -------------------------------
def tts_mutli_process(tts, text, speaker_wav, language, file_path):
    
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
    return True


def tts_single_process(tts, text, file_path, speaker_wav):
    # Run TTS using tacotron2-DDC model
    # This model is configured for a specific voice, so speaker_wav and language parameters are not needed here
    tts.tts_to_file(text=text,speaker_wav=speaker_wav, file_path=file_path)
    return True