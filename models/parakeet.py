import nemo.collections.asr as nemo_asr



def parakeet_ctc_model():
    # Load the pre-trained model
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")

    return asr_model

def parakeet_ctc_process(asr_model, audio_file):
    text = asr_model.transcribe(paths2audio_files=[audio_file], batch_size=1)

    return text
