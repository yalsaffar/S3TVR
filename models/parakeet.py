import nemo.collections.asr as nemo_asr
import torch


def parakeet_ctc_model():
    # Load the pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
    asr_model = asr_model.to(device)
    return asr_model

def parakeet_ctc_process(asr_model, audio_file):
    text = asr_model.transcribe(paths2audio_files=[audio_file], batch_size=1)

    return text
