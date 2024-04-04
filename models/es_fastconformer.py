import nemo.collections.asr as nemo_asr
import torch


def stt_es_model():
    # Load the pre-trained model
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_es_fastconformer_hybrid_large_pc")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_model = asr_model.to(device)
    return asr_model

def stt_es_process(asr_model, audio_file):
    text = asr_model.transcribe(paths2audio_files=[audio_file], batch_size=1)

    return text
