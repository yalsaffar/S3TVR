import nemo.collections.asr as nemo_asr
import torch

def stt_es_model():
    """
    Load and return the pre-trained Spanish ASR model.

    This function loads the pre-trained EncDecCTCModelBPE model from NVIDIA's NeMo collection.
    The model is configured to use a GPU if available, otherwise it defaults to CPU.

    Returns:
        nemo_asr.models.EncDecCTCModelBPE: The loaded ASR model.
            Example usage:
                asr_model = stt_es_model()
    """
    # Load the pre-trained model
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_es_fastconformer_hybrid_large_pc")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_model = asr_model.to(device)
    return asr_model

def stt_es_process(asr_model, audio_file):
    """
    Transcribe an audio file using the given ASR model.

    Args:
        asr_model (nemo_asr.models.EncDecCTCModelBPE): The ASR model to use for transcription.
            Example: asr_model = stt_es_model()
        audio_file (str): Path to the audio file to be transcribed.
            Example: "path/to/audio_file.wav"

    Returns:
        list: A list containing the transcribed text.
            Example: ["transcribed text"]
    """
    text = asr_model.transcribe(paths2audio_files=[audio_file], batch_size=1)
    return text
