import nemo.collections.asr as nemo_asr
import torch


def parakeet_ctc_model():
    """
    Load and return the pre-trained Parakeet CTC model.

    This function loads the pre-trained EncDecCTCModelBPE model from NVIDIA's NeMo collection.
    The model is configured to use a GPU if available, otherwise it defaults to CPU.

    Returns:
        nemo_asr.models.EncDecCTCModelBPE: The loaded ASR model.

    Example usage:
        asr_model = parakeet_ctc_model()
    """
    # Load the pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
    asr_model = asr_model.to(device)
    return asr_model

def parakeet_ctc_process(asr_model, audio_file):
    """
    Transcribe an audio file using the given Parakeet CTC ASR model.

    Args:
        asr_model (nemo_asr.models.EncDecCTCModelBPE): The ASR model to use for transcription.
            Example: asr_model = parakeet_ctc_model()
        audio_file (str): Path to the audio file to be transcribed.
            Example: "path/to/audio_file.wav"

    Returns:
        list: A list containing the transcribed text.
            Example: ["transcribed text"]
    
    Example usage:
        text = parakeet_ctc_process(asr_model, "path/to/audio_file.wav")
    """
    text = asr_model.transcribe(paths2audio_files=[audio_file], batch_size=1)

    return text
