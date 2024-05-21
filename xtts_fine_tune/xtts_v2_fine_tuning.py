from xtts_fine_tune.xtts_v2_data_formattor import Data_Pipeline
from xtts_fine_tune.xtts_v2_model_utils import xtts_v2_Model
import sys
import time
#!/usr/bin/env python


def Train_XTTS_V2(audio_directory, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, language):
    """
    Train the XTTS V2 model with the given parameters.

    This function initializes the data pipeline, checks the audio length, formats the data, and trains the XTTS V2 model.

    Args:
        audio_directory (str): Path to the directory containing audio files.
            Example: "path/to/audio_files/"
        num_epochs (int): Number of training epochs.
            Example: 50
        batch_size (int): Size of each training batch.
            Example: 16
        grad_acumm (int): Gradient accumulation steps.
            Example: 4
        output_path (str): Path to save the trained model outputs.
            Example: "path/to/output/"
        max_audio_length (int): Maximum allowed length of audio for training in seconds.
            Example: 3600
        language (str): Language of the audio files, either 'en' for English or 'es' for Spanish.
            Example: "en"

    Returns:
        tuple: A tuple containing paths to the configuration file, vocabulary file, fine-tuned XTTS checkpoint, and speaker wav file.
            Example: ("config_path.json", "vocab_path.json", "checkpoint.pth", "speaker.wav")

    Example usage:
        config_path, vocab_path, ft_xtts_checkpoint, speaker_wav = Train_XTTS_V2(
            "path/to/audio_files/", 50, 16, 4, "path/to/output/", 3600, "en"
        )
    """
    Data_class = Data_Pipeline(audio_directory, language)
    length_audio = Data_class.get_combined_wav_lengths()
    if length_audio > max_audio_length:
        print("The audio is not long enough to be fine tuned. Waiting....")
        time.sleep(20)
        Train_XTTS_V2(audio_directory, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, language)

    # get the directory before the current one
    audio_directory_parent = audio_directory.split("/")
    audio_directory_parent = audio_directory_parent[:-1]
    audio_directory_parent = "/".join(audio_directory_parent)
    _, train_meta, eval_meta =  Data_class.data_formatter(audio_directory_parent)
    xtts_v2 = xtts_v2_Model(train_meta, eval_meta, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, language)
    _, config_path, vocab_path, ft_xtts_checkpoint, speaker_wav = xtts_v2.train_model()

    return config_path, vocab_path, ft_xtts_checkpoint, speaker_wav

if __name__ == "__main__":
    audio_directory = sys.argv[1]
    num_epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    grad_acumm = int(sys.argv[4])
    output_path = sys.argv[5]
    max_audio_length = int(sys.argv[6])
    language = sys.argv[7]

    config_path, vocab_path, ft_xtts_checkpoint, speaker_wav = Train_XTTS_V2(audio_directory, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, language)

    # Do something with the returned values
    print(config_path, vocab_path, ft_xtts_checkpoint, speaker_wav)
