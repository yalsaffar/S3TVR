import os
import numpy as np
import traceback
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fine_tuned_xtts_v2(config_path, checkpoint_path, reference_audio_path):
    """
    Load the fine-tuned XTTS v2 model and compute speaker latents.

    Args:
        config_path (str): Path to the configuration file.
            Example: "path/to/config.json"
        checkpoint_path (str): Path to the checkpoint directory.
            Example: "path/to/checkpoint/"
        reference_audio_path (str): Path to the reference audio file.
            Example: "path/to/reference.wav"

    Returns:
        tuple: A tuple containing the model, gpt_cond_latent, and speaker_embedding.
            Example: (model, gpt_cond_latent, speaker_embedding)
    """
    print("Loading model...")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_path, use_deepspeed=True)
    model.cuda()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[reference_audio_path])
    return model, gpt_cond_latent, speaker_embedding

def Inference(model, gpt_cond_latent, speaker_embedding,path_to_save,text, temperature=0.7):
    """
    Perform inference using the fine-tuned XTTS v2 model.

    Args:
        model (Xtts): The XTTS v2 model.
            Example: model, gpt_cond_latent, speaker_embedding = load_fine_tuned_xtts_v2(config_path, checkpoint_path, reference_audio_path)
        gpt_cond_latent (torch.Tensor): GPT conditioning latent vectors.
        speaker_embedding (torch.Tensor): Speaker embedding vectors.
        path_to_save (str): Path to save the generated audio.
            Example: "path/to/output.wav"
        text (str): The input text for synthesis.
            Example: "Hello, world!"
        temperature (float, optional): Sampling temperature. Default is 0.7.
            Example: 0.7

    Returns:
        None
    """
    print("Inference...")
    out = model.inference(
        text,
        gpt_cond_latent,
        speaker_embedding,
        temperature, # Add custom parameters here # 3
    )
    torchaudio.save(path_to_save, torch.tensor(out["wav"]).unsqueeze(0), 24000)

#model, gpt_cond_latent, speaker_embedding = load_fine_tuned_xtts_v2("C:/tmp/xtts_ft/run/training/GPT_XTTS_FT-April-02-2024_05+08PM-0000000/config.json", "C:/tmp/xtts_ft/run/training/GPT_XTTS_FT-April-02-2024_05+08PM-0000000/best_model_72.pth", "old_man_segments/wavs/segment_10.wav")
    
class xtts_v2_Model():
    """
    A class to handle training of the XTTS v2 model.

    Args:
        train_csv_path (str): Path to the training CSV file.
            Example: "path/to/train.csv"
        eval_csv_path (str): Path to the evaluation CSV file.
            Example: "path/to/eval.csv"
        num_epochs (int): Number of training epochs.
            Example: 10
        batch_size (int): Size of each training batch.
            Example: 4
        grad_acumm (int): Gradient accumulation steps.
            Example: 1
        output_path (str): Path to save the trained model outputs.
            Example: "path/to/output/"
        max_audio_length (int): Maximum allowed length of audio for training in seconds.
            Example: 10
        language (str, optional): Language of the audio files, either 'en' for English or 'es' for Spanish. Default is "en".
            Example: "en"
    """
    def __init__(self, train_csv_path, eval_csv_path, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, language="en"):
        self.train_csv_path = train_csv_path
        self.eval_csv_path = eval_csv_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_acumm = grad_acumm
        self.output_path = output_path
        self.max_audio_length = max_audio_length
        self.language = language
        self.config_path = None
        self.original_xtts_checkpoint = None
        self.vocab_file = None
        self.exp_path = None
        self.speaker_wav = None


    def train_model(self):
        """
        Train the XTTS v2 model.

        Returns:
            tuple: A tuple containing a status message, config_path, vocab_file, fine-tuned XTTS checkpoint, and speaker wav file.
                Example: ("Model training done!", "path/to/config.json", "path/to/vocab.json", "path/to/best_model.pth", "path/to/speaker.wav")
        """
        #clear_gpu_cache()
        if not self.train_csv_path or not self.eval_csv_path:
            return "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !", "", "", "", ""
        try:
            # convert seconds to waveform frames
            max_audio_length = int(max_audio_length * 22050)
            self.config_path, self.original_xtts_checkpoint, self.vocab_file, self.exp_path, self.speaker_wav = train_gpt(self.language, self.num_epochs, self.batch_size, self.grad_acumm, self.train_csv_path, self.eval_csv_path, output_path=self.output_path, max_audio_length=max_audio_length)
        except:
            traceback.print_exc()
            error = traceback.format_exc()
            return f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}", "", "", "", ""

        # copy original files to avoid parameters changes issues
        os.system(f"cp {self.config_path} {self.exp_path}")
        os.system(f"cp {self.vocab_file} {self.exp_path}")

        ft_xtts_checkpoint = os.path.join(self.exp_path, "best_model.pth")
        print("Model training done!")
        #clear_gpu_cache()
        return "Model training done!", self.config_path, self.vocab_file, ft_xtts_checkpoint, self.speaker_wav

# example
#train_meta = "C:/tmp/xtts_ft/run/training/GPT_XTTS_FT-April-02-2024_05+08PM-0000000/train.csv"
#eval_meta = "C:/tmp/xtts_ft/run/training/GPT_XTTS_FT-April-02-2024_05+08PM-0000000/eval.csv"
#num_epochs = 10
#batch_size = 4
#grad_acumm = 1
#out_path = "C:/tmp/xtts_ft/run/training/GPT_XTTS_FT-April-02-2024_05+08PM-0000000"
#max_audio_length = 10
#lang = "en"
#xtts_v2 = xtts_v2_Model(train_meta, eval_meta, num_epochs, batch_size, grad_acumm, out_path, max_audio_length, lang)
#_, config_path, vocab_path, ft_xtts_checkpoint, speaker_wav = xtts_v2_Model.train_model()