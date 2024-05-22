import torch
from TTS.api import TTS
import time
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import sounddevice as sd


def xtts_v2():
    """
    Load and return the XTTS v2 model.

    This function initializes the XTTS v2 model from the 🐸TTS library.
    The model is configured to use a GPU if available, otherwise it defaults to CPU.

    Returns:
        TTS: The initialized XTTS v2 model.
        
    Example usage:
        tts = xtts_v2()
    """
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    # print(TTS().list_models())

    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


    return tts

def load_manual_xtts_v2(config_path, checkpoint_path):
    """
    Load the XTTS v2 model manually with configuration and checkpoint files.

    Args:
        config_path (str): Path to the configuration file.
            Example: "path/to/config.json"
        checkpoint_path (str): Path to the checkpoint directory.
            Example: "path/to/checkpoint/"

    Returns:
        Xtts: The loaded XTTS v2 model.
        
    Example usage:
        model = load_manual_xtts_v2("config.json", "checkpoint/")
    """
    print("Loading model...")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_path, use_deepspeed=True)
    model.cuda()
    return model

import json
import concurrent.futures

# ----------------- StreamXTTSV2 -----------------
def get_text_order(json_path, num_elements, ):
    """
    Retrieve a specified number of text elements from a JSON file and update the file.

    Args:
        json_path (str): Path to the JSON file.
            Example: "path/to/data.json"
        num_elements (int): Number of elements to retrieve.
            Example: 3

    Returns:
        list: A list of tuples containing text, order, original_path, path_to_save, and language.
        
    Example usage:
        text_order = get_text_order("data.json", 3)
    """
    with open(json_path) as f:
        data = json.load(f)
    # check if the data is empty
    if not data['text']:
        return "No more text to process"
    if len(data['text']) < num_elements:
        num_elements = len(data['text'])
    text = data['text'][:num_elements]
    order = data['order'][:num_elements]
    original_path = data['original_path'][:num_elements]
    path_to_save = data['path_to_save'][:num_elements]
    language = data['language'][:num_elements]
    # remove the first elements
    data['text'] = data['text'][num_elements:]
    data['order'] = data['order'][num_elements:]
    data['original_path'] = data['original_path'][num_elements:]
    data['path_to_save'] = data['path_to_save'][num_elements:]
    data['language'] = data['language'][num_elements:]
    # write the data back to the file
    with open(json_path, 'w') as f:
        json.dump(data, f)
    # make it return an array of arrays of text and order
    result = [i for i in zip(text, order, original_path, path_to_save, language)]
    return result

def append_text_order(json_path, text, order, original_path, path_to_save, language, original_text=None):
    """
    Append a text order to a JSON file.

    Args:
        json_path (str): Path to the JSON file.
            Example: "path/to/data.json"
        text (str): The text to append.
            Example: "Hello, world!"
        order (int): The order index.
            Example: 1
        original_path (str): Path to the original file.
            Example: "path/to/original.wav"
        path_to_save (str): Path to save the processed file.
            Example: "path/to/save.wav"
        language (str): Language of the text.
            Example: "en"
        original_text (str, optional): The original text if available.
            Example: "Hola, mundo!"

    Example usage:
        append_text_order("data.json", "Hello, world!", 1, "original.wav", "save.wav", "en", "Hola, mundo!")
    """
    with open(json_path) as f:
        data = json.load(f)
    data['text'].append(text)
    data['order'].append(order)
    data['original_path'].append(original_path)
    data['path_to_save'].append(path_to_save)
    data['language'].append(language)
    data['original_text'].append(original_text)
    with open(json_path, 'w') as f:
        json.dump(data, f)
# ----------------- StreamXTTSV2 -----------------
class StreamXTTSV2:
    """
    A class to handle streaming TTS using XTTS v2 model.

    Args:
        model (Xtts): The XTTS v2 model.
        sample_rate (int, optional): The sample rate for audio playback. Default is 24000.
        buffer_size (int, optional): The buffer size for audio playback. Default is 2.
    """
    def __init__(self, model, sample_rate=24000, buffer_size=2):
        self.model = model
        #self.gpt_cond_latent = gpt_cond_latent
        #self.speaker_embedding = speaker_embedding
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.speed = 0.95
        self.stream_chunk_size = 40
        self.buffer = torch.Tensor().to('cpu')
        self.chunk_save = torch.Tensor().to('cpu')
        self.is_playing = False
        self.tasks_order = []
        self.order = 0
        self.initial = True

    def chunk_callback(self, chunk, i, output_dir, order):
        """
        Callback function to handle each chunk of audio during streaming.

        Args:
            chunk (torch.Tensor): The audio chunk.
                Example: tensor([0.1, 0.2, 0.3])
            i (int): The chunk index.
                Example: 1
            output_dir (str): Directory to save the chunk.
                Example: "output/"
            order (int): The order index.
                Example: 1
        """
        # Accumulate chunk into buffer
        self.buffer = torch.cat((self.buffer, chunk.squeeze().to('cpu')), dim=-1)
        self.chunk_save = torch.cat((self.chunk_save, chunk.squeeze().to('cpu')), dim=-1)
        chunk_filename = output_dir + f"chunk_{i}_{order}.wav"
        print(self.sample_rate)
        torchaudio.save(chunk_filename, self.chunk_save.unsqueeze(0), self.sample_rate)
        print(f"Chunk saved as {chunk_filename}")
        self.chunk_save = torch.Tensor().to('cpu')
        
        # Check if buffer has enough chunks to start playing
        if not self.is_playing and len(self.buffer) >= self.buffer_size:
            self.start_playback()

    def start_playback(self):
        """Start audio playback."""
        self.is_playing = True
        sd.play(self.buffer.numpy(), self.sample_rate, blocking=False)
        self.buffer = torch.Tensor().to('cpu')  # Reset buffer after starting playback

    def play(self, chunks, output_dir, path_to_save, order):
        """
        Play the audio chunks and save the complete audio.

        Args:
            chunks (list): List of audio chunks.
                Example: [tensor([0.1, 0.2, 0.3]), tensor([0.4, 0.5, 0.6])]
            output_dir (str): Directory to save the chunks.
                Example: "output/"
            path_to_save (str): Path to save the complete audio file.
                Example: "output/complete.wav"
            order (int): The order index.
                Example: 1
        """
        t0 = time.time()
        

        for i, chunk in enumerate(chunks):
            #print(chunk)
            if i == 0:
                print(f"Time to first chunk: {time.time() - t0}")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
            self.chunk_callback(chunk, i, output_dir, order)
        
        # Ensure all remaining audio is played
        while sd.get_stream().active:
            time.sleep(0.1)
        if len(self.buffer) > 0:
            sd.play(self.buffer.numpy(), self.sample_rate, blocking=True)
        
        # Save the complete audio to a file
        torchaudio.save(path_to_save, self.buffer.unsqueeze(0), self.sample_rate)
        print(f"Total audio length: {self.buffer.shape[-1]}")
        print("Audio playback finished.")
        #self.order += 1
        

    def inference_and_play(self, json_path, output_dir):
        """
        Perform inference and play the generated audio.

        Args:
            json_path (str): Path to the JSON file containing text orders.
                Example: "path/to/data.json"
            output_dir (str): Directory to save the chunks.
                Example: "output/"
        """
        print("Inference...")
        

        
        self.texts = get_text_order(json_path, 3)
        
        if self.texts == "No more text to process":
            print("No more text to process")
            return
        if self.texts == "Not enough text to process":
            print("Not enough text to process")
            return
        # is it returns a list of text and order
        if self.texts is not None:
            #print(self.texts)
            self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[self.texts[0][2]])
            path_to_save = self.texts[0][3]
            #print(self.gpt_cond_latent, self.speaker_embedding)
            #print(self.texts)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                #text, order = get_text_order(texts)
                #print(text, order)
                futures = []
                print(self.texts)
                
                for text, i, path_a, path_s, lang in self.texts:
                    #print(text, i, path)
                    print(f"Processing text {i}: {text}")
                    print(f"Processing text {i}: {lang}")
                    future = executor.submit(self.model.inference_stream, text, lang, self.gpt_cond_latent, self.speaker_embedding, stream_chunk_size=self.stream_chunk_size, speed=self.speed)
                    #print(future.result())
                    futures.append(future)
                    
                
                for future, text in zip(futures, self.texts):
                    #print(text)
                    chunks = future.result()
                    print(text[1])
                    self.play(chunks, output_dir, path_to_save, text[1]) 
                    self.buffer = torch.Tensor().to('cpu')
            
            self.inference_and_play(json_path, output_dir )


def stream_prod(model, json_path, directory_path):
    """
    Stream production function for XTTS v2.

    Args:
        model (Xtts): The XTTS v2 model.
            Example: model = load_manual_xtts_v2("config.json", "checkpoint/")
        json_path (str): Path to the JSON file containing text orders.
            Example: "path/to/data.json"
        directory_path (str): Directory to save the chunks.
            Example: "output/"
    """
    streamer = StreamXTTSV2(model, buffer_size=2)
    results = streamer.inference_and_play(json_path, directory_path)
    if results is  None:
        time.sleep(3)
        stream_prod(model, json_path, directory_path)
    return "Streaming finished"


def just_inference(model, original_path, output_dir, text, lang, order):
    """
    Perform inference and save the generated audio.

    Args:
        model (Xtts): The XTTS v2 model.
            Example: model = load_manual_xtts_v2("config.json", "checkpoint/")
        original_path (str): Path to the original audio file.
            Example: "path/to/original.wav"
        output_dir (str): Directory to save the generated audio file.
            Example: "output/"
        text (str): The text to be synthesized.
            Example: "Hello, world!"
        lang (str): The language of the text.
            Example: "en"
        order (int): The order index.
            Example: 1

    Returns:
        tuple: A tuple containing the path to the saved audio file and the time to first chunk.
            Example: ("output/complete.wav", 1.23)
    """
    print("Inference...")
    path_to_save = output_dir
    t0 = time.time()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[original_path])

    chunks = model.inference_stream(
        text,
        lang,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size= 15 ,
        speed=0.95
        #temperature=0.1,
        #enable_text_splitting=True,
    )
    full_audio = torch.Tensor().to('cpu') 
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 1:
            time_to_first_chunk = time.time() - t0
            print(f"Time to first chunck: {time_to_first_chunk}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        wav_chuncks.append(chunk)
        full_audio = torch.cat((full_audio, chunk.squeeze().to('cpu')), dim=-1)
    
    
    
    # Save the complete audio to a file
    torchaudio.save(path_to_save, full_audio.unsqueeze(0), 24000)
    
    print("Inference finished")
    return path_to_save, time_to_first_chunk
    


