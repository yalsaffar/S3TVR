from scipy.io import wavfile
import noisereduce as nr
# Load your data

def noise_reduction(path, new_path): 
    """
    Perform noise reduction on an audio file and save the output.

    This function reads an audio file from the given path, performs noise reduction using the noisereduce library, 
    and saves the processed audio to a new file.

    Args:
        path (str): Path to the input audio file.
            Example: "path/to/input_audio.wav"
        new_path (str): Path to save the processed audio file.
            Example: "path/to/output_audio.wav"

    Returns:
        None

    Example usage:
        noise_reduction("input.wav", "output.wav")
    """
    rate, data = wavfile.read(path)
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(new_path, rate, reduced_noise)
    return print("Noise reduction done!")
