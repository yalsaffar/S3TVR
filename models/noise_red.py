from scipy.io import wavfile
import noisereduce as nr
# Load your data

def noise_reduction(path, new_path): # more customization can be done here
    rate, data = wavfile.read(path)
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(new_path, rate, reduced_noise)
    return print("Noise reduction done!")
