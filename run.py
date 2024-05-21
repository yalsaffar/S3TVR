import threading
import argparse
import subprocess
from models.nllb import nllb
from models.parakeet import parakeet_ctc_model
from models.es_fastconformer import stt_es_model
from models.TTS_utils import load_manual_xtts_v2
from stream_VAD import stream

def main(xtts_path, xtts_config_path, language="en", record_temp="record_temp.json", record_per="record_per.json", record_path="audio_segments/", result_dir="results", segments_dir="audio_segments"):
    """
    Main function to run the ASR stream and initiate the TTS stream production.

    Args:
        xtts_path (str): Path to the xtts model file.
            Example: "path/to/xtts_model.pt"
        xtts_config_path (str): Path to the xtts configuration file.
            Example: "path/to/xtts_config.json"
        language (str, optional): Language for the ASR model. Must be either 'en' for English or 'es' for Spanish.
            Default: 'en'
            Example: "en"
        record_temp (str, optional): Path to the temporary record JSON file.
            Default: "record_temp.json"
            Example: "path/to/record_temp.json"
        record_per (str, optional): Path to the periodic record JSON file.
            Default: "record_per.json"
            Example: "path/to/record_per.json"
        record_path (str, optional): Path to the directory where audio segments are recorded.
            Default: "audio_segments/"
            Example: "path/to/audio_segments/"
        result_dir (str, optional): Path to the directory where results are stored.
            Default: "results"
            Example: "path/to/results"
        segments_dir (str, optional): Path to the directory where audio segments are stored.
            Default: "audio_segments"
            Example: "path/to/audio_segments"
    """
    model_nllb, tokinizer_nllb = nllb()

    if language == "en":
        asr = parakeet_ctc_model()
        stream_thread = threading.Thread(target=stream, args=(asr, model_nllb, tokinizer_nllb, "english", "spanish", record_temp, record_per, result_dir, segments_dir))
    
    elif language == "es":
        asr = stt_es_model()
        stream_thread = threading.Thread(target=stream, args=(asr, model_nllb, tokinizer_nllb, "spanish", "english", record_temp, record_per, result_dir, segments_dir))

    else:
        raise ValueError("Language not supported")

    # Start the stream thread
    stream_thread.start()

    # Call the other script to start stream_prod
    subprocess.Popen(['python', 'stream_prod_main.py', xtts_path, xtts_config_path, record_temp, record_path])

    # Wait for the stream thread to complete
    stream_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stream and initiate stream_prod.")
    parser.add_argument("xtts_path", type=str, help="Path to the xtts model.")
    parser.add_argument("xtts_config_path", type=str, help="Path to the xtts config.")
    parser.add_argument("language", type=str, choices=["en", "es"], help="Language (en or es).")
    parser.add_argument("--record_temp", type=str, default="record_temp.json", help="Path to the record temp file.")
    parser.add_argument("--record_per", type=str, default="record_per.json", help="Path to the record per file.")
    parser.add_argument("--record_path", type=str, default="audio_segments/", help="Path to the record directory.")
    parser.add_argument("--result_dir", type=str, default="results", help="Path to the result directory.")
    parser.add_argument("--segments_dir", type=str, default="audio_segments", help="Path to the segments directory.")
    
    args = parser.parse_args()
    
    main(args.xtts_path, args.xtts_config_path, args.language, args.record_temp, args.record_per, args.record_path, args.result_dir, args.segments_dir)
