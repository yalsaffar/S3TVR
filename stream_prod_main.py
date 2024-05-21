import argparse
from models.TTS_utils import load_manual_xtts_v2
from models.TTS_utils import stream_prod

def main(xtts_path, xtts_config_path, record_temp="record_temp.json", record_path="audio_segments/"):
    """
    Main function to load the xtts model and run the stream production.

    Args:
        xtts_path (str): Path to the xtts model file.
            Example: "path/to/xtts_model.pt"
        xtts_config_path (str): Path to the xtts configuration file.
            Example: "path/to/xtts_config.json"
        record_temp (str, optional): Path to the temporary record JSON file.
            Default: "record_temp.json"
            Example: "path/to/record_temp.json"
        record_path (str, optional): Path to the directory where audio segments are recorded.
            Default: "audio_segments/"
            Example: "path/to/audio_segments/"
    """
    xtts_v2_model = load_manual_xtts_v2(xtts_config_path, xtts_path)
    stream_prod(xtts_v2_model, record_temp, record_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stream_prod.")
    parser.add_argument("xtts_path", type=str, help="Path to the xtts model.")
    parser.add_argument("xtts_config_path", type=str, help="Path to the xtts config.")
    parser.add_argument("record_temp", type=str, help="Path to the record temp file.")
    parser.add_argument("record_path", type=str, help="Path to the record directory.")
    
    args = parser.parse_args()
    
    main(args.xtts_path, args.xtts_config_path, args.record_temp, args.record_path)
