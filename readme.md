
# Seamless Speech-to-Speech Translation with Voice Replication (S3TVR)

## Project Overview

Seamless Speech-to-Speech Translation with Voice Replication (S3TVR) is an advanced AI cascaded feamework designed for real-time speech-to-speech translation while maintaining the speaker's voice characteristics. This project balances latency and output quality, focusing on English and Spanish languages, and involves multiple open-source models and algorithms. The system is optimized for local execution, allowing for dynamic and efficient voice translation.


<img src="https://github.com/yalsaffar/S3TVR/blob/master/workflow.gif" width="600" height="400" />

## Technologies and Framework

1. **Voice Activity Detection (VAD):** Differentiates speech from silence for efficient segmentation.
2. **Noise Reduction Model:** Enhances audio clarity.
3. **Automatic Speech Recognition (ASR):** Converts speech to text.
4. **Machine Translation Model (MT):** Translates text between languages.
5. **Text-to-Speech (TTS) Synthesis:** Converts translated text back to speech with voice replication.

## Checklist

- [x]  Zero-shot speech-to-speech translation with voice replication
- [ ]  MultiLingual: SOON
- [ ]  Clustered fine-tuned XTTS_V2: SOON
- [x]  Fine-tuning XTTS_V2 structure
- [ ]  Fine-tuned XTTS_V2 Automatic integration: SOON
- [ ]  Models direct downloading: SOON

## Installation Instructions

1. **Create a New Conda Environment:**

```bash
conda create --name s3tvr_env python=3.8
conda activate s3tvr_env
```

## Requirements:

```bash
pip install -r requirements.txt
```
## DeepSpeed Installation:

- For Linux users, follow this [installation guide](https://www.deepspeed.ai/tutorials/advanced-install/).
- For Windows users, [follow this tutorial](https://github.com/microsoft/DeepSpeed/issues/4729).

## TTS Installation:
- for TTS advanced installation, follow the [installation guide](https://github.com/coqui-ai/TTS) in their repo.


## Models Used and Adding New Models

The project utilizes several models for different tasks, including:
- **Automatic Speech Recognition (ASR):**
  - Parakeet CTC 1.1 B Model
  - STT Es FastConformer Hybrid Transducer-CTC Large P&C Model
- **Machine Translation (MT):**
  - NLLB-200
- **Text-to-Speech (TTS):**
  - XTTS V2

Each model is stored in the `models` folder. To add a new model:
1. Place the model files in the `models` folder.
2. Create a function to initialize the model.
3. Create another function to process the input with the model.
4. modify `run.py` and `stream_VAD.py` with the new models.

Example initialization and processing functions for a model:
```python
def initialize_model(model_path):
    model = SomeModelClass.from_pretrained(model_path)
    return model

def process_input(model, input_data):
    output = model(input_data)
    return output
```

## XTTS_V2 Model
- Manual Download: XTTS_V2 needs to be downloaded manually and passed to the main workflow.
- Using TTS API: Implemented methods can use TTS API with necessary workflow modifications.


## Running the S3TVR Cascaded Model
To run the model, use the following command:


``` bash
python run.py
```

Example command with arguments:

``` bash 
python run.py /path/to/xtts /path/to/xtts_config en --record_temp record_temp.json --record_per record_per.json --record_path audio_segments/ --result_dir results --segments_dir audio_segments/
```



## Overall Framework

This project is part of my bachelor's thesis for Computer Science and Artificial Intelligence at IE University. It aims to reduce linguistic barriers in real-time communication by integrating various AI models for seamless speech-to-speech translation and voice replication.

## Latency Performance

Latency is a critical metric for evaluating the efficiency of real-time speech-to-speech translation systems. The S3TVR model has been benchmarked for both English and Spanish translations, providing insights into its performance under various conditions. The following table summarizes the average, best-case, and worst-case latencies recorded for the model:

| Metric      | English       | Spanish       |
|-------------|---------------|---------------|
| **Average** | 3.09 seconds  | 3.27 seconds  |
| **Best Case** | 1.92 seconds  | 1.88 seconds  |
| **Worst Case** | 6.95 seconds  | 7.95 seconds  |

## Model Characteristics

Understanding the inherent characteristics of the S3TVR model is essential for appreciating its adaptability and performance in different scenarios. The table below outlines key features of the model, including its adaptability, customizability, latency control, and resource efficiency. These features highlight the model's design philosophy and its suitability for various applications:

| Feature           | Details                                                                              |
|-------------------|--------------------------------------------------------------------------------------|
| **Adaptability**  | Designed for easy adaptation to new models                                           |
| **Customizability** | Supports adjustments to model parameters and configurations to meet specific needs  |
| **Latency Control** | Implements strategies to minimize processing times and maintain low latency        |
| **Resource Efficiency** | Optimized for local execution with manageable resource requirements, suitable for production environments |

## Comparison with Seamless Streaming Model

To provide a comprehensive evaluation, the S3TVR model is compared against the Seamless Streaming model. This comparison focuses on key performance indicators such as latency, memory usage, translation quality, flexibility, and resource efficiency. By examining these aspects, we can understand the strengths and limitations of each model, guiding potential improvements and use-case considerations:

| Feature                       | S3TVR Model (English)       | S3TVR Model (Spanish)       | Seamless Streaming Model           |
|-------------------------------|-----------------------------|-----------------------------|------------------------------------|
| **Latency**                   | Average: 3.09 seconds       | Average: 3.27 seconds       | 4.59 to 4.73 seconds depending on threshold settings |
| **Memory Usage**              | 5618 MB                     | 3641 MB                     | Large: 2326 MB, Medium: 1151 MB    |
| **Translation Quality (BLEU Score)** | 0.36                        | 0.41                        | 0.198 to 0.203 depending on threshold |
| **Flexibility**               | High: Modular design allows easy updates | High: Modular design allows easy updates | Less modular, more rigid design  |
| **Resource Efficiency**       | Optimized for local execution | Optimized for local execution | Requires more computational resources, less optimized for local execution |


## Acknowledgments

I would like to thank my supervisor, Adrian Carrio, for his guidance and support throughout this project.

## License

This project is licensed under the MIT License.


