# Real-Time Speech Translation System Thesis

This repository is dedicated to the research and development of an advanced real-time speech translation system. The primary objective is to design a system that not only translates speech instantly across various languages but also retains the unique vocal characteristics of the speaker. This system has the potential to revolutionize international communication and aid in multilingual interactions.

## Overview

The translation system under development is poised to address the complexities of speech recognition, machine translation, and speech synthesis. By leveraging cutting-edge machine learning technologies, the goal is to create a seamless translation experience that is both accurate and efficient, capable of operating in real-time.

## Project Roadmap

Two architectural approaches are currently under consideration for the system:

### Option 1: Enhancement of SeamlessM4T Architecture

This approach is predicated on the adaptation and refinement of the SeamlessM4T model, with particular emphasis on the following components:

#### Detailed Components:

- **SeamlessM4T-NLLB**: Targeted fine-tuning for enhanced data processing in Spanish and English.
- **w2V-BERT 2.0**: Extensive testing and potential fine-tuning for Spanish language optimization.
- **VoCoder**: Evaluation for replacement or comprehensive fine-tuning; YourTTS as a possible alternative for inherent expression features.
- **Multitasking UNITY**: Strategic fine-tuning, contingent on the acquisition of clean, aligned Spanish-English datasets.
- **Speech Output**: Refinement for high-fidelity Spanish speech synthesis.
- **Sentence Detection**: In-depth analysis and potential enhancement of sentence boundary detection models.

#### Strategy and Planning:
- Fine-tune and integrate various components to improve Spanish and English translation fidelity and maintain speaker voice characteristics.
- Assess data alignment requirements and additional training needs.
- Explore advanced vocoder options to better capture speech expressions.

### Option 2: Cascaded Model Integration

This approach envisages a cascaded architecture, combining distinct models specialized in their respective domains:

#### Detailed Components:
- **Whisper-Large-v2**: Specific fine-tuning for the Spanish language, backed by targeted research to acquire robust training datasets.
- **NLLB-600M**: Adoption and fine-tuning of Facebook's streamlined model, capitalizing on its efficiency with a reduced parameter footprint.
- **YourTTS**: Intensive model selection and fine-tuning, emphasizing voice cloning capabilities within concise audio samples.

#### Strategy and Planning:
- Create a cascaded system where each model is fine-tuned for optimal Spanish language performance.
- Explore the potential of voice cloning to maintain speaker characteristics with minimal audio input.

## Data Considerations

### Data Requirements
To fine-tune and evaluate the models effectively, access to extensive, clean, and diverse datasets is essential. The data must include a range of dialects, accents, and colloquialisms to ensure comprehensive language coverage.


## Ongoing Research

The project is still in the exploratory phase, with extensive research being conducted to finalize the architectural approach. The following areas are key focal points:

- **Model Efficiency**: Ensuring the models operate in real-time without sacrificing accuracy or voice quality.
- **Data Quality**: Identifying and curating high-quality datasets for training and validation.
- **Algorithm Optimization**: Examining current algorithms for potential improvements and customizations.

