import pandas as pd
import numpy as np
import os
import sys
from models.parakeet import parakeet_ctc_model, parakeet_ctc_process
from models.es_fastconformer import stt_es_model, stt_es_process
from pydub import AudioSegment
import json
class Data_Pipeline():
    def __init__(self, wav_dir, lang):
        self.wav_dir = wav_dir
        self.lang = lang


    def get_wav_files(self):
        self.wav_files = []
        for root, dirs, files in os.walk(self.wav_dir):
            for file in files:
                if file.endswith('.wav'):
                    # make the path into the full path to the file
                    file = os.path.join(self.wav_dir, file)
                    self.wav_files.append(file)
        return self.wav_files
    def get_combined_wav_lengths(self):
        # returns a float number of the total length of all the wav files in the directory
        total_length = 0
        for file in self.wav_files:
            if file.endswith('.wav'):
                audio = AudioSegment.from_file(file)
                total_length += len(audio)
        return total_length/1000
    
    def estimated_transcription_time(self):
        # returns a float number of the estimated time based on the real time factor
        # real time factor is 1.5
        if self.lang == "en":
            return self.get_combined_wav_lengths() * 1.5
        elif self.lang == "es":
            return self.get_combined_wav_lengths() * 1.5
        return "Error: Language not supported."
    
    def load_models(self):
        if self.lang == "en":

            self.parakeet_model = parakeet_ctc_model()
            return self.parakeet_model
        elif self.lang == "es":

            self.es_model = stt_es_model()
            return self.es_model
        return "Error: Language not supported."
    
    def en_transcribe(self, audio_file):
        text = parakeet_ctc_process(self.parakeet_model, audio_file)
        return text
    
    def es_transcribe(self, audio_file):
        text = stt_es_process(self.es_model, audio_file)
        return text
    def read_transcriptions(self, json_path):
        # read the json file
        with open(json_path) as f:
            self.data = json.load(f)
        return self.data 
    
    def get_transcription(self, file_path):
        # the json file has the following keys: text, original_path, path_to_save, language, order, original_text
        # get the "original_text" of the element that has the "original_path" equal to the file_path
        for i in range(len(self.data['original_path'])):
            if self.data['original_path'][i] == file_path:
                return self.data['original_text'][i]
        return "Error: No transcription found."
    
    
    
    
    def data_formatter_with_models(self):
        self.transcriptions_df = pd.DataFrame(columns = ['wav_file', 'transcription','transcription2' ])
        if self.lang == "en":
            self.load_models()
            self.get_wav_files()
            for file in self.wav_files:
                if file.endswith('.wav'):
                    transcription = parakeet_ctc_process(self.parakeet_model, file)
                    # make the path into the full path to the file
                    #file = os.path.join(self.wav_dir, file)
                    # append transcriptions_df with the wav_file and transcription
                    self.transcriptions_df = self.transcriptions_df.append({'wav_file': file, 'transcription': transcription[0], 'transcription2': transcription[0],'speaker_name': "user0"}, ignore_index=True)
            return self.transcriptions_df
        elif self.lang == "es":
            self.load_models()
            self.get_wav_files()
            for file in self.wav_files:
                if file.endswith('.wav'):
                    self.transcriptions_df = stt_es_process(self.es_model, file)
                    # make the path into the full path to the file
                    file = os.path.join(self.wav_dir, file)
                    # append transcriptions_df with the wav_file and transcription
                    self.transcriptions_df = self.transcriptions_df.append({'wav_file': file, 'transcription': transcription[0], 'transcription2': transcription[0], 'speaker_name':"user0"}, ignore_index=True)
            return self.transcriptions_df
        
        return "Error: Language not supported."
    
    def data_formatter_without_models(self):
        self.transcriptions_df = pd.DataFrame(columns = ['wav_file', 'transcription','transcription2' ])
        self.get_wav_files()
        for file in self.wav_files:
            if file.endswith('.wav'):
                transcription = self.get_transcription(file)
                if transcription == "Error: No transcription found." and self.lang == "en":
                    transcription = parakeet_ctc_process(self.parakeet_model, file)
                elif transcription == "Error: No transcription found." and self.lang == "es":
                    transcription = stt_es_process(self.es_model, file)
            
                # make the path into the full path to the file
                #file = os.path.join(self.wav_dir, file)
                # append transcriptions_df with the wav_file and transcription
                self.transcriptions_df = self.transcriptions_df.append({'wav_file': file, 'transcription': transcription[0], 'transcription2': transcription[0], 'speaker_name': "user0"}, ignore_index=True)
        return self.transcriptions_df
    
    
    def save_transcriptions(self, output_file):
        # split the data into two data, train and eval data
        from sklearn.model_selection import train_test_split
        train_data, eval_data = train_test_split(self.transcriptions_df, test_size=0.2, random_state=42)
        # save the data into csv files
        self.path_to_train_data = output_file + "_train.csv"
        self.path_to_eval_data = output_file + "_eval.csv"
        train_data.to_csv(output_file + "_train.csv" , index=False, sep='|')
        eval_data.to_csv(output_file + "_eval.csv", index=False, sep='|')
        return "Data saved successfully.", self.path_to_train_data, self.path_to_eval_data
    
    def get_paths(self):
        # csv files, wav files directory
        return self.path_to_train_data, self.path_to_eval_data, self.wav_dir


    
    



        