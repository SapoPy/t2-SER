import os
from scipy.io import wavfile
import torch
import torchaudio
import pandas as pd     
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 data_portion=None,
                 transform=None):

        labels_path = os.path.join(audio_dir , annotations_file)
        self.audio_labels = pd.read_csv(labels_path)
        if data_portion:
            self.audio_labels = self.audio_labels[self.audio_labels['partition'] == data_portion]
        self.transform = transform
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        # se obtiene la dirección donde se almacenan el audio
        audio_path = os.path.join(self.audio_dir , self.audio_labels.iloc[idx]['path'])

        # se carga la clase y el audio
        activation = self.audio_labels.iloc[idx]['activation']
        valence = self.audio_labels.iloc[idx]['valence']
        dominance = self.audio_labels.iloc[idx]['dominance']

        labels = torch.tensor(
            [activation, valence, dominance],
            dtype=torch.float32
        )
        
        audio_signal, _ = torchaudio.load(audio_path)

        # si se entrega una transformacion se aplica al audio
        if self.transform:
            audio_signal = self.transform(audio_signal)

        # se entregan los resultados guardados en GPU
        return audio_signal[0], labels

ANNOTATIONS_FILE = 'labels.csv'
AUDIO_DIR = 'Emociones/Podcast/'

train_dataset = EmotionDataset(ANNOTATIONS_FILE,
                AUDIO_DIR, 
                data_portion='train'
                )

val_dataset = EmotionDataset(ANNOTATIONS_FILE,
                AUDIO_DIR, 
                data_portion='validation'
                )

test_dataset = EmotionDataset(ANNOTATIONS_FILE,
                AUDIO_DIR, 
                data_portion='test'
                )

if __name__ == "__main__":

    ejemplo = train_dataset[0]
    print(ejemplo)
    print(len(ejemplo[0]))
    print(ejemplo[1])