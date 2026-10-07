import torch
import torchaudio
from torch import nn
import numpy as np
MAX_LEN = int(11.008 * 16_000 / 256)  # cantidad de muestras

def fft_transform(x):
    transform = torchaudio.transforms.Spectrogram(
        n_fft=256,
        normalized=True
    )

    x = transform(x)
    x = x.transpose(1, 2)

    # Cantidad de frames originales
    n_frames = x.shape[1]

    # Crear máscara
    mask = torch.zeros(MAX_LEN, dtype=torch.bool)

    if n_frames < MAX_LEN:

        padding = torch.zeros(
            x.shape[0],
            MAX_LEN - n_frames,
            x.shape[2],
            device=x.device
        )

        x = torch.cat((x, padding), dim=1)

        # Las posiciones agregadas son padding
        mask[n_frames:] = True

    else:
        x = x[:, :MAX_LEN, :]

    return x[0], mask


def log_mel_transform(x):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=256,
        n_mels=30,
        normalized=True
    )

    x = transform(x)

    # Convertir a log-Mel
    x = torch.log(x + 1e-9)

    x = x.transpose(1, 2)

    # Cantidad de frames originales
    n_frames = x.shape[1]

    # Crear máscara
    mask = torch.zeros(MAX_LEN, dtype=torch.bool, device=x.device)

    if n_frames < MAX_LEN:

        padding = torch.zeros(
            x.shape[0],
            MAX_LEN - n_frames,
            x.shape[2],
            device=x.device
        )

        x = torch.cat((x, padding), dim=1)

        # Las posiciones agregadas son padding
        mask[n_frames:] = True

    else:
        x = x[:, :MAX_LEN, :]

    return x[0], mask


class FFTTransformer(nn.Module):
    def __init__(self, D_pos_emb=50, nhead=1):
        super().__init__()

        # token de clasificacion
        self.class_token = nn.Parameter(data=torch.randn(1, 1, 129),
                                        requires_grad=True)

        # capa lineal para proyectar dimensiones
        self.linear1 = nn.Linear(129, D_pos_emb)

        # embedding posicional
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, MAX_LEN + 1, D_pos_emb),
                                                requires_grad=True)

        # encoder de transformer
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=D_pos_emb,
                                                        nhead = nhead,
                                                        dim_feedforward=D_pos_emb,
                                                        batch_first=True)

        # capa fully connected para realizar clasificaciónn
        self.fc1 = nn.Linear(D_pos_emb, D_pos_emb)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(D_pos_emb, 3)


    def forward(self, x, padding_mask):

        B = x.shape[0]

        # CLS token
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        # Agregar posición del CLS a la máscara
        cls_mask = torch.zeros(
            B, 1,
            dtype=torch.bool,
            device=x.device
        )

        padding_mask = padding_mask.to(x.device)

        padding_mask = torch.cat(
            (cls_mask, padding_mask),
            dim=1
        )

        # Proyección
        x = self.linear1(x)

        # Embedding posicional
        x = self.pos_embedding + x

        # Transformer
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # CLS
        x = self.fc1(x[:, 0])
        x = self.activation(x)
        x = self.fc2(x)

        return x



class FFTTransformerSen(nn.Module):
    def __init__(self, D_pos_emb=50, nhead=1):
        super().__init__()

        # token de clasificacion
        self.class_token = nn.Parameter(data=torch.randn(1, 1, 129),
                                        requires_grad=True)

        # capa lineal para proyectar dimensiones
        self.linear1 = nn.Linear(129, D_pos_emb)

        # embedding posicional
        max_len = MAX_LEN + 1

        # Matriz de posiciones: [max_len, 1]
        position = torch.arange(
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        # Términos de frecuencia
        div_term = torch.exp(
            torch.arange(
                0,
                D_pos_emb,
                2,
                dtype=torch.float
            ) * (-np.log(10000.0) / D_pos_emb)
        )

        # Matriz de positional encoding
        pe = torch.zeros(
            max_len,
            D_pos_emb
        )

        # Posiciones pares -> seno
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # Posiciones impares -> coseno
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Agregamos dimensión batch
        # [1, max_len, D_pos_emb]
        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pos_embedding",
            pe
        )

        # encoder de transformer
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=D_pos_emb,
                                                        nhead = nhead,
                                                        dim_feedforward=D_pos_emb,
                                                        batch_first=True)

        # capa fully connected para realizar clasificaciónn
        self.fc1 = nn.Linear(D_pos_emb, D_pos_emb)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(D_pos_emb, 3)


    def forward(self, x, padding_mask):

        B = x.shape[0]

        # CLS token
        class_token = self.class_token.expand(B, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        # Agregar posición del CLS a la máscara
        cls_mask = torch.zeros(
            B, 1,
            dtype=torch.bool,
            device=x.device
        )

        padding_mask = padding_mask.to(x.device)

        padding_mask = torch.cat(
            (cls_mask, padding_mask),
            dim=1
        )

        # Proyección
        x = self.linear1(x)

        # Embedding posicional
        x = self.pos_embedding + x

        # Transformer
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # CLS
        x = self.fc1(x[:, 0])
        x = self.activation(x)
        x = self.fc2(x)

        return x




class MelTransformer(nn.Module):
    def __init__(self, D_pos_emb=50, nhead=1):
        super().__init__()


        # Token de clasificación
        self.class_token = nn.Parameter(
            data=torch.randn(1, 1, 30),
            requires_grad=True
        )

        # Proyectar los 30 Mel bins a D_pos_emb
        self.linear1 = nn.Linear(30, D_pos_emb)

        # Embedding posicional
        self.pos_embedding = nn.Parameter(
            torch.randn(1, MAX_LEN + 1, D_pos_emb),
            requires_grad=True
        )

        # Encoder Transformer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=D_pos_emb,
            nhead=nhead,
            dim_feedforward=D_pos_emb,
            batch_first=True
        )

        # Clasificador
        self.fc1 = nn.Linear(D_pos_emb, D_pos_emb)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(D_pos_emb, 3)

    def forward(self, x, padding_mask):

        B = x.shape[0]

        # CLS token
        class_token = self.class_token.expand(B, -1, -1)

        x = torch.cat((class_token, x), dim=1)

        # Agregar posición del CLS a la máscara
        cls_mask = torch.zeros(
            B, 1,
            dtype=torch.bool,
            device=x.device
        )

        padding_mask = padding_mask.to(x.device)

        padding_mask = torch.cat(
            (cls_mask, padding_mask),
            dim=1
        )

        # Proyección: 30 -> D_pos_emb
        x = self.linear1(x)

        # Embedding posicional
        x = self.pos_embedding + x

        # Transformer
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # CLS token
        x = self.fc1(x[:, 0])
        x = self.activation(x)
        x = self.fc2(x)

        return x


class MelTransformerSen(nn.Module):
    def __init__(self, D_pos_emb=50, nhead=1):
        super().__init__()


        # Token de clasificación
        self.class_token = nn.Parameter(
            data=torch.randn(1, 1, 30),
            requires_grad=True
        )

        # Proyectar los 30 Mel bins a D_pos_emb
        self.linear1 = nn.Linear(30, D_pos_emb)

        # Embedding posicional
        # embedding posicional
        max_len = MAX_LEN + 1

        # Matriz de posiciones: [max_len, 1]
        position = torch.arange(
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        # Términos de frecuencia
        div_term = torch.exp(
            torch.arange(
                0,
                D_pos_emb,
                2,
                dtype=torch.float
            ) * (-np.log(10000.0) / D_pos_emb)
        )

        # Matriz de positional encoding
        pe = torch.zeros(
            max_len,
            D_pos_emb
        )

        # Posiciones pares -> seno
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # Posiciones impares -> coseno
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Agregamos dimensión batch
        # [1, max_len, D_pos_emb]
        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pos_embedding",
            pe
        )

        # Encoder Transformer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=D_pos_emb,
            nhead=nhead,
            dim_feedforward=D_pos_emb,
            batch_first=True
        )

        # Clasificador
        self.fc1 = nn.Linear(D_pos_emb, D_pos_emb)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(D_pos_emb, 3)

    def forward(self, x, padding_mask):

        B = x.shape[0]

        # CLS token
        class_token = self.class_token.expand(B, -1, -1)

        x = torch.cat((class_token, x), dim=1)

        # Agregar posición del CLS a la máscara
        cls_mask = torch.zeros(
            B, 1,
            dtype=torch.bool,
            device=x.device
        )

        padding_mask = padding_mask.to(x.device)

        padding_mask = torch.cat(
            (cls_mask, padding_mask),
            dim=1
        )

        # Proyección: 30 -> D_pos_emb
        x = self.linear1(x)

        # Embedding posicional
        x = self.pos_embedding + x

        # Transformer
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # CLS token
        x = self.fc1(x[:, 0])
        x = self.activation(x)
        x = self.fc2(x)

        return x