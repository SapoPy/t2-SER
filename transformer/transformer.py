import torch
import torchaudio
from torch import nn
MAX_LEN = 512
def transformer_transform(x):
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


class CustomTransformer(nn.Module):
    def __init__(self, D_pos_emb=50):
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
                                                        nhead = 1,
                                                        dim_feedforward=D_pos_emb,
                                                        batch_first=True)

        # capa fully connected para realizar clasificaciónn
        self.fc1 = nn.Linear(D_pos_emb, D_pos_emb)
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
        x = self.fc2(x)

        return x


    