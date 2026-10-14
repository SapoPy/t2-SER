import torch
import soundfile as sf
import pandas as pd
from torch.utils.data import DataLoader
from datasetEmociones import EmotionDataset, ANNOTATIONS_FILE, AUDIO_DIR 
from transformer import (fft_transform, log_mel_transform, 
                         FFTTransformer, FFTTransformerSen,
                         MelTransformer, MelTransformerSen)


class EmotionDatasetSF(EmotionDataset):
    def __getitem__(self, idx):
        fila = self.audio_labels.iloc[idx]
        datos, _ = sf.read(self.audio_dir + fila["path"], dtype="float32")
        señal = torch.from_numpy(datos).unsqueeze(0)
        etiquetas = torch.tensor([fila["activation"],
                                  fila["valence"],
                                  fila["dominance"]], dtype=torch.float32)
        return self.transform(señal), etiquetas

def ccc(real, pred):
    mr, mp = real.mean(), pred.mean()
    vr, vp = real.var(unbiased=False), pred.var(unbiased=False)
    cov = ((real - mr) * (pred - mp)).mean()
    return (2 * cov / (vr + vp + (mr - mp) ** 2)).item()

def evaluar(modelo, batches):
    modelo.eval()
    predicciones, reales = [], []
    with torch.no_grad():
        for (x, mascara), y in batches:
            predicciones.append(modelo(x, mascara))
            reales.append(y)
    pred = torch.cat(predicciones)
    real = torch.cat(reales)
    mse = ((pred - real) ** 2).mean(dim=0)
    ccc_a = ccc(real[:, 0], pred[:, 0])
    ccc_v = ccc(real[:, 1], pred[:, 1])
    ccc_d = ccc(real[:, 2], pred[:, 2])
    return mse.tolist() + [ccc_a, ccc_v, ccc_d]

def preparar_test(transformacion):
    ds = EmotionDatasetSF(ANNOTATIONS_FILE, AUDIO_DIR,
                          data_portion="test",
                          transform=transformacion)
    return list(DataLoader(ds, batch_size=32, shuffle=False))

print("Calculando espectrogramas de test (FFT)...")
test_fft = preparar_test(fft_transform)
print("Calculando espectrogramas de test (log-Mel)...")
test_mel = preparar_test(log_mel_transform)

modelos = [
    ("FFT  aprendido 1h", FFTTransformer,    1, "FFTTransformer_params_50_1.pt",    test_fft),
    ("FFT  aprendido 5h", FFTTransformer,    5, "FFTTransformer_params_50_5.pt",    test_fft),
    ("FFT  senoidal  1h", FFTTransformerSen, 1, "FFTTransformerSen_params_50_1.pt", test_fft),
    ("FFT  senoidal  5h", FFTTransformerSen, 5, "FFTTransformerSen_params_50_5.pt", test_fft),
    ("Mel  aprendido 1h", MelTransformer,    1, "MelTransformer_params_50_1.pt",    test_mel),
    ("Mel  aprendido 5h", MelTransformer,    5, "MelTransformer_params_50_5.pt",    test_mel),
    ("Mel  senoidal  1h", MelTransformerSen, 1, "MelTransformerSen_params_50_1.pt", test_mel),
    ("Mel  senoidal  5h", MelTransformerSen, 5, "MelTransformerSen_params_50_5.pt", test_mel),
]

filas = []
for nombre, Clase, cabezas, archivo, datos in modelos:
    modelo = Clase(D_pos_emb=50, nhead=cabezas)
    modelo.load_state_dict(torch.load("transformer/" + archivo,
                                      map_location="cpu"))
    mse_a, mse_v, mse_d, ccc_a, ccc_v, ccc_d = evaluar(modelo, datos)
    filas.append([nombre, mse_a, mse_v, mse_d, (mse_a + mse_v + mse_d) / 3,
                  ccc_a, ccc_v, ccc_d, (ccc_a + ccc_v + ccc_d) / 3]) 
    print("Listo:", nombre)

etiquetas = pd.read_csv(AUDIO_DIR + ANNOTATIONS_FILE)
columnas = ["activation", "valence", "dominance"]

media_train = etiquetas[etiquetas["partition"] == "train"][columnas].mean()
reales_test = etiquetas[etiquetas["partition"] == "test"][columnas] 

mse_dummy = ((reales_test - media_train) ** 2).mean()
filas.append(["Dummy", *mse_dummy.tolist(), mse_dummy.mean(),
              0, 0, 0, 0])

tabla = pd.DataFrame(filas, columns=["Modelo",
                                     "MSE A", "MSE V", "MSE D", "MSE prom",
                                     "CCC A", "CCC V", "CCC D", "CCC prom"])
print(tabla.round(3).to_string(index=False))

