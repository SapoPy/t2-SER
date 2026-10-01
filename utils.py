import numpy as np
import torch

# funcion que ejecuta una epoca de entrenamiento,
# actualizando los parametros del modelo
def train_one_epoch(model, dataloader, optimizer, loss_fn, device='cpu'):
  train_loss = [] # se almacena la loss de cada batch

  # se opera cada batch
  for i, data in enumerate(dataloader):
    # se leen los datos
    inputs, labels = data
    # se llevan al dispositivo de ejecución
    inputs, mask = inputs
    inputs = inputs.to(device)
    mask = mask.to(device)
    labels = labels.to(device)
    # se fijan los gradientes en cero
    optimizer.zero_grad()
    # se generan las predicciones
    outputs = model(inputs, mask)

    # se calcula la funcion de perdida y los gradientes
    loss = loss_fn(outputs, labels)
    loss.backward()
    # se ajustan los pesos del modelo
    optimizer.step()
    # se guarda la loss del batch
    train_loss.append(loss.detach().cpu().numpy())

  # se entrega la loss promedio
  return np.mean(train_loss)


# funcion para calcular la loss en validacion
def get_val_loss(model, dataloader, loss_fn, device='cpu'):
  val_loss = []
  # no se calculan gradientes
  with torch.no_grad():
    # se analiza cada batch en validacion
    for i, data in enumerate(dataloader):
        inputs, labels = data
        # se llevan al dispositivo de ejecución
        inputs, mask = inputs
        inputs = inputs.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        # se generan las predicciones
        outputs = model(inputs, mask)
        # se calcula la funcion de perdida
        loss = loss_fn(outputs, labels)
        # se guarda el valor de la función de perdida
        val_loss.append(loss.detach().cpu().numpy())
  return np.mean(val_loss) # se entrega la loss promedio