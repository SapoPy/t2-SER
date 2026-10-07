import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

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


def train_model(
    model_class,
    model_params,
    train_dataset,
    val_dataset,
    learning_rate=1e-4,
    batch_size=64,
    epochs=20,
    loss_fn=None,
    optimizer_class=torch.optim.Adam,
    optimizer_params=None,
    parameters_file="best_model.pt",
    device=None,
    plot=True
):
    # Device
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Modelo
    model = model_class(**model_params).to(device)

    # Función de pérdida
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    # Parámetros adicionales del optimizador
    if optimizer_params is None:
        optimizer_params = {}

    # Optimizador
    optimizer = optimizer_class(
        model.parameters(),
        lr=learning_rate,
        **optimizer_params
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Evolución de las pérdidas
    train_loss_evol = []
    val_loss_evol = []

    # Mejor loss de validación
    best_vloss = np.inf

    # Loop de entrenamiento
    for epoch in range(epochs):

        print(f"\nÉpoca: {epoch + 1}/{epochs}")

        # Entrenamiento
        model.train()

        avg_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            device
        )

        # Validación
        model.eval()

        avg_vloss = get_val_loss(
            model,
            val_dataloader,
            loss_fn,
            device
        )

        # Guardar losses
        train_loss_evol.append(avg_loss)
        val_loss_evol.append(avg_vloss)

        print(
            f"LOSS train: {avg_loss:.6f} | "
            f"valid: {avg_vloss:.6f}"
        )

        # Guardar el mejor modelo
        if avg_vloss < best_vloss:

            best_vloss = avg_vloss

            torch.save(
                model.state_dict(),
                parameters_file
            )

            print(
                f"Modelo guardado en época {epoch + 1} "
                f"(val_loss={avg_vloss:.6f})"
            )

    # Gráfico
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            train_loss_evol,
            label="Entrenamiento"
        )

        ax.plot(
            val_loss_evol,
            label="Validación"
        )

        ax.set(
            xlabel="N° de Época",
            ylabel="Función de costo",
            title="Evolución Entrenamiento"
        )

        ax.grid(alpha=0.5)
        ax.legend()

        plt.show()

    return model, train_loss_evol, val_loss_evol, best_vloss
