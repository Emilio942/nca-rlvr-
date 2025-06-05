import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
import datetime
import os
import random

# Konfiguration
# ----------
# Diese Konstanten können nach Bedarf angepasst werden, um die Simulation zu steuern.
# Sie sind hier zentralisiert, um die Wartung und Abstimmung zu erleichtern.

# NCA Parameter
NCA_HIDDEN_CHANNELS = 12  # Anzahl der internen Kanäle für den NCA-Zustand (zusätzlich zu RGB+Life)
NCA_STATE_CHANNELS = 3 + 1 + NCA_HIDDEN_CHANNELS  # RGB (3) + Life (1) + Hidden
NCA_GRID_SIZE = 64        # Größe des quadratischen Rasters (z.B. 64x64)
NCA_STEPS_PER_GROWTH = 64 # Anzahl der NCA-Schritte pro "Wachstumsphase" oder Bewertung
NCA_CELL_FIRE_RATE = 0.5  # Wahrscheinlichkeit, dass eine Zelle in einem Schritt aktualisiert wird

# Training Parameter
GROWTH_TRAINER_LR = 1e-3   # Lernrate für den NCA-Wachstumstrainer
GROWTH_TRAINER_EPOCHS = 500 # Anzahl der Trainings-Epochen für NCA-Wachstum
RL_EPISODES = 1000         # Anzahl der Episoden für die RL-Stresstest-Umgebung (pro Generation / pro Modell)
VALIDATOR_EPOCHS = 100     # Anzahl der Trainings-Epochen für das Validierungsnetzwerk
VALIDATOR_BATCH_SIZE = 32
VALIDATOR_THRESHOLD = 0.8  # Mindest-Score, damit eine Struktur als "funktional" akzeptiert wird
VALIDATOR_MAX_SEQ_LEN_FALLBACK = NCA_STEPS_PER_GROWTH * 2 # Fallback für max_seq_len, wenn nicht gesetzt

# Evolution Parameter
MUTATION_RATE = 0.1        # Wahrscheinlichkeit einer Mutation pro Parameter
MUTATION_STRENGTH = 0.01   # Stärke der Parameterabweichung bei Mutation
POPULATION_SIZE = 5        # Anzahl der NCAs in der evolutionären Population
NUM_GENERATIONS = 5        # Anzahl der Generationen für die Evolution

# Logging & Visualisierung
LOG_FILE_PATH = "nca_rlvr_log.txt"
VISUALIZATION_FPS = 10     # Bilder pro Sekunde für die Visualisierung

# Gerätekonfiguration (CPU/GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {DEVICE}")

# Initialisiere Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_PATH, mode='w'), logging.StreamHandler()]) # 'w' mode for fresh log each run
logger = logging.getLogger(__name__)


# --- Hilfsfunktionen ---
def to_rgb(state_tensor):
    """
    Konvertiert einen internen NCA-Zustand (mit vielen Kanälen, optional mit Batch-Dimension)
    in ein RGB-Bild (3 Kanäle).
    Die ersten 3 Kanäle des Zustands werden als RGB interpretiert und auf den Bereich [0, 1] normalisiert.
    :param state_tensor: Ein Tensor des NCA-Zustands (..., Channels, H, W).
    :return: Ein Tensor des RGB-Bildes (..., 3, H, W).
    """
    # Schneide die ersten 3 Kanäle aus
    rgb = state_tensor[..., :3, :, :]
    # Normalisiere auf 0-1 Bereich. F.sigmoid ist eine gute Wahl, wenn der Output unbegrenzt ist.
    # Wenn die Kanäle bereits im 0-1 Bereich sein sollten, kann clamp besser sein.
    # Für initiale Ausgabe des Modells, das beliebige Werte ausgeben kann, ist Sigmoid sicher.
    rgb = F.sigmoid(rgb)
    return rgb

def get_living_mask(state_tensor):
    """
    Erzeugt eine Maske, die anzeigt, welche Zellen "lebendig" sind.
    Dies wird basierend auf dem Alpha-Kanal (dem 4. Kanal) bestimmt.
    Eine Zelle gilt als lebendig, wenn ihr Alpha-Wert (oder ein ähnlicher "Lebens"-Indikator) einen Schwellenwert überschreitet.
    Verwendet Max-Pooling, um auch benachbarte Zellen mit einzubeziehen, um Rauschen zu reduzieren.
    """
    alpha = state_tensor[..., 3:4, :, :] # Der 4. Kanal ist oft der Alpha-Kanal oder "Lebensstatus"
    # F.max_pool2d expects (N, C, H, W) or (C, H, W). Use unsqueeze/squeeze if needed.
    # Here, `state_tensor` might have a batch dimension, or not. `...` handles it.
    # The output of alpha is (..., 1, H, W). max_pool2d needs C, H, W.
    # If state_tensor is (C, H, W), then alpha is (1, H, W).
    # If state_tensor is (B, C, H, W), then alpha is (B, 1, H, W).
    # max_pool2d works on both.
    return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 1: Basissystem für Neural Cellular Automata (NCA)
# ----------------------------------------------------------------------------------------------------

class NCA_Model_3x3(nn.Module):
    """
    Angepasstes NCA-Modell, das 3x3 Convolutions verwendet, um die Nachbarschaft direkt zu verarbeiten.
    Dies ist die gängigere Implementierung für "Neural Cellular Automata".
    """
    def __init__(self, in_channels, hidden_channels, out_channels, logbook=None, model_id="N/A"):
        """
        Initialisiert das NCA-Modell mit 3x3 Convolution-Schichten.
        :param in_channels: Anzahl der Eingangskanäle (NCA_STATE_CHANNELS).
        :param hidden_channels: Anzahl der Kanäle in den verdeckten Schichten.
        :param out_channels: Anzahl der Ausgangskanäle (NCA_STATE_CHANNELS).
        :param logbook: Optional Logbook instance for XBA logging.
        :param model_id: Optional model identifier for logging.
        """
        super().__init__()
        self.logbook = logbook
        self.model_id = model_id
        # PyTorch Conv2d erwartet (batch, channels, H, W)
        # padding='same' sorgt dafür, dass die Output-Dimensionen die gleichen wie die Input-Dimensionen sind.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU() # Nicht-Linearität

        # XBA attributes
        # Initialize with dummy sizes, will be properly sized in the first forward pass or when state is known
        self.mean_abs_grad_conv1 = torch.zeros(hidden_channels, device=DEVICE)
        self.mean_abs_grad_conv2 = torch.zeros(out_channels, device=DEVICE)
        self.std_output_conv1 = torch.zeros(hidden_channels, device=DEVICE)
        self.std_output_conv2 = torch.zeros(out_channels, device=DEVICE)
        self.xba_logging_data = [] # Can still be used for model's internal quick logging if needed
        self._conv1_hook_handle = None
        self._conv2_hook_handle = None
        # self.logbook and self.model_id are already set above

    def forward(self, state):
        """
        Führt einen Forward-Pass des NCA-Modells durch.
        :param state: Der gesamte NCA-Zustand des Gitters (Batch_size, Channels, Height, Width)
        :return: Das berechnete Delta für die Zustandsaktualisierung.
                 (Batch_size, Channels, Height, Width)
        """
        # Ensure hooks are registered if not already
        if self._conv1_hook_handle is None:
            self._conv1_hook_handle = self.conv1.weight.register_hook(self._grad_hook_conv1)
        if self._conv2_hook_handle is None:
            self._conv2_hook_handle = self.conv2.weight.register_hook(self._grad_hook_conv2)

        # Forward pass
        out1 = self.conv1(state)
        # Calculate std_output for conv1 neurons (output channels of conv1)
        # Output shape: (batch_size, hidden_channels, H, W)
        # We want std per output channel, across batch, H, W
        # Calculate std_output regardless of requires_grad for XBA visualization purposes
        # This assumes that if XBA is active, these stats are useful.
        # If performance is critical, a flag could control this.
        self.std_output_conv1 = torch.std(out1, dim=(0, 2, 3)) # Shape: (hidden_channels,)

        x = self.relu(out1)

        out2 = self.conv2(x)
        # Calculate std_output for conv2 neurons (output channels of conv2)
        # Output shape: (batch_size, out_channels, H, W)
        self.std_output_conv2 = torch.std(out2, dim=(0, 2, 3)) # Shape: (out_channels,)

        return out2

    def _grad_hook_conv1(self, grad):
        """Hook to capture gradients for conv1 layer's weights."""
        if grad is not None:
            # grad shape for conv1.weight: (out_channels, in_channels, kH, kW)
            # We want mean absolute gradient per output neuron (filter)
            # So, calculate mean abs grad for each output channel's filter weights
            abs_grad = torch.abs(grad)
            # Mean over in_channels, kH, kW for each output filter
            self.mean_abs_grad_conv1 = torch.mean(abs_grad, dim=(1, 2, 3)) # Shape: (out_channels_conv1,)

    def _grad_hook_conv2(self, grad):
        """Hook to capture gradients for conv2 layer's weights."""
        if grad is not None:
            # grad shape for conv2.weight: (out_channels, in_channels, kH, kW)
            abs_grad = torch.abs(grad)
            self.mean_abs_grad_conv2 = torch.mean(abs_grad, dim=(1, 2, 3)) # Shape: (out_channels_conv2,)

    def identify_dead_neurons(self, epsilon=1e-6, std_threshold=1e-4):
        """
        Identifiziert Neuronen, die als "tot" betrachtet werden können.
        :param epsilon: Schwellenwert für den mittleren absoluten Gradienten.
        :param std_threshold: Schwellenwert für die Standardabweichung des Outputs.
        :return: Tuple von Tensoren: (dead_conv1_indices, dead_conv2_indices)
        """
        # Dead neurons for conv1 (output channels of conv1 are its neurons)
        dead_conv1_mask = (self.mean_abs_grad_conv1 < epsilon) & (self.std_output_conv1 < std_threshold)
        dead_conv1_indices = torch.where(dead_conv1_mask)[0]

        # Dead neurons for conv2 (output channels of conv2 are its neurons)
        dead_conv2_mask = (self.mean_abs_grad_conv2 < epsilon) & (self.std_output_conv2 < std_threshold)
        dead_conv2_indices = torch.where(dead_conv2_mask)[0]

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log to model's internal xba_logging_data with structured dicts
        if dead_conv1_indices.numel() > 0:
            log_entry_conv1 = {
                'timestamp': current_time, 'type': 'identification', 'model_id': self.model_id,
                'layer': 'conv1', 'neurons': dead_conv1_indices.tolist(),
                'details': {
                    'mean_abs_grad': self.mean_abs_grad_conv1[dead_conv1_indices].cpu().tolist(),
                    'std_output': self.std_output_conv1[dead_conv1_indices].cpu().tolist(),
                    'epsilon_threshold': epsilon, 'std_dev_threshold': std_threshold
                }
            }
            self.xba_logging_data.append(log_entry_conv1)
            if self.logbook: # Also log to global logbook if available
                self.logbook.log_xba_identification(
                    timestamp=current_time, model_id=self.model_id, layer_name='conv1',
                    neuron_indices=dead_conv1_indices,
                    details=f"MAG_mean={torch.mean(self.mean_abs_grad_conv1[dead_conv1_indices]).item():.2e}, StdOut_mean={torch.mean(self.std_output_conv1[dead_conv1_indices]).item():.2e}"
                )

        if dead_conv2_indices.numel() > 0:
            log_entry_conv2 = {
                'timestamp': current_time, 'type': 'identification', 'model_id': self.model_id,
                'layer': 'conv2', 'neurons': dead_conv2_indices.tolist(),
                'details': {
                    'mean_abs_grad': self.mean_abs_grad_conv2[dead_conv2_indices].cpu().tolist(),
                    'std_output': self.std_output_conv2[dead_conv2_indices].cpu().tolist(),
                    'epsilon_threshold': epsilon, 'std_dev_threshold': std_threshold
                }
            }
            self.xba_logging_data.append(log_entry_conv2)
            if self.logbook: # Also log to global logbook if available
                self.logbook.log_xba_identification(
                    timestamp=current_time, model_id=self.model_id, layer_name='conv2',
                    neuron_indices=dead_conv2_indices,
                    details=f"MAG_mean={torch.mean(self.mean_abs_grad_conv2[dead_conv2_indices]).item():.2e}, StdOut_mean={torch.mean(self.std_output_conv2[dead_conv2_indices]).item():.2e}"
                )

        return dead_conv1_indices, dead_conv2_indices

    def reactivate_neurons(self, dead_conv1_indices, dead_conv2_indices, reactivation_method='reinitialize'):
        """
        Reaktiviert tote Neuronen.
        :param dead_conv1_indices: Indizes der toten Neuronen in conv1.
        :param dead_conv2_indices: Indizes der toten Neuronen in conv2.
        :param reactivation_method: 'reinitialize' oder 'inject_noise'.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with torch.no_grad():
            # Reactivate conv1 neurons
            if dead_conv1_indices.numel() > 0:
                for idx in dead_conv1_indices: # Iterate for precise application
                    if reactivation_method == 'reinitialize':
                        nn.init.kaiming_normal_(self.conv1.weight[idx, :, :, :])
                        if self.conv1.bias is not None:
                             nn.init.zeros_(self.conv1.bias[idx])
                    elif reactivation_method == 'inject_noise':
                        noise_w = torch.randn_like(self.conv1.weight[idx, :, :, :]) * 0.01
                        self.conv1.weight[idx, :, :, :].add_(noise_w)
                        if self.conv1.bias is not None:
                            noise_b = torch.randn_like(self.conv1.bias[idx]) * 0.01
                            self.conv1.bias[idx].add_(noise_b)

                if self.logbook:
                    self.logbook.log_xba_action(
                        timestamp=current_time, model_id=self.model_id, layer_name='conv1',
                        neuron_indices=dead_conv1_indices, action_type=reactivation_method,
                        details=f"{dead_conv1_indices.numel()} neurons affected"
                    )
                # Always log to model's internal xba_logging_data with structured dict
                self.xba_logging_data.append({
                    'timestamp': current_time, 'type': 'reactivation', 'model_id': self.model_id,
                    'layer': 'conv1', 'neurons': dead_conv1_indices.tolist(),
                    'method': reactivation_method,
                    'count': dead_conv1_indices.numel()
                })

            # Reactivate conv2 neurons
            if dead_conv2_indices.numel() > 0:
                for idx in dead_conv2_indices: # Iterate for precise application
                    if reactivation_method == 'reinitialize':
                        nn.init.kaiming_normal_(self.conv2.weight[idx, :, :, :])
                        if self.conv2.bias is not None:
                            nn.init.zeros_(self.conv2.bias[idx])
                    elif reactivation_method == 'inject_noise':
                        noise_w = torch.randn_like(self.conv2.weight[idx, :, :, :]) * 0.01
                        self.conv2.weight[idx, :, :, :].add_(noise_w)
                        if self.conv2.bias is not None:
                            noise_b = torch.randn_like(self.conv2.bias[idx]) * 0.01
                            self.conv2.bias[idx].add_(noise_b)

                if self.logbook:
                    self.logbook.log_xba_action(
                        timestamp=current_time, model_id=self.model_id, layer_name='conv2',
                        neuron_indices=dead_conv2_indices, action_type=reactivation_method,
                        details=f"{dead_conv2_indices.numel()} neurons affected"
                    )
                # Always log to model's internal xba_logging_data with structured dict
                self.xba_logging_data.append({
                    'timestamp': current_time, 'type': 'reactivation', 'model_id': self.model_id,
                    'layer': 'conv2', 'neurons': dead_conv2_indices.tolist(),
                    'method': reactivation_method,
                    'count': dead_conv2_indices.numel()
                })

    def remove_hooks(self):
        """Removes gradient hooks if they exist."""
        if self._conv1_hook_handle is not None:
            self._conv1_hook_handle.remove()
            self._conv1_hook_handle = None
        if self._conv2_hook_handle is not None:
            self._conv2_hook_handle.remove()
            self._conv2_hook_handle = None


def nca_step_function(state, model, current_step=0, xba_enabled=False, xba_check_interval=10,
                        epsilon=1e-6, std_threshold=1e-4, reactivation_method='reinitialize'):
    """
    Führt einen Simulationsschritt basierend auf lokalen Regeln des NCA-Modells aus.
    Dieser Schritt ist differenzierbar.
    :param state: Der aktuelle Zustand des NCA-Gitters (Batch_size, Channels, Height, Width).
    :param model: Das trainierte NCA_Model (NCA_Model_3x3 Instanz).
    :param current_step: Der aktuelle Simulationsschritt (für XBA).
    :param xba_enabled: Ob XBA aktiviert ist.
    :param xba_check_interval: Intervall für XBA-Prüfungen.
    :param epsilon: Epsilon-Wert für XBA identify_dead_neurons.
    :param std_threshold: Std_threshold-Wert für XBA identify_dead_neurons.
    :param reactivation_method: Methode für XBA reactivate_neurons.
    :return: Der neue Zustand des NCA-Gitters nach einem Schritt.
    """
    # XBA Logic
    if xba_enabled and (current_step % xba_check_interval == 0) and current_step > 0:
        # Ensure model is in training mode for gradients to be available if XBA is used during training
        # However, nca_step_function can be called with torch.no_grad() context,
        # in which case mean_abs_grad might not be up-to-date.
        # XBA is typically more relevant during training.
        # For now, we assume if xba_enabled, the necessary grads for mean_abs_grad were computed.

        # It's important that model(state) runs before identify_dead_neurons if identify_dead_neurons
        # relies on std_output from the current forward pass, and that backward() has been called
        # for mean_abs_grad to be populated.
        # Given nca_step_function is one step, backward() is usually called after many such steps.
        # This implies mean_abs_grad is from a *previous* training iteration's backward pass.
        # std_output is from the *current* forward pass if model(state) is called before this block.
        # Let's call model(state) first, then XBA.
        pass # XBA logic will be inserted after delta_state calculation

    # 1. Lebendigkeitsmaske vor der Aktualisierung (um tote Zellen nicht zu aktualisieren)
    # Diese Maske wird verwendet, um zu verhindern, dass die Aktualisierung an bereits toten Zellen stattfindet,
    # die keine Nachbarverbindung zu lebenden Zellen haben.
    pre_living_mask = get_living_mask(state)

    # 2. Berechne das Delta mit dem NCA-Modell
    delta_state = model(state) # This populates std_output if model is in training mode

    # XBA Logic - Placed after model(state) call to use current std_output
    if xba_enabled and (current_step % xba_check_interval == 0) and current_step > 0:
        # Note: mean_abs_grad would be from the *last optimizer step*.
        # std_output is from the *current forward pass* (model(state) above).
        # This timing is typical for such interventions.
        if hasattr(model, 'identify_dead_neurons') and hasattr(model, 'reactivate_neurons'):
            dead_conv1_indices, dead_conv2_indices = model.identify_dead_neurons(epsilon, std_threshold)
            if dead_conv1_indices.numel() > 0 or dead_conv2_indices.numel() > 0:
                logger.info(f"[XBA Interv@step {current_step}] Identifying dead neurons. Conv1: {dead_conv1_indices.tolist()}, Conv2: {dead_conv2_indices.tolist()}")
                model.reactivate_neurons(dead_conv1_indices, dead_conv2_indices, reactivation_method)
                logger.info(f"[XBA Interv@step {current_step}] Attempted reactivation using '{reactivation_method}'.")
        else:
            logger.warning(f"[XBA Interv@step {current_step}] XBA enabled but model missing XBA methods.")


    # 3. Zufällige Aktualisierungsmaske (Stochastic Update)
    # Ein wichtiges Element in NCAs für Selbstorganisation und Reduzierung von Mustern.
    # Wichtig: Die update_mask sollte nur auf Zellen angewendet werden, die gemäß pre_living_mask lebendig sind.
    update_mask = (torch.rand(state.shape[0], 1, state.shape[2], state.shape[3], device=DEVICE) < NCA_CELL_FIRE_RATE).float()
    
    # Kombiniere die Zufallsmaske mit der Lebendigkeitsmaske, damit nur lebendige Zellen aktualisiert werden
    # und das Modell nicht "versucht", tote Zellen zu aktivieren, wenn diese nicht am Rande des Wachstums sind.
    # Aber Vorsicht: Wenn die Pre-Mask zu restriktiv ist, verhindert sie Wachstum.
    # Der NCA-Artikel wendet die Fire Rate *direkt* auf das Delta an, nachdem das Delta von toten Zellen (alpha=0) maskiert wurde.
    
    # Für ein Wachstum, das sich ausbreiten kann, sollte die `pre_living_mask`
    # eher dazu dienen, Zellen zu "schützen" oder zu "ignorieren", die offensichtlich tot sind
    # und keine Reaktion von ihren Nachbarn erhalten.
    # Eine einfachere und gängigere Methode ist, das Delta nur für aktive Zellen zu berechnen
    # oder die Maske erst *nach* dem Delta auf den Zustand anzuwenden.

    # Im originalen NCA-Paper wird das Delta für *alle* Zellen berechnet, dann wird die Update-Maske (Fire Rate)
    # angewendet, und schließlich eine Lebendigkeitsmaske, die Zellen entfernt, die zu "schwach" werden.
    # Die "Self-Correction" der NCAs kommt daher, dass das Modell auch tote Zellen zu lebendigen machen kann,
    # wenn der Alpha-Wert über den Threshold steigt.

    # Korrektur des gravierenden Logikfehlers 1.1:
    # Die `state` Aktualisierung sollte nicht durch `pre_living_mask` begrenzt werden,
    # um Wachstum in neue Bereiche zu ermöglichen. Die `pre_living_mask` wird nur zur Steuerung
    # der `update_mask` verwendet, nicht zur direkten Einschränkung der Zustandsänderung.
    
    # Wende das Delta auf den Zustand an, aber nur für die zufällig ausgewählten Zellen.
    state = state + delta_state * update_mask

    # 5. Post-Living-Maske: Zellen sterben, wenn ihr Alpha-Wert unter den Schwellenwert fällt.
    # Dies ist der Mechanismus, durch den Zellen "sterben" und Strukturen begrenzt werden.
    post_living_mask = get_living_mask(state)
    state = state * post_living_mask.float() # Zellen, die "sterben", werden auf 0 gesetzt

    # Stelle sicher, dass die ersten 3 Kanäle (RGB) im Bereich [0, 1] bleiben
    state[:, :3, :, :] = torch.clamp(state[:, :3, :, :], 0.0, 1.0)
    # Der Alpha-Kanal (4. Kanal) kann auch geklemmt werden, z.B. auf [0, 1]
    state[:, 3:4, :, :] = torch.clamp(state[:, 3:4, :, :], 0.0, 1.0)

    return state

def initial_seed_generator(grid_size, num_channels, batch_size=1, seed_type='center_pixel'):
    """
    Generiert die anfängliche Startkonfiguration für das NCA-Gitter.
    :param grid_size: Die Größe des quadratischen Gitters (z.B. 64).
    :param num_channels: Die Anzahl der Kanäle im NCA-Zustand.
    :param batch_size: Anzahl der gleichzeitigen NCA-Simulationen.
    :param seed_type: Art des Start-Seeds ('center_pixel', 'random_noise', 'predefined_shape').
    :return: Ein Tensor des initialen Zustands (Batch_size, Channels, Height, Width).
    """
    initial_state = torch.zeros(batch_size, num_channels, grid_size, grid_size, device=DEVICE)

    if seed_type == 'center_pixel':
        center_x, center_y = grid_size // 2, grid_size // 2
        # Setze einen zentralen Punkt auf volle Intensität (z.B. Weiß und volle "Lebenskraft")
        initial_state[:, :3, center_y, center_x] = 1.0  # RGB auf 1 (Weiß)
        initial_state[:, 3, center_y, center_x] = 1.0  # Alpha/Life auf 1 (Voll lebendig)
    elif seed_type == 'random_noise':
        # Füge etwas Rauschen hinzu, um das Wachstum zu initiieren
        initial_state = torch.rand(batch_size, num_channels, grid_size, grid_size, device=DEVICE) * 0.1
        initial_state[:, 3, :, :] = (initial_state[:, 3, :, :] > 0.05).float() # Einfache Lebendigkeitsmaske
    elif seed_type == 'predefined_shape':
        # Beispiel: Eine kleine Box in der Mitte
        box_size = grid_size // 8
        start_x = (grid_size - box_size) // 2
        end_x = start_x + box_size
        start_y = (grid_size - box_size) // 2
        end_y = start_y + box_size
        initial_state[:, :3, start_y:end_y, start_x:end_x] = 0.5 # Graue Box
        initial_state[:, 3, start_y:end_y, start_x:end_x] = 1.0 # Volle Lebenskraft
    else:
        raise ValueError(f"Unbekannter seed_type: {seed_type}")

    return initial_state

class Growth_Trainer:
    """
    Trainiert das NCA-Modell so, dass eine Zielstruktur erreicht wird.
    """
    def __init__(self, nca_model, target_image_path=None, target_image=None):
        """
        Initialisiert den Growth_Trainer.
        :param nca_model: Die Instanz des NCA_Model_3x3, die trainiert werden soll.
        :param target_image_path: Pfad zum Zielbild (z.B. PNG).
        :param target_image: Optional ein direkt übergebener Tensor des Zielbildes.
        """
        self.model = nca_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=GROWTH_TRAINER_LR)
        self.criterion = nn.MSELoss() # Oder L1Loss, je nach gewünschtem Ergebnis

        if target_image_path:
            # Lade und verarbeite das Zielbild
            try:
                from PIL import Image
                img = Image.open(target_image_path).convert('RGB')
                img = img.resize((NCA_GRID_SIZE, NCA_GRID_SIZE), Image.Resampling.LANCZOS)
                self.target_image = torch.tensor(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                logger.info(f"Zielbild '{target_image_path}' geladen und auf {NCA_GRID_SIZE}x{NCA_GRID_SIZE} skaliert.")
            except ImportError:
                logger.error("PIL (Pillow) ist nicht installiert. Kann keine Bilder laden. Bitte 'pip install Pillow' ausführen.")
                self.target_image = torch.ones(1, 3, NCA_GRID_SIZE, NCA_GRID_SIZE, device=DEVICE) * 0.5 # Dummy-Ziel
            except FileNotFoundError:
                logger.error(f"Zielbild '{target_image_path}' nicht gefunden. Verwende Dummy-Ziel.")
                self.target_image = torch.ones(1, 3, NCA_GRID_SIZE, NCA_GRID_SIZE, device=DEVICE) * 0.5 # Dummy-Ziel
        elif target_image is not None:
            self.target_image = target_image.to(DEVICE)
        else:
            # Standard-Zielbild (z.B. ein Kreis oder Quadrat, falls kein Bild gegeben)
            self.target_image = torch.zeros(1, 3, NCA_GRID_SIZE, NCA_GRID_SIZE, device=DEVICE)
            # Einfaches Quadrat als Ziel
            sq_size = NCA_GRID_SIZE // 4
            sq_start = NCA_GRID_SIZE // 2 - sq_size // 2
            sq_end = sq_start + sq_size
            self.target_image[:, :, sq_start:sq_end, sq_start:sq_end] = 1.0 # Weißes Quadrat
            logger.info("Kein Zielbild oder Pfad angegeben. Verwende Standard-Quadrat als Ziel.")

        # Das Zielbild muss die gleiche Form haben wie die RGB-Kanäle des NCA-Zustands
        if self.target_image.shape[1] != 3 or self.target_image.shape[2] != NCA_GRID_SIZE or \
           self.target_image.shape[3] != NCA_GRID_SIZE:
            raise ValueError(f"Zielbild muss (1, 3, {NCA_GRID_SIZE}, {NCA_GRID_SIZE}) sein. Aktuell: {self.target_image.shape}")


    def train_step(self, num_steps=NCA_STEPS_PER_GROWTH):
        """
        Führt einen Trainingsschritt für das NCA-Modell aus.
        Generiert einen Start-Seed, simuliert Wachstum und berechnet den Verlust zum Zielbild.
        :param num_steps: Anzahl der NCA-Schritte für die Simulation.
        :return: Den aktuellen Verlust.
        """
        self.optimizer.zero_grad()
        self.model.train() # Ensure model is in training mode for hooks and XBA stats

        # Generiere einen initialen Seed
        state = initial_seed_generator(NCA_GRID_SIZE, NCA_STATE_CHANNELS, batch_size=1, seed_type='center_pixel')
        
        # Simuliere NCA-Wachstum über mehrere Schritte
        for step_idx in range(num_steps):
            # XBA default parameters for Growth_Trainer
            state = nca_step_function(
                state, self.model,
                current_step=step_idx + 1, # Step counts usually start from 1
                xba_enabled=True, # Enable XBA during growth training
                xba_check_interval=10, # Check every 10 steps
                epsilon=1e-7, # Slightly lower epsilon for more sensitivity
                std_threshold=1e-5, # Slightly lower std_threshold
                reactivation_method='reinitialize'
            )

        # Extrahiere die RGB-Kanäle des finalen Zustands
        final_rgb = to_rgb(state)

        # Berechne den Verlust zum Zielbild
        loss = self.criterion(final_rgb, self.target_image)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, epochs=GROWTH_TRAINER_EPOCHS):
        """
        Führt den gesamten Trainingsprozess für das NCA-Modell durch.
        :param epochs: Anzahl der Epochen.
        """
        logger.info(f"Beginne NCA Wachstumstraining für {epochs} Epochen...")
        for epoch in range(epochs):
            loss = self.train_step()
            if (epoch + 1) % (epochs // 10 or 1) == 0:
                logger.info(f"Epoche {epoch+1}/{epochs}, Loss: {loss:.6f}")
        logger.info("NCA Wachstumstraining abgeschlossen.")

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 2: Reinforcement-Umgebung zur Stresstest-Simulation
# ----------------------------------------------------------------------------------------------------

class StressTest_Env(gym.Env):
    """
    Gymnasium-Umgebung zur Stresstest-Simulation für NCA-generierte Strukturen.
    Der Agent (ein beliebiges RL-Modell, das mit dieser Umgebung interagiert) kann
    Aktionen ausführen, um die NCA-Struktur zu beeinflussen (z.B. Schäden verursachen).
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': VISUALIZATION_FPS}

    def __init__(self, nca_model_for_env, render_mode=None):
        """
        Initialisiert die Stresstest-Umgebung.
        :param nca_model_for_env: Die NCA_Model_3x3 Instanz, deren Wachstum in dieser Umgebung getestet wird.
        :param render_mode: "human" für Anzeige, "rgb_array" für Array-Rückgabe, None für keine Anzeige.
        """
        super().__init__()
        self.nca_model = nca_model_for_env
        self.grid_size = NCA_GRID_SIZE
        self.state_channels = NCA_STATE_CHANNELS
        self.current_nca_state = None
        self.render_mode = render_mode

        # Beobachtungsraum: Der aktuelle NCA-Zustand (RGB für visuelle Repräsentation)
        # Die RGB-Ausgabe von to_rgb ist (3, H, W). Box erwartet dies für numpy-Arrays.
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.grid_size, self.grid_size), dtype=np.float32)

        # Aktionsraum: Beispielaktionen, die der RL-Agent ausführen kann.
        # Aktion 0: Nichts tun (Wachstum fortsetzen)
        # Aktion 1: Zufälligen Bereich beschädigen (Zellen auf 0 setzen)
        # Aktion 2: Rotation (weniger sinnvoll für 2D NCA, aber als Beispiel)
        # Aktion 3: Druck anwenden (z.B. Zellen am Rand entfernen)
        self.action_space = spaces.Discrete(4) # 0: No-op, 1: Damage, 2: Rotate, 3: Pressure

        self.fig, self.ax = None, None
        self.im = None

    def _get_obs(self):
        """
        Gibt den Beobachtungszustand für den RL-Agenten zurück (RGB-Kanäle des NCA-Zustands).
        Die `to_rgb` Funktion gibt einen (1, 3, H, W) Tensor zurück, da `self.current_nca_state`
        eine Batch-Dimension hat. Für Gymnasium wird jedoch (3, H, W) erwartet, daher `squeeze(0)`.
        """
        return to_rgb(self.current_nca_state).squeeze(0).cpu().numpy()

    def _get_info(self):
        """Gibt zusätzliche Informationen über den Zustand der Umgebung zurück."""
        # Könnte z.B. die Anzahl der lebenden Zellen oder die Komplexität der Struktur enthalten
        living_cells = get_living_mask(self.current_nca_state).sum().item()
        return {"living_cells": living_cells}

    def reset(self, seed=None, options=None):
        """
        Setzt die Umgebung auf den Anfangszustand zurück.
        Generiert einen neuen initialen NCA-Seed und lässt ihn einige Schritte wachsen,
        bevor der Agent die Kontrolle übernimmt.
        """
        super().reset(seed=seed)
        self.current_nca_state = initial_seed_generator(self.grid_size, self.state_channels, batch_size=1, seed_type='center_pixel')
        
        # Lasse den NCA einige Schritte "vorwachsen", um eine initiale Struktur zu bilden
        with torch.no_grad(): # Kein Gradient Tracking während der Reset-Phase
            self.nca_model.eval() # Set model to eval mode if not training
            for step_idx in range(NCA_STEPS_PER_GROWTH // 2): # Halbe Anzahl der normalen Wachstumsschritte
                self.current_nca_state = nca_step_function(
                    self.current_nca_state, self.nca_model,
                    current_step=step_idx + 1,
                    xba_enabled=False, # XBA typically not used in no_grad/eval contexts like reset
                                       # or if used, ensure grads are not required by XBA parts
                    xba_check_interval=10,
                    epsilon=1e-6,
                    std_threshold=1e-4,
                    reactivation_method='reinitialize'
                )
        
        observation = self._get_obs()
        info = self._get_info()
        logger.debug(f"Umgebung zurückgesetzt. Lebende Zellen: {info['living_cells']}")

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        """
        Führt einen Simulationsschritt in der Umgebung aus.
        :param action: Die vom RL-Agenten gewählte Aktion.
        :return: (observation, reward, terminated, truncated, info)
        """
        previous_living_cells = get_living_mask(self.current_nca_state).sum().item()

        # 1. Wende die Aktion des Agenten an
        if action == 1:  # Schaden zufügen
            logger.debug("Aktion: Schaden zufügen")
            # Zufälligen quadratischen Bereich auf 0 setzen (simuliert Zerstörung)
            damage_size = random.randint(self.grid_size // 10, self.grid_size // 5)
            start_x = random.randint(0, self.grid_size - damage_size)
            start_y = random.randint(0, self.grid_size - damage_size)
            self.current_nca_state[:, :, start_y:start_y+damage_size, start_x:start_x+damage_size] = 0.0
        elif action == 2: # Rotation (simuliert eine äußere Kraft)
            logger.debug("Aktion: Rotation")
            # Einfache Rotationssimulation: z.B. Pixel verschieben
            self.current_nca_state = torch.roll(self.current_nca_state, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(2, 3))
        elif action == 3: # Druck anwenden (simuliert äußeren Druck/Begrenzung)
            logger.debug("Aktion: Druck anwenden")
            # Zellen am Rand entfernen (z.B. die äußersten 2 Pixel)
            self.current_nca_state[:, :, :2, :] = 0.0
            self.current_nca_state[:, :, -2:, :] = 0.0
            self.current_nca_state[:, :, :, :2] = 0.0
            self.current_nca_state[:, :, :, -2:] = 0.0
        # Aktion 0 (No-op) erfordert keine direkte Manipulation des Zustands hier.

        # 2. Lasse den NCA für einen Schritt "heilen" oder weiterwachsen
        with torch.no_grad(): # Kein Gradient Tracking während der RL-Simulation
            self.nca_model.eval() # Set model to eval mode
            # current_step for StressTest_Env step, assuming each call is one logical step in its context
            # If StressTest_Env has its own step counter, it should be used.
            # For now, let's assume a single step, so current_step=1 or 0.
            # If XBA were to be active here, a step counter would be needed for `xba_check_interval`.
            # For now, disabling XBA as it's usually for training.
            self.current_nca_state = nca_step_function(
                self.current_nca_state, self.nca_model,
                current_step=1, # Placeholder, ideally env would track its own steps if XBA is on
                xba_enabled=False, # XBA typically not used in no_grad/eval contexts
                xba_check_interval=10,
                epsilon=1e-6,
                std_threshold=1e-4,
                reactivation_method='reinitialize'
            )
        
        # 3. Belohnungsberechnung
        current_living_cells = get_living_mask(self.current_nca_state).sum().item()
        
        reward = 0.0
        terminated = False
        truncated = False # truncated = True, wenn max_steps erreicht ist (durch den RL-Loop gesteuert)

        if current_living_cells > (NCA_GRID_SIZE * NCA_GRID_SIZE * 0.01): # Mindestens 1% der Zellen leben
            reward += 1.0 # Überlebensbonus
            if action == 1: # Wenn Schaden angewendet wurde, belohne für Regeneration
                if current_living_cells > previous_living_cells * 0.9: # Wenn sich die Struktur gut erholt hat
                    reward += 2.0
            elif action == 0: # Belohne Stabilität bei No-op
                if current_living_cells >= previous_living_cells:
                    reward += 1.0
        else:
            reward -= 5.0 # Große Bestrafung bei Kollaps
            terminated = True # Episode beendet, wenn Struktur stirbt

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Rendert den aktuellen Zustand der Umgebung."""
        if self.render_mode == 'human':
            rgb_array = self._get_obs() # Numpy Array (3, H, W)
            rgb_array = np.transpose(rgb_array, (1, 2, 0)) # (H, W, 3) für Matplotlib

            if self.fig is None:
                plt.ion() # Interaktiver Modus
                self.fig, self.ax = plt.subplots(1, 1, figsize=(self.grid_size/10, self.grid_size/10))
                self.im = self.ax.imshow(rgb_array)
                self.ax.axis('off')
                self.fig.tight_layout(pad=0)
            else:
                self.im.set_data(rgb_array)
            self.fig.canvas.draw_idle()
            plt.pause(0.01) # Kleine Pause für die Visualisierung
        elif self.render_mode == 'rgb_array':
            rgb_array = self._get_obs() # Numpy Array (3, H, W)
            return np.transpose(rgb_array, (1, 2, 0)) # (H, W, 3)
        return None

    def close(self):
        """Schließt die Rendering-Fenster."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None

class Reinforcement_Loop:
    """
    Simuliert einen einfachen Reinforcement Learning Loop.
    Dies ist ein Placeholder, der einen sehr einfachen Agenten (z.B. Zufallsagent oder eine
    Grundlagen-Policy, die versucht, die Struktur zu erhalten) beinhaltet.
    Für echte RL würde hier ein komplexerer Algorithmus (PPO, DQN etc.) zum Einsatz kommen.
    """
    def __init__(self, env, agent_policy=None):
        self.env = env
        # Einfacher Zufallsagent als Placeholder
        self.agent_policy = agent_policy if agent_policy else lambda obs: self.env.action_space.sample()

    def run_episode(self, max_steps=NCA_STEPS_PER_GROWTH):
        """
        Führt eine einzelne RL-Episode durch.
        :param max_steps: Maximale Anzahl von Schritten in einer Episode.
        :return: Totalbelohnung der Episode, Liste der Zustände, ob terminated
        """
        obs, info = self.env.reset()
        total_reward = 0
        done = False
        # Speichere den initialen Zustand als (NCA_STATE_CHANNELS, H, W) Tensor
        episode_states = [self.env.current_nca_state.squeeze(0).cpu()] 
        
        for step in range(max_steps):
            action = self.agent_policy(obs) # Agent wählt eine Aktion
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            # Speichere den Zustand nach dem Schritt als (NCA_STATE_CHANNELS, H, W) Tensor
            episode_states.append(self.env.current_nca_state.squeeze(0).cpu()) 
            
            if terminated or truncated:
                done = True
                break
        
        logger.debug(f"Episode abgeschlossen. Schritte: {step+1}, Belohnung: {total_reward:.2f}, Terminated: {terminated}")
        return total_reward, episode_states, terminated

class SimRunner:
    """
    Führt eine Vielzahl von Simulationen (RL-Episoden) durch, um Daten zu sammeln
    und das Verhalten von NCA-generierten Strukturen unter Stress zu bewerten.
    """
    def __init__(self, nca_model_for_sims, num_simulations=RL_EPISODES):
        self.nca_model = nca_model_for_sims
        self.num_simulations = num_simulations
        self.env = StressTest_Env(nca_model_for_env=self.nca_model, render_mode=None) # Keine visuelle Ausgabe hier
        self.rl_loop = Reinforcement_Loop(self.env)
        
        self.simulation_results = [] # Speichert (total_reward, terminated, episode_states)
        self.positive_examples = []  # Für den Validator: stabile/funktionale Wachstumssequenzen
        self.negative_examples = []  # Für den Validator: tumorartige/kollabierende Wachstumssequenzen

    def run_simulations(self):
        """
        Führt die angegebenen Anzahl von Simulationen aus und sammelt Ergebnisse.
        """
        logger.info(f"Führe {self.num_simulations} RL-Stresstests durch...")
        for i in range(self.num_simulations):
            total_reward, episode_states_full, terminated = self.rl_loop.run_episode()
            
            # Konvertiere die Liste der vollen NCA-Zustände in eine Liste von RGB-Tensoren (3, H, W)
            # und stacke sie zu einem einzigen Tensor (Seq_len, 3, H, W) für den Validator
            # to_rgb(s) gibt (3, H, W) zurück, da s hier (NCA_STATE_CHANNELS, H, W) ist.
            episode_rgb_sequence = torch.stack([to_rgb(s) for s in episode_states_full])

            self.simulation_results.append({
                "id": i,
                "total_reward": total_reward,
                "terminated": terminated,
                "episode_states_rgb": episode_rgb_sequence # Speichere die RGB-Sequenz
            })
            
            # Sammle Beispiele für den Validator
            # Eine hohe Belohnung und nicht-terminated -> Positivbeispiel
            # Eine niedrige Belohnung und terminated -> Negativbeispiel
            if total_reward > 5.0 and not terminated: # Beispiel-Schwelle für "gut"
                self.positive_examples.append(episode_rgb_sequence)
            elif total_reward < 0.0 or terminated: # Beispiel-Schwelle für "schlecht"
                self.negative_examples.append(episode_rgb_sequence)

            if (i + 1) % (self.num_simulations // 10 or 1) == 0:
                logger.info(f"Simulationsfortschritt: {i+1}/{self.num_simulations}. Letzte Belohnung: {total_reward:.2f}")

        logger.info("RL-Stresstests abgeschlossen.")
        self.env.close() # Schließe die Umgebung nach den Simulationen
        return self.simulation_results, self.positive_examples, self.negative_examples

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 3: Validierungsnetzwerk (RLVR)
# ----------------------------------------------------------------------------------------------------

class Validator_Net(nn.Module):
    """
    CNN- oder Transformer-basiertes Validierungsnetzwerk, das auf Sequenzen von Wachstumsbildern
    trainiert wird, um "funktional" vs. "tumorartig" zu unterscheiden.
    Für die Sequenzverarbeitung verwenden wir eine Kombination aus 2D-CNN und einem GRU/LSTM.
    """
    def __init__(self, input_channels_per_frame=3, hidden_dim=128, num_classes=2):
        """
        Initialisiert das Validierungsnetzwerk.
        :param input_channels_per_frame: Anzahl der Kanäle pro Frame (z.B. 3 für RGB).
        :param hidden_dim: Dimension der verdeckten Schichten des RNN.
        :param num_classes: Anzahl der Ausgabeklassen (z.B. 2: Funktional/Tumorartig).
        """
        super().__init__()
        # Feature-Extraktor für jeden Frame (Bild)
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels_per_frame, 32, kernel_size=3, stride=2, padding=1), # Output: 32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # Output: 16x16x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 8x8x128
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Output: 1x1x128 -> (Batch, 128)
        )
        # Anpassung der Dimension nach dem CNN-Feature-Extraktor
        self.feature_output_dim = 128 # Dies ist die Output-Größe des CNN-Teils

        # LSTM oder GRU für die Sequenzverarbeitung
        self.rnn = nn.GRU(self.feature_output_dim, hidden_dim, batch_first=True)

        # Klassifikationskopf
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Führt einen Forward-Pass des Validierungsnetzwerks durch.
        :param x: Eine Sequenz von Frames (Batch_size, Sequence_length, Channels, Height, Width).
        :return: Logits für die Klassifikation (Batch_size, num_classes).
        """
        batch_size, seq_len, C, H, W = x.size()

        # Reshape, um alle Frames als Batch zu verarbeiten
        x = x.view(batch_size * seq_len, C, H, W)

        # Extrahiere Features mit CNN
        cnn_features = self.cnn_feature_extractor(x)
        cnn_features = cnn_features.view(batch_size, seq_len, self.feature_output_dim) # Reshape zurück zur Sequenzform

        # Verarbeite Sequenz mit RNN
        rnn_output, _ = self.rnn(cnn_features)

        # Nimm den letzten Output des RNN für die Klassifikation
        final_rnn_output = rnn_output[:, -1, :] # (Batch_size, hidden_dim)

        # Klassifiziere
        logits = self.classifier(final_rnn_output)
        return logits

class Validator_Training:
    """
    Trainiert das Validierungsnetzwerk auf gesammelten Positiv- und Negativbeispielen.
    """
    def __init__(self, validator_net, pos_examples, neg_examples):
        self.net = validator_net.to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.max_trained_seq_len = 0 # Wird während der Dataset-Erstellung gesetzt

        self.dataset = self._create_dataset(pos_examples, neg_examples)
        if self.dataset: # Nur wenn Dataset nicht leer ist
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=VALIDATOR_BATCH_SIZE, shuffle=True)
        else:
            self.dataloader = None # Leere DataLoader

    def _create_dataset(self, pos_examples, neg_examples):
        """
        Erstellt ein Dataset aus positiven (Label 0: funktional) und negativen (Label 1: tumorartig) Beispielen.
        Stellt sicher, dass alle Sequenzen die gleiche Länge haben, durch Padding oder Clipping.
        """
        all_sequences = []
        all_labels = []

        max_seq_len = 0
        if pos_examples: max_seq_len = max(max_seq_len, max(s.shape[0] for s in pos_examples))
        if neg_examples: max_seq_len = max(max_seq_len, max(s.shape[0] for s in neg_examples))
        
        # Sicherstellen, dass max_seq_len nicht 0 ist, falls keine Beispiele vorhanden
        if max_seq_len == 0:
            logger.warning("Keine Beispiele für Validator-Training. Dataset ist leer.")
            return []

        self.max_trained_seq_len = max_seq_len # Speichere die maximale Sequenzlänge

        # Padding/Clipping der Sequenzen
        def pad_sequence(seq, target_len):
            if seq.shape[0] > target_len:
                return seq[:target_len] # Clipping
            else:
                # Padding mit Nullen am Ende der Sequenz
                padding_needed = target_len - seq.shape[0]
                padding_shape = (padding_needed, seq.shape[1], seq.shape[2], seq.shape[3])
                padding = torch.zeros(padding_shape, dtype=seq.dtype)
                return torch.cat((seq, padding), dim=0)

        for seq in pos_examples:
            all_sequences.append(pad_sequence(seq, max_seq_len))
            all_labels.append(0) # 0 für "funktional"
        
        for seq in neg_examples:
            all_sequences.append(pad_sequence(seq, max_seq_len))
            all_labels.append(1) # 1 für "tumorartig"

        if not all_sequences:
            return [] # Leeres Dataset, falls keine Beispiele

        return torch.utils.data.TensorDataset(
            torch.stack(all_sequences).to(DEVICE),
            torch.tensor(all_labels).long().to(DEVICE)
        )

    def train(self, epochs=VALIDATOR_EPOCHS):
        """
        Führt den Trainingsprozess für das Validierungsnetzwerk durch.
        """
        if self.dataloader is None or len(self.dataset) == 0:
            logger.warning("Keine Trainingsdaten für Validator. Training übersprungen.")
            return

        logger.info(f"Beginne Validierungsnetzwerk-Training für {epochs} Epochen...")
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, labels) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                outputs = self.net(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.dataloader)
            if (epoch + 1) % (epochs // 10 or 1) == 0:
                logger.info(f"Validator Epoche {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
        logger.info("Validierungsnetzwerk-Training abgeschlossen.")
        self.net.eval() # Setze Netzwerk in den Evaluierungsmodus

class Validator_Filter:
    """
    Filtert neue Strukturen aus dem NCA-System basierend auf dem Validierungsnetzwerk.
    """
    def __init__(self, validator_net, threshold=VALIDATOR_THRESHOLD):
        self.net = validator_net.to(DEVICE)
        self.threshold = threshold
        self.net.eval() # Immer im Evaluierungsmodus
        self._max_seq_len = 0 # Initialisiert auf 0, wird nach dem Training des Validators gesetzt

    def set_max_seq_len(self, length):
        """Setzt die maximale Sequenzlänge, die der Validator erwartet."""
        self._max_seq_len = length
        logger.info(f"Validator_Filter: max_seq_len auf {self._max_seq_len} gesetzt.")

    def check_structure(self, frames_sequence: list[torch.Tensor]):
        """
        Prüft eine gegebene Wachstumssequenz auf "Funktionalität" vs. "Tumorartigkeit".
        :param frames_sequence: Eine Liste von RGB-Zuständen (C, H, W) über die Zeit.
                                Jeder Tensor sollte (3, H, W) Dimensionen haben.
        :return: (is_functional, score), wobei score die Wahrscheinlichkeit für "funktional" ist.
        """
        if not frames_sequence:
            logger.warning("Leere Sequenz an Validator_Filter übergeben.")
            return False, 0.0

        # Bestimme die Ziel-Sequenzlänge. Nutze die vom Training übertragene Länge, sonst Fallback.
        target_seq_len = self._max_seq_len if self._max_seq_len > 0 else VALIDATOR_MAX_SEQ_LEN_FALLBACK
        if self._max_seq_len == 0:
             logger.warning(f"Validator_Filter: _max_seq_len nicht gesetzt. Verwende Fallback {target_seq_len}.")

        input_sequence_processed = []
        for frame_tensor in frames_sequence: # Jedes Element ist ein (C, H, W) Tensor
            input_sequence_processed.append(frame_tensor.to(DEVICE))
        
        # Pad/Clip the input sequence to the target_seq_len
        if len(input_sequence_processed) > target_seq_len:
            input_sequence_processed = input_sequence_processed[:target_seq_len] # Clipping
        elif len(input_sequence_processed) < target_seq_len:
            padding_needed = target_seq_len - len(input_sequence_processed)
            example_frame_shape = input_sequence_processed[0].shape # (C, H, W)
            padding_tensors = [torch.zeros(example_frame_shape, dtype=input_sequence_processed[0].dtype, device=DEVICE) for _ in range(padding_needed)]
            input_sequence_processed.extend(padding_tensors)

        sequence_tensor = torch.stack(input_sequence_processed)
        input_tensor = sequence_tensor.unsqueeze(0) # Add batch dimension: (1, Seq_len, C, H, W)

        with torch.no_grad():
            outputs = self.net(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            # Wahrscheinlichkeit für Klasse 0 ("funktional")
            functional_score = probabilities[0, 0].item() 
            
        is_functional = functional_score > self.threshold
        return is_functional, functional_score

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 4: Feedback-Loop / Selbstverbesserung
# ----------------------------------------------------------------------------------------------------

class Selection_Loop:
    """
    Verwaltet die Auswahl erfolgreicher NCA-Modelle oder initialer Bedingungen.
    """
    def __init__(self, population_size=POPULATION_SIZE):
        self.population_size = population_size
        self.best_nca_models = deque(maxlen=population_size) # Speichert die besten Modelle (z.B. nach Belohnung sortiert)

    def register_nca_model(self, nca_model, score):
        """
        Registriert ein NCA-Modell mit einem zugehörigen Score (z.B. aggregierte RL-Belohnung).
        :param nca_model: Das NCA-Modell, das bewertet wurde.
        :param score: Der Gesamtscore (z.B. durchschnittliche Belohnung aus SimRunner).
        """
        # Kopiere das Modell, um Probleme mit Referenzen zu vermeiden
        # Pass logbook and model_id from the original model if available
        logbook_to_pass = getattr(nca_model, 'logbook', None)
        # For model_id, we might want to assign a new ID to copies, or specify copy status
        # For now, let's try to keep the original ID for traceability, or append a suffix.
        original_model_id = getattr(nca_model, 'model_id', "unknown_orig")
        copied_model_id = f"{original_model_id}_sel_copy"


        model_copy = NCA_Model_3x3(
            NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS,
            logbook=logbook_to_pass, model_id=copied_model_id
        ).to(DEVICE)
        model_copy.load_state_dict(nca_model.state_dict())
        
        # Füge das Modell und seinen Score hinzu
        self.best_nca_models.append({'model': model_copy, 'score': score})
        # Sortiere nach Score (höher ist besser)
        self.best_nca_models = deque(sorted(self.best_nca_models, key=lambda x: x['score'], reverse=True)[:self.population_size])
        
        # Corrected f-string syntax for the list comprehension part
        top_scores_str = ", ".join([f"{item['score']:.2f}" for item in self.best_nca_models])
        logger.debug(f"NCA-Modell mit Score {score:.2f} registriert. Aktuelle Population Top-Scores: [{top_scores_str}]")

    def get_best_model(self):
        """Gibt das aktuell beste Modell zurück."""
        if not self.best_nca_models:
            return None
        return self.best_nca_models[0] # Gibt das Dict {'model': ..., 'score': ...} zurück

    def get_population(self):
        """Gibt die aktuelle Population der besten Modelle zurück."""
        return list(self.best_nca_models)

class Mutation_Engine:
    """
    Erzeugt Varianten erfolgreicher NCAs durch kleine Parameterabweichungen.
    """
    def __init__(self, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    def mutate(self, nca_model, logbook=None, new_model_id="mutated_N/A"):
        """
        Mutiert die Parameter eines NCA-Modells.
        :param nca_model: Das zu mutierende NCA-Modell.
        :param logbook: Optional Logbook instance to assign to the mutated model.
        :param new_model_id: Optional new model_id for the mutated model.
        :return: Ein neues, mutiertes NCA-Modell.
        """
        # The parent model's logbook is nca_model.logbook.
        # If a specific logbook is passed for the child, use it, else inherit from parent.
        logbook_for_child = logbook if logbook is not None else getattr(nca_model, 'logbook', None)

        mutated_model = NCA_Model_3x3(
            NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS,
            logbook=logbook_for_child, model_id=new_model_id
        ).to(DEVICE)
        mutated_model.load_state_dict(nca_model.state_dict()) # Kopiere die Gewichte des Elternmodells

        with torch.no_grad():
            for param in mutated_model.parameters():
                if random.random() < self.mutation_rate:
                    # Füge gaußsches Rauschen zu den Parametern hinzu
                    noise = torch.randn_like(param) * self.mutation_strength
                    param.add_(noise)
        logger.debug("NCA-Modell mutiert.")
        return mutated_model

class Evolution_Controller:
    """
    Kontrolliert den gesamten evolutionären Prozess: Selektion, Mutation, Verstärkung.
    """
    def __init__(self, num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE):
        self.num_generations = num_generations
        self.population_size = population_size
        self.selection_loop = Selection_Loop(population_size=population_size)
        self.mutation_engine = Mutation_Engine()

        self.current_nca_population = []
        self.validator_net = Validator_Net(input_channels_per_frame=3, hidden_dim=128, num_classes=2).to(DEVICE)
        self.validator_filter = Validator_Filter(self.validator_net)
        self.logbook = Logbook() # Initialisiere Logbook hier

    def initialize_population(self):
        """Erstellt die initiale Population von NCA-Modellen."""
        self.current_nca_population = []
        for i in range(self.population_size):
            model_id = f"gen0_model{i}" # Assign a unique ID for each model
            new_nca_model = NCA_Model_3x3(
                NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS,
                logbook=self.logbook, model_id=model_id
            ).to(DEVICE)
            # Optional: Initiales Wachstumstraining für jedes neue Modell, um eine Grundfähigkeit zu etablieren
            # Für eine Demo lassen wir dies aus, um die Evolution schneller zu sehen.
            self.current_nca_population.append(new_nca_model)
        logger.info(f"Initialisiere Population mit {self.population_size} NCA-Modellen (Logbook und Model ID injiziert).")

    def run_evolution(self):
        """Führt den evolutionären Prozess über mehrere Generationen aus."""
        logger.info(f"Beginne Evolution über {self.num_generations} Generationen.")
        self.initialize_population()

        for generation in range(self.num_generations):
            logger.info(f"--- Generation {generation+1}/{self.num_generations} ---")
            
            # Phase 1: Jedes NCA-Modell wird getestet
            all_positive_examples = []
            all_negative_examples = []
            generation_model_scores = [] # Speichert {model, avg_reward, pos_ex, neg_ex}

            for i, nca_model in enumerate(self.current_nca_population):
                logger.info(f"Teste Modell {i+1}/{len(self.current_nca_population)} in Stresstest-Umgebung (RL-Episoden pro Modell: {RL_EPISODES // POPULATION_SIZE})...")
                sim_runner = SimRunner(nca_model_for_sims=nca_model, num_simulations=RL_EPISODES // self.population_size)
                results, pos_ex, neg_ex = sim_runner.run_simulations()
                
                avg_reward = np.mean([res['total_reward'] for res in results]) if results else 0.0
                
                all_positive_examples.extend(pos_ex)
                all_negative_examples.extend(neg_ex)

                # Calculate XBA effectiveness metric for this model
                num_xba_reactivations = 0
                if hasattr(nca_model, 'xba_logging_data'):
                    for log_item in nca_model.xba_logging_data:
                        if isinstance(log_item, dict) and log_item.get('type') == 'reactivation':
                            num_xba_reactivations += log_item.get('count', 0) # Sum counts from each reactivation event

                generation_model_scores.append({
                    'model': nca_model,
                    'avg_reward': avg_reward,
                    'pos_ex': pos_ex,
                    'neg_ex': neg_ex,
                    'num_xba_reactivations': num_xba_reactivations # Store XBA metric
                })

                logger.info(f"Modell {i+1} ({nca_model.model_id}) abgeschlossen. Avg. Belohnung: {avg_reward:.2f}, XBA Reactivations: {num_xba_reactivations}")
                
            # Phase 2: Trainiere den Validator mit den gesammelten Beispielen dieser Generation
            if all_positive_examples and all_negative_examples:
                logger.info(f"Trainiere Validator mit {len(all_positive_examples)} Positiv- und {len(all_negative_examples)} Negativbeispielen.")
                validator_trainer = Validator_Training(self.validator_net, all_positive_examples, all_negative_examples)
                validator_trainer.train()
                # Übertrage die maximale Sequenzlänge an den Validator_Filter
                if validator_trainer.max_trained_seq_len > 0:
                    self.validator_filter.set_max_seq_len(validator_trainer.max_trained_seq_len)
            else:
                logger.warning("Nicht genug Beispiele für Validator-Training in dieser Generation. Validator wird nicht trainiert.")

            # Phase 3: Selektion und Bewertung der Modelle mit dem Validator
            current_generation_scores = []
            for item in generation_model_scores:
                nca_model = item['model']
                avg_reward = item['avg_reward']
                
                # Wähle die besten Sequenzen des Modells für die Validierung
                # `item['pos_ex']` ist eine Liste von Tensoren, wobei jeder Tensor (Seq_len, 3, H, W) ist.
                # Wir nehmen die 5 längsten/besten Sequenzen zur Validierung.
                best_sequences_for_validation = sorted(item['pos_ex'], key=lambda s: s.shape[0], reverse=True)[:5]

                total_validation_score = 0
                num_validated_sequences = 0
                if best_sequences_for_validation:
                    for seq_tensor in best_sequences_for_validation: # seq_tensor ist (Seq_len, 3, H, W)
                        # Validator_Filter.check_structure erwartet eine Liste von einzelnen Frames (3, H, W)
                        frames_list = [frame for frame in seq_tensor] # Konvertiere Tensor(S, C, H, W) zu List[Tensor(C, H, W)]
                        is_functional, val_score = self.validator_filter.check_structure(frames_list)
                        total_validation_score += val_score
                        num_validated_sequences += 1
                
                validation_avg_score = total_validation_score / num_validated_sequences if num_validated_sequences > 0 else 0.0

                # Kombiniere RL-Belohnung und Validierungs-Score für den endgültigen Score
                # Der Validierungs-Score ist von 0 bis 1. Multipliziere ihn, um seinen Einfluss zu gewichten.
                xba_effectiveness_score = item.get('num_xba_reactivations', 0)
                xba_weight = 0.05 # Weight for XBA score, tune as needed

                combined_score = avg_reward + (validation_avg_score * 10) + (xba_effectiveness_score * xba_weight)

                current_generation_scores.append(combined_score)
                # Pass the original model object (nca_model) to register_nca_model
                self.selection_loop.register_nca_model(nca_model, combined_score)

            # Logge die Zusammenfassung der Generation
            best_gen_score = max(current_generation_scores) if current_generation_scores else 0.0
            avg_gen_score = np.mean(current_generation_scores) if current_generation_scores else 0.0
            avg_xba_score_this_gen = np.mean([s.get('num_xba_reactivations', 0) for s in generation_model_scores]) if generation_model_scores else 0.0

            self.logbook.log_generation_summary(
                generation=generation+1,
                best_score=best_gen_score,
                avg_score=avg_gen_score,
                num_pos_examples=len(all_positive_examples),
                num_neg_examples=len(all_negative_examples),
                avg_xba_reactivations=avg_xba_score_this_gen
            )

            # Phase 4: Mutation und Erzeugung der nächsten Generation
            next_generation_population = []
            # Übernehme die besten Modelle direkt (Eliten-Strategie)
            # Models from get_population() are dicts {'model': model_instance, 'score': ...}
            # The model_instance within this dict is a copy made by register_nca_model,
            # which should have logbook and a copied model_id.
            elites_dicts = self.selection_loop.get_population()
            for elite_dict in elites_dicts:
                next_generation_population.append(elite_dict['model']) # Add the model instance

            # Fülle den Rest der Population mit Mutationen der besten Modelle auf
            num_elites = len(next_generation_population)
            num_mutants_to_create = self.population_size - num_elites
            
            if elites_dicts: # Ensure there are elites to choose from for mutation
                for i in range(num_mutants_to_create):
                    parent_model_dict = random.choice(elites_dicts) # Choose from the list of elite dicts
                    parent_model = parent_model_dict['model']
                    # Create a new unique ID for the mutated child
                    mutated_model_id = f"gen{generation+1}_mutant{i}_from_{parent_model.model_id}"
                    mutated_child = self.mutation_engine.mutate(
                        parent_model,
                        logbook=self.logbook,
                        new_model_id=mutated_model_id
                    )
                    next_generation_population.append(mutated_child)
            else: # Fallback if no elites (e.g. first generation if selection loop was empty)
                 # This case should ideally not happen if population_size > 0 and init_population ran
                logger.warning(f"Generation {generation+1}: No elites found to mutate from. Filling with new models.")
                for i in range(num_mutants_to_create):
                    model_id = f"gen{generation+1}_newfill{i}"
                    new_model = NCA_Model_3x3(
                        NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS,
                        logbook=self.logbook, model_id=model_id
                    ).to(DEVICE)
                    next_generation_population.append(new_model)

            self.current_nca_population = next_generation_population[:self.population_size] # Ensure correct size
            logger.info(f"Generation {generation+1} abgeschlossen. Top NCA-Score dieser Population: {self.selection_loop.get_best_model()['score']:.2f}")

        logger.info("Evolution abgeschlossen. Das beste NCA-Modell ist verfügbar.")
        return self.selection_loop.get_best_model()['model']

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 5: Visualisierung & Debugging
# ----------------------------------------------------------------------------------------------------

class Growth_Visualizer:
    """
    Zeigt die Entwicklung jeder Struktur frameweise an.
    """
    def __init__(self, grid_size=NCA_GRID_SIZE, show_xba_effects=True, xba_highlight_color=None): # xba_highlight_color currently unused
        self.grid_size = grid_size
        self.show_xba_effects = show_xba_effects
        self.xba_highlight_color = xba_highlight_color # Placeholder for now
        self.fig, self.ax = plt.subplots(figsize=(grid_size/10, grid_size/10))
        self.im = self.ax.imshow(np.zeros((grid_size, grid_size, 3)), cmap='viridis')
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.animation = None

    def animate_growth(self, nca_model, num_steps=NCA_STEPS_PER_GROWTH, title="NCA Growth",
                         xba_enabled_viz=False, xba_check_interval_viz=10,
                         epsilon_viz=1e-6, std_threshold_viz=1e-4,
                         reactivation_method_viz='reinitialize'):
        """
        Erzeugt eine Animation des NCA-Wachstumsprozesses.
        :param nca_model: Das NCA-Modell zur Simulation.
        :param num_steps: Anzahl der Simulationsschritte.
        :param title: Titel der Animation.
        :param xba_enabled_viz: Ob XBA während der Visualisierung aktiv sein soll.
        :param xba_check_interval_viz: Intervall für XBA-Prüfungen in der Visualisierung.
        :param epsilon_viz: Epsilon für XBA in der Visualisierung.
        :param std_threshold_viz: Std_threshold für XBA in der Visualisierung.
        :param reactivation_method_viz: Reactivation-Methode für XBA in der Visualisierung.
        """
        logger.info(f"Starte Visualisierung des NCA-Wachstums '{title}' (XBA Viz: {xba_enabled_viz})...")
        states_sequence = []
        xba_messages_sequence = [] # To store XBA messages for each frame

        original_xba_log_len = len(nca_model.xba_logging_data) if hasattr(nca_model, 'xba_logging_data') else 0

        state = initial_seed_generator(self.grid_size, NCA_STATE_CHANNELS, batch_size=1, seed_type='center_pixel')
        states_sequence.append(to_rgb(state).squeeze(0).cpu().numpy().transpose(1, 2, 0)) # Initialzustand
        xba_messages_sequence.append("") # No XBA action at step 0

        # Ensure model is in eval mode for visualization, but XBA might need std_output.
        # The change in NCA_Model_3x3.forward to always calculate std_output helps here.
        nca_model.eval()

        with torch.no_grad(): # Visualization should not track gradients
            for i in range(num_steps):
                current_xba_message_parts = []

                # Store length of XBA logs before potential XBA intervention for this step
                # This helps in fetching only new logs generated in this step
                log_len_before_this_step_xba = len(nca_model.xba_logging_data)

                # Call nca_step_function - XBA logic within nca_step_function will run if xba_enabled_viz is True
                # This is important as it calls model.forward() which populates std_outputs
                state = nca_step_function(
                    state, nca_model,
                    current_step=i + 1,
                    xba_enabled=xba_enabled_viz,
                    xba_check_interval=xba_check_interval_viz,
                    epsilon=epsilon_viz,
                    std_threshold=std_threshold_viz,
                    reactivation_method=reactivation_method_viz
                )

                # Collect XBA messages if XBA was active and did something *in this step*
                # The XBA logic (identify/reactivate) is now inside nca_step_function.
                # We need to retrieve messages logged by those calls for the *current* step.
                if xba_enabled_viz and ((i + 1) % xba_check_interval_viz == 0):
                    new_logs = nca_model.xba_logging_data[log_len_before_this_step_xba:]
                    for log_entry in new_logs:
                        if log_entry.get('type') == 'identification':
                            current_xba_message_parts.append(
                                f"XBA ID: {log_entry['layer']} ({len(log_entry['neurons'])} neurons)"
                            )
                        elif log_entry.get('type') == 'reactivation':
                             current_xba_message_parts.append(
                                f"XBA Act: {log_entry['layer']} ({log_entry['count']} by {log_entry['method']})"
                            )

                xba_messages_sequence.append("; ".join(current_xba_message_parts) if current_xba_message_parts else "")
                states_sequence.append(to_rgb(state).squeeze(0).cpu().numpy().transpose(1, 2, 0))

                if i % (num_steps // 10 or 1) == 0:
                    logger.debug(f"Visualisierung Fortschritt: {i+1}/{num_steps}")

        # Clean up any XBA logs generated *only* for this visualization run, if desired
        # For simplicity, we are not doing that here. If xba_logging_data grows too large,
        # one might consider passing a temporary list or filtering by a run_id.

        def update(frame_info):
            frame_idx_val, frame_data, xba_msg = frame_info # Unpack
            self.im.set_data(frame_data)
            full_title = f"{title}\nStep: {frame_idx_val}"
            if self.show_xba_effects and xba_msg:
                full_title += f"\n{xba_msg}"
            self.ax.set_title(full_title)
            return self.im,

        # Create frames by zipping states, messages, and step numbers
        # Step numbers for display should start from 0 or 1 as per convention.
        # range(len(states_sequence)) makes step_idx_val go from 0 to num_steps.
        animation_frames = zip(range(len(states_sequence)), states_sequence, xba_messages_sequence)

        self.animation = FuncAnimation(self.fig, update,
                                       frames=animation_frames,
                                       interval=1000/VISUALIZATION_FPS, blit=True)
        plt.show(block=False)
        logger.info(f"Visualisierung abgeschlossen. Titel: {title}")

    def save_animation(self, filename="nca_growth.gif"):
        """Speichert die generierte Animation als GIF."""
        if self.animation:
            logger.info(f"Speichere Animation als {filename}...")
            # Use 'pillow' writer for GIFs, requires Pillow library.
            try:
                self.animation.save(filename, writer='pillow', fps=VISUALIZATION_FPS)
                logger.info(f"Animation gespeichert als {filename}")
            except Exception as e:
                logger.error(f"Fehler beim Speichern der Animation: {e}")
        else:
            logger.warning("Keine Animation generiert. 'animate_growth' zuerst aufrufen.")

class Tumor_Detector_Debugger:
    """
    Hebt "abnormales" Verhalten hervor (z.B. ungebremstes Wachstum, Zellhaufen, keine Struktur).
    Dies kann manuell oder mit einem einfachen Regelwerk geschehen.
    Für eine fortgeschrittene Debugging-Funktion würde man hier die Fähigkeiten des Validator_Net nutzen.
    """
    def __init__(self, validator_filter):
        self.validator_filter = validator_filter

    def analyze_episode(self, episode_id, episode_states_full, total_reward, terminated):
        """
        Analysiert eine einzelne Episode und markiert potenzielles "Tumor"-Verhalten.
        :param episode_id: ID der Episode.
        :param episode_states_full: Liste der NCA-Zustände (NCA_STATE_CHANNELS, H, W) über die Zeit.
        :param total_reward: Gesamte Belohnung der Episode.
        :param terminated: Ob die Episode terminiert wurde.
        """
        logger.info(f"Debugging Episode {episode_id}: Total Reward={total_reward:.2f}, Terminated={terminated}")
        
        # Konvertiere Zustände zu RGB für den Validator
        # episode_states_full ist eine List[Tensor(NCA_STATE_CHANNELS, H, W)]
        # to_rgb(s) für s=(C,H,W) gibt (3,H,W) zurück (kein Batch-Dim)
        episode_rgb_states_list = [to_rgb(s) for s in episode_states_full] 
        
        is_functional, score = self.validator_filter.check_structure(episode_rgb_states_list)
        
        if is_functional:
            logger.info(f"  -> Validator-Score: {score:.4f} (Als funktional eingestuft)")
        else:
            logger.warning(f"  -> Validator-Score: {score:.4f} (Potenziell tumorartig/Dysfunktional!)")
            # Hier könnten zusätzliche Debug-Informationen ausgegeben werden, z.B.
            # - Zellzahländerung über Zeit
            # - Kompaktheitsmetriken
            # - Visualisierung der kritischen Frames

        # Einfache Regeln für Debugging ohne Validator:
        # Hier muss der letzte Zustand des Typs (NCA_STATE_CHANNELS, H, W) wieder in (1, NCA_STATE_CHANNELS, H, W) gebracht werden
        # für get_living_mask, da es batch-fähig ist.
        final_living_cells = get_living_mask(episode_states_full[-1].unsqueeze(0)).sum().item()
        initial_living_cells = get_living_mask(episode_states_full[0].unsqueeze(0)).sum().item()
        
        max_cells = NCA_GRID_SIZE * NCA_GRID_SIZE
        if final_living_cells > max_cells * 0.8:
            logger.warning(f"  -> Warnung: Sehr hohes Zellwachstum ({final_living_cells} Zellen). Möglicherweise unkontrolliert.")
        if final_living_cells < max_cells * 0.05 and not terminated:
            logger.warning(f"  -> Warnung: Wenig lebende Zellen ({final_living_cells}) obwohl nicht terminiert. Fast tot.")

class Logbook:
    """
    Protokolliert jede Generation, Scores und Auswahlgründe.
    Verwendet das eingebaute Logging-Modul.
    """
    def __init__(self):
        # Verwenden Sie den bereits initialisierten Logger für dieses Modul
        self.logger = logging.getLogger(__name__) 

    def log_generation_summary(self, generation, best_score, avg_score, num_pos_examples, num_neg_examples, avg_xba_reactivations=0.0):
        """Protokolliert eine Zusammenfassung der Generation."""
        self.logger.info(f"--- Generationszusammenfassung {generation} ---")
        self.logger.info(f"  Bester Modell-Score: {best_score:.4f}")
        self.logger.info(f"  Durchschnittlicher Population-Score: {avg_score:.4f}")
        self.logger.info(f"  Gesammelte Positivbeispiele für Validator: {num_pos_examples}")
        self.logger.info(f"  Gesammelte Negativbeispiele für Validator: {num_neg_examples}")
        self.logger.info(f"  Durchschnittliche XBA-Reaktivierungen: {avg_xba_reactivations:.2f}")
        self.logger.info("----------------------------------")

    def log_model_selection(self, model_id, score, is_selected):
        """Protokolliert die Auswahl eines Modells."""
        status = "Ausgewählt" if is_selected else "Nicht ausgewählt"
        self.logger.debug(f"Modell {model_id}: Score {score:.4f} - Status: {status}")

    def log_xba_identification(self, timestamp, model_id, layer_name, neuron_indices, details=""):
        """Protokolliert die Identifizierung toter Neuronen durch XBA."""
        # Ensure neuron_indices is a list for consistent logging
        if isinstance(neuron_indices, torch.Tensor):
            neuron_indices_list = neuron_indices.tolist()
        elif isinstance(neuron_indices, np.ndarray):
            neuron_indices_list = neuron_indices.tolist()
        else:
            neuron_indices_list = neuron_indices # Assuming it's already a list or similar

        log_message = (
            f"[XBA ID] Timestamp: {timestamp} | Model ID: {model_id} | Layer: {layer_name} | "
            f"Identified Neurons: {neuron_indices_list} | Details: {details}"
        )
        self.logger.info(log_message)

    def log_xba_action(self, timestamp, model_id, layer_name, neuron_indices, action_type, details=""):
        """Protokolliert eine XBA-Reaktivierungsaktion."""
        # Ensure neuron_indices is a list for consistent logging
        if isinstance(neuron_indices, torch.Tensor):
            neuron_indices_list = neuron_indices.tolist()
        elif isinstance(neuron_indices, np.ndarray):
            neuron_indices_list = neuron_indices.tolist()
        else:
            neuron_indices_list = neuron_indices # Assuming it's already a list or similar

        log_message = (
            f"[XBA Action] Timestamp: {timestamp} | Model ID: {model_id} | Layer: {layer_name} | "
            f"Neurons: {neuron_indices_list} | Action: {action_type} | Details: {details}"
        )
        self.logger.info(log_message)


# ----------------------------------------------------------------------------------------------------
# Haupt-Ausführungsscript (Pipeline)
# ----------------------------------------------------------------------------------------------------

def main():
    """
    Hauptfunktion, die den gesamten NCA-RLVR-Prozess orchestriert.
    """
    logger.info("Starte Projekt: Validiertes Wachstum mit Neural Cellular Automata (NCA-RLVR)")
    
    # 1. Initialisiere den Evolution Controller
    # Dieser Controller wird alle anderen Module (NCA-Modell, RL-Umgebung, Validator) orchestrieren.
    evolution_controller = Evolution_Controller(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE)

    # Starte den evolutionären Prozess
    best_nca_model_overall = evolution_controller.run_evolution()

    if best_nca_model_overall:
        logger.info("\n--- Evolution abgeschlossen ---")
        logger.info(f"Bestes NCA-Modell gefunden. Parameter des besten Modells: {sum(p.numel() for p in best_nca_model_overall.parameters())} Parameter")
        
        # 2. Visualisiere das Wachstum des besten Modells
        visualizer = Growth_Visualizer()
        visualizer.animate_growth(best_nca_model_overall, num_steps=NCA_STEPS_PER_GROWTH * 2, title="Bestes NCA-Wachstum")
        visualizer.save_animation("best_nca_growth.gif")

        # 3. Führe eine abschließende Stresstest-Simulation mit dem besten Modell durch
        final_stress_env = StressTest_Env(nca_model_for_env=best_nca_model_overall, render_mode='human')
        final_rl_loop = Reinforcement_Loop(final_stress_env)
        
        logger.info("\nFühre abschließenden Stresstest mit dem besten Modell durch (visuell)...")
        # Hier werden die vollen NCA-Zustände zurückgegeben, da der Debugger sie erwartet
        final_reward, final_states_full, final_terminated = final_rl_loop.run_episode(max_steps=NCA_STEPS_PER_GROWTH * 2)
        logger.info(f"Abschließender Stresstest Belohnung: {final_reward:.2f}, Terminiert: {final_terminated}")
        
        # Debugging des finalen Verhaltens mit dem bestehenden Debugger
        debugger = Tumor_Detector_Debugger(evolution_controller.validator_filter)
        debugger.analyze_episode(0, final_states_full, final_reward, final_terminated)

        final_stress_env.close()
        plt.show(block=True) # Stelle sicher, dass das Matplotlib-Fenster offen bleibt
    else:
        logger.error("Kein bestes NCA-Modell gefunden. Evolution möglicherweise fehlgeschlagen.")

if __name__ == "__main__":
    # Stelle sicher, dass das Geräte-Flag richtig gesetzt ist
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # Beschleunigt Convs, wenn Input-Größe konstant ist

    # main() # Comment out main execution for tests

    # --- XBA Test Suite ---
    test_logbook = Logbook() # Global logbook for tests

    def create_test_model(logbook_instance=None, model_id="test_model_default"):
        # Use smaller channel numbers for faster tests if desired, but stick to config for now
        model = NCA_Model_3x3(
            NCA_STATE_CHANNELS,
            NCA_HIDDEN_CHANNELS,
            NCA_STATE_CHANNELS,
            logbook=logbook_instance,
            model_id=model_id
        ).to(DEVICE)
        model.eval() # Default to eval mode for tests unless training specific parts
        return model

    def test_identify_dead_neurons():
        print("\nRunning test_identify_dead_neurons...")
        model = create_test_model(logbook_instance=test_logbook, model_id="identify_test")
        epsilon = 0.01
        std_threshold = 0.01

        # All healthy
        model.mean_abs_grad_conv1.fill_(0.1)
        model.std_output_conv1.fill_(0.1)
        model.mean_abs_grad_conv2.fill_(0.1)
        model.std_output_conv2.fill_(0.1)
        dead1, dead2 = model.identify_dead_neurons(epsilon, std_threshold)
        assert dead1.numel() == 0, "Test Case 1.1 Failed: All healthy conv1"
        assert dead2.numel() == 0, "Test Case 1.2 Failed: All healthy conv2"
        print("  Test Case 1 (All Healthy): Passed")

        # Some dead in conv1
        model.mean_abs_grad_conv1[0] = 0.001
        model.std_output_conv1[0] = 0.001
        model.mean_abs_grad_conv1[1] = 0.002
        model.std_output_conv1[1] = 0.002
        dead1, dead2 = model.identify_dead_neurons(epsilon, std_threshold)
        assert dead1.numel() == 2 and 0 in dead1 and 1 in dead1, f"Test Case 2.1 Failed: Some dead conv1. Got: {dead1.tolist()}"
        assert dead2.numel() == 0, "Test Case 2.2 Failed: All healthy conv2 (error in conv1 test)"
        print("  Test Case 2 (Some Dead Conv1): Passed")

        # Check logs (simple check for entries in model's own log)
        assert any("identification" in entry.get('type', "") for entry in model.xba_logging_data if isinstance(entry, dict)), "Test Case 2.3 Failed: identification log missing"
        model.xba_logging_data.clear() # Clear for next test part

        # Low MAG, high StdOut (healthy)
        model.mean_abs_grad_conv1[2] = 0.001
        model.std_output_conv1[2] = 0.1 # Healthy std_output
        dead1, dead2 = model.identify_dead_neurons(epsilon, std_threshold)
        assert 2 not in dead1, "Test Case 3.1 Failed: Low MAG, high StdOut should be healthy"
        print("  Test Case 3 (Low MAG, High StdOut): Passed")

        # High MAG, low StdOut (healthy)
        model.mean_abs_grad_conv1[3] = 0.1
        model.std_output_conv1[3] = 0.001 # Healthy MAG
        dead1, dead2 = model.identify_dead_neurons(epsilon, std_threshold)
        assert 3 not in dead1, "Test Case 4.1 Failed: High MAG, low StdOut should be healthy"
        print("  Test Case 4 (High MAG, Low StdOut): Passed")
        print("test_identify_dead_neurons: All tests passed.")

    def test_reactivate_neurons_reinitialize():
        print("\nRunning test_reactivate_neurons_reinitialize...")
        model = create_test_model(logbook_instance=test_logbook, model_id="reinit_test")
        dead_indices_c1 = torch.tensor([0, 1], device=DEVICE).long()
        dead_indices_c2 = torch.tensor([2, 3], device=DEVICE).long()

        initial_weights_c1_dead0 = model.conv1.weight.data[0].clone()
        initial_weights_c2_dead2 = model.conv2.weight.data[2].clone()

        model.reactivate_neurons(dead_indices_c1, dead_indices_c2, 'reinitialize')

        assert not torch.equal(model.conv1.weight.data[0], initial_weights_c1_dead0), "Test Case 1.1 Failed: Conv1 neuron 0 not reinitialized"
        assert not torch.equal(model.conv2.weight.data[2], initial_weights_c2_dead2), "Test Case 1.2 Failed: Conv2 neuron 2 not reinitialized"
        # Check if bias (if exists) was zeroed
        if model.conv1.bias is not None:
            assert torch.all(model.conv1.bias.data[dead_indices_c1] == 0), "Test Case 1.3 Failed: Conv1 bias not zeroed"
        print("  Test Case 1 (Weights Changed & Bias Zeroed): Passed")

        # Check logs
        react_logs = [log for log in model.xba_logging_data if isinstance(log, dict) and log.get('type') == 'reactivation']
        assert len(react_logs) >= 1, "Test Case 1.4 Failed: Reactivation log missing/incomplete"
        assert any(log['layer'] == 'conv1' and log['method'] == 'reinitialize' for log in react_logs), "Test Case 1.5 Failed: Conv1 reinit log incorrect"
        assert any(log['layer'] == 'conv2' and log['method'] == 'reinitialize' for log in react_logs), "Test Case 1.6 Failed: Conv2 reinit log incorrect"
        print("  Test Case 2 (Logging): Passed")
        print("test_reactivate_neurons_reinitialize: All tests passed.")

    def test_reactivate_neurons_inject_noise():
        print("\nRunning test_reactivate_neurons_inject_noise...")
        model = create_test_model(logbook_instance=test_logbook, model_id="noise_test")
        dead_indices_c1 = torch.tensor([0], device=DEVICE).long()
        dead_indices_c2 = torch.tensor([], device=DEVICE).long() # No dead neurons for conv2 this time

        initial_weights_c1_dead0 = model.conv1.weight.data[0].clone()
        # Sum of weights before adding noise (as a simple check)
        sum_before = initial_weights_c1_dead0.sum()

        model.reactivate_neurons(dead_indices_c1, dead_indices_c2, 'inject_noise')

        sum_after = model.conv1.weight.data[0].sum()
        # Weights should have changed, but not drastically (noise is small)
        assert not torch.equal(model.conv1.weight.data[0], initial_weights_c1_dead0), "Test Case 1.1 Failed: Conv1 neuron 0 weights did not change"
        assert not torch.isclose(sum_before, sum_after, atol=1e-5), "Test Case 1.2 Failed: Sum of weights likely unchanged, noise might be zero."
        print("  Test Case 1 (Weights Changed by Noise): Passed")

        # Check logs
        react_logs = [log for log in model.xba_logging_data if isinstance(log, dict) and log.get('type') == 'reactivation']
        assert len(react_logs) >= 1, "Test Case 2.1 Failed: Reactivation log missing"
        assert any(log['layer'] == 'conv1' and log['method'] == 'inject_noise' for log in react_logs), "Test Case 2.2 Failed: Conv1 inject_noise log incorrect"
        print("  Test Case 2 (Logging): Passed")
        print("test_reactivate_neurons_inject_noise: All tests passed.")

    def test_xba_hooks_and_stats_collection():
        print("\nRunning test_xba_hooks_and_stats_collection...")
        # Use a model configured for training to ensure hooks are active and grads are computed
        model = create_test_model(logbook_instance=test_logbook, model_id="hooks_test")
        model.train() # Set to train mode for gradients

        # Dummy input and loss calculation for backward pass
        # Use smaller grid for test speed if NCA_GRID_SIZE is large
        test_grid_size = 16
        dummy_state = torch.randn(2, NCA_STATE_CHANNELS, test_grid_size, test_grid_size, device=DEVICE, requires_grad=True)

        # Perform forward pass
        output_delta = model(dummy_state)

        # Assert std_outputs are populated (not all zeros)
        # These are calculated during forward pass if model.train() or if requires_grad is true on input
        assert model.std_output_conv1.abs().sum() > 1e-9, "Test Case 1.1 Failed: std_output_conv1 not populated"
        assert model.std_output_conv2.abs().sum() > 1e-9, "Test Case 1.2 Failed: std_output_conv2 not populated"
        print("  Test Case 1 (StdDev Output Collection): Passed")

        # Create a dummy loss and perform backward pass
        loss = output_delta.sum()
        loss.backward()

        # Assert mean_abs_grads are populated (not all zeros)
        assert model.mean_abs_grad_conv1.abs().sum() > 1e-9, "Test Case 2.1 Failed: mean_abs_grad_conv1 not populated by hook"
        assert model.mean_abs_grad_conv2.abs().sum() > 1e-9, "Test Case 2.2 Failed: mean_abs_grad_conv2 not populated by hook"
        print("  Test Case 2 (Gradient Hook Collection): Passed")

        model.remove_hooks() # Test hook removal
        assert model._conv1_hook_handle is None and model._conv2_hook_handle is None, "Test Case 3.1 Failed: Hooks not removed"
        print("  Test Case 3 (Hook Removal): Passed")

        print("test_xba_hooks_and_stats_collection: All tests passed.")


    def test_xba_integration_in_nca_step():
        print("\nRunning test_xba_integration_in_nca_step...")
        model = create_test_model(logbook_instance=test_logbook, model_id="integration_test")
        model.train() # XBA typically runs during training

        test_grid_size = 16
        # Use a zero initial state to try and force std_output to be zero.
        # This assumes biases are zero or near zero after initialization for conv1.
        initial_state = torch.zeros(1, NCA_STATE_CHANNELS, test_grid_size, test_grid_size, device=DEVICE)

        # Setup conditions for XBA to trigger for conv1 neuron 0
        # mean_abs_grad is manually set. std_output will be calculated by model(state) call
        # inside nca_step_function. With zero input, std_output should be zero.
        model.mean_abs_grad_conv1.fill_(0.1) # Make other neurons healthy
        model.mean_abs_grad_conv1[0] = 1e-8  # This neuron's MAG is dead

        # std_output will be determined by the forward pass.
        # We expect std_output_conv1[0] to be ~0 due to zero input state.

        initial_weights_c1_dead0 = model.conv1.weight.data[0].clone()

        # Call nca_step_function where XBA should activate
        _ = nca_step_function(
            initial_state, model,
            current_step=10, # current_step % xba_check_interval == 0
            xba_enabled=True,
            xba_check_interval=10,
            epsilon=1e-7, # Ensure our setup triggers this
            std_threshold=1e-5, # Ensure our setup triggers this
            reactivation_method='reinitialize'
        )

        # Assert that reactivate_neurons was effectively called for conv1 neuron 0
        # 1. Check if weights changed
        assert not torch.equal(model.conv1.weight.data[0], initial_weights_c1_dead0), "Test Case 1.1 Failed: Weights of dead neuron did not change"

        # 2. Check model's internal XBA log for reactivation entry
        react_log_found = False
        for log_entry in model.xba_logging_data:
            if isinstance(log_entry, dict) and \
               log_entry.get('type') == 'reactivation' and \
               log_entry.get('layer') == 'conv1' and \
               0 in log_entry.get('neurons', []):
                react_log_found = True
                break
        assert react_log_found, "Test Case 1.2 Failed: Reactivation log for conv1 neuron 0 not found in model.xba_logging_data"
        print("  Test Case 1 (XBA Triggered and Neuron Reactivated): Passed")

        print("test_xba_integration_in_nca_step: All tests passed.")

    def run_all_xba_tests():
        print("--- Starting XBA Test Suite ---")
        test_identify_dead_neurons()
        test_reactivate_neurons_reinitialize()
        test_reactivate_neurons_inject_noise()
        test_xba_hooks_and_stats_collection()
        test_xba_integration_in_nca_step()
        print("\n--- XBA Test Suite Completed ---")

    # Instead of main(), run the tests directly if this script is executed
    run_all_xba_tests()
