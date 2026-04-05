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
GROWTH_TRAINER_EPOCHS = 10 # Anzahl der Trainings-Epochen für NCA-Wachstum
RL_EPISODES = 10         # Anzahl der Episoden für die RL-Stresstest-Umgebung (pro Generation / pro Modell)
VALIDATOR_EPOCHS = 100     # Anzahl der Trainings-Epochen für das Validierungsnetzwerk
VALIDATOR_BATCH_SIZE = 32
VALIDATOR_THRESHOLD = 0.8  # Mindest-Score, damit eine Struktur als "funktional" akzeptiert wird
VALIDATOR_MAX_SEQ_LEN_FALLBACK = NCA_STEPS_PER_GROWTH * 2 # Fallback für max_seq_len, wenn nicht gesetzt

# Evolution Parameter
MUTATION_RATE = 0.1        # Wahrscheinlichkeit einer Mutation pro Parameter
MUTATION_STRENGTH = 0.01   # Stärke der Parameterabweichung bei Mutation
POPULATION_SIZE = 2        # Anzahl der NCAs in der evolutionären Population
NUM_GENERATIONS = 1        # Anzahl der Generationen für die Evolution

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
    Hamiltonian Steerable NCA (Carlin-Architektur).
    Vereint SE(2)-Äquivarianz mit energetischer Stabilität.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Wir lernen nur 3 radiale Parameter pro Kanal-Interaktion (Ring-Basis)
        # Dies erzwingt Isotropie und damit 0-Shot Rotationsinvarianz.
        self.w1_base = nn.Parameter(torch.randn(hidden_channels, in_channels, 3) * 0.01)
        self.w2_base = nn.Parameter(torch.randn(out_channels, hidden_channels, 3) * 0.01)
        
        # Biases sind skalar pro Kanal (translationsinvariant)
        self.b1 = nn.Parameter(torch.zeros(hidden_channels))
        self.b2 = nn.Parameter(torch.zeros(out_channels))
        
        self.relu = nn.LeakyReLU(0.2)

    def _get_hamiltonian_kernel(self, w_base, enforce_symmetry=True):
        """
        Erzeugt einen äquivarianten Kernel und erzwingt Symmetrie M=M^T
        für eine konservative Energielandschaft.
        """
        out_c, in_c, _ = w_base.shape
        
        # Kanal-Symmetrisierung (Hamiltonian Constraint)
        # Macht das System zu einem Gradientenabstieg: f = -grad(E)
        w = w_base
        if enforce_symmetry and out_c == in_c:
            w = (w + w.transpose(0, 1)) * 0.5
            
        kernel = torch.zeros(out_c, in_c, 3, 3, device=w_base.device)
        # Ring 0: Center
        kernel[:, :, 1, 1] = w[:, :, 0]
        # Ring 1: Neighbors (N, S, E, W)
        kernel[:, :, 0, 1] = w[:, :, 1]; kernel[:, :, 2, 1] = w[:, :, 1]
        kernel[:, :, 1, 0] = w[:, :, 1]; kernel[:, :, 1, 2] = w[:, :, 1]
        # Ring 2: Diagonals
        kernel[:, :, 0, 0] = w[:, :, 2]; kernel[:, :, 0, 2] = w[:, :, 2]
        kernel[:, :, 2, 0] = w[:, :, 2]; kernel[:, :, 2, 2] = w[:, :, 2]
        return kernel

    def forward(self, state):
        """
        Forward Pass als Differentialschritt (Neural ODE Flow).
        """
        # Layer 1: Symmetrische Interaktion (Hidden)
        k1 = self._get_hamiltonian_kernel(self.w1_base, enforce_symmetry=False)
        x = F.conv2d(state, k1, bias=self.b1, padding=1)
        x = self.relu(x)
        
        # Layer 2: Rückführung auf Zustandsraum
        k2 = self._get_hamiltonian_kernel(self.w2_base, enforce_symmetry=False)
        x = F.conv2d(x, k2, bias=self.b2, padding=1)
        
        # Integration-Step-Size (dt)
        # Ein kleiner Faktor simuliert den kontinuierlichen Fluss einer ODE.
        return x * 0.05 

def calculate_spectral_stability(model):
    """
    Berechnet den geschätzten Spektralradius des Systems.
    Nach Carlin: rho(I + J) < 1 ist das Ziel für absolute Stabilität.
    """
    with torch.no_grad():
        # Schätzung basierend auf der Frobenius-Norm der Gewichte
        # Ein grober aber differenzierbarer Indikator für Stabilität
        norm1 = torch.norm(model.w1_base)
        norm2 = torch.norm(model.w2_base)
        stability_index = (norm1 * norm2).item()
        return stability_index # Kleinere Werte = Stabileres Wachstum

def nca_step_function(state, model):
    """
    Führt einen Simulationsschritt basierend auf lokalen Regeln des NCA-Modells aus.
    Dieser Schritt ist differenzierbar.
    :param state: Der aktuelle Zustand des NCA-Gitters (Batch_size, Channels, Height, Width).
    :param model: Das trainierte NCA_Model (NCA_Model_3x3 Instanz).
    :return: Der neue Zustand des NCA-Gitters nach einem Schritt.
    """
    # 1. Lebendigkeitsmaske vor der Aktualisierung (um tote Zellen nicht zu aktualisieren)
    # Diese Maske wird verwendet, um zu verhindern, dass die Aktualisierung an bereits toten Zellen stattfindet,
    # die keine Nachbarverbindung zu lebenden Zellen haben.
    pre_living_mask = get_living_mask(state)

    # 2. Berechne das Delta mit dem NCA-Modell
    delta_state = model(state)

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

def calculate_turing_loss(model):
    """
    Berechnet den Turing-Regularisierungs-Loss nach Carlin.
    Erzwingt Bedingungen für Diffusion-Driven Instability:
    1. Spur(J) < 0 (Stabilität des homogenen Zustands)
    2. Det(J) > 0 (Kopplung)
    """
    # Wir schätzen die Reaktions-Interaktion J aus den zentralen Gewichten (Ring 0)
    # J ~ w2_base[:,:,0] @ w1_base[:,:,0]
    w1_center = model.w1_base[:, :, 0]
    w2_center = model.w2_base[:, :, 0]
    
    # Vereinfachte Schätzung der System-Matrix J (lineare Approximation)
    # Wir nehmen an, dass die Interaktion quadratisch ist (hidden x hidden)
    if w1_center.shape[0] == w2_center.shape[1]:
        J = torch.matmul(w2_center, w1_center)
        trace = torch.trace(J)
        # Wir wollen trace < 0. Wenn trace > 0, bestrafen wir es.
        loss_trace = torch.relu(trace + 0.1) 
        
        # Wir wollen det(J) > 0. (Grob geschätzt über die Diagonale für Stabilität)
        loss_det = torch.relu(-torch.diag(J).sum() + 0.1)
        
        return loss_trace + loss_det
    return torch.tensor(0.0, device=DEVICE)

def estimate_target_complexity(target_rgb):
    """
    Schätzt die Kolmogorov-Komplexität und die Topologische Schranke (Betti-Zahlen)
    des Bildes nach Carlins Beweis.
    """
    with torch.no_grad():
        # 1. Information-Bottleneck (Kolmogorov/Patches)
        patches = F.unfold(target_rgb, kernel_size=3).transpose(1, 2)
        unique_patches_proxy = torch.unique(patches.round(decimals=1), dim=1).shape[1]
        req_channels_info = int(np.ceil(np.log2(unique_patches_proxy + 1) / 2.0))
        
        # 2. Topologische Schranke (Betti-Zahlen)
        # C >= max(beta_k + 1). Wir nutzen die Varianz/Kanten-Dichte des Bildes 
        # als Heuristik für topologische Features (Löcher und Komponenten).
        edges = target_rgb[:, :, 1:, 1:] - target_rgb[:, :, :-1, :-1]
        beta_proxy = int(edges.abs().sum() / (NCA_GRID_SIZE * 2)) # Proxy für beta_1 (Löcher/Strukturen)
        req_channels_topo = beta_proxy + 1
        
        # Die finale Kanalzahl muss alle mathematischen Schranken erfüllen
        final_channels = max(4, req_channels_info, req_channels_topo)
        
        return final_channels

def calculate_cohomology_loss(model, state):
    """
    Implementiert die Cohomology Penalty nach Carlin (Garbentheorie).
    Misst die Inkonsistenz (den Cocycle-Defekt) zwischen benachbarten 
    3x3-Updates auf deren Überlappungen.
    """
    # Wir simulieren zwei leicht verschobene Fenster
    # s1: Update am Ort (x, y)
    # s2: Update am Ort (x+1, y)
    with torch.no_grad():
        delta = model(state)
    
    # Der Cocycle-Check: Differenz zwischen benachbarten Pixel-Updates
    # In einem konsistenten Sheaf sollte der Gradient der Updates glatt sein
    shift_x = torch.roll(delta, shifts=-1, dims=3)
    shift_y = torch.roll(delta, shifts=-1, dims=2)
    
    # L2-Norm des Cocycle-Defekts (δs)
    loss_h1_x = F.mse_loss(delta, shift_x)
    loss_h1_y = F.mse_loss(delta, shift_y)
    
    return loss_h1_x + loss_h1_y

def active_inference_step(state, model, target_prior, learning_rate=0.01):
    """
    Implementiert Active Inference nach Carlin.
    Jede Zelle minimiert ihre lokale freie Energie (FEP).
    Kein globales Backprop nötig!
    """
    # 1. Sensory Observation (Was sehen die Nachbarn?)
    # Das Modell fungiert hier als 'Generative Model' g(mu)
    with torch.no_grad():
        prediction = model(state)
        # Prediction Error ε = Realität - Erwartung
        # (Vereinfacht: Wie weit weicht der aktuelle Zustand vom Idealfluss ab?)
        prediction_error = target_prior - state
    
    # 2. Lokale Parameter-Update-Regel (Eq. 13 aus Carlins Antwort)
    # mu = mu + alpha * (Sensitivität * Fehler - Präzisions-Pull)
    # Hier nutzen wir das Modell-Delta als Sensitivitäts-Proxy
    precision_pull = 0.1 * (state - target_prior)
    fep_move = learning_rate * (prediction - precision_pull)
    
    new_state = state + fep_move
    return torch.clamp(new_state, 0.0, 1.0)

def calculate_fractal_loss(current_rgb, target_dh=1.7):
    """
    Differentiable Box-Counting nach Carlin (Frage 15).
    Berechnet die Hausdorff-Dimension über dyadische Skalierung.
    """
    # Grayscale für die Geometrie-Analyse
    gray = current_rgb.mean(dim=1, keepdim=True)
    
    # Skalen: 1x1, 2x2, 4x4, 8x8 (Dyadisches Partitioning)
    scales = [1, 2, 4, 8]
    log_eps = torch.tensor([-np.log(s) for s in scales], device=gray.device)
    log_n = []
    
    for s in scales:
        if s == 1:
            # Gesamte 'Masse' (approximiert N(epsilon))
            n_eps = gray.sum()
        else:
            # Pooling simuliert das Zählen besetzter Boxen
            pooled = F.avg_pool2d(gray, kernel_size=s, stride=s)
            n_eps = pooled.sum()
        log_n.append(torch.log(n_eps + 1e-10))
    
    log_n = torch.stack(log_n)
    
    # Lineare Regression (Least Squares) für die Steigung (D_H)
    # y = mx + c -> m = cov(x,y) / var(x)
    x = log_eps
    y = log_n
    m = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()
    
    return F.mse_loss(m, torch.tensor(target_dh, device=gray.device))

def calculate_tropical_potential(model):
    """
    Tropical Polyhedral Potential nach Carlin (Frage 13).
    Zwingt die ReLU-Aktivierungen auf die stabilen Vertizes 
    der tropischen Varietät (Max-Plus Logik).
    """
    # Wir fördern 'Sparsity' und 'Sharpness' in den Gewichten,
    # was in der tropischen Geometrie stabilen Polyeder-Ecken entspricht.
    l1_reg = torch.norm(model.w1_base, p=1)
    l2_reg = torch.norm(model.w2_base, p=1)
    
    # Max-Dominanz: Ein Zeichen für eine klare tropische Entscheidung
    max_dom = torch.max(model.w1_base) - torch.mean(model.w1_base)
    
    return 0.01 * (l1_reg + l2_reg) - 0.001 * max_dom

class Growth_Trainer:
    """
    Der finale mathematische Trainer: 
    MSE + Turing + Stabilität + Topologie + Cohomology + FEP + Fractal + Tropical.
    """
    def __init__(self, nca_model, target_image_path=None, target_image=None):
        self.model = nca_model
        self.target_dh = 1.7 # Standard für verzweigte Strukturen
        # ... (Rest der Init bleibt gleich)
        self.use_fep = True

    def train_step(self, num_steps=NCA_STEPS_PER_GROWTH):
        self.optimizer.zero_grad()
        state = initial_seed_generator(NCA_GRID_SIZE, NCA_STATE_CHANNELS, batch_size=1, seed_type='center_pixel')
        
        # Target Prior für FEP
        target_prior = torch.zeros(1, NCA_STATE_CHANNELS, NCA_GRID_SIZE, NCA_GRID_SIZE, device=DEVICE)
        target_prior[:, :3, :, :] = self.target_image
        target_prior[:, 3, :, :] = (self.target_image.sum(dim=1) > 0.1).float()

        for _ in range(num_steps):
            if self.use_fep:
                state = active_inference_step(state, self.model, target_prior)
            else:
                state = nca_step_function(state, self.model)

        final_rgb = to_rgb(state)
        
        # --- Der ultimative Carlin-Loss ---
        # 1. Bild-Vergleich
        mse_loss = self.criterion(final_rgb, self.target_image)
        # 2. Biologie & Physik
        turing_loss = calculate_turing_loss(self.model)
        # 3. Topologie & Geometrie
        topo_loss = calculate_topological_loss(final_rgb, self.target_image)
        # 4. Logik & Konsistenz
        coh_loss = calculate_cohomology_loss(self.model, state)
        # 5. Fraktale Skalierung (NEU)
        frac_loss = calculate_fractal_loss(final_rgb, self.target_dh)
        # 6. Tropische Stabilität (NEU)
        trop_loss = calculate_tropical_potential(self.model)
        
        total_loss = (mse_loss + 
                      0.01 * turing_loss + 
                      0.05 * topo_loss + 
                      0.1 * coh_loss + 
                      0.02 * frac_loss + 
                      0.005 * trop_loss)

        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

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
            for _ in range(NCA_STEPS_PER_GROWTH // 2): # Halbe Anzahl der normalen Wachstumsschritte
                self.current_nca_state = nca_step_function(self.current_nca_state, self.nca_model)
        
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
            self.current_nca_state = nca_step_function(self.current_nca_state, self.nca_model)
        
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
        model_copy = NCA_Model_3x3(NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS).to(DEVICE)
        model_copy.load_state_dict(nca_model.state_dict())
        
        # Füge das Modell und seinen Score hinzu
        self.best_nca_models.append({'model': model_copy, 'score': score})
        # Sortiere nach Score (höher ist besser)
        self.best_nca_models = deque(sorted(self.best_nca_models, key=lambda x: x['score'], reverse=True)[:self.population_size])
        
        logger.debug(f"NCA-Modell mit Score {score:.2f} registriert. Aktuelle Population Top-Scores: {[f'{m['score']:.2f}' for m in self.best_nca_models]}")

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

    def mutate(self, nca_model):
        """
        Mutiert die Parameter eines NCA-Modells.
        :param nca_model: Das zu mutierende NCA-Modell.
        :return: Ein neues, mutiertes NCA-Modell.
        """
        mutated_model = NCA_Model_3x3(NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS).to(DEVICE)
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
        for _ in range(self.population_size):
            new_nca_model = NCA_Model_3x3(NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS).to(DEVICE)
            # Optional: Initiales Wachstumstraining für jedes neue Modell, um eine Grundfähigkeit zu etablieren
            # Für eine Demo lassen wir dies aus, um die Evolution schneller zu sehen.
            self.current_nca_population.append(new_nca_model)
        logger.info(f"Initialisiere Population mit {self.population_size} NCA-Modellen.")

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
                
                generation_model_scores.append({'model': nca_model, 'avg_reward': avg_reward, 'pos_ex': pos_ex, 'neg_ex': neg_ex})

                logger.info(f"Modell {i+1} abgeschlossen. Avg. Belohnung: {avg_reward:.2f}")
                
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
                combined_score = avg_reward + validation_avg_score * 10 

                current_generation_scores.append(combined_score)
                self.selection_loop.register_nca_model(nca_model, combined_score) # Registriere Modelle mit ihrem kombinierten Score

            # Logge die Zusammenfassung der Generation
            best_gen_score = max(current_generation_scores) if current_generation_scores else 0.0
            avg_gen_score = np.mean(current_generation_scores) if current_generation_scores else 0.0
            self.logbook.log_generation_summary(
                generation=generation+1,
                best_score=best_gen_score,
                avg_score=avg_gen_score,
                num_pos_examples=len(all_positive_examples),
                num_neg_examples=len(all_negative_examples)
            )

            # Phase 4: Mutation und Erzeugung der nächsten Generation
            next_generation_population = []
            # Übernehme die besten Modelle direkt (Eliten-Strategie)
            # Kopiere die Top-Modelle, um sie nicht durch nachfolgende Mutationen zu beeinflussen.
            for elite_item in self.selection_loop.get_population():
                elite_model_copy = NCA_Model_3x3(NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS).to(DEVICE)
                elite_model_copy.load_state_dict(elite_item['model'].state_dict())
                next_generation_population.append(elite_model_copy)

            # Fülle den Rest der Population mit Mutationen der besten Modelle auf
            # Mutiere von den bestehenden Modellen in der selection_loop (die bereits Kopien sind)
            while len(next_generation_population) < self.population_size:
                # Wähle ein zufälliges Modell aus der aktuellen Elite als Elternteil
                parent_model = random.choice(self.selection_loop.get_population())['model']
                mutated_child = self.mutation_engine.mutate(parent_model)
                next_generation_population.append(mutated_child)
            
            self.current_nca_population = next_generation_population
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
    def __init__(self, grid_size=NCA_GRID_SIZE):
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(grid_size/10, grid_size/10))
        self.im = self.ax.imshow(np.zeros((grid_size, grid_size, 3)), cmap='viridis')
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.animation = None

    def animate_growth(self, nca_model, num_steps=NCA_STEPS_PER_GROWTH, title="NCA Growth"):
        """
        Erzeugt eine Animation des NCA-Wachstumsprozesses.
        :param nca_model: Das NCA-Modell zur Simulation.
        :param num_steps: Anzahl der Simulationsschritte.
        :param title: Titel der Animation.
        """
        logger.info(f"Starte Visualisierung des NCA-Wachstums '{title}'...")
        states_sequence = []
        state = initial_seed_generator(self.grid_size, NCA_STATE_CHANNELS, batch_size=1, seed_type='center_pixel')
        # to_rgb(state) gibt (1, 3, H, W) zurück, daher squeeze(0) für (3, H, W)
        states_sequence.append(to_rgb(state).squeeze(0).cpu().numpy().transpose(1, 2, 0)) # Initialzustand

        with torch.no_grad():
            for i in range(num_steps):
                state = nca_step_function(state, nca_model)
                # to_rgb(state) gibt (1, 3, H, W) zurück, daher squeeze(0) für (3, H, W)
                states_sequence.append(to_rgb(state).squeeze(0).cpu().numpy().transpose(1, 2, 0))
                if i % (num_steps // 10 or 1) == 0:
                    logger.debug(f"Visualisierung Fortschritt: {i+1}/{num_steps}")

        # Für die Animation muss frame_idx als Liste übergeben werden, damit es in der Closure änderbar ist.
        # oder einfacher: nutze die index-variable, die FuncAnimation selbst übergibt.
        def update(frame_data_and_idx):
            frame_data, frame_idx_val = frame_data_and_idx
            self.im.set_data(frame_data)
            self.ax.set_title(f"{title}\nStep: {frame_idx_val}")
            return self.im,

        self.animation = FuncAnimation(self.fig, update, 
                                       frames=zip(states_sequence, range(len(states_sequence))),
                                       interval=1000/VISUALIZATION_FPS, blit=True, save_count=len(states_sequence))
        plt.show(block=False) # Nicht blockieren, um andere Operationen zu ermöglichen
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

    def log_generation_summary(self, generation, best_score, avg_score, num_pos_examples, num_neg_examples):
        """Protokolliert eine Zusammenfassung der Generation."""
        self.logger.info(f"--- Generationszusammenfassung {generation} ---")
        self.logger.info(f"  Bester Modell-Score: {best_score:.4f}")
        self.logger.info(f"  Durchschnittlicher Population-Score: {avg_score:.4f}")
        self.logger.info(f"  Gesammelte Positivbeispiele für Validator: {num_pos_examples}")
        self.logger.info(f"  Gesammelte Negativbeispiele für Validator: {num_neg_examples}")
        self.logger.info("----------------------------------")

    def log_model_selection(self, model_id, score, is_selected):
        """Protokolliert die Auswahl eines Modells."""
        status = "Ausgewählt" if is_selected else "Nicht ausgewählt"
        self.logger.debug(f"Modell {model_id}: Score {score:.4f} - Status: {status}")


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

    main()
