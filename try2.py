
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
    rgb = state_tensor[..., :3, :, :]
    rgb = F.sigmoid(rgb)
    return rgb

def get_living_mask(state_tensor):
    """
    Erzeugt eine Maske, die anzeigt, welche Zellen "lebendig" sind.
    Dies wird basierend auf dem Alpha-Kanal (dem 4. Kanal) bestimmt.
    Eine Zelle gilt als lebendig, wenn ihr Alpha-Wert einen Schwellenwert überschreitet.
    Verwendet Max-Pooling, um auch benachbarte Zellen mit einzubeziehen und Rauschen zu reduzieren.
    """
    alpha = state_tensor[..., 3:4, :, :]
    return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 1: Basissystem für Neural Cellular Automata (NCA)
# ----------------------------------------------------------------------------------------------------

class NCA_Model_3x3(nn.Module):
    """
    Definiert das differenzierbare CNN-Modell für einen Zellulären Automaten,
    das eine 3x3 Nachbarschaft direkt über 3x3 Convolution-Schichten verarbeitet.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initialisiert das NCA-Modell.
        :param in_channels: Anzahl der Eingangskanäle (NCA_STATE_CHANNELS).
        :param hidden_channels: Anzahl der Kanäle in den verdeckten Schichten.
        :param out_channels: Anzahl der Ausgangskanäle (NCA_STATE_CHANNELS).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, state):
        """
        Führt einen Forward-Pass des NCA-Modells durch.
        :param state: Der gesamte NCA-Zustand des Gitters (Batch_size, Channels, Height, Width).
        :return: Das berechnete Delta für die Zustandsaktualisierung.
        """
        x = self.conv1(state)
        x = self.relu(x)
        x = self.conv2(x)
        return x

def nca_step_function(state, model):
    """
    Führt einen Simulationsschritt basierend auf lokalen Regeln des NCA-Modells aus.
    Dieser Schritt ist differenzierbar und erlaubt Wachstum in neue Bereiche.
    :param state: Der aktuelle Zustand des NCA-Gitters (Batch_size, Channels, Height, Width).
    :param model: Das trainierte NCA_Model_3x3.
    :return: Der neue Zustand des NCA-Gitters nach einem Schritt.
    """
    # 1. Berechne das Delta mit dem NCA-Modell für alle Zellen
    delta_state = model(state)

    # 2. Zufällige Aktualisierungsmaske (Stochastic Update) für Selbstorganisation
    update_mask = (torch.rand(state.shape[0], 1, state.shape[2], state.shape[3], device=DEVICE) < NCA_CELL_FIRE_RATE).float()
    
    # 3. Wende das Delta auf den Zustand an, aber nur für die zufällig ausgewählten Zellen.
    state = state + delta_state * update_mask

    # 4. Post-Living-Maske: Zellen sterben, wenn ihr Alpha-Wert unter den Schwellenwert fällt.
    # Dies ist der entscheidende Schritt, der Wachstum in neue Bereiche erlaubt, aber auch das Sterben schwacher Zellen erzwingt.
    post_living_mask = get_living_mask(state)
    state = state * post_living_mask.float()

    # 5. Stelle sicher, dass Kanalwerte im gültigen Bereich bleiben.
    state[:, :3, :, :] = torch.clamp(state[:, :3, :, :], 0.0, 1.0) # RGB
    state[:, 3:4, :, :] = torch.clamp(state[:, 3:4, :, :], 0.0, 1.0) # Alpha/Life

    return state

def initial_seed_generator(grid_size, num_channels, batch_size=1, seed_type='center_pixel'):
    """
    Generiert die anfängliche Startkonfiguration für das NCA-Gitter.
    :param grid_size: Die Größe des quadratischen Gitters.
    :param num_channels: Die Anzahl der Kanäle im NCA-Zustand.
    :param batch_size: Anzahl der gleichzeitigen NCA-Simulationen.
    :param seed_type: Art des Start-Seeds ('center_pixel', 'random_noise', 'predefined_shape').
    :return: Ein Tensor des initialen Zustands (Batch_size, Channels, Height, Width).
    """
    initial_state = torch.zeros(batch_size, num_channels, grid_size, grid_size, device=DEVICE)

    if seed_type == 'center_pixel':
        center_x, center_y = grid_size // 2, grid_size // 2
        initial_state[:, :3, center_y, center_x] = 1.0  # RGB auf 1 (Weiß)
        initial_state[:, 3, center_y, center_x] = 1.0  # Alpha/Life auf 1 (Voll lebendig)
    elif seed_type == 'random_noise':
        initial_state = torch.rand(batch_size, num_channels, grid_size, grid_size, device=DEVICE) * 0.1
        initial_state[:, 3, :, :] = (initial_state[:, 3, :, :] > 0.05).float()
    elif seed_type == 'predefined_shape':
        box_size = grid_size // 8
        start_x = (grid_size - box_size) // 2
        end_x = start_x + box_size
        start_y = (grid_size - box_size) // 2
        end_y = start_y + box_size
        initial_state[:, :3, start_y:end_y, start_x:end_x] = 0.5
        initial_state[:, 3, start_y:end_y, start_x:end_x] = 1.0
    else:
        raise ValueError(f"Unbekannter seed_type: {seed_type}")

    return initial_state

class Growth_Trainer:
    """
    Trainiert das NCA-Modell so, dass eine Zielstruktur erreicht wird.
    """
    def __init__(self, nca_model, target_image_path=None, target_image=None):
        self.model = nca_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=GROWTH_TRAINER_LR)
        self.criterion = nn.MSELoss()

        if target_image_path:
            try:
                from PIL import Image
                img = Image.open(target_image_path).convert('RGB')
                img = img.resize((NCA_GRID_SIZE, NCA_GRID_SIZE), Image.Resampling.LANCZOS)
                self.target_image = torch.tensor(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                logger.info(f"Zielbild '{target_image_path}' geladen und auf {NCA_GRID_SIZE}x{NCA_GRID_SIZE} skaliert.")
            except ImportError:
                logger.error("PIL (Pillow) ist nicht installiert. Kann keine Bilder laden. Bitte 'pip install Pillow' ausführen.")
                self.target_image = torch.ones(1, 3, NCA_GRID_SIZE, NCA_GRID_SIZE, device=DEVICE) * 0.5
            except FileNotFoundError:
                logger.error(f"Zielbild '{target_image_path}' nicht gefunden. Verwende Dummy-Ziel.")
                self.target_image = torch.ones(1, 3, NCA_GRID_SIZE, NCA_GRID_SIZE, device=DEVICE) * 0.5
        elif target_image is not None:
            self.target_image = target_image.to(DEVICE)
        else:
            self.target_image = torch.zeros(1, 3, NCA_GRID_SIZE, NCA_GRID_SIZE, device=DEVICE)
            sq_size = NCA_GRID_SIZE // 4
            sq_start = NCA_GRID_SIZE // 2 - sq_size // 2
            sq_end = sq_start + sq_size
            self.target_image[:, :, sq_start:sq_end, sq_start:sq_end] = 1.0
            logger.info("Kein Zielbild oder Pfad angegeben. Verwende Standard-Quadrat als Ziel.")

        if self.target_image.shape[1:] != (3, NCA_GRID_SIZE, NCA_GRID_SIZE):
            raise ValueError(f"Zielbild muss (1, 3, {NCA_GRID_SIZE}, {NCA_GRID_SIZE}) sein. Aktuell: {self.target_image.shape}")

    def train_step(self, num_steps=NCA_STEPS_PER_GROWTH):
        self.optimizer.zero_grad()
        state = initial_seed_generator(NCA_GRID_SIZE, NCA_STATE_CHANNELS, batch_size=1, seed_type='center_pixel')
        for _ in range(num_steps):
            state = nca_step_function(state, self.model)
        final_rgb = to_rgb(state)
        loss = self.criterion(final_rgb, self.target_image)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs=GROWTH_TRAINER_EPOCHS):
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
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': VISUALIZATION_FPS}

    def __init__(self, nca_model_for_env, render_mode=None):
        super().__init__()
        self.nca_model = nca_model_for_env
        self.grid_size = NCA_GRID_SIZE
        self.state_channels = NCA_STATE_CHANNELS
        self.current_nca_state = None
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.grid_size, self.grid_size), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.fig, self.ax, self.im = None, None, None

    def _get_obs(self):
        return to_rgb(self.current_nca_state).squeeze(0).cpu().numpy()

    def _get_info(self):
        return {"living_cells": get_living_mask(self.current_nca_state).sum().item()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_nca_state = initial_seed_generator(self.grid_size, self.state_channels, batch_size=1, seed_type='center_pixel')
        with torch.no_grad():
            for _ in range(NCA_STEPS_PER_GROWTH // 2):
                self.current_nca_state = nca_step_function(self.current_nca_state, self.nca_model)
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        previous_living_cells = get_living_mask(self.current_nca_state).sum().item()

        if action == 1:  # Schaden zufügen
            damage_size = random.randint(self.grid_size // 10, self.grid_size // 5)
            start_x = random.randint(0, self.grid_size - damage_size)
            start_y = random.randint(0, self.grid_size - damage_size)
            self.current_nca_state[:, :, start_y:start_y+damage_size, start_x:start_x+damage_size] = 0.0
        elif action == 2: # Rotation
            self.current_nca_state = torch.roll(self.current_nca_state, shifts=(random.randint(-2, 2), random.randint(-2, 2)), dims=(2, 3))
        elif action == 3: # Druck
            self.current_nca_state[:, :, :2, :] = 0.0; self.current_nca_state[:, :, -2:, :] = 0.0
            self.current_nca_state[:, :, :, :2] = 0.0; self.current_nca_state[:, :, :, -2:] = 0.0

        with torch.no_grad():
            self.current_nca_state = nca_step_function(self.current_nca_state, self.nca_model)
        
        current_living_cells = get_living_mask(self.current_nca_state).sum().item()
        
        # Hinweis: Diese Belohnungsfunktion ist eine einfache Baseline. Für komplexeres Verhalten
        # könnte sie um Form-, Integritäts- oder Ziel-Metriken erweitert werden.
        reward = 0.0
        terminated = False
        if current_living_cells > (NCA_GRID_SIZE * NCA_GRID_SIZE * 0.01):
            reward += 1.0 # Überlebensbonus
            if action == 1 and current_living_cells > previous_living_cells * 0.9: reward += 2.0 # Regenerationsbonus
            elif action == 0 and current_living_cells >= previous_living_cells: reward += 1.0 # Stabilitätsbonus
        else:
            reward -= 5.0 # Strafe für Kollaps
            terminated = True

        if self.render_mode == 'human': self.render()
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == 'human':
            rgb_array = np.transpose(self._get_obs(), (1, 2, 0))
            if self.fig is None:
                plt.ion(); self.fig, self.ax = plt.subplots(1, 1, figsize=(self.grid_size/10, self.grid_size/10))
                self.im = self.ax.imshow(rgb_array); self.ax.axis('off'); self.fig.tight_layout(pad=0)
            else: self.im.set_data(rgb_array)
            self.fig.canvas.draw_idle(); plt.pause(0.01)
        elif self.render_mode == 'rgb_array':
            return np.transpose(self._get_obs(), (1, 2, 0))
        return None

    def close(self):
        if self.fig: plt.close(self.fig); self.fig, self.ax, self.im = None, None, None

class Reinforcement_Loop:
    """
    Simuliert einen Reinforcement Learning Loop mit einem einfachen Zufallsagenten.
    """
    def __init__(self, env, agent_policy=None):
        self.env = env
        # Hinweis: Dies ist ein einfacher Zufallsagent. Für ein fortgeschrittenes System
        # würde hier ein lernender RL-Agent (z.B. PPO, DQN) integriert.
        self.agent_policy = agent_policy if agent_policy else lambda obs: self.env.action_space.sample()

    def run_episode(self, max_steps=NCA_STEPS_PER_GROWTH):
        """
        Führt eine einzelne RL-Episode durch.
        :param max_steps: Maximale Anzahl von Schritten in einer Episode.
        :return: Totalbelohnung der Episode, Liste der Zustände, ob terminated
        """
        obs, info = self.env.reset()
        total_reward = 0
        episode_states = [self.env.current_nca_state.squeeze(0).cpu()]
        for step in range(max_steps):
            action = self.agent_policy(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            episode_states.append(self.env.current_nca_state.squeeze(0).cpu())
            if terminated or truncated: break
        return total_reward, episode_states, terminated

class SimRunner:
    """
    Führt RL-Episoden durch, um Daten (Positiv-/Negativbeispiele) für den Validator zu sammeln.
    """
    def __init__(self, nca_model_for_sims, num_simulations=RL_EPISODES):
        self.nca_model = nca_model_for_sims
        self.num_simulations = num_simulations
        self.env = StressTest_Env(nca_model_for_env=self.nca_model, render_mode=None)
        self.rl_loop = Reinforcement_Loop(self.env)
        
    def run_simulations(self):
        logger.info(f"Führe {self.num_simulations} RL-Stresstests durch...")
        simulation_results, positive_examples, negative_examples = [], [], []
        for i in range(self.num_simulations):
            total_reward, episode_states_full, terminated = self.rl_loop.run_episode()
            episode_rgb_sequence = torch.stack([to_rgb(s) for s in episode_states_full])
            simulation_results.append({"id": i, "total_reward": total_reward, "terminated": terminated, "episode_states_rgb": episode_rgb_sequence})
            if total_reward > 5.0 and not terminated: positive_examples.append(episode_rgb_sequence)
            elif total_reward < 0.0 or terminated: negative_examples.append(episode_rgb_sequence)
            if (i + 1) % (self.num_simulations // 10 or 1) == 0:
                logger.info(f"Simulationsfortschritt: {i+1}/{self.num_simulations}. Letzte Belohnung: {total_reward:.2f}")
        self.env.close()
        return simulation_results, positive_examples, negative_examples

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 3: Validierungsnetzwerk (RLVR)
# ----------------------------------------------------------------------------------------------------

class Validator_Net(nn.Module):
    """
    CNN-GRU Netzwerk zur Klassifikation von Wachstumssequenzen als "funktional" vs. "tumorartig".
    """
    def __init__(self, input_channels_per_frame=3, hidden_dim=128, num_classes=2):
        super().__init__()
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels_per_frame, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.feature_output_dim = 128
        self.rnn = nn.GRU(self.feature_output_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        cnn_features = self.cnn_feature_extractor(x).view(batch_size, seq_len, self.feature_output_dim)
        rnn_output, _ = self.rnn(cnn_features)
        logits = self.classifier(rnn_output[:, -1, :])
        return logits

class Validator_Training:
    """
    Trainiert das Validierungsnetzwerk auf gesammelten Positiv- und Negativbeispielen.
    """
    def __init__(self, validator_net, pos_examples, neg_examples):
        self.net = validator_net.to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.max_trained_seq_len = 0
        self.dataset = self._create_dataset(pos_examples, neg_examples)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=VALIDATOR_BATCH_SIZE, shuffle=True) if self.dataset else None

    def _create_dataset(self, pos_examples, neg_examples):
        all_sequences, all_labels = [], []
        max_seq_len = max([s.shape[0] for s in pos_examples] + [s.shape[0] for s in neg_examples]) if pos_examples or neg_examples else 0
        if max_seq_len == 0: logger.warning("Keine Beispiele für Validator-Training."); return []
        self.max_trained_seq_len = max_seq_len

        def pad_sequence(seq, target_len):
            if seq.shape[0] > target_len: return seq[:target_len]
            padding = torch.zeros((target_len - seq.shape[0], *seq.shape[1:]), dtype=seq.dtype)
            return torch.cat((seq, padding), dim=0)

        for seq in pos_examples: all_sequences.append(pad_sequence(seq, max_seq_len)); all_labels.append(0)
        for seq in neg_examples: all_sequences.append(pad_sequence(seq, max_seq_len)); all_labels.append(1)
        return torch.utils.data.TensorDataset(torch.stack(all_sequences).to(DEVICE), torch.tensor(all_labels).long().to(DEVICE))

    def train(self, epochs=VALIDATOR_EPOCHS):
        if not self.dataloader: logger.warning("Keine Trainingsdaten für Validator. Training übersprungen."); return
        logger.info(f"Beginne Validierungsnetzwerk-Training für {epochs} Epochen..."); self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, labels in self.dataloader:
                self.optimizer.zero_grad(); outputs = self.net(data); loss = self.criterion(outputs, labels)
                loss.backward(); self.optimizer.step(); total_loss += loss.item()
            if (epoch + 1) % (epochs // 10 or 1) == 0: logger.info(f"Validator Epoche {epoch+1}/{epochs}, Avg Loss: {total_loss / len(self.dataloader):.6f}")
        logger.info("Validierungsnetzwerk-Training abgeschlossen."); self.net.eval()

class Validator_Filter:
    """Filtert NCA-Strukturen basierend auf dem Score des Validierungsnetzwerks."""
    def __init__(self, validator_net, threshold=VALIDATOR_THRESHOLD):
        self.net = validator_net.to(DEVICE); self.threshold = threshold; self.net.eval()
        self._max_seq_len = 0

    def set_max_seq_len(self, length):
        self._max_seq_len = length
        logger.info(f"Validator_Filter: max_seq_len auf {self._max_seq_len} gesetzt.")

    def check_structure(self, frames_sequence: list[torch.Tensor]):
        if not frames_sequence: logger.warning("Leere Sequenz an Validator_Filter übergeben."); return False, 0.0
        target_seq_len = self._max_seq_len if self._max_seq_len > 0 else VALIDATOR_MAX_SEQ_LEN_FALLBACK
        if self._max_seq_len == 0: logger.warning(f"Validator_Filter: _max_seq_len nicht gesetzt. Verwende Fallback {target_seq_len}.")

        processed_seq = list(frames_sequence)
        if len(processed_seq) > target_seq_len: processed_seq = processed_seq[:target_seq_len]
        elif len(processed_seq) < target_seq_len:
            padding = [torch.zeros_like(processed_seq[0]) for _ in range(target_seq_len - len(processed_seq))]
            processed_seq.extend(padding)

        input_tensor = torch.stack(processed_seq).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probabilities = F.softmax(self.net(input_tensor), dim=1)
            functional_score = probabilities[0, 0].item()
        return functional_score > self.threshold, functional_score

# ----------------------------------------------------------------------------------------------------
# 📁 Modul 4: Feedback-Loop / Selbstverbesserung
# ----------------------------------------------------------------------------------------------------

class Selection_Loop:
    """Verwaltet die Population der besten NCA-Modelle."""
    def __init__(self, population_size=POPULATION_SIZE):
        self.population_size = population_size
        self.best_nca_models = deque(maxlen=population_size)

    def register_nca_model(self, nca_model, score):
        model_copy = NCA_Model_3x3(NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS).to(DEVICE)
        model_copy.load_state_dict(nca_model.state_dict())
        self.best_nca_models.append({'model': model_copy, 'score': score})
        self.best_nca_models = deque(sorted(self.best_nca_models, key=lambda x: x['score'], reverse=True))

    def get_best_model(self):
        return self.best_nca_models[0] if self.best_nca_models else None

    def get_population(self):
        return list(self.best_nca_models)

class Mutation_Engine:
    """Erzeugt Varianten von NCA-Modellen durch Parameter-Mutation."""
    def __init__(self, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
        self.mutation_rate, self.mutation_strength = mutation_rate, mutation_strength

    def mutate(self, nca_model):
        mutated_model = NCA_Model_3x3(NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS).to(DEVICE)
        mutated_model.load_state_dict(nca_model.state_dict())
        with torch.no_grad():
            for param in mutated_model.parameters():
                if random.random() < self.mutation_rate:
                    param.add_(torch.randn_like(param) * self.mutation_strength)
        return mutated_model

class Evolution_Controller:
    """Orchestriert den gesamten evolutionären Prozess."""
    def __init__(self, num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE):
        self.num_generations, self.population_size = num_generations, population_size
        self.selection_loop = Selection_Loop(population_size)
        self.mutation_engine = Mutation_Engine()
        self.validator_net = Validator_Net().to(DEVICE)
        self.validator_filter = Validator_Filter(self.validator_net)
        self.logbook = Logbook()

    def initialize_population(self):
        self.current_nca_population = [NCA_Model_3x3(NCA_STATE_CHANNELS, NCA_HIDDEN_CHANNELS, NCA_STATE_CHANNELS).to(DEVICE) for _ in range(self.population_size)]
        logger.info(f"Initialisiere Population mit {self.population_size} NCA-Modellen.")

    def run_evolution(self):
        logger.info(f"Beginne Evolution über {self.num_generations} Generationen.")
        self.initialize_population()
        for generation in range(self.num_generations):
            logger.info(f"--- Generation {generation+1}/{self.num_generations} ---")
            
            all_pos_ex, all_neg_ex, gen_scores = [], [], []
            for i, nca_model in enumerate(self.current_nca_population):
                logger.info(f"Teste Modell {i+1}/{len(self.current_nca_population)}...")
                sim_runner = SimRunner(nca_model, num_simulations=RL_EPISODES // self.population_size)
                _, pos_ex, neg_ex = sim_runner.run_simulations()
                all_pos_ex.extend(pos_ex); all_neg_ex.extend(neg_ex)
                gen_scores.append({'model': nca_model, 'pos_ex': pos_ex, 'neg_ex': neg_ex})
                
            if all_pos_ex and all_neg_ex:
                validator_trainer = Validator_Training(self.validator_net, all_pos_ex, all_neg_ex)
                validator_trainer.train()
                if validator_trainer.max_trained_seq_len > 0:
                    self.validator_filter.set_max_seq_len(validator_trainer.max_trained_seq_len)
            else: # Added for clarity when validator is not trained
                logger.warning("Nicht genug Beispiele für Validator-Training in dieser Generation. Validator wird nicht trainiert.")
            
            gen_combined_scores = []
            for item in gen_scores:
                num_pos = len(item['pos_ex'])
                num_neg = len(item['neg_ex'])
                
                # Einfacher Fitness-Score: Anzahl positiver Beispiele minus negative
                # Dies bevorzugt Modelle, die robust sind (mehr Positivbeispiele) und weniger scheitern (weniger Negativbeispiele)
                fitness_score = num_pos - num_neg

                total_val_score = 0
                # Only run validation if there are positive examples for this model
                if item['pos_ex']:
                    for seq_tensor in item['pos_ex']:
                        # Pass the sequence as a list of individual frames (C, H, W)
                        _, val_score = self.validator_filter.check_structure([frame for frame in seq_tensor])
                        total_val_score += val_score
                    avg_val_score = total_val_score / num_pos
                else:
                    avg_val_score = 0.0
                
                combined_score = fitness_score + avg_val_score * 5 # Gewichteter Validator-Score
                gen_combined_scores.append(combined_score)
                self.selection_loop.register_nca_model(item['model'], combined_score)

            self.logbook.log_generation_summary(generation+1, max(gen_combined_scores, default=0), np.mean(gen_combined_scores), len(all_pos_ex), len(all_neg_ex))
            
            next_gen = [item['model'] for item in self.selection_loop.get_population()] # Eliten
            while len(next_gen) < self.population_size:
                parent = random.choice(self.selection_loop.get_population())['model']
                next_gen.append(self.mutation_engine.mutate(parent))
            self.current_nca_population = next_gen
            
            if self.selection_loop.get_best_model():
                 logger.info(f"Generation {generation+1} abgeschlossen. Top NCA-Score dieser Population: {self.selection_loop.get_best_model()['score']:.2f}")

        logger.info("Evolution abgeschlossen."); best_model_info = self.selection_loop.get_best_model()
        return best_model_info['model'] if best_model_info else None

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
        
        # Initialzustand separat hinzufügen
        states_sequence.append(to_rgb(state).squeeze(0).cpu().numpy().transpose(1, 2, 0))

        with torch.no_grad():
            for i in range(num_steps):
                state = nca_step_function(state, nca_model)
                states_sequence.append(to_rgb(state).squeeze(0).cpu().numpy().transpose(1, 2, 0))

        # Die Update-Funktion empfängt ein Tupel (frame_data, frame_idx_val)
        def update(frame_data_and_idx_tuple):
            frame_data, frame_idx_val = frame_data_and_idx_tuple # Tupel entpacken
            self.im.set_data(frame_data)
            self.ax.set_title(f"{title}\nStep: {frame_idx_val}")
            return self.im, # Wichtig: FuncAnimation erwartet ein Iterable von Artists

        # FuncAnimation direkt mit der update-Funktion aufrufen und zip für frames verwenden
        self.animation = FuncAnimation(self.fig, update,
                                       frames=zip(states_sequence, range(len(states_sequence))),
                                       interval=1000/VISUALIZATION_FPS, blit=True)
        plt.show(block=False)
        logger.info(f"Visualisierung abgeschlossen. Titel: {title}")

    def save_animation(self, filename="nca_growth.gif"):
        """Speichert die generierte Animation als GIF."""
        if self.animation:
            logger.info(f"Speichere Animation als {filename}...")
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
