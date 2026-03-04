# Cognitive Neural Architecture (CNA)
### Simulation of Suboptimality Task Conditions

> **Drift Diffusion Model Parameters · 2 Load × 5 Speed Conditions · DDM outputs: `a` (boundary separation) and `v` (drift rate)**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Architecture Initialization](#2-architecture-initialization)
3. [Module Implementations](#3-module-implementations)
   - [3.1 Sensory Encoder](#31-sensory-encoder)
   - [3.2 Working Memory Buffer](#32-working-memory-buffer)
   - [3.3 Relational Reasoner](#33-relational-reasoner)
   - [3.4 Executive Controller](#34-executive-controller)
   - [3.5 Vigilance Tracker](#35-vigilance-tracker)
   - [3.6 Leaky Competing Accumulator (LCA)](#36-leaky-competing-accumulator-lca)
   - [3.7 Urgency Module](#37-urgency-module)
4. [Training Cases & Protocol](#4-training-cases--protocol)
   - [4.1 Dataset Generation](#41-dataset-generation)
   - [4.2 Loss Function](#42-loss-function)
   - [4.3 Training Loop](#43-training-loop)
   - [4.4 Condition Cell Reference](#44-condition-cell-reference)
5. [EEG-Based Initialization](#5-eeg-based-initialization)
6. [Full Training Setup](#6-full-training-setup)

---

## 1. Architecture Overview

The **Cognitive Neural Architecture (CNA)** is a modular recurrent system designed to replicate the behavioural and DDM-measurable outputs of the Suboptimality Task under varying cognitive load and speed-accuracy tradeoff conditions. Each module corresponds to a distinct cognitive construct measured by the task battery.

**Experimental structure:** 2 load conditions (no-load, dual-task) × 5 speed conditions → **10 condition cells** per participant. Each cell yields boundary separation `a` and drift rate `v`, which the network must produce as interpretable emergent quantities.

> **DDM → Network mapping**
> - **Drift rate `v`** maps to the signal-to-noise ratio in evidence accumulation — how cleanly the accumulator converges toward a decision.
> - **Boundary separation `a`** maps to the commitment threshold — how much accumulated evidence is needed before a response gate fires.

### Module Summary

| Module    | Class                  | Key Parameters                        | Cognitive Analog                    |
|-----------|------------------------|---------------------------------------|-------------------------------------|
| Input     | `SensoryEncoder`       | `in=128, out=256, layers=4`           | Visual/spatial stimulus processing  |
| WM        | `WorkingMemoryBuffer`  | `slots=4, slot_dim=128`               | Op Span, Sym Span, Rot Span         |
| Fluid     | `RelationalReasoner`   | `heads=8, depth=4`                    | RAPM, Number Series, Letter Sets    |
| Attention | `ExecutiveController`  | `hidden=256, inhibition=True`         | Squared Simon, Antisaccade, VA      |
| Sustained | `VigilanceTracker`     | `decay=0.02, min=0.1`                 | SART, SAC-T, PVT                    |
| Decision  | `LeakyAccumulator`     | `n_acc=2, leak=0.1, noise=0.15`       | LCA → RT + choice → `a` and `v`    |
| Speed     | `UrgencyModule`        | `n_levels=5, scale_range=[0.5, 2.0]`  | 5 speed-pressure conditions         |

---

## 2. Architecture Initialization

All modules are initialized with principled parameter distributions informed by their cognitive analogs. Random seeds are fixed per participant to allow reproducible simulation of individual differences.

```python
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ArchitectureConfig:
    """Hyperparameters for all CNA modules."""
    # Sensory encoder
    sensory_input_dim: int = 128
    sensory_hidden_dim: int = 256
    sensory_layers: int = 4

    # Working memory
    wm_n_slots: int = 4             # capacity limit
    wm_slot_dim: int = 128
    wm_load_penalty: float = 0.3    # noise added per extra load unit

    # Relational reasoner
    rr_n_heads: int = 8
    rr_depth: int = 4
    rr_dropout: float = 0.1

    # Executive / attention control
    ec_hidden_dim: int = 256
    ec_inhibition_strength: float = 1.2

    # Vigilance tracker
    vig_decay_rate: float = 0.02    # per trial
    vig_min_level: float = 0.1
    vig_recovery_rate: float = 0.05

    # Leaky competing accumulator
    lca_n_accumulators: int = 2
    lca_leak: float = 0.10
    lca_noise_std: float = 0.15
    lca_inhibition: float = 0.08

    # Urgency / speed-accuracy
    n_speed_levels: int = 5
    urgency_scale_range: Tuple = (0.5, 2.0)


class CognitiveNeuralArchitecture(nn.Module):
    """
    Full modular architecture for Suboptimality Task simulation.
    Produces RT + choice per trial; DDM parameters a and v
    are extracted post-hoc via HDDM fitting on simulated RT distributions.
    """

    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder     = SensoryEncoder(cfg)
        self.wm_buffer   = WorkingMemoryBuffer(cfg)
        self.reasoner    = RelationalReasoner(cfg)
        self.executive   = ExecutiveController(cfg)
        self.vigilance   = VigilanceTracker(cfg)
        self.accumulator = LeakyAccumulator(cfg)
        self.urgency     = UrgencyModule(cfg)

        self._init_weights()

    def _init_weights(self):
        """Principled initialization per module."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'inhibit' in name:
                    # Inhibitory weights: negative, small
                    nn.init.normal_(module.weight, mean=-0.1, std=0.05)
                elif 'accumulator' in name:
                    # Accumulators: near-zero start
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for pname, param in module.named_parameters():
                    if 'weight' in pname:
                        nn.init.orthogonal_(param)  # ortho init for RNNs
                    elif 'bias' in pname:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1 (standard LSTM trick)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

    def forward(self, stimulus, load_condition, speed_level,
                hx=None, trial_number=0):
        """
        Args:
            stimulus      : (B, T, input_dim)  perceptual input sequence
            load_condition: (B,) int in {0, 1}  0=no-load, 1=dual-task
            speed_level   : (B,) int in {0..4}  0=most cautious, 4=most urgent
            hx            : recurrent hidden state (for sustained attention)
            trial_number  : int  used by vigilance decay
        Returns:
            rt      : (B,) simulated response time
            choice  : (B,) 0 or 1
            extras  : dict with intermediate states for analysis
        """
        # 1. Encode stimulus
        features = self.encoder(stimulus)           # (B, hidden)

        # 2. Working memory: load degrades encoding quality
        wm_out, wm_noise = self.wm_buffer(features, load_condition)

        # 3. Relational reasoning over WM contents
        relational = self.reasoner(wm_out)          # (B, hidden)

        # 4. Executive control: suppress prepotent response
        controlled, conflict = self.executive(relational, stimulus)

        # 5. Vigilance gates signal gain
        vig_level = self.vigilance.get_level(trial_number)
        gated = controlled * vig_level              # scalar gate

        # 6. Urgency collapses threshold over time
        urgency = self.urgency(speed_level)         # (B,)

        # 7. Leaky accumulator: produces RT + choice
        rt, choice, acc_trace = self.accumulator(
            gated, urgency, noise_scale=wm_noise
        )

        return rt, choice, {
            'wm_noise': wm_noise, 'conflict': conflict,
            'vig_level': vig_level, 'acc_trace': acc_trace
        }
```

---

## 3. Module Implementations

### 3.1 Sensory Encoder

A feedforward stack that processes the stimulus array into a compact feature vector. No recurrence — stimulus encoding is assumed instantaneous relative to the decision process.

```python
class SensoryEncoder(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.sensory_input_dim, cfg.sensory_hidden_dim),
            nn.LayerNorm(cfg.sensory_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.sensory_hidden_dim, cfg.sensory_hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):   # x: (B, T, D) → mean-pool → (B, D)
        return self.net(x.mean(dim=1))
```

---

### 3.2 Working Memory Buffer

Implements slot-based storage with explicit capacity limits. The `load_condition` parameter injects Gaussian noise proportional to the load penalty, degrading the effective drift rate. Models dual-task interference observed in Operation Span tasks.

```python
class WorkingMemoryBuffer(nn.Module):
    """Slot-attention style WM with capacity limits and load noise."""
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.n_slots      = cfg.wm_n_slots
        self.slot_dim     = cfg.wm_slot_dim
        self.load_penalty = cfg.wm_load_penalty

        # Slot competition — slots learned, compete via softmax
        self.slot_keys   = nn.Parameter(torch.randn(cfg.wm_n_slots, cfg.wm_slot_dim))
        self.slot_values = nn.Parameter(torch.randn(cfg.wm_n_slots, cfg.wm_slot_dim))
        self.proj_in     = nn.Linear(cfg.sensory_hidden_dim, cfg.wm_slot_dim)
        self.proj_out    = nn.Linear(cfg.wm_slot_dim, cfg.sensory_hidden_dim)

    def forward(self, features, load_condition):
        query = self.proj_in(features)              # (B, slot_dim)
        # Attention over slots
        attn  = torch.softmax(
            query @ self.slot_keys.T / self.slot_dim**0.5, dim=-1
        )                                           # (B, n_slots)
        readout = attn @ self.slot_values           # (B, slot_dim)
        out = self.proj_out(readout)                # (B, sensory_hidden_dim)

        # Load noise: dual-task degrades signal
        noise_scale = load_condition.float() * self.load_penalty
        noise = torch.randn_like(out) * noise_scale.unsqueeze(-1)
        return out + noise, noise_scale             # also return scale for LCA
```

---

### 3.3 Relational Reasoner

A multi-head self-attention transformer block operating over the WM readout. Models the relational reasoning demands of fluid intelligence tasks (RAPM, Number Series). Depth can be varied to simulate different task difficulties.

```python
class RelationalReasoner(nn.Module):
    """Transformer block for relational / pattern reasoning."""
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        d = cfg.sensory_hidden_dim
        self.attn  = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.rr_n_heads,
            dropout=cfg.rr_dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):                   # x: (B, D) → unsqueeze for attn
        x = x.unsqueeze(1)                  # (B, 1, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1)                 # (B, D)
```

---

### 3.4 Executive Controller

Models attention control (Squared Simon, Antisaccade) via a two-pathway architecture: a bottom-up salience pathway and a top-down inhibitory control signal. Conflict between pathways degrades the quality of evidence passed to the accumulator.

```python
class ExecutiveController(nn.Module):
    """Biased competition model of attention control."""
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        d = cfg.sensory_hidden_dim
        # Top-down control signal
        self.control  = nn.Linear(d, d)
        # Bottom-up salience (prepotent response)
        self.salience = nn.Linear(d, d)
        # Inhibitory gate
        self.inhibit  = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Tanh(),
            nn.Linear(d, d),
        )
        self.inh_strength = cfg.ec_inhibition_strength
        self.output_norm  = nn.LayerNorm(d)

    def forward(self, features, raw_stimulus):
        td = self.control(features)               # top-down goal signal
        bu = self.salience(raw_stimulus.mean(1))  # bottom-up prepotent

        # Conflict: cosine distance between pathways
        conflict = 1.0 - torch.cosine_similarity(td, bu, dim=-1)  # (B,)

        # Inhibitory gate suppresses prepotent response
        gate   = self.inhibit(torch.cat([td, bu], dim=-1))
        output = td + self.inh_strength * gate    # biased competition result
        return self.output_norm(output), conflict
```

---

### 3.5 Vigilance Tracker

A resource depletion model where a scalar "arousal" state decays over trials (PVT, SART). The vigilance level gates the gain of the evidence signal, producing the lapses and slow responses characteristic of sustained attention tasks.

```python
class VigilanceTracker(nn.Module):
    """
    Resource depletion model for sustained attention.
    Vigilance decays over trials and recovers partially between blocks.
    """
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.decay    = cfg.vig_decay_rate
        self.min_vig  = cfg.vig_min_level
        self.recovery = cfg.vig_recovery_rate
        self.register_buffer('level', torch.tensor(1.0))

    def get_level(self, trial_number: int) -> float:
        # Exponential decay with floor
        lvl = max(self.min_vig,
                  1.0 * np.exp(-self.decay * trial_number))
        self.level = torch.tensor(lvl)
        return lvl

    def reset_block(self):
        # Partial recovery between blocks
        new_level = float(self.level) + self.recovery
        self.level = torch.tensor(min(1.0, new_level))
```

---

### 3.6 Leaky Competing Accumulator (LCA)

The core decision module. Two accumulators race with mutual inhibition; the first to reach a dynamic threshold (modulated by urgency) wins. RT is the time to threshold; the winning accumulator determines choice. This directly instantiates DDM equations in discrete stochastic form.

```python
class LeakyAccumulator(nn.Module):
    """
    Leaky Competing Accumulator → produces RT distributions
    from which DDM parameters a and v are extracted via HDDM.
    """
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.n_acc   = cfg.lca_n_accumulators
        self.leak    = cfg.lca_leak
        self.noise   = cfg.lca_noise_std
        self.inhibit = cfg.lca_inhibition
        self.dt      = 0.001       # 1 ms time step
        self.max_time = 3.0        # 3 s max RT

        # Project evidence to accumulator drive
        self.drive_proj = nn.Linear(256, self.n_acc)

    def forward(self, evidence, urgency, noise_scale=0.0):
        B = evidence.size(0)
        drive   = self.drive_proj(evidence)     # (B, n_acc): net input per acc
        n_steps = int(self.max_time / self.dt)

        x      = torch.zeros(B, self.n_acc)    # accumulator states
        rt     = torch.full((B,), self.max_time)
        choice = torch.zeros(B, dtype=torch.long)
        done   = torch.zeros(B, dtype=torch.bool)
        trace  = []

        # Dynamic threshold collapses with urgency
        base_threshold = 1.0
        threshold = base_threshold / (urgency + 1e-6)  # (B,)

        total_noise = self.noise + noise_scale.float()  # load inflates noise

        for t in range(n_steps):
            t_sec = t * self.dt
            # LCA update: dx = (-leak*x - inhibit*sum + drive + noise) * dt
            inhib = x.sum(dim=1, keepdim=True) - x  # competitor sum
            noise = torch.randn_like(x) * total_noise * self.dt**0.5
            dx    = (-self.leak * x
                     - self.inhibit * inhib
                     + drive
                     + noise) * self.dt
            x     = torch.clamp(x + dx, min=0)  # rectify (no negative activation)
            trace.append(x.detach())

            # Check threshold crossing for undecided trials
            crossed = (x.max(dim=1).values >= threshold) & ~done
            if crossed.any():
                rt[crossed]     = t_sec
                choice[crossed] = x[crossed].argmax(dim=1)
                done            = done | crossed
            if done.all(): break

        return rt, choice, torch.stack(trace, dim=1)  # (B,), (B,), (B, T, n_acc)
```

---

### 3.7 Urgency Module

Maps the 5 discrete speed-pressure levels to a scalar urgency signal. Higher urgency **collapses the LCA threshold**, recreating the speed-accuracy tradeoff that manifests as reduced `a` in DDM fits under high speed pressure.

```python
class UrgencyModule(nn.Module):
    """Learns a mapping from speed level index to urgency scalar."""
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        lo, hi = cfg.urgency_scale_range
        # Fixed linear schedule (can be made learnable)
        self.register_buffer(
            'urgency_levels',
            torch.linspace(lo, hi, cfg.n_speed_levels)  # [0.5, ..., 2.0]
        )

    def forward(self, speed_level_idx):     # (B,) int
        return self.urgency_levels[speed_level_idx]  # (B,) float
```

---

## 4. Training Cases & Protocol

### 4.1 Dataset Generation

Training data is generated synthetically to match the factorial structure of the Suboptimality Task. Each "participant" is a fixed random seed that sets individual-difference parameters (WM capacity, vigilance decay rate, inhibition strength). Targets are RT distributions per condition cell, with DDM parameters extracted as fitting targets.

```python
class SuboptimalityDataset(torch.utils.data.Dataset):
    """
    Each sample = one trial from one condition cell.
    The model must reproduce RT + accuracy matching human DDM outputs.
    """
    LOAD_CONDITIONS  = [0, 1]            # 0=no-load, 1=dual-task
    SPEED_CONDITIONS = [0, 1, 2, 3, 4]  # 0=slowest, 4=fastest

    def __init__(self, n_participants=50, trials_per_cell=200,
                 ddm_params_file='ddm_outputs.csv'):
        self.n_p = n_participants
        self.tpc = trials_per_cell
        # Load real DDM params (a and v) as regression targets
        import pandas as pd
        self.ddm = pd.read_csv(ddm_params_file)  # cols: pid, load, speed, a, v
        self.trials = self._build_trial_list()

    def _build_trial_list(self):
        trials = []
        for pid in range(self.n_p):
            for load in self.LOAD_CONDITIONS:
                for speed in self.SPEED_CONDITIONS:
                    row = self.ddm[
                        (self.ddm.pid == pid) &
                        (self.ddm.load == load) &
                        (self.ddm.speed == speed)
                    ].iloc[0]
                    for _ in range(self.tpc):
                        trials.append({
                            'pid': pid,
                            'load': load,
                            'speed': speed,
                            'target_a': row['a'],   # boundary separation
                            'target_v': row['v'],   # drift rate
                        })
        return trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        t = self.trials[idx]
        # Stimulus: random perceptual noise around a signal (simplified)
        stimulus = torch.randn(10, 128)  # 10 time steps, 128-dim input
        if t['load'] == 1:
            stimulus += torch.randn_like(stimulus) * 0.5  # dual-task degrades
        return {
            'stimulus': stimulus,
            'load'    : torch.tensor(t['load'],  dtype=torch.long),
            'speed'   : torch.tensor(t['speed'], dtype=torch.long),
            'pid'     : t['pid'],
            'target_a': torch.tensor(t['target_a'], dtype=torch.float32),
            'target_v': torch.tensor(t['target_v'], dtype=torch.float32),
        }
```

---

### 4.2 Loss Function

Three terms are combined: **(1)** RT distribution matching (Wasserstein-1 on binned RT quantiles), **(2)** DDM parameter regression via a differentiable EZ-diffusion surrogate, **(3)** accuracy (proportion correct) per condition cell.

```python
class CognitiveArchitectureLoss(nn.Module):
    def __init__(self, lambda_rt=1.0, lambda_ddm=2.0, lambda_acc=0.5):
        super().__init__()
        self.lambda_rt  = lambda_rt
        self.lambda_ddm = lambda_ddm
        self.lambda_acc = lambda_acc

    def forward(self, rt_pred, choice_pred,
                rt_target, choice_target, target_a, target_v):

        # 1. RT distribution loss (MSE on quantiles: fast, robust)
        quantiles = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        q_pred    = torch.quantile(rt_pred, quantiles)
        q_target  = torch.quantile(rt_target, quantiles)
        rt_loss   = F.mse_loss(q_pred, q_target)

        # 2. DDM parameter regression via differentiable EZ-DDM surrogate
        a_pred, v_pred = ezdiffusion_surrogate(rt_pred, choice_pred)
        ddm_loss = (F.mse_loss(a_pred, target_a) +
                    F.mse_loss(v_pred, target_v))

        # 3. Accuracy loss
        acc_pred   = choice_pred.float().mean()
        acc_target = choice_target.float().mean()
        acc_loss   = F.mse_loss(acc_pred, acc_target)

        total = (self.lambda_rt  * rt_loss +
                 self.lambda_ddm * ddm_loss +
                 self.lambda_acc * acc_loss)

        return total, {'rt': rt_loss, 'ddm': ddm_loss, 'acc': acc_loss}


def ezdiffusion_surrogate(rt, choice):
    """EZ-diffusion closed-form estimator (Wagenmakers et al. 2007).
    Extracts a and v analytically from mean RT, RT variance, and accuracy.
    Differentiable w.r.t. rt and choice tensors."""
    acc = choice.float().mean().clamp(0.51, 0.99)  # avoid boundary
    vrt = rt.var()
    L   = torch.log(acc / (1 - acc))
    v   = torch.sign(acc - 0.5) * (L * (acc**2 * L - acc * L + acc - 0.5) / vrt)**0.25
    a   = L / v
    return a, v
```

---

### 4.3 Training Loop

```python
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0

    for batch in loader:
        stimulus = batch['stimulus'].to(device)     # (B, T, D)
        load     = batch['load'].to(device)         # (B,)
        speed    = batch['speed'].to(device)        # (B,)
        target_a = batch['target_a'].to(device)     # (B,)
        target_v = batch['target_v'].to(device)     # (B,)

        rt_pred, choice_pred, _ = model(stimulus, load, speed)

        # Ground truth RT from DDM forward model (for distribution matching)
        rt_target = simulate_ddm(target_a, target_v, n=rt_pred.shape[0])

        loss, breakdown = loss_fn(
            rt_pred, choice_pred,
            rt_target, (rt_pred < rt_target.mean()).long(),
            target_a, target_v
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def simulate_ddm(a, v, n=200, dt=0.001, noise=1.0):
    """Forward DDM: simulate RT samples given a and v tensors."""
    rt_samples = []
    for a_i, v_i in zip(a, v):
        x, t = 0.0, 0.0
        while abs(x) < a_i.item() and t < 3.0:
            x += v_i.item() * dt + noise * np.random.randn() * dt**0.5
            t += dt
        rt_samples.append(t)
    return torch.tensor(rt_samples)
```

---

### 4.4 Condition Cell Reference

Expected effects of each condition on internal network dynamics and DDM parameter outputs:

| Load       | Speed      | Expected `v`         | Expected `a`       | Dominant Effect              | Training Signal                  |
|------------|------------|----------------------|--------------------|------------------------------|----------------------------------|
| 0 (none)   | 0 (slow)   | High                 | Wide (cautious)    | None — baseline              | High accuracy, slow RT           |
| 0 (none)   | 2 (medium) | High                 | Moderate           | Urgency moderates threshold  | Moderate RT, good accuracy       |
| 0 (none)   | 4 (fast)   | High                 | Narrow (urgent)    | Urgency collapses threshold  | Fast RT, slight accuracy drop    |
| 1 (dual)   | 0 (slow)   | Reduced (WM noise)   | Wide               | WM noise degrades drive      | Slow RT, lower accuracy          |
| 1 (dual)   | 2 (medium) | Reduced              | Moderate           | WM + urgency interact        | Moderate RT, lower accuracy      |
| 1 (dual)   | 4 (fast)   | Reduced              | Narrow             | Worst case: noise + urgency  | Fast RT, highest error rate      |

---

## 5. EEG-Based Initialization

Resting-state EEG spectral features can initialize individual-difference parameters in the network, bridging the empirical dataset to model predictions.

```python
class EEGInitializer:
    """
    Maps resting-state EEG spectral features to CNA module parameters.
    Implements the bridge between empirical EEG data and model initialization.
    """
    # Canonical EEG-cognition mappings (literature-derived)
    EEG_MAPPINGS = {
        'frontal_theta':   'wm_buffer.load_penalty',  # ↑θ → ↑WM engagement
        'parietal_alpha':  'vigilance.decay',          # ↑α → slower vigilance decay
        'frontal_beta':    'executive.inh_strength',   # ↑β → ↑inhibitory control
        'occipital_alpha': 'encoder.noise_floor',      # ↑α → noisier encoding
    }

    def __init__(self, eeg_features: dict):
        self.features = eeg_features  # {'frontal_theta': float, ...}

    def initialize_model(self, model: CognitiveNeuralArchitecture):
        # Normalize EEG features to parameter ranges
        ft = self.features.get('frontal_theta',   5.0)   # μV²/Hz
        pa = self.features.get('parietal_alpha',  10.0)
        fb = self.features.get('frontal_beta',    15.0)
        oa = self.features.get('occipital_alpha', 8.0)

        # Working memory capacity: high frontal theta → lower noise
        model.wm_buffer.load_penalty = max(0.1, 0.5 - 0.02 * ft)

        # Vigilance: high parietal alpha → slower decay (more sustained)
        model.vigilance.decay = max(0.005, 0.04 - 0.001 * pa)

        # Executive control: high frontal beta → stronger inhibition
        model.executive.inh_strength = 0.5 + 0.05 * fb

        # Sensory noise floor: high occipital alpha → noisier inputs
        noise_floor = 0.05 + 0.005 * oa
        model.encoder.net[0].weight.data.add_(
            torch.randn_like(model.encoder.net[0].weight) * noise_floor
        )
        return model


# Usage:
# eeg_feat = extract_resting_eeg_features(raw_eeg_array)  # your pipeline
# init     = EEGInitializer(eeg_feat)
# model_p  = init.initialize_model(CognitiveNeuralArchitecture(cfg))
```

---

## 6. Full Training Setup

```python
if __name__ == '__main__':
    cfg    = ArchitectureConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model   = CognitiveNeuralArchitecture(cfg).to(device)
    loss_fn = CognitiveArchitectureLoss(lambda_rt=1.0,
                                        lambda_ddm=2.0,
                                        lambda_acc=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )

    dataset = SuboptimalityDataset(
        n_participants=50,
        trials_per_cell=200,
        ddm_params_file='ddm_outputs.csv'
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4
    )

    for epoch in range(100):
        loss = train_epoch(model, loader, optimizer, loss_fn, device)
        scheduler.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Loss: {loss:.4f}')
            # Evaluate: compute R² between predicted and real a, v
            # across all 10 condition cells
            eval_ddm_recovery(model, dataset, device)
```

> **Evaluation target:** The primary metric is R² between network-recovered DDM parameters (`a`, `v`) and ground-truth participant DDM fits, computed separately for each of the 10 condition cells. A well-fitted model should capture both the main effects of load and speed, and their interaction.