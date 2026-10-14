# Minimal reference inference for so101_bc_history1.
# Deliberately dependency-light: torch + numpy. Read model_card.json for the contract.
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path

HERE = Path(__file__).parent
CARD = json.load(open(HERE / 'model_card.json'))

class BCPolicy(nn.Module):
    def __init__(self, i, o, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, o), nn.Tanh())
    def forward(self, x): return self.net(x)

_s_mean = np.array(CARD['observation']['state_mean'], np.float32)
_s_std  = np.array(CARD['observation']['state_std'],  np.float32)
_a_min  = np.array(CARD['action']['action_min'], np.float32)
_a_max  = np.array(CARD['action']['action_max'], np.float32)
_n_hist = len(CARD['observation']['order']) - 1

def load():
    net = BCPolicy(CARD['observation']['dim'], CARD['action']['dim'])
    net.load_state_dict(torch.load(HERE / 'policy.pt', map_location='cpu'))
    net.eval()
    return net

def act(net, history):
    # history: recent joint-state arrays, OLDEST FIRST. Returns position targets.
    frames = history[-(_n_hist + 1):]
    while len(frames) < _n_hist + 1:          # episode start: repeat the first frame
        frames = [frames[0]] + frames
    x = np.concatenate([(np.asarray(f, np.float32) - _s_mean) / _s_std for f in frames])
    with torch.no_grad():
        y = net(torch.tensor(x)[None]).numpy()[0]
    return (y + 1) / 2 * (_a_max - _a_min) + _a_min

if __name__ == '__main__':
    net = load()
    s = np.array(CARD['observation']['state_mean'], np.float32)
    print('action for a mean-pose observation:', np.round(act(net, [s, s]), 2))
