import numpy as np
import matplotlib.pyplot as plt
from aisp.nsa import RNSA

np.random.seed(42)
self_data = np.random.normal(loc=0.5, scale=0.08, size=(100, 2))
self_data = np.clip(self_data, 0, 1)
y_train = np.ones(len(self_data), dtype=int)

RAIO = 0.08
nsa = RNSA(N=60, r=RAIO, seed=42)
nsa.fit(self_data, y_train)

test_normal  = np.clip(np.random.normal(loc=0.5, scale=0.08, size=(30, 2)), 0, 1)
test_anomaly = np.random.uniform(0.0, 1.0, size=(30, 2))
X_test       = np.vstack([test_normal, test_anomaly])
predictions  = nsa.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Painel 1: detectores ───────────────────────────────────────────────────
axes[0].scatter(self_data[:, 0], self_data[:, 1],
                color='green', zorder=4, s=40, label='Self (normal)')
detectors_array = nsa.detectors[1]
for det in detectors_array:
    circle = plt.Circle(
        (det.position[0], det.position[1]),
        RAIO,
        color='red', alpha=0.15, zorder=2
    )
    axes[0].add_patch(circle)
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
axes[0].set_aspect('equal')
axes[0].set_title('Detectores NSA gerados\n(vermelho = zona non-self)')
axes[0].legend()

# ── Painel 2: classificação ────────────────────────────────────────────────
# ↓ CORREÇÃO: comparar com a string '1', não o inteiro 1
colors = ['#2ecc71' if p == '1' else '#e74c3c' for p in predictions]
for point, color in zip(X_test, colors):
    axes[1].scatter(*point, color=color, s=60, zorder=3)
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].set_aspect('equal')
axes[1].set_title('Detecção de anomalias')
handles = [plt.Line2D([0],[0], marker='o', color='w',
           markerfacecolor=c, markersize=10, label=l)
           for c, l in [('#2ecc71','Normal (self)'), ('#e74c3c','Anomalia (non-self)')]]
axes[1].legend(handles=handles)

plt.suptitle('Negative Selection Algorithm (RNSA)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('nsa_demo.png', dpi=150)
plt.show()
print(f"Detectores gerados: {len(detectors_array)}")
print(f"Predições: {predictions}")
