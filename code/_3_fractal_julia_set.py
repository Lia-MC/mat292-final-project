import numpy as np
import matplotlib.pyplot as plt

def calculate_total_survival(external_combined_risk, resource_score, environmental_risk,
                             weights=[0.25, 0.375, 0.375], risk_threshold=0.5, k=8):
    """
    Calculate normalized total survival probability (0-1)
    external_combined_risk: combined human + animal risk (0-1)
    resource_score: -5 to +5
    environmental_risk: 0-1
    """
    S_external = 1 - external_combined_risk
    S_resources = max(0, min(1, (resource_score + 5) / 10))
    S_environment = 1 / (1 + np.exp(k * (environmental_risk - risk_threshold)))
    total_survival = (weights[0] * S_external +
                      weights[1] * S_resources +
                      weights[2] * S_environment)
    return max(0, min(1, total_survival))


# Julia set config
N = 600
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
# Julia constant:
c = -0.8 + 0.156j

img = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        z = Z[i,j]
        steps = 0
        # standard Julia set iteration
        while abs(z) < 2 and steps < 60:
            z = z**2 + c
            steps += 1
        # mapping:
        risk = min(1, steps / 60)
        resource = np.cos(z.real) * 2 + 2 # use real part for variety
        environment = abs(np.sin(z.imag)) # use imag component
        img[i,j] = calculate_total_survival(risk, resource, environment)

plt.imshow(img, cmap="plasma", extent=[-1.5,1.5,-1.5,1.5])
plt.title("Julia Survival Fractal Pattern (Color = Model Survival Probability)")
plt.colorbar(label='Survival Probability')
plt.axis('off')
plt.show()