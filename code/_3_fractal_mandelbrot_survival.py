import numpy as np
import matplotlib.pyplot as plt

def calculate_total_survival(external_combined_risk, resource_score, environmental_risk,
                            weights=[0.25,0.375,0.375], risk_threshold=0.5, k=8):
    S_external = 1 - external_combined_risk
    S_resources = max(0, min(1, (resource_score + 5) / 10))
    S_environment = 1 / (1 + np.exp(k*(environmental_risk - risk_threshold)))
    total_survival = (weights[0]*S_external + weights[1]*S_resources + weights[2]*S_environment)
    return max(0, min(1, total_survival))

N = 500
x = np.linspace(-2, 1, N)
y = np.linspace(-1.5, 1.5, N)
X, Y = np.meshgrid(x, y)
C = X + 1j*Y

img = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        c = C[i,j]
        z = 0
        steps = 0
        while abs(z) < 2 and steps < 60:
            z = z**2 + c
            steps += 1
        # Map fractal info to model parameters:
        risk = min(1, steps/60)
        resource = np.sin(c.real)*2 + 2
        environment = abs(np.cos(c.imag))
        img[i,j] = calculate_total_survival(risk, resource, environment)

plt.imshow(img, cmap='viridis', extent=[-2,1,-1.5,1.5])
plt.title("Mandelbrot Survival Fractal")
plt.colorbar(label='Survival Probability')
plt.axis('off')
plt.show()