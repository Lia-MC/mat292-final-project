import numpy as np
import matplotlib.pyplot as plt

# function from our model
def calculate_total_survival(external_combined_risk, resource_score, environmental_risk,
                             weights=[0.25, 0.375, 0.375], risk_threshold=0.5, k=8):
    S_external = 1 - external_combined_risk
    S_resources = max(0, min(1, (resource_score + 5) / 10))  
    S_environment = 1 / (1 + np.exp(k * (environmental_risk - risk_threshold)))
    total_survival = (weights[0] * S_external +
                      weights[1] * S_resources +
                      weights[2] * S_environment)
    return max(0, min(1, total_survival))

# fractal image generation
N = 400

xs = np.linspace(0, 1, N) # external_combined_risk
ys = np.linspace(-5, 5, N) # resource score
zs = np.linspace(0, 1, N) # environmental risk

# generate grid for fractal landscape where pixel colors = total_survival
fractal_img = np.zeros((N, N, 3))

for i, ext in enumerate(xs):
    for j, res in enumerate(ys):
        env = zs[(i+j)%N] 
        surv = calculate_total_survival(ext, res, env)
        # color maps: blue = safe, yellow = moderate, red = dangerous
        color = plt.cm.plasma(surv)
        fractal_img[j, i, :] = color[:3]

im = plt.imshow(fractal_img, origin="lower", extent=[0,1,-5,5])
plt.xlabel("Combined External Threat Risk (0=safe, 1=dangerous)")
plt.ylabel("Resource Score (-5=none, +5=max)")
plt.title("Fractal Landscape of Survival Probability")
plt.colorbar(im, label="Survival Probability")
plt.show()