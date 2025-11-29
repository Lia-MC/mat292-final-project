import numpy as np
import matplotlib.pyplot as plt

# simplified function from our model
def calculate_total_survival(external_combined_risk, resource_score, environmental_risk,
                             weights=[0.25,0.375,0.375], risk_threshold=0.5, k=8):
    S_external = 1 - external_combined_risk
    S_resources = max(0, min(1, (resource_score + 5) / 10))
    S_environment = 1 / (1 + np.exp(k*(environmental_risk - risk_threshold)))
    total_survival = (weights[0]*S_external + weights[1]*S_resources + weights[2]*S_environment)
    return max(0, min(1, total_survival))

def fractal_tree(ax, x, y, angle, depth, risk, resource, env):
    if depth == 0: return
    length = 0.8 + resource*0.1
    color = plt.cm.inferno(calculate_total_survival(risk, resource, env))
    x2, y2 = x + length*np.cos(angle), y + length*np.sin(angle)
    ax.plot([x,x2],[y,y2],color=color,lw=depth*0.7)
    # randomize parameters for branches
    nresource = np.clip(resource + np.random.uniform(-2,1),-5,5)
    nrisk = np.clip(risk + np.random.uniform(-0.2,0.2),0,1)
    nenv = np.clip(env + np.random.uniform(-0.2,0.2),0,1)
    fractal_tree(ax, x2, y2, angle-np.pi/8, depth-1, nrisk, nresource, nenv)
    fractal_tree(ax, x2, y2, angle+np.pi/8, depth-1, nrisk, nresource, nenv)

fig, ax = plt.subplots(figsize=(9,10))
fractal_tree(ax, 0, 0, np.pi/2, 11, risk=0.15, resource=3.0, env=0.2)
plt.axis('off')
plt.title("Fractal Tree of Survival Probability")
plt.show()