import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors

def calculate_total_survival(external_combined_risk, resource_score, environmental_risk,
                            weights=[0.25,0.375,0.375], risk_threshold=0.5, k=8):
    S_external = 1 - external_combined_risk
    S_resources = max(0, min(1, (resource_score + 5) / 10))
    S_environment = 1 / (1 + np.exp(k*(environmental_risk - risk_threshold)))
    total_survival = (weights[0]*S_external + weights[1]*S_resources + weights[2]*S_environment)
    return max(0, min(1, total_survival))

# generate random scenario nodes:
n_nodes = 80
np.random.seed(42)
params = []
for _ in range(n_nodes):
    risk = np.random.uniform(0, 1)
    resource = np.random.uniform(-5, 5)
    env = np.random.uniform(0, 1)
    surv = calculate_total_survival(risk, resource, env)
    params.append((risk, resource, env, surv))

# graph
G = nx.Graph()
for idx, (r, res, env, surv) in enumerate(params):
    color = plt.cm.plasma(surv)
    G.add_node(idx, survival=surv, color=color, pos=(res, env))
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        # add edge if nodes have similar survival probability
        if abs(params[i][3] - params[j][3]) < 0.13:
            G.add_edge(i,j)

fig, ax = plt.subplots(figsize=(11,9))
node_colors = [G.nodes[n]['color'] for n in G.nodes]
node_pos = {n:(G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]) for n in G.nodes}

nx.draw_networkx_nodes(G, pos=node_pos, node_color=node_colors, node_size=140, alpha=0.85, ax=ax)
nx.draw_networkx_edges(G, pos=node_pos, alpha=0.22, ax=ax)
ax.set_title("Survival Probability Network Fractal (Constellation Graph)")
ax.set_xlabel("Resource Score")
ax.set_ylabel("Environmental Risk")

sm = cm.ScalarMappable(cmap="plasma", norm=colors.Normalize(vmin=0, vmax=1))
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Survival Probability")

plt.show()