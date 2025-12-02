import numpy as np
import matplotlib.pyplot as plt
import _4_model as model


# 1. health decay: time series graph
def plot_health_time_series_and_phase(user_inputs):
    # run model for individual risks
    indiv = model.run_individual_model(user_inputs)
    t = indiv["t"]
    R = indiv["R_t"]
    H = indiv["H_t"]

    # generate the coupled model
    plt.figure()
    plt.plot(R, H)
    plt.xlabel("R (remaining years)")
    plt.ylabel("H (health index)")
    plt.title("Coupled Model Phase Portrait: H vs R")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return indiv

# plot the health bifurcation digram (as H -> infinity)
def plot_health_bifurcation(user_inputs, param="s",
                            param_values=np.linspace(0.0, 1.0, 30)):
    H_inf_values = []

    for val in param_values:
        ui = user_inputs.copy()
        ui[param] = float(val)
        indiv = model.run_individual_model(ui)
        H_inf = indiv["H_t"][-1]
        H_inf_values.append(H_inf)

    plt.figure()
    plt.scatter(param_values, H_inf_values, s=15)
    plt.xlabel(param, fontsize=16)
    plt.ylabel("H", fontsize=16)
    # plt.title(f"Health Bifurcation Diagram: Final H vs {param}", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # core individual inputs (sample params)
    user_inputs = {
        "age": 29,
        "country": "Canada",
        "s": 0.3,
        "a": 0.6,
        "m": 0.4,
        "delta_w": 0.02,
        "H0": 0.9,
    }

    # 1. health decay plots: time series and coupled phase portrait
    indiv = plot_health_time_series_and_phase(user_inputs)

    # 2. health bifurcation
    plot_health_bifurcation(user_inputs, param="s", # can change to "a", "m", "H0" to see bifurcations for other params in model for individual risk!
                            param_values=np.linspace(0.0, 1.0, 25))

if __name__ == "__main__":
    main()
