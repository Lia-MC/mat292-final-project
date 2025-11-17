# graphs.py
import numpy as np
import matplotlib.pyplot as plt
import model


# =============== 1. HEALTH DECAY: TIME SERIES + BIFURCATION ===============

def plot_health_time_series_and_phase(user_inputs):
    """
    Health decay – Time series + coupled phase portrait (H vs R).
    """
    indiv = model.run_individual_model(user_inputs)
    t = indiv["t"]
    R = indiv["R_t"]
    H = indiv["H_t"]

    # Time series: H(t) and R(t)
    plt.figure()
    plt.plot(t, H, label="H(t): Health index")
    plt.plot(t, R, label="R(t): Remaining years")
    plt.xlabel("Time (years)")
    plt.ylabel("State")
    plt.title("Health Decay and Remaining Life – Time Series")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Coupled model: Phase portrait H vs R
    plt.figure()
    plt.plot(R, H)
    plt.xlabel("R (remaining years)")
    plt.ylabel("H (health index)")
    plt.title("Coupled Model Phase Portrait: H vs R")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return indiv


def plot_health_bifurcation(user_inputs, param="s",
                            param_values=np.linspace(0.0, 1.0, 30)):
    """
    Health decay – 'bifurcation'-style plot:
    vary parameter (s or a) and plot asymptotic H∞.
    This is more like a parameter–steady-state diagram, which is what you want.
    """
    H_inf_values = []

    for val in param_values:
        ui = user_inputs.copy()
        ui[param] = float(val)
        indiv = model.run_individual_model(ui)
        H_inf = indiv["H_t"][-1]
        H_inf_values.append(H_inf)

    plt.figure()
    plt.scatter(param_values, H_inf_values, s=15)
    plt.xlabel(param)
    plt.ylabel("Asymptotic H (H∞)")
    plt.title(f"Health 'Bifurcation' Diagram – Final H vs {param}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========= 2. ENVIRONMENT: BIFURCATION + PARAMETER-PLANE “PHASE PORTRAIT” =========

def environment_bifurcation(country, base_inputs):
    """
    Environment – Bifurcation: vary β and see environmental_risk.
    base_inputs should contain:
      alpha, beta, gamma, delta1, delta2, delta3, delta4,
      natural_disaster, temp_difference, drought_risk, population_density, Y0
    """
    env_model = model.EnvironmentalRiskModel()

    betas = np.linspace(0.6, 0.8, 40)
    risks = []

    for beta in betas:
        r = env_model.calculate_normalized_risk(
            country,
            alpha=base_inputs["alpha"],
            beta=beta,
            gamma=base_inputs["gamma"],
            delta1=base_inputs["delta1"],
            natural_disaster=base_inputs["natural_disaster"],
            delta2=base_inputs["delta2"],
            temp_difference=base_inputs["temp_difference"],
            delta3=base_inputs["delta3"],
            drought_risk=base_inputs["drought_risk"],
            delta4=base_inputs["delta4"],
            population_density=base_inputs["population_density"],
            Y0=base_inputs["Y0"],
        )
        risks.append(r)

    plt.figure()
    plt.plot(betas, risks, marker="o")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Environmental risk (normalized)")
    plt.title("Environment Bifurcation: Risk vs β")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def environment_phase_plane(country, base_inputs):
    """
    Environment – parameter-plane "phase portrait":
    plot risk over (β, γ) plane as a contour plot.
    Not a dynamical phase portrait, but a param-space portrait.
    """
    env_model = model.EnvironmentalRiskModel()

    beta_vals = np.linspace(0.6, 0.8, 40)
    gamma_vals = np.linspace(0.2, 0.4, 40)
    B, G = np.meshgrid(beta_vals, gamma_vals)
    Risk = np.zeros_like(B)

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            beta = B[i, j]
            gamma = G[i, j]
            r = env_model.calculate_normalized_risk(
                country,
                alpha=base_inputs["alpha"],
                beta=beta,
                gamma=gamma,
                delta1=base_inputs["delta1"],
                natural_disaster=base_inputs["natural_disaster"],
                delta2=base_inputs["delta2"],
                temp_difference=base_inputs["temp_difference"],
                delta3=base_inputs["delta3"],
                drought_risk=base_inputs["drought_risk"],
                delta4=base_inputs["delta4"],
                population_density=base_inputs["population_density"],
                Y0=base_inputs["Y0"],
            )
            Risk[i, j] = r

    plt.figure()
    cp = plt.contourf(B, G, Risk, levels=20)
    plt.colorbar(cp, label="Environmental risk")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\gamma$")
    plt.title("Environment Parameter Plane: Risk over (β, γ)")
    plt.tight_layout()
    plt.show()


# ======= 3. EXTERNAL THREAT: SENSITIVITY + ANIMAL PHASE-PLANE =======

def external_threat_sensitivity(country, years=50,
                                homicide_rates=np.linspace(0.1, 20, 20),
                                G=1.0, E=1.0):
    """
    External threat – Sensitivity: vary homicide rate H and see final V(years).
    Uses HumanThreatModel ODE directly.
    """
    ht = model.HumanThreatModel()
    final_probs = []

    for H in homicide_rates:
        lambda_base = (H / 100000.0) * G * E
        t_span = (0, years)
        t, V = ht.runge_kutta_4(
            ht.human_threat_ode,
            y0=0.0,
            t_span=t_span,
            args=(lambda_base,),
            n_steps=1000,
        )
        final_probs.append(V[-1])

    plt.figure()
    plt.plot(homicide_rates, final_probs, marker="o")
    plt.xlabel("Homicide rate per 100,000")
    plt.ylabel("Final cumulative probability V(T)")
    plt.title(f"External Threat Sensitivity – {country}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def animal_phase_plane(level="medium", time_span=50):
    """
    External threat – Animal phase-plane:
    plot animal threat vs survival using AnimalThreatModel.
    This directly gives you a phase-plane (y vs x).
    """
    at = model.AnimalThreatModel()

    params = at.animal_parameters[level]
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]

    if level == "low":
        initial_threat = 0.01
    elif level == "medium":
        initial_threat = 0.1
    else:
        initial_threat = 0.3

    y0 = np.array([1.0, initial_threat])
    t_span = (0, time_span)
    t, y = at.runge_kutta_4_system(
        at.animal_threat_ode,
        y0=y0,
        t_span=t_span,
        args=(a, b, c, d),
        n_steps=1000,
    )

    x = y[0]  # survival
    z = y[1]  # animal threat level

    plt.figure()
    plt.plot(x, z)
    plt.xlabel("x: Survival")
    plt.ylabel("y: Animal threat level")
    plt.title(f"Animal Phase-Plane – Predator level: {level}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========================== 4. DRIVER ==========================

def main():
    # Core individual inputs (you can change for a demo)
    user_inputs = {
        "age": 29,
        "country": "Canada",
        "s": 0.3,
        "a": 0.6,
        "m": 0.4,
        "delta_w": 0.02,
        "H0": 0.9,
    }

    # 1) Health decay – time series + coupled phase portrait
    indiv = plot_health_time_series_and_phase(user_inputs)

    # 1b) Health "bifurcation" – vary s or a
    plot_health_bifurcation(user_inputs, param="s",
                            param_values=np.linspace(0.0, 1.0, 25))
    # You could also do param="a" to show effect of activity

    # 2) Environment – we need some base environmental inputs
    # For a real run, you could call get_environmental_risk_inputs(country)
    # once, then reuse the returned values here. For now, you can plug in
    # some numbers from a previous run or your report.
    country = "Canada"
    base_env_inputs = {
        "natural_disaster": 0.2,
        "temp_difference": 1.0,
        "drought_risk": 1.0,
        "population_density": 100.0,
        "alpha": 0.3,
        "beta": 0.7,
        "gamma": 0.3,
        "delta1": 0.0055,
        "delta2": 0.0055,
        "delta3": 0.0055,
        "delta4": 0.0055,
        "Y0": 0.5,
    }

    environment_bifurcation(country, base_env_inputs)
    environment_phase_plane(country, base_env_inputs)

    # 3) External threat – sensitivity + animal phase-plane
    external_threat_sensitivity(country, years=50,
                                homicide_rates=np.linspace(0.1, 20, 20),
                                G=1.0, E=1.0)
    animal_phase_plane(level="medium", time_span=50)


if __name__ == "__main__":
    main()
