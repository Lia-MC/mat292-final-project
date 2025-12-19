import numpy as np
import matplotlib.pyplot as plt
import _5_model as model


def plot_health_phase_with_fit(user_inputs, poly_degree=3):

    # run model
    indiv = model.run_individual_model(user_inputs)
    R = np.asarray(indiv["R_t"])
    H = np.asarray(indiv["H_t"])

    # sort for smooth curve
    idx = np.argsort(R)
    R = R[idx]
    H = H[idx]

    # polynomial fit
    coeffs = np.polyfit(R, H, deg=poly_degree)
    R_fit = np.linspace(R.min(), R.max(), 500)
    H_fit = np.polyval(coeffs, R_fit)

    # plot
    plt.figure(figsize=(8, 6))

    plt.scatter(
        R, H,
        color="tab:blue",
        s=40,
        label="Data"
    )

    plt.plot(
        R_fit, H_fit,
        color="tab:orange", 
        linewidth=2.5,
        label="Fit"
    )

    plt.xlabel("R (remaining years)", fontsize=14)
    plt.ylabel("H (health index)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    user_inputs = {
        "age": 29,
        "country": "Canada",
        "s": 0.3,
        "a": 0.6,
        "m": 0.4,
        "delta_w": 0.02,
        "H0": 0.9,
    }

    plot_health_phase_with_fit(user_inputs, poly_degree=3)


if __name__ == "__main__":
    main()
