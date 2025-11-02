import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters (example)
# -------------------------
A0 = 30.0         # current age
L_max = 85.0      # maximum lifespan estimate
R0 = L_max - A0   # initial remaining years
H0 = 0.9          # initial health index (0..1)

# Risk factors (example)
s = 0.3           # disease severity (0..1)
delta_w = 0.12    # waist-to-hip difference
m = 0.25          # metabolic risk (0..1)
a = 0.6           # activity (0..1)

# Health dynamics coefficients
beta_s = 0.8
beta_w = 0.4
beta_m = 0.5
beta_a = 0.9
# decay toward 0 if risk present; activity moves H toward 1.

# Gompertz hazard params
h0 = 0.005        # baseline hazard at age A0
g = 0.06          # rate hazard increases with age

# sensitivity of mortality to poor health
eta = 2.0

# -------------------------
# Define derivatives for the system y = [R, H]
# -------------------------
def derivatives(y, t):
    R, H = y
    age = A0 + t
    # health ODE: simple linear form pushing H down with risk, up with activity
    dH_dt = - (beta_s * s + beta_w * delta_w + beta_m * m) * H + beta_a * a * (1 - H)
    # Gompertz hazard (age-dependent)
    hazard = h0 * np.exp(g * (age - A0))
    # R ODE: proportional to R times hazard, amplified when H is low
    dR_dt = - R * hazard * (1.0 + eta * (1.0 - H))
    return np.array([dR_dt, dH_dt])

# -------------------------
# RK4 integrator for systems
# -------------------------
def rk4_system(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0,:] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = f(y[i-1,:], t[i-1])
        k2 = f(y[i-1,:] + 0.5*dt*k1, t[i-1] + 0.5*dt)
        k3 = f(y[i-1,:] + 0.5*dt*k2, t[i-1] + 0.5*dt)
        k4 = f(y[i-1,:] + dt*k3, t[i-1] + dt)
        y[i,:] = y[i-1,:] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # keep H within [0,1] and R nonnegative
        y[i,1] = np.clip(y[i,1], 0.0, 1.0)
        y[i,0] = max(y[i,0], 0.0)
    return y

# -------------------------
# Run simulation
# -------------------------
t_end = 80.0  # years forward to simulate (e.g., up to age A0 + 80)
n_steps = 4000
t = np.linspace(0.0, t_end, n_steps)

y0 = np.array([R0, H0])
sol = rk4_system(derivatives, y0, t)
R_sol = sol[:,0]
H_sol = sol[:,1]
age = A0 + t

# -------------------------
# Plot results
# -------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(age, R_sol)
plt.xlabel("Age (years)")
plt.ylabel("Remaining years R(t)")
plt.title("Remaining Years")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(age, H_sol)
plt.xlabel("Age (years)")
plt.ylabel("Health index H(t)")
plt.title("Health Index")
plt.ylim(-0.05,1.05)
plt.grid(True)

plt.tight_layout()
plt.show()

# use rk4 to compute r(t)
def ageeqtn(r, t):
    drdt = -1
    return drdt

# Python program to implement Runge Kutta method
# A sample differential equation "dy / dx = (x - y)/2"
def dydx(x, y):
    return ((x - y)/2)

# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rungeKutta(x0, y0, x, h):
    # Count number of iterations using step size or
    # step height h
    n = (int)((x - x0)/h) 
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)

        # Update next value of y
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

        # Update next value of x
        x0 = x0 + h
    return y

# Driver method
x0 = 0
y = 1
x = 2
h = 0.2
print ('The value of y at x is:', rungeKutta(x0, y, x, h))

# This code is contributed by Prateek Bhindwar