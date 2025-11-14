'''
Each pixel in the image represents one simulated survival scenario with specific values for risk, resource availability, 
and environmental risk.
The color shows the survival probability as calculated by my model for those parameters, using the "plasma" 
colormap (see colorbar on the right)


This is AI explanation:
"The Julia Survival Fractal Pattern image visually represents the outputs of my survival probability model over 
a multidimensional parameter space. Each pixel uses the mathematics of Julia set fractals to generate test values 
for external risk, resource score, and environmental risk, which are then run through the survival model. The 
resulting survival probability is mapped to color, revealing how survival likelihood changes across these complex 
scenarios. The fractal boundaries show regions of rapid transition, where the model is especially sensitive to input changes."

Note: You cannot map the x and y axes directly to specific parameters since the Julia set generates complex stuff
'''


import numpy as np
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

import grpc
import atexit

@atexit.register
def cleanup_grpc():
    try:
        grpc._channel._shutdown_all()  # quietly close gRPC threads
    except Exception:
        pass

# surpressing the warning cuz i dont like to see it
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

countries = {
    "Done, no other countries to select": [0, 0.0],
    "Hong Kong": [1, 85.51],
    "Japan": [2, 84.71],
    "South Korea": [3, 84.33],
    "French Polynesia": [4, 84.07],
    "Andorra": [5, 84.04],
    "Switzerland": [6, 83.95],
    "Australia": [7, 83.92],
    "Singapore": [8, 83.74],
    "Italy": [9, 83.72],
    "Spain": [10, 83.67],
    "Réunion": [11, 83.55],
    "France": [12, 83.33],
    "Norway": [13, 83.31],
    "Malta": [14, 83.3],
    "Guernsey": [15, 83.27],
    "Sweden": [16, 83.26],
    "Macao": [17, 83.08],
    "United Arab Emirates": [18, 82.91],
    "Iceland": [19, 82.69],
    "Canada": [20, 82.63],
    "Martinique": [21, 82.56],
    "Israel": [22, 82.41],
    "Ireland": [23, 82.41],
    "Qatar": [24, 82.37],
    "Portugal": [25, 82.36],
    "Bermuda": [26, 82.31],
    "Luxembourg": [27, 82.23],
    "Netherlands": [28, 82.16],
    "Belgium": [29, 82.11],
    "New Zealand": [30, 82.09],
    "Guadeloupe": [31, 82.05],
    "Austria": [32, 81.96],
    "Denmark": [33, 81.93],
    "Finland": [34, 81.91],
    "Greece": [35, 81.86],
    "Puerto Rico": [36, 81.69],
    "Cyprus": [37, 81.65],
    "Slovenia": [38, 81.6],
    "Germany": [39, 81.38],
    "United Kingdom": [40, 81.3],
    "Bahrain": [41, 81.28],
    "Chile": [42, 81.17],
    "Maldives": [43, 81.04],
    "Isle of Man": [44, 81.0],
    "Costa Rica": [45, 80.8],
    "Taiwan": [46, 65.96],
    "Kuwait": [47, 80.41],
    "Cayman Islands": [48, 80.36],
    "Faroe Islands": [49, 80.18],
    "Oman": [50, 80.03],
    "Czechia": [51, 79.83],
    "Jersey": [52, 79.71],
    "Albania": [53, 79.6],
    "Panama": [54, 79.59],
    "United States": [55, 79.3],
    "Estonia": [56, 79.15],
    "New Caledonia|merged": [57, 78.77],
    "Saudi Arabia": [58, 78.73],
    "Poland": [59, 78.63],
    "Croatia": [60, 78.58],
    "Slovakia": [61, 78.34],
    "Uruguay": [62, 78.14],
    "Cuba": [63, 78.08],
    "Kosovo": [64, 78.03],
    "China": [65, 77.95],
    "Bosnia and Herzegovina": [66, 77.85],
    "Lebanon": [67, 77.82],
    "Jordan": [68, 77.81],
    "Peru": [69, 77.74],
    "Colombia": [70, 77.72],
    "Iran": [71, 77.65],
    "Antigua and Barbuda": [72, 77.6],
    "Sri Lanka": [73, 77.48],
    "Argentina": [74, 77.39],
    "North Macedonia": [75, 77.39],
    "Ecuador": [76, 77.39],
    "Guam": [77, 77.21],
    "Turkey": [78, 77.16],
    "Montenegro": [79, 77.09],
    "Hungary": [80, 77.02],
    "French Guiana": [81, 76.98],
    "Curaçao": [82, 76.8],
    "Serbia": [83, 76.77],
    "Malaysia": [84, 76.66],
    "Tunisia": [85, 76.51],
    "Thailand": [86, 76.41],
    "Aruba": [87, 76.35],
    "Algeria": [88, 76.26],
    "Latvia": [89, 76.19],
    "Barbados": [90, 76.18],
    "Cabo Verde": [91, 76.06],
    "Mayotte": [92, 76.05],
    "Lithuania": [93, 76.03],
    "Romania": [94, 75.94],
    "Brazil": [95, 75.85],
    "Armenia": [96, 75.68],
    "Bulgaria": [97, 75.64],
    "US Virgin Islands": [98, 75.47],
    "Brunei": [99, 75.33],
    "Morocco": [100, 75.31],
    "Grenada": [101, 75.2],
    "Mexico": [102, 75.07],
    "Nicaragua": [103, 74.95],
    "Mauritius": [104, 74.93],
    "Bangladesh": [105, 74.67],
    "Vietnam": [106, 74.59],
    "Bahamas": [107, 74.55],
    "Georgia": [108, 74.5],
    "Belarus": [109, 74.43],
    "Azerbaijan": [110, 74.43],
    "Kazakhstan": [111, 74.4],
    "Paraguay": [112, 73.84],
    "Dominican Republic": [113, 73.72],
    "North Korea": [114, 73.64],
    "Suriname": [115, 73.63],
    "Belize": [116, 73.57],
    "Trinidad and Tobago": [117, 73.49],
    "Ukraine": [118, 73.42],
    "Russia": [119, 73.15],
    "Bhutan": [120, 72.97],
    "Tonga": [121, 72.89],
    "Honduras": [122, 72.88],
    "Seychelles": [123, 72.86],
    "Saint Lucia": [124, 72.7],
    "Guatemala": [125, 72.6],
    "Venezuela": [126, 72.51],
    "Uzbekistan": [127, 72.39],
    "Iraq": [128, 72.32],
    "Syria": [129, 72.12],
    "El Salvador": [130, 72.1],
    "India": [131, 72.0],
    "Tajikistan": [132, 71.79],
    "Mongolia": [133, 71.73],
    "Samoa": [134, 71.7],
    "Kyrgyzstan": [135, 71.68],
    "Egypt": [136, 71.63],
    "Jamaica": [137, 71.48],
    "Vanuatu": [138, 71.48],
    "Western Sahara": [139, 71.39],
    "Saint Vincent and the Grenadines": [140, 71.23],
    "Moldova": [141, 71.2],
    "Indonesia": [142, 71.15],
    "Dominica": [143, 71.13],
    "Cambodia": [144, 70.67],
    "Solomon Islands": [145, 70.53],
    "Nepal": [146, 70.35],
    "Guyana": [147, 70.18],
    "Turkmenistan": [148, 70.07],
    "Greenland": [149, 70.06],
    "Sao Tome and Principe": [150, 69.72],
    "Libya": [151, 69.34],
    "Yemen": [152, 69.3],
    "Botswana": [153, 69.16],
    "Laos": [154, 68.96],
    "Senegal": [155, 68.68],
    "Eritrea": [156, 68.62],
    "Bolivia": [157, 68.58],
    "Mauritania": [158, 68.48],
    "Gabon": [159, 68.34],
    "Uganda": [160, 68.25],
    "Rwanda": [161, 67.78],
    "Timor-Leste": [162, 67.69],
    "Pakistan": [163, 67.65],
    "Namibia": [164, 67.39],
    "Malawi": [165, 67.35],
    "Fiji": [166, 67.32],
    "Ethiopia": [167, 67.31],
    "F.S. Micronesia": [168, 67.2],
    "Tanzania": [169, 67.0],
    "Myanmar": [170, 66.89],
    "Comoros": [171, 66.78],
    "Kiribati": [172, 66.47],
    "Zambia": [173, 66.35],
    "Sudan": [174, 66.33],
    "South Africa": [175, 66.14],
    "Papua New Guinea": [176, 66.13],
    "Afghanistan": [177, 66.03],
    "Djibouti": [178, 65.99],
    "Gambia": [179, 65.86],
    "Congo, Rep.": [180, 65.77],
    "Ghana": [181, 65.5],
    "Palestine": [182, 65.17],
    "Haiti": [183, 64.94],
    "Angola": [184, 64.62],
    "Eswatini": [185, 64.12],
    "Guinea-Bissau": [186, 64.08],
    "Equatorial Guinea": [187, 63.71],
    "Cameroon": [188, 63.7],
    "Burundi": [189, 63.65],
    "Kenya": [190, 63.65],
    "Madagascar": [191, 63.63],
    "Mozambique": [192, 63.61],
    "Zimbabwe": [193, 62.77],
    "Togo": [194, 62.74],
    "Liberia": [195, 62.16],
    "Cote d'Ivoire": [196, 61.94],
    "DR Congo": [197, 61.9],
    "Sierra Leone": [198, 61.79],
    "Niger": [199, 61.18],
    "Burkina Faso": [200, 61.09],
    "Benin": [201, 60.77],
    "Guinea": [202, 60.74],
    "Mali": [203, 60.44],
    "Somalia": [204, 58.82],
    "South Sudan": [205, 57.62],
    "CAR": [206, 57.41],
    "Lesotho": [207, 57.38],
    "Chad": [208, 55.07],
    "Nigeria": [209, 54.46],
}

# baseline hazards at age A0
baseline_hazards = {
    0: 0.000,
    5: 0.050,
    6: 0.104,
    8: 0.113,
    9: 0.237, 
    10: 0.073,
    12: 0.090,
    14: 0.108,
    15: 0.116,
    17: 0.132,
    18: 0.285,
    21: 0.185,
    26: 0.382,
    35: 0.232,
    36: 0.443,
    38: 0.279,
    48: 0.299,
    52: 0.560,
    56: 0.382,
    68: 0.421,
    72: 0.467,
    84: 0.599,
    108: 0.805
}

def age():
    global model
    curage = int(input("What is your current age? "))

    for country in countries:
        index = countries[country][0]
        print(f"{index}. {country}")

    countryindex = [-1]
    while countryindex[-1] != 0:
        choice = int(input("Select the number corresponding to your country (0 to finish): "))
        countryindex.append(choice)

    Lmax = 0.0
    numcountries = 0
    # Find corresponding Lmax values
    for idx in countryindex[1:]:
        if idx == 0:
            break
        for country, values in countries.items():
            if values[0] == idx:
                Lmax += values[1]
                numcountries += 1

    L_max = Lmax / numcountries 

    # medical conditions severity
    # LLM COMPUTES THIS
    # s = (0, 1) # pick value between these based on user inputs
    users = input("List any medical conditions you have and their severity for each one: ")
    prompt = "ONLY OUTPUT A NUMBER CORRESPONDING TO s, NO NUMBERS. Based on the following user medical conditions and their severity, compute an overall disease severity score between 0 and 1, where 0 means no conditions and 1 means extremely severe conditions: " + users
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        s = numbers_as_floats[0]
    else:
        s = 0.5
    print("s =", s)

    # activity levels
    # LLM COMPUTES THIS
    # a = (0, 1) # pick value between these based on user inputs
    usera = input("Describe your typical weekly physical activity levels: ")
    prompt = "ONLY OUTPUT A NUMBER CORRESPONDING TO a, NO NUMBERS. Based on the following user description of their typical weekly physical activity levels, compute an activity score between 0 and 1, where 0 means no activity and 1 means extremely active: " + usera
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        a = numbers_as_floats[0]
    else:
        a = 0.5
    print("a =", a)

    # WHR stuff
    w = float(input("Enter your waist to hip ratio: "))
    wopt = 0.875 # average optimal whr for all people
    delta_w = w - wopt

    # metabolic risk score
    # LLM COMPUTES THIS
    # m = (0, 1) # pick value between these based on user inputs
    userm = input("Provide any relevant metabolic health information (e.g., blood pressure, cholesterol, blood sugar levels): ")
    prompt = "ONLY OUTPUT A NUMBER CORRESPONDING TO m, NO NUMBERS. Based on the following user metabolic health information, compute a metabolic risk score between 0 and 1, where 0 means no metabolic risk and 1 means extremely high metabolic risk: " + userm
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        m = numbers_as_floats[0]
    else:
        m = 0.5
    print("m =", m)

    # LLM COMPUTES THIS
    # H0 = 0.9          # initial health index (0..1) 
    userh = input("Provide any additional information about your current health status: ")
    prompt = "ONLY OUTPUT A NUMBER CORRESPONDING TO H0, NO NUMBERS. Based on the following user overall assessment of their current health status, compute an initial health index between 0 and 1, where 0 means very poor health and 1 means excellent health: " + users + usera + userm + userh
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        H0 = numbers_as_floats[0]
    else:
        H0 = 0.9 # default
    print("H0 =", H0)

    # Parameters
    A0 = curage         # current age
    R0 = L_max - A0   # initial remaining years

    # baseline hazard at age A0 -> look it up
    # Sort keys to make sure we can find the interval
    keys = sorted(baseline_hazards.keys())

    # If below the minimum or above the maximum, clamp
    if A0 <= keys[0]:
        h0 = baseline_hazards[keys[0]]
    elif A0 >= keys[-1]:
        h0 = baseline_hazards[keys[-1]]
    else:
        # Find the two nearest keys around A0
        for i in range(len(keys) - 1):
            x1, x2 = keys[i], keys[i + 1]
            if x1 <= A0 <= x2:
                y1, y2 = baseline_hazards[x1], baseline_hazards[x2]
                # Linear interpolation formula
                h0 = y1 + (A0 - x1) * (y2 - y1) / (x2 - x1)
                break

    # Health dynamics coefficients
    # decay toward 0 if risk present; activity moves H toward 1
    # calibrate based on data
    k_s = 1 # 0.5 to 5 
    k_a = 0.5 # 0.2 to 1
    k_w = 0.25 # 0.1 to 0.5
    k_m = 0.5 # 0.2 to 1

    # Gompertz hazard params
    # calibrate based on data
    k_g = 0.06          # rate hazard increases with age

    # sensitivity of mortality to poor health
    # calibrate based on data
    k_h = 2.0

    # Define derivatives for the system y = [R, H]
    def derivatives(y, t):
        R, H = y
        age = A0 + t
        # health ODE: simple linear form pushing H down with risk, up with activity
        dH_dt = - (k_s * s + k_w * delta_w + k_m * m) * H + k_a * a * (1 - H)
        # Gompertz hazard (age-dependent)
        hazard = h0 * np.exp(k_g * (age - A0))
        # R ODE: proportional to R times hazard, amplified when H is low
        dR_dt = - R * hazard * (1.0 + k_h * (1.0 - H))
        return np.array([dR_dt, dH_dt])

    # RK4 integrator for systems
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
            y[i,1] = np.clip(y[i,1], 0.0, 1.0) # keep H within [0,1] and R nonnegative
            y[i,0] = max(y[i,0], 0.0)
        return y

    # Run simulation
    t_end = 80.0  # years forward to simulate (e.g., up to age A0 + 80)
    n_steps = 4000
    t = np.linspace(0.0, t_end, n_steps)

    y0 = np.array([R0, H0])
    sol = rk4_system(derivatives, y0, t)
    R_sol = sol[:,0]
    H_sol = sol[:,1]
    age = A0 + t

    # Remaining life expectancy estimate: when R approaches zero
    final_R = R_sol[-1]
    predicted_lifespan = A0 + R0 - final_R  # or simply A0 + t_end if you want max simulated age

    # Print results
    print("\n--- Simulation Results ---")
    # print(f"Predicted remaining years of life: {R_sol[-1]:.2f}")
    print(f"Predicted remaining years of life: {(predicted_lifespan-A0):.2f}")
    print(f"Predicted total lifespan: {predicted_lifespan:.2f}")
    # print(f"Final health index at age {age[-1]:.1f}: {H_sol[-1]:.2f}")

    return [A0, predicted_lifespan-A0]

class HumanThreatModel:
    def __init__(self):
        # Removed gender and ethnicity multipliers since we're getting them via user input
        pass

    def get_homicide_rate(self, country):
        """Get homicide rate per 100,000 for a country"""
        print("\nPlease go to: https://data.worldbank.org/indicator/VC.IHR.PSRC.P5")
        print(f"Find the homicide rate for {country} (per 100,000 people)")
        value = float(input("Enter the homicide rate: "))
        return value

    def human_threat_ode(self, V, t, lambda_base):
        """ODE for cumulative probability of being a homicide victim"""
        return lambda_base * (1 - V)

    def runge_kutta_4(self, f, y0, t_span, args, n_steps=1000):
        """4th Order Runge-Kutta method"""
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = y0

        for i in range(n_steps):
            k1 = h * f(y[i], t[i], *args)
            k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h, *args)
            k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h, *args)
            k4 = h * f(y[i] + k3, t[i] + h, *args)
            y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t, y

    def get_user_multipliers(self):
        """Get gender and ethnicity multipliers from user input"""
        print("\n=== Gender Multiplier ===")
        G = float(input("Enter one of the following numbers for gender:\n"
                        "if female: 1\nif male: 4\nif other or non-conforming: 5\n"
                        "Your choice: "))

        print("\n=== Ethnicity Multiplier ===")
        E = float(input("Enter one of the following numbers for ethnicity:\n"
                        "if black: 10\nif white: 8.75\nif hispanic: 5.5\nif other: 4\n"
                        "Your choice: "))

        return G, E

    def calculate_human_threat(self, years, country, n_steps=1000):      # i think done ################################################################################### Connect to years to R pls
        """
        Calculate cumulative homicide probability using Runge-Kutta
        """

        # Get user multipliers
        G, E = self.get_user_multipliers()

        # Get homicide rate
        H = self.get_homicide_rate(country)
        lambda_base = (H / 100000) * G * E  # Combine all multipliers

        # Solve ODE using Runge-Kutta
        t, V = self.runge_kutta_4(
            self.human_threat_ode,
            y0=0.0,  # Initial V(0) = 0
            t_span=(0, years),
            args=(lambda_base,),  # Only pass lambda_base now
            n_steps=n_steps
        )

        return t, V, country

class AnimalThreatModel:
    def __init__(self):
        # FIXED parameters
        self.animal_parameters = {
            'low': {'a': 0.3, 'b': 0.2, 'c': 0.5, 'd': 0.05},
            'medium': {'a': 1.0, 'b': 0.3, 'c': 0.2, 'd': 0.3},
            'high': {'a': 2.0, 'b': 0.2, 'c': 0.08, 'd': 1.2}
        }

        # Pre-calculate maximum threat levels for each predator type
        self.max_threat_levels = self._calculate_max_threat_levels()

    def _calculate_max_threat_levels(self):
        """Calculate maximum possible threat levels for each predator type"""
        max_levels = {}
        for level, params in self.animal_parameters.items():
            a, b, c, d = params['a'], params['b'], params['c'], params['d']
            max_threat = self._find_actual_max_threat(a, b, c, d)
            max_levels[level] = max_threat
        return max_levels

    def _find_actual_max_threat(self, a, b, c, d, time_span=200, n_steps=2000):
        """Find the actual maximum threat level by running a long simulation"""
        y0 = np.array([1.0, 0.1])

        t, y = self.runge_kutta_4_system(
            self.animal_threat_ode,
            y0=y0,
            t_span=(0, time_span),
            args=(a, b, c, d),
            n_steps=n_steps
        )

        return np.max(y[1])

    def animal_threat_ode(self, variables, t, a, b, c, d):
        """Lotka-Volterra ODE system for animal threat"""
        x, y = variables  # x: survival probability, y: animal threat level
        dxdt = a * x - b * x * y
        dydt = -c * y + d * x * y
        return np.array([dxdt, dydt])

    def runge_kutta_4_system(self, f, y0, t_span, args, n_steps=1000):
        """4th Order Runge-Kutta for systems of ODEs"""
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((2, n_steps + 1))
        y[:, 0] = y0

        for i in range(n_steps):
            k1 = h * f(y[:, i], t[i], *args)
            k2 = h * f(y[:, i] + 0.5 * k1, t[i] + 0.5 * h, *args)
            k3 = h * f(y[:, i] + 0.5 * k2, t[i] + 0.5 * h, *args)
            k4 = h * f(y[:, i] + k3, t[i] + h, *args)
            y[:, i + 1] = y[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t, y

    def get_predator_level_from_user(self):
        """Get predator level from user input"""
        print("\n=== Animal Threat Assessment ===")
        print("Choose the predator intensity in your environment:")
        print("1. Low - Small animals (squirrels, rabbits)")
        print("2. Medium - Medium predators (foxes, coyotes)")
        print("3. High - Large predators (wolves, bears)")

        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice == '1':
                return 'low'
            elif choice == '2':
                return 'medium'
            elif choice == '3':
                return 'high'
            else:
                print("Invalid choice! Please enter 1, 2, or 3")

    def calculate_animal_threat(self, time_span, predator_level=None, initial_survival=1.0, n_steps=1000): # i think done ########################################################## Connect to time_span to R pls
        """
        Calculate animal threat seriousness using Runge-Kutta
        """
        # Get predator level from user if not provided
        if predator_level is None:
            predator_level = self.get_predator_level_from_user()

        # Set initial threat based on predator level
        if predator_level == 'low':
            initial_threat = 0.01
        elif predator_level == 'medium':
            initial_threat = 0.1
        else:  # high
            initial_threat = 0.3

        # Get parameters and max threat level
        params = self.animal_parameters[predator_level]
        max_threat = self.max_threat_levels[predator_level]
        a, b, c, d = params['a'], params['b'], params['c'], params['d']

        # Initial conditions
        y0 = np.array([initial_survival, initial_threat])

        # Solve ODE system
        t, y = self.runge_kutta_4_system(
            self.animal_threat_ode,
            y0=y0,
            t_span=(0, time_span),
            args=(a, b, c, d),
            n_steps=n_steps
        )

        threat_levels = y[1]
        avg_threat = np.mean(threat_levels)

        # Use predator-specific maximums for scaling, but map to the appropriate third
        ratio = avg_threat / max_threat if max_threat > 0 else 0.0

        # Map each predator type to its designated third of the seriousness scale
        if predator_level == 'low':
            # Low predators: 0-33.3% range
            seriousness_score = ratio * 0.333
        elif predator_level == 'medium':
            # Medium predators: 33.3-66.7% range
            seriousness_score = 0.333 + (ratio * 0.334)
        else:  # high
            # High predators: 66.7-100% range
            seriousness_score = 0.667 + (ratio * 0.333)

        # Ensure we stay within bounds
        seriousness_score = min(1.0, max(0.0, seriousness_score))

        return seriousness_score, predator_level, initial_threat, max_threat, avg_threat


def analyze_external_threats(years, country): # i think done ################################################################################################################################################ Connect years to R pls
    """Combine both threat models for comprehensive analysis"""

    human_model = HumanThreatModel()
    animal_model = AnimalThreatModel()

    # Human threat analysis - now handles its own user input
    human_time, human_risk, country = human_model.calculate_human_threat(years=years, country=country)

    # Animal threat analysis - now handles its own user input and returns seriousness score
    animal_seriousness, animal_level, initial_threat, max_threat, avg_threat = animal_model.calculate_animal_threat(
        time_span=years)

    # Convert animal seriousness to risk (seriousness score is already 0-1 scale representing threat level)
    animal_risk = animal_seriousness

    # Combine results - weighted combination (you can adjust weights as needed)
    # Human risk is probability of homicide, animal risk is threat seriousness
    combined_risk = 0.5 * human_risk[-1] + 0.5 * animal_risk

    return {
        'country': country,
        'human_risk': human_risk[-1],  # Final cumulative probability
        'animal_risk': animal_risk,
        'animal_level': animal_level,
        'combined_risk': combined_risk,
        'time_points': human_time,
        'human_risk_over_time': human_risk
    }

def resource_availability():
    """Get resource availability scores from user input"""
    print("\n=== Resource Availability Assessment ===")
    print("Please answer the following questions about your resources:\n")

    scores = {}

    # Water availability
    print("1. Do you have drinkable water available to you?")
    print("   [1] Always (7 days a week)")
    print("   [2] Most days (5-6 days a week)")
    print("   [3] Some days (4 days a week)")
    print("   [4] Rarely (1-3 days a week)")
    print("   [5] Barely ever")

    water_choice = input("   Enter your choice (1-5): ").strip()
    if water_choice == '1':
        scores['water'] = 1.0
    elif water_choice == '2':
        scores['water'] = 0.5
    elif water_choice == '3':
        scores['water'] = 0.0
    elif water_choice == '4':
        scores['water'] = -0.5
    elif water_choice == '5':
        scores['water'] = -1.0
    else:
        print("   Invalid choice, defaulting to 'Always'")
        scores['water'] = 1.0

    # Food availability
    print("\n2. Do you have food available to you?")
    print("   [1] Always (7 days a week)")
    print("   [2] Most days (5-6 days a week)")
    print("   [3] Some days (4 days a week)")
    print("   [4] Rarely (1-3 days a week)")
    print("   [5] Barely ever")

    food_choice = input("   Enter your choice (1-5): ").strip()
    if food_choice == '1':
        scores['food'] = 1.0
    elif food_choice == '2':
        scores['food'] = 0.5
    elif food_choice == '3':
        scores['food'] = 0.0
    elif food_choice == '4':
        scores['food'] = -0.5
    elif food_choice == '5':
        scores['food'] = -1.0
    else:
        print("   Invalid choice, defaulting to 'Always'")
        scores['food'] = 1.0

    # Medical availability
    print("\n3. Do you have medical attention/hospitals near you?")
    print("   [1] Very close (< 10 km)")
    print("   [2] Close (10-30 km)")
    print("   [3] Moderate distance (30-50 km)")
    print("   [4] Far (50-100 km)")
    print("   [5] Very far (> 100 km)")

    medical_choice = input("   Enter your choice (1-5): ").strip()
    if medical_choice == '1':
        scores['medical'] = 1.0
    elif medical_choice == '2':
        scores['medical'] = 0.5
    elif medical_choice == '3':
        scores['medical'] = 0.0
    elif medical_choice == '4':
        scores['medical'] = -0.5
    elif medical_choice == '5':
        scores['medical'] = -1.0
    else:
        print("   Invalid choice, defaulting to 'Very close'")
        scores['medical'] = 1.0

    # Shelter availability
    print("\n4. Do you have shelter available (anywhere to sleep)?")
    print("   [1] Always (7 days a week)")
    print("   [2] Most days (5-6 days a week)")
    print("   [3] Some days (4 days a week)")
    print("   [4] Rarely (1-3 days a week)")
    print("   [5] Barely ever")

    shelter_choice = input("   Enter your choice (1-5): ").strip()
    if shelter_choice == '1':
        scores['shelter'] = 1.0
    elif shelter_choice == '2':
        scores['shelter'] = 0.5
    elif shelter_choice == '3':
        scores['shelter'] = 0.0
    elif shelter_choice == '4':
        scores['shelter'] = -0.5
    elif shelter_choice == '5':
        scores['shelter'] = -1.0
    else:
        print("   Invalid choice, defaulting to 'Always'")
        scores['shelter'] = 1.0

    # Economic situation
    print("\n5. How's your economic situation?")
    print("   [1] Comfortable")
    print("   [2] Tight")
    print("   [3] Difficult")

    economic_choice = input("   Enter your choice (1-3): ").strip()
    if economic_choice == '1':
        scores['economic'] = 1.0
    elif economic_choice == '2':
        scores['economic'] = 0.0
    elif economic_choice == '3':
        scores['economic'] = -1.0
    else:
        print("   Invalid choice, defaulting to 'Comfortable'")
        scores['economic'] = 1.0

    # Calculate overall score
    scores['overall'] = sum(scores.values()) / len(scores)

    return scores

class EnvironmentalRiskModel:
    global model
    def calculate_normalized_risk(self, country, alpha, beta, gamma, delta1, natural_disaster, delta2, temp_difference, delta3,
                                  drought_risk, delta4, population_density, Y0):
        lambda_prev = Y0
        # Stabilize through iterations
        for _ in range(10):
            lambda_current = (alpha +
                              beta * Y0 +
                              gamma * lambda_prev +
                              delta1 * natural_disaster +
                              delta2 * temp_difference +
                              delta3 * drought_risk +
                              delta4 * population_density)
            lambda_prev = lambda_current

        # Calculate maximum possible risk (worst-case scenario)
        max_alpha = 0.5
        max_beta = 0.8
        max_gamma = 0.4
        max_delta = 0.01

        # Worst-case inputs (all at maximum risk)
        max_natural_disaster = 1.0  # 100% disaster risk
        max_temp_difference = 5.0  # Maximum temperature anomaly
        max_drought_risk = 5.0  # Maximum drought risk score (0-5 scale)
        max_population_density = 22000.0  # Macau's density (~22,000 people/km²) --> Highest population density source: wikipedia
        
        # LLM COMPUTES THIS
        # max_Y0 = 10.0  # High initial disaster frequency ############################################################################################################################### Ask AI to get this prompt below, change that 10 to the actual number
        prompt =  ''' For max_Y0
        Estimate the highest monthly frequency of natural disasters in {country}. 
        Consider disasters like: earthquakes, floods, hurricanes, wildfires.
        Return ONLY a single decimal number representing the highest amount of events in one month.
        Base this on historical data (such as the last 50 years) and geographical risk factors.
        '''
        response = model.generate_content(prompt)
        text_output = response.text.strip()
        numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
        numbers_as_floats = [float(s) for s in numbers_as_strings]
        if numbers_as_floats:
            max_Y0 = numbers_as_floats[0]
        else:
            max_Y0 = 10.0 # default
        print("max_Y0 =", max_Y0)

        # Calculate maximum possible risk
        lambda_prev_max = max_Y0
        for _ in range(10):
            lambda_current_max = (max_alpha +
                                  max_beta * max_Y0 +
                                  max_gamma * lambda_prev_max +
                                  max_delta * max_natural_disaster +
                                  max_delta * max_temp_difference +
                                  max_delta * max_drought_risk +
                                  max_delta * max_population_density)
            lambda_prev_max = lambda_current_max

        # Normalize to 0-1 range
        normalized_risk = max(0, min(1, lambda_current / lambda_current_max))

        return normalized_risk

def get_environmental_risk_inputs(country):
    """Get all environmental risk inputs from user and calculate normalized risk (0-1)"""
    print("=== ENVIRONMENTAL RISK ASSESSMENT ===")

    print(f"\nFor {country}, please provide the following risk factors:")
    natural_disaster = float(input("Natural disaster risk at https://worldpopulationreview.com/country-rankings/natural-disaster-risk-by-country (0 to 100): ")) / 100
    temp_difference = float(input("Water temperature anomaly (or closest to it) at https://www.ospo.noaa.gov/products/ocean/cb/sst5km/?product=ssta (-5 to 5): "))
    drought_risk = float(input("Drought risk score at https://worldpopulationreview.com/country-rankings/droughts-by-country (0 to 5): "))
    population_density = float(input("Population density per km^2 (first column) at https://worldpopulationreview.com/country-rankings/countries-by-density: "))

    ############################################################################################################################################################################################################### Get AI to pick alpha, beta, gamma, delta1, delta2, delta3, delta4, and Y0. Later delete the section that uses user input here (it's just for testing)
    # LLM COMPUTES THIS
    # Y0         
    prompt = f"""
            Estimate the average monthly frequency of natural disasters in {country}. 
            Consider disasters like: earthquakes, floods, hurricanes, wildfires.
            Return ONLY a single decimal number representing average events per month.
            Base this on historical data and geographical risk factors.

            Examples:
            Japan (high seismic+typhoon risk): 1.8-2.2
            Philippines (high typhoon risk): 2.0-2.5  
            USA (varied regional risks): 1.2-1.6
            UK (low disaster risk): 0.3-0.6

            Country: {country}
            Answer: """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        Y0 = numbers_as_floats[0]
    else:
        Y0 = 3 # default
    print("Y0 =", Y0)

    # LLM COMPUTES THIS
    # alpha         
    prompt = f"""
            Pick numerical value for the following parameters that work in the following {country}: Alpha between 0.1 and 0.5 
            and it's the baseline safety (i.e. safety when nothing else is happening).
            Return ONLY a single decimal number representing alpha.
            Base this on historical data and geographical risk factors.
            Country: {country}
            """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        alpha = numbers_as_floats[0]
    else:
        alpha = 0.3 # default
    print("alpha =", alpha)

    # LLM COMPUTES THIS
    # beta         
    prompt = f"""
            Pick numerical value for the following parameters that work in the following {country}: 
            for beta between 0.6 and 0.8 and it's recent disasters event (i.e. how much last month’s disasters affect this month’s).
            Return ONLY a single decimal number representing beta.
            Base this on historical data and geographical risk factors.
            Country: {country}
            """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        beta = numbers_as_floats[0]
    else:
        beta = 0.7 # default
    print("beta =", beta)

    # LLM COMPUTES THIS
    # gamma         
    prompt = f"""
            Pick numerical value for the following parameters that work in the following {country}: 
            for gamma between 0.2 and 0.4 and it's persistent risk memory (i.e. how much the risk persists over time)
            Return ONLY a single decimal number representing gamma.
            Base this on historical data and geographical risk factors.
            Country: {country}
            """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        gamma = numbers_as_floats[0]
    else:
        gamma = 0.3 # default
    print("gamma =", gamma)

    # LLM COMPUTES THIS
    # delta1         
    prompt = f"""
            Pick numerical value for the following parameters that work in the following {country}: 
            for all deltas between 0.001 - 0.01, which are for how external factors affect and they are: 
            delta1 --> natural disaster risk factor per country, 
            Return ONLY a single decimal number representing delta1.
            Base this on historical data and geographical risk factors.
            Country: {country}
            """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        delta1 = numbers_as_floats[0]
    else:
        delta1 = 0.0055 # default for all deltas
    print("delta1 =", delta1)

    # LLM COMPUTES THIS
    # delta2         
    prompt = f"""
            Pick numerical value for the following parameters that work in the following {country}: 
            for all deltas between 0.001 - 0.01, which are for how external factors affect and they are: 
            delta2 --> the difference between today's temperature and the long term average for ocean, 
            Return ONLY a single decimal number representing delta2.
            Base this on historical data and geographical risk factors.
            Country: {country}
            """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        delta2 = numbers_as_floats[0]
    else:
        delta2 = 0.0055 # default for all deltas
    print("delta2 =", delta2)

    # LLM COMPUTES THIS
    # delta3         
    prompt = f"""
            Pick numerical value for the following parameters that work in the following {country}: 
            for all deltas between 0.001 - 0.01, which are for how external factors affect and they are: 
            delta3 --> the drought risk score for {country}
            Return ONLY a single decimal number representing delta3.
            Base this on historical data and geographical risk factors.
            """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        delta3 = numbers_as_floats[0]
    else:
        delta3 = 0.0055 # default for all deltas
    print("delta3 =", delta3)

    # LLM COMPUTES THIS
    # delta4         
    prompt = f"""
            Pick numerical value for the following parameters that work in the following {country}: 
            for all deltas between 0.001 - 0.01, which are for how external factors affect and they are: 
            delta4 --> population density of country: {country}. 
            Return ONLY a single decimal number representing delta4.
            Base this on historical data and geographical risk factors.
            """
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        delta4 = numbers_as_floats[0]
    else:
        delta4 = 0.0055 # default for all deltas
    print("delta4 =", delta4)

    # '''
    # Prompts (2 prompts):  ********************************************************************************************Note that both prompts you need to input the country
    # 1) For y0 
    # prompt = f"""
    #     Estimate the average monthly frequency of natural disasters in {country}. 
    #     Consider disasters like: earthquakes, floods, hurricanes, wildfires.
    #     Return ONLY a single decimal number representing average events per month.
    #     Base this on historical data and geographical risk factors.

    #     Examples:
    #     Japan (high seismic+typhoon risk): 1.8-2.2
    #     Philippines (high typhoon risk): 2.0-2.5  
    #     USA (varied regional risks): 1.2-1.6
    #     UK (low disaster risk): 0.3-0.6

    #     Country: {country}
    #     Answer: """
    #     ############### Last two lines to but the answer Ig, I don't really know how that works

    # 2) For the parameters (alpha, beta, gamma, and all deltas) ***********************************************************note that country is the selected country
    # Pick numerical values for the following parameters that work in the following {country}: Alpha between 0.1 and 0.5 
    # and it's the baseline safety (i.e. safety when nothing else is happening), for beta between 0.6 and 0.8 and it's 
    # recent disasters event (i.e. how much last month’s disasters affect this month’s) for gamma between 0.2 and 0.4 and 
    # it's persistent risk memory (i.e. how much the risk persists over time) for all deltas between 0.001 - 0.01, which 
    # are for how external factors affect and they are: delta1 --> natural disaster risk factor per country, delta2 --> the 
    # difference between today’s temperature and the long term average for ocean, delta3 --> the drought risk score, 
    # delta4 --> population density per country. This is all to model natural disaster risk
    # '''

    # print(f"\nFor {country}, please provide the model parameters:")
    # alpha = float(input("Alpha (baseline safety 0.1-0.5): "))
    # beta = float(input("Beta (recent disasters effect 0.6-0.8): "))
    # gamma = float(input("Gamma (persistent risk memory 0.2-0.4): "))
    # delta1 = float(input("Delta1 (natural disaster weight 0.001-0.01): "))
    # delta2 = float(input("Delta2 (temperature weight 0.001-0.01): "))
    # delta3 = float(input("Delta3 (drought weight 0.001-0.01): "))
    # delta4 = float(input("Delta4 (population density weight 0.001-0.01): "))
    # Y0 = float(input("Initial disaster frequency (Y0): "))

        # Calculate normalized risk — do NOT shadow the LLM client name `model`
    env_model = EnvironmentalRiskModel()
    normalized_risk = env_model.calculate_normalized_risk(
        country, alpha, beta, gamma,
        delta1, natural_disaster, delta2, temp_difference,
        delta3, drought_risk, delta4, population_density, Y0
    )


    # Return both the normalized risk and all inputs for reference
    return {
        'environmental_risk': normalized_risk,  # Between 0 and 1
        'country': country,
        'inputs': {
            'natural_disaster': natural_disaster,
            'temp_difference': temp_difference,
            'drought_risk': drought_risk,
            'population_density': population_density,
            'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'delta1': delta1, 'delta2': delta2, 'delta3': delta3, 'delta4': delta4,
            'Y0': Y0
        }
    }


def calculate_total_survival(external_combined_risk, resource_scores, environmental_risk,
                             weights=[0.25, 0.375, 0.375], risk_threshold=0.5, k=8):
    """
    Calculate normalized total survival probability (0-1)
    external_combined_risk: combined human + animal risk from analyze_external_threats() (0-1)
    resource_scores: dict with water, food, medical, shelter, economic scores (-1 to +1)
    environmental_risk: normalized risk from EnvironmentalRiskModel (0-1, where 0=safe, 1=dangerous)
    weights: [external_threats_weight, resource_weight, environment_weight]
    risk_threshold: environmental risk level where survival probability drops to 50%
    k: steepness of the environmental risk transition
    """

    # 1. External threats survival (inverse of combined human + animal risk)
    S_external = 1 - external_combined_risk

    # 2. Resource availability (normalize -5 to +5 → 0 to 1)
    resource_keys = ['water', 'food', 'medical', 'shelter', 'economic']
    resource_values = [resource_scores[key] for key in resource_keys if key in resource_scores]
    resource_sum = sum(resource_values)
    S_resources = max(0, min(1, (resource_sum + 5) / 10))

    # 3. Environmental risk survival
    S_environment = 1 / (1 + np.exp(k * (environmental_risk - risk_threshold)))

    # 4. Weighted SUM (additive instead of multiplicative)
    total_survival = (weights[0] * S_external +
                      weights[1] * S_resources +
                      weights[2] * S_environment)

    # Ensure result is between 0 and 1
    return max(0, min(1, total_survival))


# Complete analysis function
def run_complete_survival_analysis(REMAINING):
    """
    Complete survival analysis integrating all models
    """
    print("=== COMPREHENSIVE SURVIVAL PROBABILITY ANALYSIS ===\n")

    # 0. Get country
    country = input("\nEnter your country: ")

    # 1. Get combined external threats (human + animal)
    print("Step 1: External Threat Assessment")
    external_results = analyze_external_threats(years=REMAINING, country=country) ###################################################################################################################################### Update years to R pls
    combined_external_risk = external_results['combined_risk']  # Use the combined risk

    # 2. Get resource availability
    print("\nStep 2: Resource Availability Assessment")
    resource_scores = resource_availability()

    # 3. Get environmental risk
    print("\nStep 3: Environmental Risk Assessment")
    environmental_data = get_environmental_risk_inputs(country)
    environmental_risk = environmental_data['environmental_risk']

    # 4. Calculate total survival probability
    total_survival = calculate_total_survival(
        external_combined_risk=combined_external_risk,
        resource_scores=resource_scores,
        environmental_risk=environmental_risk
    )

    # 5. Display comprehensive results
    print(f"\n=== FINAL SURVIVAL PROBABILITY RESULTS ===")
    print(f"Country: {external_results['country']}")
    print(f"Combined External Threat Risk: {combined_external_risk:.4f} ({combined_external_risk * 100:.1f}%)")
    print(f"  - Human Threat: {external_results['human_risk']:.4f}")
    print(f"  - Animal Threat ({external_results['animal_level']}): {external_results['animal_risk']:.4f}")
    print(f"Resource Availability Score: {resource_scores['overall']:.3f}")
    print(f"Environmental Risk: {environmental_risk:.3f}")
    print(f"---")
    print(f"TOTAL SURVIVAL PROBABILITY: {total_survival:.4f} ({total_survival * 100:.1f}%)")

    return {
        'total_survival': total_survival,
        'external_combined_risk': combined_external_risk,
        'human_risk': external_results['human_risk'],
        'animal_risk': external_results['animal_risk'],
        'resource_score': resource_scores['overall'],
        'environmental_risk': environmental_risk,
        'country': external_results['country']
    }

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


# -- Julia set config --
N = 600
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.5, 1.5, N)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
# Julia constant
c = -0.8 + 0.156j

img = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        z = Z[i,j]
        steps = 0
        # Standard Julia set iteration
        while abs(z) < 2 and steps < 60:
            z = z**2 + c
            steps += 1
        # Use fractal values to 'simulate': escape speed ~ risk, etc
        # Here’s an example mapping:
        risk = min(1, steps / 60)                # risk grows with slower escape
        resource = np.cos(z.real) * 2 + 2        # use real part for variety, scale to roughly -2..+6
        environment = abs(np.sin(z.imag))        # use imag part (0..1)
        img[i,j] = calculate_total_survival(risk, resource, environment)

plt.imshow(img, cmap="plasma", extent=[-1.5,1.5,-1.5,1.5])
plt.title("Julia Survival Fractal Pattern (Color = Model Survival Probability)")
plt.colorbar(label='Survival Probability')
plt.axis('off')
plt.show()