import numpy as np
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# the LLM model!!!
model = genai.GenerativeModel('gemini-2.5-flash')

import grpc
import atexit

# surpressing the warning aka bug fix
@atexit.register
def cleanup_grpc():
    try:
        grpc._channel._shutdown_all()
    except Exception:
        pass

# surpressing the warning aka bug fix (again)
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
    # find corresponding Lmax values for each countries
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
        s = max(0, min(s, 1))
    else:
        s = 0.5 # default
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
        a = max(0, min(a, 1))
    else:
        a = 0.5 # default
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
        m = max(0, min(m, 1))
    else:
        m = 0.5 # default
    print("m =", m)

    # initial health index 
    # LLM COMPUTES THIS
    # H0 = 0.9          
    # h0 (0, 1) # pick value between these based on user inputs
    userh = input("Provide any additional information about your current health status: ")
    prompt = "ONLY OUTPUT A NUMBER CORRESPONDING TO H0, NO NUMBERS. Based on the following user overall assessment of their current health status, compute an initial health index between 0 and 1, where 0 means very poor health and 1 means excellent health: " + users + usera + userm + userh
    response = model.generate_content(prompt)
    text_output = response.text.strip()
    numbers_as_strings = re.findall(r'\d+\.?\d*', text_output)
    numbers_as_floats = [float(s) for s in numbers_as_strings]
    if numbers_as_floats:
        H0 = numbers_as_floats[0]
        H0 = max(0, min(H0, 1))
    else:
        H0 = 0.9 # default
    print("H0 =", H0)

    # other parameters
    A0 = curage # current age
    R0 = L_max - A0 # initial remaining years

    # sort keys to make sure we can find interval
    keys = sorted(baseline_hazards.keys())

    # quick algorithm to find h0 at age A0 via linear interpolation
    if A0 <= keys[0]:
        h0 = baseline_hazards[keys[0]]
    elif A0 >= keys[-1]:
        h0 = baseline_hazards[keys[-1]]
    else:
        for i in range(len(keys) - 1):
            x1, x2 = keys[i], keys[i + 1]
            if x1 <= A0 <= x2:
                y1, y2 = baseline_hazards[x1], baseline_hazards[x2]
                # linear interpolation formula:
                h0 = y1 + (A0 - x1) * (y2 - y1) / (x2 - x1)
                break

    # health dynamics coefficients
    # calibrated based on data
    k_s = 1 # 0.5 to 5 
    k_a = 0.5 # 0.2 to 1
    k_w = 0.25 # 0.1 to 0.5
    k_m = 0.5 # 0.2 to 1

    # gompertz hazard params
    # calibrated based on data
    k_g = 0.06 # because rate hazard increases with age

    # sensitivity of mortality to poor health
    # calibrated based on data
    k_h = 2.0

    # define derivatives for the system y = [R, H]
    def derivatives(y, t):
        R, H = y
        age = A0 + t
        # health ODE
        dH_dt = - (k_s * s + k_w * delta_w + k_m * m) * H + k_a * a * (1 - H)
        # gompertz hazard equation
        hazard = h0 * np.exp(k_g * (age - A0))
        # R ODE
        dR_dt = - R * hazard * (1.0 + k_h * (1.0 - H))
        return np.array([dR_dt, dH_dt])

    # RK4 algorithm for system of ODEs
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
            y[i,1] = np.clip(y[i,1], 0.0, 1.0)
            y[i,0] = max(y[i,0], 0.0)
        return y

    # run simulation
    t_end = L_max # simulate up to max age user can be (based on earlier calculations of L_max)
    n_steps = 4000
    t = np.linspace(0.0, t_end, n_steps)

    y0 = np.array([R0, H0])
    sol = rk4_system(derivatives, y0, t)
    R_sol = sol[:,0]
    H_sol = sol[:,1]
    age = A0 + t

    # remaining life expectancy estimate: when R approaches zero
    final_R = max(0, R_sol[-1])
    predicted_lifespan = A0 + R0 - final_R

    # RESULTS (commented out some of the less relevant ones)
    print("\nSimulation Results!!!")
    # print(f"Predicted remaining years of life: {R_sol[-1]:.2f}")
    print(f"Predicted remaining years of life: {(final_R):.2f}")
    print(f"Predicted total lifespan: {predicted_lifespan:.2f}")
    # print(f"Final health index at age {age[-1]:.1f}: {H_sol[-1]:.2f}")

    return [A0, predicted_lifespan-A0]

class HumanThreatModel:
    def __init__(self):
        # removed gender and ethnicity multipliers since we're getting them via user input
        pass

    # calculate homicide rate 
    def get_homicide_rate(self, country):
        print("\nPlease go to: https://data.worldbank.org/indicator/VC.IHR.PSRC.P5")
        print(f"Find the homicide rate for {country} (per 100,000 people)")
        value = float(input("Enter the homicide rate: "))
        return value

    # calculate human threat ODE based on inputs
    def human_threat_ode(self, V, t, lambda_base):
        return lambda_base * (1 - V)

    # RK4 algorithm for single ODE
    def runge_kutta_4(self, f, y0, t_span, args, n_steps=1000):
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

    # get multipliers via user input for gender and ethnicity
    def get_user_multipliers(self):
        # gender multiplier
        G = float(input("Enter one of the following numbers for gender:\n"
                        "if female: 1\nif male: 3.51\nif other or non-conforming: 5\n"
                        "Your choice: "))

        # ethnicity multiplier
        E = float(input("Enter one of the following numbers for ethnicity:\n"
                        "if black: 7.94\nif white: 1.93\nif hispanic: 1.78\nif other: 1\n"
                        "Your choice: "))

        return G, E

    # human threat risk factor calculation using above functions!
    def calculate_human_threat(self, years, country, n_steps=1000):

        # get necessary parameters/variables
        G, E = self.get_user_multipliers()
        H = self.get_homicide_rate(country)
        lambda_base = (H / 100000) * G * E

        # call rk4 solver
        t, V = self.runge_kutta_4(
            self.human_threat_ode,
            y0=0.0,
            t_span=(0, years),
            args=(lambda_base,),
            n_steps=n_steps
        )

        return t, V, country

class AnimalThreatModel:
    def __init__(self):
        # FIXED parameters based on data calibration
        self.animal_parameters = {
            'low': {'a': 0.3, 'b': 0.2, 'c': 0.5, 'd': 0.05},
            'medium': {'a': 1.0, 'b': 0.3, 'c': 0.2, 'd': 0.3},
            'high': {'a': 2.0, 'b': 0.2, 'c': 0.08, 'd': 1.2}
        }

        # pre-calculate max threat levels for each predator type
        self.max_threat_levels = self._calculate_max_threat_levels()

    # calculate maximum possible threat levels for each predator type
    def _calculate_max_threat_levels(self):
        max_levels = {}
        for level, params in self.animal_parameters.items():
            a, b, c, d = params['a'], params['b'], params['c'], params['d']
            max_threat = self._find_actual_max_threat(a, b, c, d)
            max_levels[level] = max_threat
        return max_levels

    # using rk4 solve for the max threat parameter
    def _find_actual_max_threat(self, a, b, c, d, time_span=200, n_steps=2000):
        y0 = np.array([1.0, 0.1])

        t, y = self.runge_kutta_4_system(
            self.animal_threat_ode,
            y0=y0,
            t_span=(0, time_span),
            args=(a, b, c, d),
            n_steps=n_steps
        )

        return np.max(y[1])

    # lokta-volterra ODE system for animal threat param
    def animal_threat_ode(self, variables, t, a, b, c, d):
        x, y = variables # x: survival probability, y: animal threat level
        dxdt = a * x - b * x * y
        dydt = -c * y + d * x * y
        return np.array([dxdt, dydt])

    # rk4 system solver for animal threat
    def runge_kutta_4_system(self, f, y0, t_span, args, n_steps=1000):
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

    # get predator level info based on user input
    def get_predator_level_from_user(self):
        print("\nAnimal Threat Assessment")
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
                print("Invalid choice! Enter 1, 2, or 3")

    # calculate animal threat seriousness
    def calculate_animal_threat(self, time_span, predator_level=None, initial_survival=1.0, n_steps=1000):
        if predator_level is None:
            predator_level = self.get_predator_level_from_user()

        # set initial threat score based on predator level
        if predator_level == 'low':
            initial_threat = 0.01
        elif predator_level == 'medium':
            initial_threat = 0.1
        else:
            initial_threat = 0.3 # high

        # get parameters and max threat level
        params = self.animal_parameters[predator_level]
        max_threat = self.max_threat_levels[predator_level]
        a, b, c, d = params['a'], params['b'], params['c'], params['d']

        # initial conditions
        y0 = np.array([initial_survival, initial_threat])

        # solve ODE system by calling rk4
        t, y = self.runge_kutta_4_system(
            self.animal_threat_ode,
            y0=y0,
            t_span=(0, time_span),
            args=(a, b, c, d),
            n_steps=n_steps
        )

        threat_levels = y[1]
        avg_threat = np.mean(threat_levels)

        # ratio for scaling for predator-specific maximums, map to the appropriate third
        ratio = avg_threat / max_threat if max_threat > 0 else 0.0

        # map each predator type to its designated third of the seriousness scale
        if predator_level == 'low':
            # low risk predators: 0-33.3% range
            seriousness_score = ratio * 0.333
        elif predator_level == 'medium':
            # medium risk predators: 33.3-66.7% range
            seriousness_score = 0.333 + (ratio * 0.334)
        else:  # high
            # high risk predators: 66.7-100% range
            seriousness_score = 0.667 + (ratio * 0.333)

        # ensure we stay within bounds
        seriousness_score = min(1.0, max(0.0, seriousness_score))

        return seriousness_score, predator_level, initial_threat, max_threat, avg_threat

def analyze_external_threats(years, country):
    # combine both threat models for comprehensive analysis
    human_model = HumanThreatModel()
    animal_model = AnimalThreatModel()

    # run human threat analysis
    human_time, human_risk, country = human_model.calculate_human_threat(years=years, country=country)

    # run animal threat analysis
    animal_seriousness, animal_level, initial_threat, max_threat, avg_threat = animal_model.calculate_animal_threat(
        time_span=years)

    animal_risk = animal_seriousness

    # weighted combination 
    # weights calibrated based on data
    combined_risk = 0.5 * human_risk[-1] + 0.5 * animal_risk

    return {
        'country': country,
        'human_risk': human_risk[-1], # final cumulative probability
        'animal_risk': animal_risk,
        'animal_level': animal_level,
        'combined_risk': combined_risk,
        'time_points': human_time,
        'human_risk_over_time': human_risk
    }
# resource availability scores
def resource_availability():
    print("\nResource Availability Assessment")

    scores = {}

    # water availability
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

    # food availability
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

    # medical availability
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

    # shelter availability
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

    # economic situation
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

    # calculate overall score
    scores['overall'] = sum(scores.values()) / len(scores)

    return scores

# environmental risk calculator model
class EnvironmentalRiskModel:
    global model
    def calculate_normalized_risk(self, country, alpha, beta, gamma, delta1, natural_disaster, delta2, temp_difference, delta3,
                                  drought_risk, delta4, population_density, Y0):
        lambda_prev = Y0

        for _ in range(10):
            lambda_current = (alpha +
                              beta * Y0 +
                              gamma * lambda_prev +
                              delta1 * natural_disaster +
                              delta2 * temp_difference +
                              delta3 * drought_risk +
                              delta4 * population_density)
            lambda_prev = lambda_current

        # calculate max possible risk (to see like the worst-case scenario)
        max_alpha = 0.5
        max_beta = 0.8
        max_gamma = 0.4
        max_delta = 0.01

        # worst-case inputs (aka all at maximum risk)
        max_natural_disaster = 1.0 # 100% disaster risk
        max_temp_difference = 5.0 # maximum temperature anomaly
        max_drought_risk = 5.0 # maximum drought risk score (0-5 scale)
        max_population_density = 22000.0 # Macau's density (~22,000 people/km²) --> Highest population density source: wikipedia
        
        # Initial disaster frequency
        # LLM COMPUTES THIS
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
            max_Y0 = min(max_Y0, 10)
        else:
            max_Y0 = 10.0 # default
        print("max_Y0 =", max_Y0)

        # calculate maximum possible risk
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

        # normalize to 0-1 range
        normalized_risk = max(0, min(1, lambda_current / lambda_current_max))

        return normalized_risk

# calculate normalized environmental risk based on user inputs
def get_environmental_risk_inputs(country):
    print("ENVIRONMENTAL RISK ASSESSMENT")

    print(f"\nFor {country}, please provide the following risk factors:")
    natural_disaster = float(input("Natural disaster risk at https://worldpopulationreview.com/country-rankings/natural-disaster-risk-by-country (0 to 100): ")) / 100
    temp_difference = float(input("Water temperature anomaly (or closest to it) at https://www.ospo.noaa.gov/products/ocean/cb/sst5km/?product=ssta (-5 to 5): "))
    drought_risk = float(input("Drought risk score at https://worldpopulationreview.com/country-rankings/droughts-by-country (0 to 5): "))
    population_density = float(input("Population density per km^2 (first column) at https://worldpopulationreview.com/country-rankings/countries-by-density: "))

    # LLM generates appropriate alpha, beta, gamma, delta1, delta2, delta3, delta4, and Y0 based on user info
    
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
        Y0 = min(Y0, 10)
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
        alpha = max(0.1, min(alpha, 0.5))
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
        beta = max(0.6, min(beta, 0.8))
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
        gamma = max(0.2, min(gamma, 0.4))
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
        delta1 = max(0.001, min(delta1, 0.01))
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
        delta2 = max(0.001, min(delta2, 0.01))
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
        delta3 = max(0.001, min(delta3, 0.01))
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
        delta4 = max(0.001, min(delta4, 0.01))
    else:
        delta4 = 0.0055 # default for all deltas
    print("delta4 =", delta4)

    # calculate normalized risk
    env_model = EnvironmentalRiskModel()
    normalized_risk = env_model.calculate_normalized_risk(
        country, alpha, beta, gamma,
        delta1, natural_disaster, delta2, temp_difference,
        delta3, drought_risk, delta4, population_density, Y0
    )

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

    # external threats survival (inverse of combined human + animal risk)
    S_external = 1 - external_combined_risk

    # resource availability with weighted importance
    resource_weights = {
        'water': 0.30, # 30%
        'food': 0.25, # 25% 
        'medical': 0.15, # 15%
        'shelter': 0.20, # 20%
        'economic': 0.10 # 10%
    }

    # calculate weighted resource score
    weighted_resource_sum = 0
    for resource, weight in resource_weights.items():
        weighted_resource_sum += resource_scores[resource] * weight

    # normalize from [-1, +1] range to [0, 1] range
    S_resources = max(0, min(1, (weighted_resource_sum + 1) / 2))

    # environmental risk survival
    S_environment = 1 / (1 + np.exp(k * (environmental_risk - risk_threshold)))

    # weighted SUM (additive instead of multiplicative)
    total_survival = (weights[0] * S_external +
                      weights[1] * S_resources +
                      weights[2] * S_environment)

    # wnsure result is between 0 and 1
    return max(0, min(1, total_survival))


# complete analysis function
def run_complete_survival_analysis(REMAINING):
    print("COMPREHENSIVE SURVIVAL PROBABILITY ANALYSIS\n")

    country = input("\nEnter your country: ")

    # get combined external threats (human + animal)
    external_results = analyze_external_threats(years=REMAINING, country=country)
    combined_external_risk = external_results['combined_risk']  # use combined risk

    # get resource availability
    resource_scores = resource_availability()

    # get environmental risk
    environmental_data = get_environmental_risk_inputs(country)
    environmental_risk = environmental_data['environmental_risk']

    # calculate total survival probability
    total_survival = calculate_total_survival(
        external_combined_risk=combined_external_risk,
        resource_scores=resource_scores,
        environmental_risk=environmental_risk
    )

    # display comprehensive results
    print(f"\nFINAL SURVIVAL PROBABILITY RESULTS!!!")
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

def run_individual_model(user_inputs):
    """
    Non-interactive version of the age() logic.
    Reads values from user_inputs instead of calling input().
    Returns time series for R(t), H(t) and the predicted lifespan.
    """

    A0 = int(user_inputs["age"])
    country_name = user_inputs["country"]

    if country_name not in countries:
        raise ValueError(f"Country '{country_name}' not found in countries dict.")

    # Country-based life expectancy → remaining years
    L_max = countries[country_name][1]
    R0 = L_max - A0

    # Individual parameters (use passed values, otherwise defaults)
    s = float(user_inputs.get("s", 0.5))       # disease severity
    a = float(user_inputs.get("a", 0.5))       # activity
    m = float(user_inputs.get("m", 0.5))       # metabolic risk
    delta_w = float(user_inputs.get("delta_w", 0.0))
    H0 = float(user_inputs.get("H0", 0.9))

    # Baseline hazard interpolation
    keys = sorted(baseline_hazards.keys())
    if A0 <= keys[0]:
        h0 = baseline_hazards[keys[0]]
    elif A0 >= keys[-1]:
        h0 = baseline_hazards[keys[-1]]
    else:
        for i in range(len(keys) - 1):
            x1, x2 = keys[i], keys[i + 1]
            if x1 <= A0 <= x2:
                y1, y2 = baseline_hazards[x1], baseline_hazards[x2]
                h0 = y1 + (A0 - x1) * (y2 - y1) / (x2 - x1)
                break

    # Coefficients (same as in age())
    k_s = 1.0
    k_a = 0.5
    k_w = 0.25
    k_m = 0.5
    k_g = 0.06
    k_h = 2.0

    def derivatives(y, t):
        R, H = y
        age = A0 + t
        dH_dt = - (k_s * s + k_w * delta_w + k_m * m) * H + k_a * a * (1 - H)
        hazard = h0 * np.exp(k_g * (age - A0))
        dR_dt = - R * hazard * (1.0 + k_h * (1.0 - H))
        return np.array([dR_dt, dH_dt])

    def rk4_system(f, y0, t):
        y = np.zeros((len(t), len(y0)))
        y[0, :] = y0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            k1 = f(y[i - 1, :], t[i - 1])
            k2 = f(y[i - 1, :] + 0.5 * dt * k1, t[i - 1] + 0.5 * dt)
            k3 = f(y[i - 1, :] + 0.5 * dt * k2, t[i - 1] + 0.5 * dt)
            k4 = f(y[i - 1, :] + dt * k3, t[i - 1] + dt)
            y[i, :] = y[i - 1, :] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y[i, 1] = np.clip(y[i, 1], 0.0, 1.0)
            y[i, 0] = max(y[i, 0], 0.0)
        return y

    # Simulate 80 years into the future
    t_end = 80.0
    n_steps = 4000
    t = np.linspace(0.0, t_end, n_steps)

    y0 = np.array([R0, H0])
    sol = rk4_system(derivatives, y0, t)
    R_sol = sol[:, 0]
    H_sol = sol[:, 1]

    final_R = R_sol[-1]
    predicted_lifespan = A0 + R0 - final_R

    return {
        "t": t,
        "R_t": R_sol,
        "H_t": H_sol,
        "predicted_lifespan": predicted_lifespan,
        "params": {
            "s": s, "a": a, "m": m,
            "delta_w": delta_w,
            "H0": H0,
            "A0": A0,
            "L_max": L_max
        }
    }


if __name__ == "__main__":
    # AGE = age()[0]
    REMAINING = age()[1]
    results = run_complete_survival_analysis(REMAINING)
    tasdf = REMAINING * results['total_survival']
    print(f"\nEstimated Remaining Survival Years: {tasdf:.2f} years")