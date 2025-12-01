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

# data of countries and life expectancies
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

def getcountry():
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

    return L_max

def get_s():
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

    return s, users

def get_a():
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

    return a, usera

def get_whr():
    # waist to hip ratio info
    w = float(input("Enter your waist to hip ratio: "))
    wopt = 0.875 # average optimal whr for all people
    delta_w = w - wopt
    return delta_w

def get_m():
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

    return m, userm

def get_H0(users, usera, userm):
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

    return H0

def get_h0(A0):
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

    return h0

def age():
    global model
    curage = int(input("What is your current age? "))

    L_max = getcountry()

    s, users = get_s()

    a, usera = get_a()

    delta_w = get_whr()

    m, userm = get_m()

    H0 = get_H0(users, usera, userm)

    # other parameters
    A0 = curage # current age
    R0 = L_max - A0 # initial remaining years

    h0 = get_h0(A0)

    # health dynamics coefficients
    # calibrated based on data
    k_s = 3 # 0.5 to 5 
    k_a = 0.7 # 0.2 to 1
    k_w = 0.35 # 0.1 to 0.5
    k_m = 0.7 # 0.2 to 1

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

    # remaining life expectancy estimate: when R approaches zero
    final_R = max(0, R_sol[-1])
    predicted_lifespan = A0 + R0 - final_R
    final_R = predicted_lifespan - A0

    # results
    print("\nSimulation Results!!!")
    print(f"Predicted remaining years of life: {(final_R):.2f}")
    print(f"Predicted total lifespan: {predicted_lifespan:.2f}")

    return [A0, predicted_lifespan-A0]

if __name__ == "__main__":
    # AGE = age()[0] # current age
    REMAINING = age()[1] # estimated remaining years of survival