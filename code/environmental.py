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
                        "if female: 1\nif male: 4\nif other or non-conforming: 5\n"
                        "Your choice: "))

        # ethnicity multiplier
        E = float(input("Enter one of the following numbers for ethnicity:\n"
                        "if black: 10\nif white: 8.75\nif hispanic: 5.5\nif other: 4\n"
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

if __name__ == "__main__":
    REMAINING = 70 # for testing purposes, the output based model for individual risk
    results = run_complete_survival_analysis(REMAINING)
    tasdf = REMAINING * results['total_survival']
    print(f"\nEstimated Remaining Survival Years: {tasdf:.2f} years")