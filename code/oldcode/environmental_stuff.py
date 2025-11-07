import numpy as np

class HumanThreatModel:
    def __init__(self):
        # Gender multipliers
        self.gender_multipliers = {
            'female': 1.0,
            'male': 4.0,
            'other': 5.0
        }

        # Ethnicity multipliers
        self.ethnicity_multipliers = {
            'black': 10.0,
            'white': 8.75,
            'hispanic': 5.5,
            'other': 4.0
        }

    def get_homicide_rate(self, country):
        """Get homicide rate per 100,000 for a country"""
        value = 1 ### User interface pls, make them go https://data.worldbank.org/indicator/VC.IHR.PSRC.P5 there and find their country
        return value

    def human_threat_ode(self, V, t, lambda_base, G, E):
        """ODE for cumulative probability of being a homicide victim"""
        return lambda_base * G * E * (1 - V)

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

    def calculate_human_threat(self, country, gender, ethnicity, years=80, n_steps=1000):
        """
        Calculate cumulative homicide probability using Runge-Kutta
        """
        # Get parameters
        H = self.get_homicide_rate(country)
        lambda_base = H / 100000
        G = self.gender_multipliers[gender]
        E = self.ethnicity_multipliers[ethnicity]

        # Solve ODE using Runge-Kutta
        t, V = self.runge_kutta_4(
            self.human_threat_ode,
            y0=0.0,  # Initial V(0) = 0
            t_span=(0, years),
            args=(lambda_base, G, E),
            n_steps=n_steps
        )

        return t, V


class AnimalThreatModel:
    def __init__(self):
        # Pre-defined parameters for different animal types
        self.animal_parameters = { ####### Ask AI to give you these values but this is smth that they should look like
            'bear': {'a': 0.1, 'b': 0.8, 'c': 0.3, 'd': 0.6},
            'wolf': {'a': 0.2, 'b': 0.7, 'c': 0.4, 'd': 0.5},
            'shark': {'a': 0.05, 'b': 0.9, 'c': 0.2, 'd': 0.3},
            'snake': {'a': 0.15, 'b': 0.6, 'c': 0.5, 'd': 0.4},

        }

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

    def calculate_animal_threat(self, animal_type, initial_survival=1.0, initial_threat=0.1, time_span=50,
                                n_steps=1000):
        """
        Calculate animal threat dynamics using Runge-Kutta

        Parameters:
        animal_type: type of animal ('bear', 'wolf', etc.)
        initial_survival: initial survival probability (default 1.0 = 100%)
        initial_threat: initial threat level (0-1)
        time_span: time period to simulate
        n_steps: number of integration steps
        """
        # Get parameters for the animal
        params = self.animal_parameters.get(animal_type)
        a, b, c, d = params['a'], params['b'], params['c'], params['d']

        # Initial conditions - start with 100% survival
        y0 = np.array([initial_survival, initial_threat])

        # Solve ODE system using Runge-Kutta
        t, y = self.runge_kutta_4_system(
            self.animal_threat_ode,
            y0=y0,
            t_span=(0, time_span),
            args=(a, b, c, d),
            n_steps=n_steps
        )

        return t, y[0], y[1]  # time, survival_prob, threat_level


def analyze_external_threats(country, gender, ethnicity, animal_type, years=80):
    """Combine both threat models for comprehensive analysis"""

    human_model = HumanThreatModel()
    animal_model = AnimalThreatModel()

    # Human threat analysis
    human_time, human_risk = human_model.calculate_human_threat(
        country, gender, ethnicity, years
    )

    # Animal threat analysis
    animal_time, survival_prob, threat_level = animal_model.calculate_animal_threat(
        animal_type, time_span=years
    )

    # Combine results - simple additive model
    combined_risk = human_risk + (1 - survival_prob)

    return {
        'human_risk': human_risk,
        'animal_survival': survival_prob,
        'animal_threat': threat_level,
        'combined_risk': combined_risk,
        'time': human_time
    }


def resource_availability():
    # get user input asking the following questions
    # Do you have drinkable water available to you?
        # If always 1, if between 6-5, 0.5 if 4 days a week 0, if 3-1 -0.5 if barely ever -1
    # Do you have food available to you?
        # If always 1, if between 6-5, 0.5 if 4 days a week 0, if 3-1 -0.5 if barely ever -1
    # Do you have medical attention/hospitals near you
        # If < 10km 1, if 10-30km 0.5, if 30-50km 0, if 50-100km -0.5, if >100km -1
    # Do you have shelter available (anywhere to sleep)?
        # If always 1, if between 6-5, 0.5 if 4 days a week 0, if 3-1 -0.5 if barely ever -1
    # How's the economical situation
        # If comfortable 1, if tight 0, if cooked -1
    scores = {}

    # Get info
    scores['water'] = 1
    scores['food'] = 1
    scores['medical'] = 1
    scores['shelter'] = 1
    scores['economic'] = 1

    scores['overall'] = sum(scores.values()) / len(scores)
    return scores

class EnvironmentalRiskModel:
    def calculate_risk(self, alpha, beta, gamma, delta1, season, delta2, climate_index, delta3, environmental_stress, delta4, population_density, Y0):
        lambda_prev = Y0
        # Stabilize through iterations
        for _ in range(10):
            lambda_current = (alpha +
                            beta * Y0 +
                            gamma * lambda_prev +
                            delta1 * season +
                            delta2 * climate_index +
                            delta3 * environmental_stress +
                            delta4 * population_density)
            lambda_prev = lambda_current
        return lambda_current

# Get all inputs pretty sure this works
print("Enter model parameters:")
season = float(input("Season risk (0-100): ")) / 100
climate_index = float(input("Climate index: "))
environmental_stress = float(input("Environmental stress (0-100): ")) / 100
population_density = float(input("Population density (0-100): ")) / 100

### Get AI to pick alpha, beta, gamma, delta1, delta2, delta3, delta4, and Y0.
'''
Prompts: 
for y0 
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

For the parameters (alpha, beta, gamma, and all deltas)
Pick values for the following parameters that work in the following {country}: canada for alpha between 0.1 and 0.5 and it's the baseline safety, safety when nothing else is happening for beta between 0.6 and 0.8 and it's recent disasters event, how much last month’s disasters affect this month’s for gamma between 0.2 and 0.4 and it's persistent risk memory, how much the risk persists over time for all deltas between 0.001 - 0.01 and which are for how external factors affect and they are: delta1: natural disaster risk factor per country delta2: the difference between today’s temperature and the long term average for ocean delta3: the drought risk score delta4: population density per country This is all to model natural disaster risk

Note that both prompts you need to input the country
'''
'''
# This is just to test so it works
alpha = float(input("Alpha: "))
beta = float(input("Beta: "))
gamma = float(input("Gamma: "))
delta1 = float(input("Delta1 (season): "))
delta2 = float(input("Delta2 (climate): "))
delta3 = float(input("Delta3 (environmental stress): "))
delta4 = float(input("Delta4 (population density): "))
Y0 = float(input("Initial events (Y0): "))
'''

# Calculate risk
model = EnvironmentalRiskModel()
risk = model.calculate_risk(alpha, beta, gamma, delta1, season, delta2, climate_index,
                           delta3, environmental_stress, delta4, population_density, Y0)

print(f"Average Risk Level: {risk:.3f}")