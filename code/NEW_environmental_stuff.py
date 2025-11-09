import numpy as np


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
    '''
    Prompts (2 prompts):  ********************************************************************************************Note that both prompts you need to input the country
    1) For y0 
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
        ############### Last two lines to but the answer Ig, I don't really know how that works

    2) For the parameters (alpha, beta, gamma, and all deltas) ***********************************************************note that country is the selected country
    Pick numerical values for the following parameters that work in the following {country}: Alpha between 0.1 and 0.5 
    and it's the baseline safety (i.e. safety when nothing else is happening), for beta between 0.6 and 0.8 and it's 
    recent disasters event (i.e. how much last month’s disasters affect this month’s) for gamma between 0.2 and 0.4 and 
    it's persistent risk memory (i.e. how much the risk persists over time) for all deltas between 0.001 - 0.01, which 
    are for how external factors affect and they are: delta1 --> natural disaster risk factor per country, delta2 --> the 
    difference between today’s temperature and the long term average for ocean, delta3 --> the drought risk score, 
    delta4 --> population density per country. This is all to model natural disaster risk
    '''

    print(f"\nFor {country}, please provide the model parameters:")
    alpha = float(input("Alpha (baseline safety 0.1-0.5): "))
    beta = float(input("Beta (recent disasters effect 0.6-0.8): "))
    gamma = float(input("Gamma (persistent risk memory 0.2-0.4): "))
    delta1 = float(input("Delta1 (natural disaster weight 0.001-0.01): "))
    delta2 = float(input("Delta2 (temperature weight 0.001-0.01): "))
    delta3 = float(input("Delta3 (drought weight 0.001-0.01): "))
    delta4 = float(input("Delta4 (population density weight 0.001-0.01): "))
    Y0 = float(input("Initial disaster frequency (Y0): "))

    # Calculate normalized risk
    model = EnvironmentalRiskModel()
    normalized_risk = model.calculate_normalized_risk(country, alpha, beta, gamma, delta1, natural_disaster, delta2,
                                                      temp_difference, delta3, drought_risk, delta4, population_density, Y0)

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


# Run the complete analysis
if __name__ == "__main__":
    REMAINING=0 # FIX WHEN MERGING
    results = run_complete_survival_analysis(REMAINING)