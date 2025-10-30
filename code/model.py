import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Example: dy/dt = -k * y
# def model(y, t, k):
#     dydt = -k * y
#     return dydt

# data

countries = {
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

# get user inputs and initial processing of user inputs

curage = int(input("What is your current age? "))

for country in countries:
    index = countries[country][0]
    print(f"{index}. {country}")
    countryindex = int(input("Select the number corresponding to your country: "))

for country in countries:
    if countries[country][0] == countryindex:
        Lmax = countries[country][1]

# medical conditions severity
# LLM COMPUTES THIS
s = (0, 1) # pick value between these based on user inputs

# activity levels
# LLM COMPUTES THIS
a = (0, 1) # pick value between these based on user inputs

# WHR stuff
w = int(input("Enter your waist to hip ratio: "))
wopt = 0.875 # average optimal whr for all people
deltaw = w - wopt

# metabolic risk score
# LLM COMPUTES THIS
m = (0, 1) # pick value between these based on user inputs

# coefficients k
ks = 1 # 0.5 to 5
ka = 0.5 # 0.2 to 1
kw = 0.25 # 0.1 to 0.5
km = 0.5 # 0.2 to 1

# initial value

Rnot = Lmax - curage