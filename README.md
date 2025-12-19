# LIFE SPAN PREDICTOR: ALGORITHMS, FRACTALS, AND GRAPHS
## Project Description
This project develops a program to model human lifespan using differential equations as a dynamic quantity influenced by individual health and environmental conditions, and assesses the reliability of the model through thorough analysis. The program, written in Python, links two models: an individual survival model that simulates health decline, and an environmental hazard model that estimates external risk based on resources, crime and animal threats, and the surrounding climate. Each model gathers data through user input, interpolation, and calculations to generate parameters through computation and the use of gemini-2.5-flash, a Large Language Model. These parameters are substituted into the differential equations that govern the remaining lifespan, and the systems are solved numerically using the NumPy library. While the original model uses an implementation of the fourth-order Runge-Kutta method, analysis of the model’s performance was conducted by comparing to second- and sixth-order Runge-Kutta methods, as well as Adams-Bashforth of order 4 and Backward Differentiation Formula of order 2. Further analysis of the model’s behaviour was conducted by comparing the output and trends to existing data sets and existing research, and generating visual representations of the impacts of slightly altering the parameters using the Matplotlib library. By analyzing the limitations of the model, it was made clear that while the model follows expected trends, the output is unreliable due to its chaotic nature. For future work, improvements can be made to reduce this chaotic behaviour; however, accurately predicting lifespan remains a formidable task due to the randomness of life. 

## Demo Video
Please see below a video of a sample run of our code, in case of technical difficulties.
[![Demo Video!](https://img.youtube.com/vi/8Ze7asPSlRc/0.jpg)](https://www.youtube.com/watch?v=8Ze7asPSlRc)

There are many files in this repository. Each key file has a separate set of instructions to run them. Please see below for the list of files and their descriptions. Below that, you can find the instructions for each respective set of files.

## NOTICE:
To run or read any of our code, please first follow the steps below:
1. Clone the repository
2. Generate an API key at https://aistudio.google.com/app/api-keys
3. Navigate to the code folder. This is where all relevant, updated code is stored.
4. Locally make a .env file and write GOOGLE_API_KEY = {fill this in with your API key, remove curly braces}
5. Put the following command in your terminal: pip install -r requirements.txt

(WARNING: If at any point there is an error that you have run out of API credits, please generate a new API key and rerun our program. We recomend generating a new API key for every run, to prevent it running out in the middle of a trial, as this has happened several times before.)

## File Organization
_0_requirements.txt: Lists the dependencies that are not included in the default Python Library. Please install them as necessary.

_1_final_model.py: This is our final lifespan predictor model with an RK4 solver.

_2_environmental.py: This is the envionmental risk score calculator portion of our final model with an RK4 solver.

_2_individual.py: This is the individual risk score calculator portion of our final model with an RK4 solver.

_3_fractal_julia_set.py: Generates the Julia fractal modelling chaotic behaviour of the results dictated by the parameters.

_3_fractal_mandelbrot.py: Generates a mandelbrot survival fractal representing chaotic behaviour of the results dictated by the parameters.

_3_fractal_network.py: Generates a network diagram relating environmental parameters and their combined impact on survival probability.

_3_fractal_survivalprob_vs_resources.py: Generates visual representation of impact of resources and external threat risks on survival probability.

_3_fractal_tree_of_survival_prob.py: Generates visual tree representation of the chaotic behaviour of the impacts of parameters on the results.

_4_adams.py: Adams-Bashforth of order 4 solver version of the model.

_4_bdf2.py: Backward Differentiation Formula of order 2  solver version of the model.

_4_rk2.py: 2nd order Runge-Kutta solver version of the model.

_4_rk6.py: 4th order Runge-Kutta solver version of the model.

_5_graphing_params.py: Graph of figure 1 from final report.

_5_graphing_trends.py: Graph of figure 2 from final report.

_5_model.py: Model compatible with the graphing files above.

_6_sample_output.txt: Sample output of our model.

_7_sample_output.txt: Another sample output of our model.

## Instructions to run files

### Interactive Model
To run our interactive model that generates the expected lifespan based on user input parameters, please follow the steps below:
1. Navigate to code/_1_final_model.py
2. Run the code locally by inputting the following into your terminal: python code/_1_final_model.py
- If you would like to run parts of our model (the individual risk portion or the environmental risk portion), please run the respective files by inputting into your terminal either python code/_2_individual.py or python code/_2_environmental.py
- If you prefer to read a sample output of our interactive model, please feel free to read code/_5_sample_output.txt.
- If the interactive model does not run on your laptop due to the lack of dependencies or other technical issues, please see our demo video near the beginning of this README file.

### Fractals
To generate the fractals, please follow the steps below: 
1. Based on your desired fractal, navigate to code/_3_fractal_julia_set.py (Figure 3) OR code/_3_fractal_mandelbrot_survival.py OR code/_3_fractal_network.py OR code/_3_fractal_survivalprob_vs_resources.py OR code/_3_fractal_tree_of_survival_prob.py
2. Run the code locally by inputting the following into your terminal: python code/[filename of whichever file you choose with the square brackets removed].py

### Varying Numerical Solvers
To run the code for different numerical solvers, please follow the steps below: 
1. Navigate to the desired numerical solver file: code/_4_adams.py, code/_4_bdf2.py, code/_2_rk2.py, OR code/_6_rk6.py. Use code/_1_final_model.py for RK4 solver
2. Run the code locally by inputting the following into your terminal: python code/[filename of whichever file you choose with the square brackets removed].py

### Runtime comparison graphs
To generate the graph of runtime comparisons (Figure 1), please follow the steps below: 
1. Navigate to code/_5_graphing_trends.py
2. Run the code locally by inputting the following into your terminal: python code/_6_graphing_trends.py

### Parameter comparison graphs
NOTE: Our original version of the graph was unfortunately not correctly saved, so this isn't corresponding to the exact graph we have on our final report since we cannot remember the exact coefficients of the fitting curve, however, the data points are the same and the general fit trend is similar.
To generate the graph of parameter comparisons (Figure 2), please follow the steps below: 
1. Navigate to code/_5_graphing_trends.py
2. Run the code locally by inputting the following into your terminal: python code/_6_graphing_trends.py

## Sample Outputs
For sample outputs of the code from some of our previous runs, please read _6_sample_output.txt or _7_sample_output.txt.

## Miscellaneous
The remaining folders and files document our progress. Please disregard these. There is no important information directly related to our final project, but we wanted to keep it for future reference.

We hope you enjoy learning about your remaining life span!
