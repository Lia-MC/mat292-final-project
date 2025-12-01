# LIFE SPAN PREDICTOR: ALGORITHMS, FRACTALS, AND GRAPHS

There are many files in this repository. Each key file has a separate set of instructions to run them. Please see below for each respective set of instruction. 

To run or read any of our code, please first follow the steps below:
1. Clone the repository
2. Generate an API key at https://aistudio.google.com/app/api-keys
3. Navigate to the code folder. This is where all relevant, updated code is stored.
4. Locally make a .env file and write GOOGLE_API_KEY = {fill this in with your API key, remove curly braces}
5. Put the following command in your terminal: pip install -r requirements.txt

(WARNING: If at any point there is an error that you have run out of API credits, please generate a new API key and rerun our program. We recomend generating a new API key for every run, to prevent it running out in the middle of a trial, as this has happened several times before.)


To run our interactive model that generates the expected lifespan based on user input parameters, please follow the steps below:
1. Navigate to code/_1_final_model.py
2. Run the code locally by inputting the following into your terminal: python code/_1_final_model.py
- If you would like to run parts of our model (the individual risk portion or the environmental risk portion), please run the respective files by inputting into your terminal either python code/_2_individual.py or python code/_2_environmental.py
- If you prefer to read a sample output of our interactive model, please feel free to read code/_5_sample_output.txt.
- If the interactive model does not run on your laptop due to the lack of dependencies or other technical issues, please see our demo video:
<p align="center">
  <video width="500" src="https://www.youtube.com/watch?v=8Ze7asPSlRc">
</p>

To generate the fractals, please follow the steps below: 
1. Based on your desired fractal, navigate to code/_4_fractal_julia_set.py OR code/_4_fractal_mandelbrot_survival.py OR code/_4_fractal_network.py OR code/_4_fractal_survivalprob_vs_resources.py OR code/_4_fractal_tree_of_survival_prob.py
2. Run the code locally by inputting the following into your terminal: python code/[filename of whichever file you choose with the square brackets removed].py


To generate other (non-fractal) graphs, please follow the steps below: 
1. Navigate to code/graphs.py
2. Run the code locally by inputting the following into your terminal: python code/_4_graphs.py


The remaining folders and files document our progress. Please disregard these. There is no important information directly related to our final project, but we wanted to keep it for future reference.


We hope you enjoy learning about your remaining life span!
