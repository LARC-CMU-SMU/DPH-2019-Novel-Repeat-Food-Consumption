# Characterizing and Predicting Repeat Food Consumption Behavior for Just-in-Time Interventions

This repository contains the Jupyter notebooks and source codes used for reproducing the results published in our [Digital Health 2019 paper](https://www.researchgate.net/publication/335880405_Characterizing_and_Predicting_Repeat_Food_Consumption_Behavior_for_Just-in-Time_Interventions):

Yue Liu, Helena Lee, Palakorn Achananuparp, Ee-Peng Lim, Tzu-Ling Cheng, and Shou-De Lin. 2019. Characterizing and Predicting Repeat Food Consumption Behavior for Just-in-Time Interventions.  In <em>Proceedings of the 9th International Conference on Digital Public Health</em> (DPH2019). ACM, New York, NY, USA,  11-20. DOI: https://doi.org/10.1145/3357729.3357736

Please contact [Liu Yue](mailto:yueliu@smu.edu.sg?cc=liuyue715@hotmail.com) if you have any questions or problems.

## Requirements
The notebooks have been tested in Python 3.7 via Anaconda with the following packages:

* fpmc==0.0.0
* hpfrec==0.2.2.13

See requirements.txt for a complete list.


## Pipeline

### Step 0: Data import
[Download the data](https://drive.google.com/open?id=1J2I9UlHiQrA0S8DM2Z3YUl4qr6UDBORr) and extract the CSV and TSV files to the `data` directory.

### Step 1: Data preparation
Run the notebook `1-0-Data preparation.ipynb` to perpare the datasets for the recommendation task.

__Outputs:__ Several CSV files will be generated and stored in the `data` folder.

### Step 2: Exploitary data analysis
Run the notebooks `1-*.ipynb` to perform data analysis of repeat and novel consumption. The notebook requires data files from previous steps in the `data` folder.

__Outputs:__ Several reports will be generated and stored in the `figure` folder.

### Step 3: Paramater tuning of recommenders
Run the notebook `2-*.ipynb` to perform parameter tuning for the recommendation models. The notebook requires data files from previous steps in the `data` folder.

__Outputs:__ Several files will be generated and stored in the `output/param` folder.

### Step 4: Performing recommendations
Run the notebook `3-*.ipynb` to perform the recommendation tasks. The notebook requires data files from previous steps in the `data` and `output/param` folders.

__Outputs:__ Several files will be generated and stored in the `model` and `output/result` folder.
