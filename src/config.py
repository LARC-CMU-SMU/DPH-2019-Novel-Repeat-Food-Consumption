import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figure')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
PARAM_DIR = os.path.join(BASE_DIR, 'output', 'param')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'result')

for DIR in [DATA_DIR, FIG_DIR, MODEL_DIR, PARAM_DIR, OUTPUT_DIR]:
    if not os.path.exists(DIR):
        os.makedirs(DIR, exist_ok=True)
