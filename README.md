# Battery Cycle Life Predictor

Tuned Random Forest that predicts total battery cycle life from the first 50 cycles.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Retrain (optional)
    python train.py battery_total_cycle_life_full.csv

## Docker
    docker build -t battery-life .
    docker run -p 8501:8501 battery-life

## Deploy options
- Streamlit Community Cloud: push this folder to GitHub, choose app.py as the main file.
- Hugging Face Spaces (Streamlit SDK): upload the folder.
- AWS EC2: install Docker, run the commands above, open port 8501 in the security group.
