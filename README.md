# Behavior-Based Continuous Authentication for Mobile Banking

This project is a Streamlit dashboard that demonstrates a sophisticated system for continuous, behavior-based authentication in a mobile banking application. It uses a dual-model approach (Isolation Forest and LSTM Autoencoder) to detect anomalies and provides rich, interactive visualizations to analyze the results.

## Project Structure

The project has been streamlined into a clean, easy-to-run format:

- `app.py`: The main and only Streamlit application file. It contains the data generation logic, model training, risk assessment, and the interactive visual dashboard.
- `requirements.txt`: A list of all Python dependencies required to run the project.
- `README.md`: This file.

## How it Works

1.  **Launch the App**: Run the Streamlit application from your terminal.
2.  **Run Analysis**: In the app's sidebar, click the "Run Full Analysis" button.
3.  **Backend Processing**: In the background, the app will:
    - Generate synthetic behavioral data for normal, anomalous, and fraudulent sessions.
    - Train an `IsolationForest` model to detect point-in-time anomalies.
    - Train an `LSTM Autoencoder` to learn the user's normal sequential behavior.
    - Score all test sessions using both models to generate a combined risk score.
4.  **Visualize Insights**: The main dashboard will populate with a series of interactive charts (Sankey, Polar, Bubble, and Heatmap) that provide deep insights into the model's performance and the nature of the different behavioral patterns.

## 🚀 Running the Application

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

Your web browser will open with the visual dashboard, ready for analysis. 