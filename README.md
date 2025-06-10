# 💰 Black-Scholes Options Pricing Visualizer

This is a Streamlit web application that implements the Black-Scholes option pricing model for European call and put options. It allows users to interactively adjust key parameters and visualize how changes in these parameters impact option prices through dynamic heatmaps.

## ✨ Features

* **Real-time Option Pricing:** Get instant theoretical prices for European Call and Put options based on user-defined inputs.
* **Interactive Global Parameters:** Adjust stock price (S), strike price (K), time to expiration (T), risk-free rate (r), and volatility (sigma) using sliders in the sidebar.
* **Dynamic Heatmaps:** Explore the sensitivity of option prices across a range of two chosen parameters, with the other parameters held constant (and adjustable via local sliders).
    * **Call Option Heatmaps:**
        * Stock Price (S) vs. Volatility ($\sigma$)
        * Strike Price (K) vs. Time to Expiration (T)
        * Stock Price (S) vs. Risk-Free Rate (r)
    * **Put Option Heatmaps:**
        * Strike Price (K) vs. Time to Expiration (T)
        * Stock Price (S) vs. Volatility ($\sigma$)
        * Strike Price (K) vs. Risk-Free Rate (r)
* **Current Point Price:** For each heatmap, a dedicated metric displays the exact option price at a specific point, which can be adjusted independently with its own set of sliders.
* **Clear Visualizations:** Uses `matplotlib` and `seaborn` to generate intuitive heatmaps with color gradients to represent price changes.

## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository (if applicable) or save the code:**
    If this code is part of a Git repository:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```
    Otherwise, save the provided Python code as `app.py` and the `requirements.txt` file (see below) in the same directory.

2.  **Create a virtual environment (recommended):**
    This helps manage dependencies for your project.
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required libraries:**
    Create a `requirements.txt` file in the same directory as `app.py` with the following content:

    ```
    streamlit>=1.33.0
    numpy>=1.26.0
    scipy>=1.11.0
    matplotlib>=3.8.0
    seaborn>=0.13.0
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    Your web browser should automatically open a new tab with the Streamlit application. If not, copy and paste the provided local URL (e.g., `http://localhost:8501`) into your browser.

## ⚙️ Black-Scholes Model Assumptions

The Black-Scholes model is a theoretical framework that relies on several key assumptions:

* **European Options:** The model is for European options, meaning they can only be exercised at expiration.
* **No Dividends:** The underlying asset does not pay dividends during the option's life.
* **Constant Volatility:** The volatility of the underlying asset's returns is constant and known.
* **Constant Risk-Free Rate:** The risk-free interest rate is constant and known.
* **Efficient Markets:** Market movements are random and cannot be predicted.
* **No Transaction Costs:** There are no commissions or fees for buying or selling the option.
* **Lognormal Distribution:** The price of the underlying asset follows a lognormal distribution.
