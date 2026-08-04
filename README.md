# RiskFlow Analytics

### Dynamic GARCH-CVaR Forecaster for Commodity Markets

Most financial risk models assume the market is always calm. That works fine—until it doesn't. 

**RiskFlow Analytics** is a financial risk dashboard I built to fix this blind spot. Instead of relying on static averages, it uses a dynamic risk engine to detect market panic in real time. Ultimately, it takes complex market data and answers one crucial question: **"How much money could I actually lose tomorrow?"**

---

## 🚀 How It Works

*   **Live Data Ingestion:** Pulls the latest daily pricing for volatile commodities like Crude Oil (CL=F), Gold (GC=F), and Silver (SI=F) using `yfinance`.
*   **Dynamic Volatility Forecasting:** Runs a **GARCH(1,1)** algorithm that actively tracks market momentum. If the market spikes today, the model instantly updates to forecast higher turbulence for tomorrow.
*   **Value at Risk (VaR 99%):** Establishes a mathematical line in the sand. (*e.g., We are 99% confident our daily losses will not exceed this percentage.*)
*   **Conditional VaR (CVaR 99%):** Calculates the nightmare scenario. (*e.g., If a 1-in-100 market disaster actually happens, this is the average expected loss.*)

## 💡 Why Build This?

The biggest problem with quantitative finance isn't the math—it's how the risk is communicated. 

I built RiskFlow Analytics to take heavy, graduate-level mathematics and package them into a clean, readable user interface. It takes raw arrays and GARCH models and turns them into straightforward metrics so that anyone, from a portfolio manager to a retail investor, can instantly understand their true financial exposure at a glance.

---

## 🛠️ Tech Stack

*   **Language:** Python 3
*   **Frontend UI:** Streamlit
*   **Data Ingestion:** yfinance
*   **Mathematical Modeling:** arch (for GARCH volatility), NumPy, Pandas

---

## 💻 Installation & Setup

To run RiskFlow Analytics locally, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/RiskFlow-Analytics.git](https://github.com/yourusername/RiskFlow-Analytics.git)
cd RiskFlow-Analytics
    ```

**2. Create and activate a virtual environment**
*   **Mac/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

*   **Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

**3. Install the required dependencies**
```bash
pip install -r requirements.txt
    ```

**4. Run the application**
```bash
streamlit run app.py
    ```