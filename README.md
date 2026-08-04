# RiskFlow Analytics

### Dynamic GARCH-CVaR Forecaster for Commodity Markets

Traditional financial risk models often assume markets are static and calm, blinding them to sudden crashes. **RiskFlow Analytics** is a dynamic, full-stack financial risk dashboard designed to fix this. It builds a smarter, reactive risk engine that detects market shocks in real-time and translates complex mathematical data into highly accessible, immediately readable metrics.

At its core, this engine answers a single, critical question for an investor: **"How much money could I realistically lose tomorrow?"**

---

## 🚀 What This Project Does

RiskFlow Analytics automates three main tasks:

*   **Live Data Ingestion:** Connects to live market APIs (via `yfinance`) to pull the latest daily pricing for volatile commodities like Crude Oil (CL=F), Gold (GC=F), and Silver (SI=F).
*   **Dynamic Volatility Forecasting:** Instead of using simple historical averages, it runs a recursive algorithm (**GARCH(1,1)**) that actively tracks market panic. If the market spikes today, the engine instantly forecasts higher turbulence for tomorrow.
*   **Downside Risk Calculation:** It calculates precise "lines in the sand" for potential losses using Filtered Historical Simulation:
    *   **Value at Risk (VaR 99%):** The maximum expected loss under normal market conditions. (e.g., *We are 99% sure our losses tomorrow will not exceed this percentage.*)
    *   **Conditional VaR (CVaR 99%):** The average expected loss if a true 1-in-100 market disaster occurs. (e.g., *If we cross the VaR threshold, this is how bad the nightmare scenario gets.*)

## 💡 The Value It Brings

The quantitative finance industry often struggles to make risk readable. The core value of RiskFlow Analytics is bridging robust, algorithm-heavy architecture with a clean visual interface. By framing graduate-level mathematics within an intuitive UI, this tool ensures that anyone—from portfolio managers to retail investors—can immediately understand their true financial exposure at a glance, without needing to decipher raw data arrays or complex equations.

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