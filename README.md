# Walmart Demand Forecasting: A Data-Driven Case Study

This project showcases a real-world application of data science and machine learning to solve a critical business problem in the retail sector: demand forecasting.

---

## 📚 **Overview**
Demand forecasting plays a pivotal role in retail business operations. This project leverages Walmart's sales data and applies advanced machine learning and deep learning models to predict sales with high accuracy. The insights gained can drive significant business improvements, including optimized inventory, targeted marketing, and enhanced operational efficiency.

---

## 🎯 **Project Objectives**
1. Accurately forecast weekly sales for Walmart stores.
2. Identify key trends and patterns in sales data.
3. Provide actionable insights to optimize inventory, staffing, and marketing strategies.

---

## 📊 **Dataset Description**
This project uses data from the Kaggle competition [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting):

- **train.csv**: Contains weekly sales data for various stores and departments.
- **stores.csv**: Provides metadata on store type and size.
- **features.csv**: Includes additional information such as temperature, fuel price, CPI, unemployment rate, and holiday flags.

---

## ⚙️ **Methodology**
### **Data Preprocessing**
- Handling missing values using median imputation.
- Removing outliers and addressing negative sales values.
- Feature engineering (e.g., creating `Total_MarkDown`, splitting date into `Year`, `Month`, `Week`).
- One-hot encoding for categorical features.
- Data normalization using MinMaxScaler.
- Feature selection with Recursive Feature Elimination (RFE).

### **Machine Learning Models**
- Linear Regression
- Random Forest Regression
- K-Neighbors Regression
- XGBoost Regression

### **Model Evaluation Metrics**
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## 🚀 **Key Results**
- **Top Performing Model**: Random Forest Regression with 97.89% accuracy (R² score).
- **Other Models**: XGBoost (94.21%), Linear Regression (92.28%), K-Neighbors (91.97%).
- Detailed comparison highlights the strength of ensemble-based methods in handling complex datasets.

---

## 📈 **Visual Insights**
Key visualizations include:
- Sales trends over time.
- Performance by department and store type.
- Seasonal patterns and holiday effects on sales.

These visualizations provide actionable insights for data-driven decision-making.

---

## 💼 **Business Impact**
- **Improved Decision-Making**: Insights into trends and performance drive smarter inventory, staffing, and marketing strategies.
- **Personalized Marketing**: Seasonal and weather-based insights enable targeted campaigns.
- **Operational Efficiency**: Accurate forecasts minimize overstocking and stockouts.
- **Risk Mitigation**: Proactively address underperforming areas.

---

## 🛠️ **Tools & Technologies**
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Deployment**: Streamlit for creating an interactive application

---

## 📂 **Repository Structure**
```
├── data/                 # Dataset files
├── notebooks/            # Jupyter notebooks for EDA and modeling
├── app/                  # Streamlit app files
├── models/               # Saved model files
├── visuals/              # Visualization assets
├── README.md             # Project documentation
```

---

## 🌟 **Getting Started**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/walmart-demand-forecasting.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

---

## 🤝 **Contributing**
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📬 **Contact**
For questions or collaboration, reach out to me on [LinkedIn](https://www.linkedin.com/in/dhrutijoshi16/) or via email at 16dhrutijoshi@gmail.com.

---

## ⭐ **Acknowledgments**
- Dataset provided by Walmart via Kaggle.
- Inspiration from real-world retail use cases.

---

## ⭐ **If you find this project helpful, please give it a star!** ⭐
