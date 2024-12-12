import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Load Data and Models
data = pd.read_csv('./data/preprocessed_walmart_dataset.csv')
models = {
    "Linear Regression": "./models/linear_regressor.pkl",
    "Random Forest": "./models/randomforest_regressor.pkl",
    "XGBoost": "./models/xgboost_regressor.pkl"
}

# Function to load a model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Sidebar: Model Selection
st.sidebar.title("Select a Model")
selected_model_name = st.sidebar.selectbox(
    "Choose the forecasting model:",
    options=list(models.keys())
)
model = load_model(models[selected_model_name])

# Main Page: Overview
st.title("Walmart Time-Series Sales Forecasting")
st.write("This app predicts weekly sales for Walmart stores using various regression models.")
st.markdown("### Data Overview")
st.dataframe(data.head())

# Show Preprocessed Data Insights
if st.checkbox("Show Data Visualizations"):
    st.subheader("Select an Insight to View")

    # Button for Average Monthly Sales
    if st.button("Average Monthly Sales"):
        st.subheader("Average Monthly Sales")
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Month',y='Weekly_Sales',data=data)
        plt.ylabel('Sales',fontsize=14)
        plt.xlabel('Months',fontsize=14)
        plt.title('Average Monthly Sales',fontsize=16)
        st.pyplot(plt)

        st.markdown("""
            **This bar chart illustrates average monthly sales, revealing a steady increase throughout the year and a significant peak in December.**
            
            **Business Insights:** 
            - Seasonal Trends: December's peak reflects heightened holiday shopping demand.
            - Steady Growth: The gradual rise in sales suggests effective customer engagement or seasonal product interest.
            
            **Recommendations:**
            - Optimize Resources: Increase staff and streamline logistics for high-demand months.
            - Plan Promotions: Target campaigns ahead of December to capitalize on seasonal demand.
            """)
    
    if st.button("Departmental Sales Performance"):
        st.subheader("Departmental Sales Performance")
        plt.figure(figsize=(20,8))
        sns.barplot(x='Dept',y='Weekly_Sales',data=data)
        plt.title('Average Sales per Department', fontsize=18)
        plt.ylabel('Sales', fontsize=16)
        plt.xlabel('Department', fontsize=16)
        plt.show()
        st.pyplot(plt)

        st.markdown("""
            **This bar chart illustrates sales performance across various departments. Certain departments significantly outperform others, with noticeable peaks in specific categories.**
                    
            **Business Insights:** 
            - Top-Performing Departments: Departments with higher sales peaks should be prioritized for stocking and marketing efforts to maximize revenue.
            - Underperforming Areas: Departments with low sales require analysis to uncover customer preferences or address operational gaps.
            - Diversification: A heavy reliance on top-performing departments suggests the need to diversify product offerings to reduce risks.
                    
            **Recommendations:**
            - Targeted Investments: Allocate resources to high-performing departments to maximize their growth potential.
            - Departmental Promotions: Implement campaigns to increase sales in underperforming categories.
            - Optimize Product Mix: Adjust inventory and product offerings based on customer behavior and emerging trends.
            """)

    
    if st.button("Store Sales Performance"):
        st.subheader("Store Sales Performance")
        plt.figure(figsize=(20,8))
        sns.barplot(x='Store',y='Weekly_Sales',data=data)
        plt.title('Average Sales per Store', fontsize=18)
        plt.ylabel('Sales', fontsize=16)
        plt.xlabel('Store', fontsize=16)
        plt.show()
        st.pyplot(plt)

        st.markdown("""
            **This bar chart illustrates sales performance across various stores. Certain stores significantly outperform others, with noticeable peaks in specific locations.**
                    
            **Business Insights:** 
            - Top-Performing Stores: Stores with higher average sales are consistently outperforming others, likely due to high-demand locations or strong strategies.
            - Underperforming Stores: Stores with notably lower sales indicate potential challenges such as location, customer engagement, or inventory issues.
            - Variation in Performance: Sales variation suggests the need for tailored strategies per store.
                                
            **Recommendations:**
            - Focus on Strengths: Invest in top-performing stores to boost revenue.
            - Address Weaknesses: Analyze and improve underperforming stores with targeted promotions and adjustments.
            - Standardize Success: Apply best practices from high-performing stores across the network.
            - Customize Locally: Adapt inventory and marketing to each store’s demographics.
            """)

    
    if st.button("Effect of Temperature on Sales"):
        st.subheader("Effect of Temperature on Sales")
        plt.figure(figsize=(10,8))
        sns.distplot(data['Temperature'])
        plt.title('Effect of Temperature',fontsize=15)
        plt.xlabel('Temperature',fontsize=14)
        plt.ylabel('Density',fontsize=14)
        st.pyplot(plt)

        st.markdown("""
            **The density plot reveals a clear relationship between temperature and sales, with a peak around moderate temperatures (60–80°F), showing that sales are impacted by climatic conditions.**
                    
            **Business Insights:** 
            - Weather-Driven Demand: Sales peak at mild temperatures (60–80°F), indicating climatic influence on shopping behavior.
            - Seasonal Planning: Use weather trends to forecast demand and manage inventory efficiently.
                                
            **Recommendations:**
            - Weather-Based Forecasting: Incorporate weather data into demand models for precision.
            - Marketing Strategy: Align campaigns with favorable weather to boost sales.
            """)
        

data1 = pd.read_csv('./data/final_data.csv')

if 'Date' in data1.columns:
    data1['Date'] = pd.to_datetime(data1['Date'])  # Ensure it's in datetime format
    data1['Year'] = data1['Date'].dt.year
    data1['Month'] = data1['Date'].dt.month
    data1['Week'] = data1['Date'].dt.isocalendar().week
    # Drop the original Date column, since we now have useful features
    data1 = data1.drop(['Date'], axis=1)

# Prepare features and target
X = data1.drop(['Weekly_Sales'], axis=1)  # Features (excluding target)
Y = data1['Weekly_Sales']  # Target (Weekly Sales)

# Train-Test Split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=50)


# Show Model Predictions
if st.checkbox("Show Model Predictions vs Real Values"):
    if selected_model_name == "Linear Regression":
        # Predictions for Linear Regression
        lr = model  # Assuming model is already loaded
        y_pred = lr.predict(X_test[:200])

        # Plotting
        st.markdown("### 📈 Linear Regression: Predictions vs Real Values")
        plt.figure(figsize=(20,8))
        plt.plot(lr.predict(X_test[:200]), label="prediction", linewidth=2.0,color='blue')
        plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
        # Adding plot title and labels
        plt.title("Linear Regression: Predictions vs Real Sales", fontsize=16, weight='bold', pad=20)
        plt.xlabel("Sample Index", fontsize=12, labelpad=10)
        plt.ylabel("Weekly Sales", fontsize=12, labelpad=10)
        plt.legend(loc="best")
        plt.show()
        st.pyplot(plt)
        # Adding explanatory text for better engagement
        st.markdown("""
        **What does this graph show?**
        - The blue line represents the **predicted weekly sales** using the Linear Regression model.
        - The dashed coral line shows the **actual weekly sales** for the same period.
        - Notice how the predictions align with the actual values, showcasing the model's performance.
        """)

        # Add model evaluation
        if st.checkbox("Evaluate Model Performance"):
            # Evaluate predictions
            y_pred = lr.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)
            
            # Create a DataFrame for the metrics
            evaluation_metrics = pd.DataFrame({
                "Metric": ["Mean Absolute Error (MAE)", 
                        "Mean Squared Error (MSE)", 
                        "Root Mean Squared Error (RMSE)", 
                        "R² Score"],
                "Value": [f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
            })

            # Display table in Streamlit
            st.markdown("### 📊 Model Evaluation Metrics")
            st.write(f"**Model Selected**: {selected_model_name}")
            st.table(evaluation_metrics)

        
    elif selected_model_name == "Random Forest":
        # Predictions for Random Forest
        rf = model  # Assuming model is already loaded
        y_pred_rf = rf.predict(X_test[:200])
        # Plotting
        st.markdown("### 📈 Random Forest: Predictions vs Real Values")
        plt.figure(figsize=(20,8))
        plt.plot(rf.predict(X_test[:200]), label="prediction", linewidth=2.0,color='blue')
        plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
        plt.legend(loc="best")
        plt.show()
        st.pyplot(plt)

        # Add model evaluation
        if st.checkbox("Evaluate Model Performance"):
            # Evaluate predictions
            y_pred_rf = rf.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred_rf)
            mse = metrics.mean_squared_error(y_test, y_pred_rf)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred_rf)
            
            # Create a DataFrame for the metrics
            evaluation_metrics = pd.DataFrame({
                "Metric": ["Mean Absolute Error (MAE)", 
                        "Mean Squared Error (MSE)", 
                        "Root Mean Squared Error (RMSE)", 
                        "R² Score"],
                "Value": [f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
            })

            # Display table in Streamlit
            st.markdown("### 📊 Model Evaluation Metrics")
            st.write(f"**Model Selected**: {selected_model_name}")
            st.table(evaluation_metrics)


    else:
        # Predictions for Random Forest
        xgf = model  # Assuming model is already loaded
        y_pred_xgf = xgf.predict(X_test[:200])
        # Plotting
        st.markdown("### 📈 XGBoost: Predictions vs Real Values")
        plt.figure(figsize=(20,8))
        plt.plot(xgf.predict(X_test[:200]), label="prediction", linewidth=2.0,color='blue')
        plt.plot(y_test[:200].values, label="real_values", linewidth=2.0,color='lightcoral')
        plt.legend(loc="best")
        plt.show()
        st.pyplot(plt)

        # Add model evaluation
        if st.checkbox("Evaluate Model Performance"):
            # Evaluate predictions
            y_pred_xgf = xgf.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred_xgf)
            mse = metrics.mean_squared_error(y_test, y_pred_xgf)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred_xgf)
            
            # Create a DataFrame for the metrics
            evaluation_metrics = pd.DataFrame({
                "Metric": ["Mean Absolute Error (MAE)", 
                        "Mean Squared Error (MSE)", 
                        "Root Mean Squared Error (RMSE)", 
                        "R² Score"],
                "Value": [f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
            })

            # Display table in Streamlit
            st.markdown("### 📊 Model Evaluation Metrics")
            st.write(f"**Model Selected**: {selected_model_name}")
            st.table(evaluation_metrics)