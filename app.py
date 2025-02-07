import streamlit as st
import pandas as pd
import numpy as np
import joblib
import chardet
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load models and scalers
def load_model_model(model_name):
    return joblib.load("models", f"{model_name}.pkl")
    models = {
    "Isolation Forest": load_model_model("iso_forest_model"),
    "Local Outlier Factor": load_model_model("lof_model"),
    "One-Class SVM": load_model_model("best_model"),
    "ARIMA": load_model_model("arima_model")
}

# Function to build transaction graph and compute centrality features
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['From'], row['To'], weight=row['Quantity'])

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
   
    df['Degree_Centrality'] = df['From'].map(degree_centrality).fillna(0)
    df['Betweenness_Centrality'] = df['From'].map(betweenness_centrality).fillna(0)
    df['Closeness_Centrality'] = df['From'].map(closeness_centrality).fillna(0)
    return df

def preprocess_data(df):
    feature_orders = [
        'Quantity', 'Quantity_Log', 'Time_Diff', 'Rolling_Mean_Quantity',
        'Rolling_Std_Quantity', 'Is_Self_Trade', 'Is_Circular_Trade',
        'Is_High_Activity', 'Quantity_Capped', 'Degree_Centrality',
        'Betweenness_Centrality', 'Closeness_Centrality'
    ]
    
    # Convert 'DateTime (UTC)' to datetime format
    df['DateTime (UTC)'] = pd.to_datetime(df['DateTime (UTC)'], dayfirst=True, errors='coerce')
    
    # Extract timestamp features
    df['Year'] = df['DateTime (UTC)'].dt.year
    df['Month'] = df['DateTime (UTC)'].dt.month
    df['Day'] = df['DateTime (UTC)'].dt.day
    df['Hour'] = df['DateTime (UTC)'].dt.hour
    df['Minute'] = df['DateTime (UTC)'].dt.minute
    df['Day_Of_Week'] = df['DateTime (UTC)'].dt.dayofweek
    
    # Sort by DateTime to calculate time intervals
    df = df.sort_values(by='DateTime (UTC)')
    df['Time_Diff'] = df['DateTime (UTC)'].diff().dt.total_seconds().fillna(0)
    
    # Apply log transformation for Quantity
    df['Quantity_Log'] = np.log1p(df['Quantity'])
    
    # Self trade detection
    df['Is_Self_Trade'] = (df['From'] == df['To']).astype(int)
    
    # Identify circular trades
    df['Is_Circular_Trade'] = False
    grouped_pairs = df.groupby(['From', 'To'])
    for (from_addr, to_addr), group in grouped_pairs:
        if len(group) > 1:
            df.loc[group.index, 'Is_Circular_Trade'] = True
    
    # Rolling window calculations
    window_size = 5
    df['Rolling_Mean_Quantity'] = df['Quantity'].rolling(window=window_size).mean()
    df['Rolling_Std_Quantity'] = df['Quantity'].rolling(window=window_size).std()
    
    df['Rolling_Mean_Quantity'] = df['Rolling_Mean_Quantity'].fillna(df['Rolling_Mean_Quantity'].mean())
    df['Rolling_Std_Quantity'] = df['Rolling_Std_Quantity'].fillna(df['Rolling_Std_Quantity'].mean())
    
    # Outlier detection based on IQR
    Q1 = df['Quantity'].quantile(0.25)
    Q3 = df['Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['Is_Outlier'] = (df['Quantity'] < lower_bound) | (df['Quantity'] > upper_bound)
    df['Quantity_Capped'] = df['Quantity'].clip(lower=lower_bound, upper=upper_bound)
    
    # High activity detection
    transaction_counts = df.groupby('From').size().reset_index(name='Transaction_Count')
    high_activity_threshold = transaction_counts['Transaction_Count'].quantile(0.95)
    high_activity_addresses = transaction_counts[transaction_counts['Transaction_Count'] > high_activity_threshold]['From']
    df['Is_High_Activity'] = df['From'].isin(high_activity_addresses)
    
    # Numerical columns to scale
    numerical_columns = ['Quantity', 'Time_Diff']
    df[numerical_columns] = df[numerical_columns].apply(pd.to_numeric, errors='coerce')
    df[numerical_columns] = df[numerical_columns].fillna(0)
    
    # Initialize MinMaxScaler and scale the selected columns
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Build the graph with centrality features
    df = build_graph(df)
    
    # Ensure the required columns are present
    for feature in feature_orders:
        if feature not in df.columns:
            df[feature] = 0
    
    # Fill missing values
    df.fillna(0, inplace=True)
    
        # Handle categorical columns to prevent the error with new categories
    for col in df.select_dtypes(include='category').columns:
        if '0' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('0')  # Add '0' as a category if it's not already present
        df[col].fillna('0', inplace=True)  # Fill 

    return df[feature_orders]

# Load models
models = {
    "Isolation Forest": load_model_model("iso_forest_model"),
    "Local Outlier Factor": load_model_model("lof_model"),
    "One-Class SVM": load_model_model("best_model"),
    "ARIMA": load_model_model("arima_model")  # Ensure proper saving/loading
}
# Load the pre-trained LSTM model, ignoring the loss function
lstm_model = load_model('lstm_anomaly_detector.h5', compile=False)

# Function to plot anomalies
def plot_anomalies_iso(df, model_name):
    normal_df = df[df['Prediction'] == "Normal"]
    anomaly_df = df[df['Prediction'] == "Anomaly"]

    plt.figure(figsize=(10, 5))
    plt.scatter(normal_df.index, normal_df['Quantity'], color='green', label="Normal", alpha=0.5)
    plt.scatter(anomaly_df.index, anomaly_df['Quantity'], color='red', label="Anomaly", alpha=0.5)
    plt.xlabel("Transaction Index")
    plt.ylabel("Quantity")
    plt.title(f"Anomalies detected by {model_name}")
    plt.legend()
    st.pyplot(plt)


# Function to plot anomalies
def plot_anomalies(df, model_name):
    normal_df = df[df['Prediction'] == "Normal"]
    anomaly_df = df[df['Prediction'] == "Anomaly"]

    plt.figure(figsize=(10, 5))
    plt.scatter(normal_df.index, normal_df['Quantity'], color='red', label="Anomaly", alpha=0.5)
    plt.scatter(anomaly_df.index, anomaly_df['Quantity'], color='green', label="Normal", alpha=0.5)
    plt.xlabel("Transaction Index")
    plt.ylabel("Quantity")
    plt.title(f"Anomalies detected by {model_name}")
    plt.legend()
    st.pyplot(plt)

# Function to preprocess the new dataset
def preprocess_data_for_lstm(data):
    # Ensure DateTime column exists
    if 'DateTime (UTC)' not in data.columns or 'Quantity' not in data.columns:
        st.error("Dataset must contain 'DateTime (UTC)' and 'Quantity' columns.")
        return None

    # Convert DateTime (UTC) to datetime format
    data['DateTime'] = pd.to_datetime(data['DateTime (UTC)'])
    data.set_index('DateTime', inplace=True)
    
    # Resample data to daily quantities
    daily_quantity = data['Quantity'].resample('D').sum().to_frame()
    
    return daily_quantity

# Function to create sequences from data
def create_sequences(data, window_size=30):
    X = []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
    return np.array(X)

# Function to preprocess data for ARIMA (ensure it has time-series format)
def preprocess_for_arima(df):
    df['DateTime (UTC)'] = pd.to_datetime(df['DateTime (UTC)'])
    df = df.sort_values(by='DateTime (UTC)')  # Ensure sorted order
    df.set_index('DateTime (UTC)', inplace=True)  # Set DateTime as index
    return df

def predict_anomalies_arima(df, arima_model):
    if arima_model is None:
        st.error("ARIMA model is not loaded properly.")
        return df
    
    series = df['Quantity']
    
    try:
        # ARIMA Forecast for the same period
        forecast = arima_model.forecast(steps=len(series))
        
        # Ensure forecast and dataset length match
        forecast = forecast[:len(series)] if len(forecast) > len(series) else forecast
        series = series[:len(forecast)] if len(forecast) < len(series) else series
        df = df.iloc[:len(forecast)]  # Adjust DataFrame length
        df["Prediction"] = np.where(abs(series.values - forecast) > 2 * np.std(series), "Anomaly", "Normal")

        # ---- Compute Rolling Mean & Bollinger Bands ----
        rolling_mean = series.rolling(window=15).mean()  # Rolling Mean
        rolling_std = series.rolling(window=15).std()
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)

        # ---- Plot Results ----
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series.index, series, label='Quantity', color='blue')
        ax.plot(rolling_mean.index, rolling_mean, label='Rolling Mean', color='orange')
        ax.fill_between(series.index, upper_band, lower_band, alpha=0.2, color='gray', label='Bollinger Bands')
        ax.plot(series.index, forecast, color='red', label='ARIMA Forecast')

        ax.set_title('Quantity Analysis with Bollinger Bands & ARIMA Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Quantity')
        
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        st.pyplot(fig)  # Display in Streamlit

    except Exception as e:
        st.error(f"Error in plotting: {e}")
        return df

    return df

# Ensemble model
def ensemble_prediction(processed_data):
    # Get individual predictions
    iso_forest = load_model_model("iso_forest_model")
    lof = load_model_model("lof_model")
    svm = load_model_model("best_model")
    
    iso_forest_pred = iso_forest.predict(processed_data)
    lof_pred = lof.predict(processed_data)
    svm_pred = svm.predict(processed_data)
    
    # Combine predictions with voting
    ensemble_preds = []
    for i in range(len(processed_data)):
        # Voting logic: count the number of "Anomaly" votes
        votes = [iso_forest_pred[i], lof_pred[i], svm_pred[i]]
        vote_count = votes.count(1)  # -1 means "Anomaly"
        if vote_count >= 2:  # If 2 or more models vote "Anomaly", classify as Anomaly
            ensemble_preds.append("Anomaly")
        else:
            ensemble_preds.append("Normal")
    
    return ensemble_preds

# Streamlit UI
st.title("Wash Trading Detection System")

# Upload File
uploaded_file = st.file_uploader("Upload CSV, TSV, or Excel file", type=["csv", "tsv", "xlsx"])

# Model Selection Dropdown
model_choice = st.selectbox("Select Model", ["Select", "One-Class SVM", "Isolation Forest", "Local Outlier Factor", "Ensemble", "LSTM","ARIMA","All"])

# Function to handle model predictions
if st.button("Predict"):

    if uploaded_file is None:
        st.error("Please upload a file before predicting.")
        st.stop()

    rawdata = uploaded_file.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding'] if result['encoding'] is not None else 'ISO-8859-1'
    uploaded_file.seek(0)  # Reset file pointer

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding=encoding)
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, encoding=encoding, sep='\t')
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("Unsupported file format. Please upload a CSV, TSV, or Excel file.")
        st.stop()

    st.write("Preview of uploaded data:")
    st.write(df.head())
    predictions = {}

    if model_choice != "Select":
        processed_data = preprocess_data(df)

        if model_choice == "Ensemble":
            ensemble_preds = ensemble_prediction(processed_data)
            df["Prediction"] = ensemble_preds
            st.write("Ensemble Predictions:")
            st.write(df)
            if 'NEAR_token_transaction_data' in uploaded_file.name:
                plot_anomalies(df, "Ensemble")
            else:
                plot_anomalies_iso(df, "Ensemble")
            normal_count = df['Prediction'].value_counts().get('Normal', 0)
            anomaly_count = df['Prediction'].value_counts().get('Anomaly', 0)
            st.write(f"Total Normal Points: {normal_count}")
            st.write(f"Total Anomalies: {anomaly_count}")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {model_choice} Predictions", csv, f"{model_choice}_predictions.csv", "text/csv", key=f'download-{model_choice}')

        elif model_choice == "All":
            svm_model = models["One-Class SVM"]
            svm_predictions = svm_model.predict(processed_data)
            predictions[model_choice] = ["Normal" if p == -1 else "Anomaly" for p in svm_predictions]
            result_df = df.copy()
            result_df["Prediction"] = predictions[model_choice]
            if 'NEAR_token_transaction_data' in uploaded_file.name:
                plot_anomalies(result_df, "One-Class SVM")
            else:
                plot_anomalies_iso(result_df, "One-Class SVM")
            normal_count = result_df['Prediction'].value_counts().get('Normal', 0)
            anomaly_count = result_df['Prediction'].value_counts().get('Anomaly', 0)
            st.write(f"One-Class SVM Predictions: ")
            st.write(f"Total Normal Points: {normal_count}")
            st.write(f"Total Anomalies: {anomaly_count}")

            lof_model = models["Local Outlier Factor"]
            lof_predictions = lof_model.predict(processed_data)
            predictions[model_choice] = ["Normal" if p == -1 else "Anomaly" for p in lof_predictions]
            result_df = df.copy()
            result_df["Prediction"] = predictions[model_choice]
            if 'NEAR_token_transaction_data' in uploaded_file.name:
                plot_anomalies(result_df, "Local Outlier Factor")
            else:
                plot_anomalies_iso(result_df, "Local Outlier Factor")
            normal_count = result_df['Prediction'].value_counts().get('Normal', 0)
            anomaly_count = result_df['Prediction'].value_counts().get('Anomaly', 0)
            st.write(f"Local Outlier Factor Predictions: ")
            st.write(f"Total Normal Points: {normal_count}")
            st.write(f"Total Anomalies: {anomaly_count}")

            iso_model = models["Isolation Forest"]
            iso_predictions = iso_model.predict(processed_data)
            predictions[model_choice] = ["Normal" if p == -1 else "Anomaly" for p in iso_predictions]
            result_df = df.copy()
            result_df["Prediction"] = predictions[model_choice]
            plot_anomalies_iso(result_df, "Isolation Forest")
            normal_count = result_df['Prediction'].value_counts().get('Normal', 0)
            anomaly_count = result_df['Prediction'].value_counts().get('Anomaly', 0)
            st.write(f"Isolation Forest Predictions: ")
            st.write(f"Total Normal Points: {normal_count}")
            st.write(f"Total Anomalies: {anomaly_count}")

            ensemble_preds = ensemble_prediction(processed_data)
            df["Prediction"] = ensemble_preds
            if 'NEAR_token_transaction_data' in uploaded_file.name:
                plot_anomalies(df, "Ensemble")
            else:
                plot_anomalies_iso(df, "Ensemble")
            normal_count = df['Prediction'].value_counts().get('Normal', 0)
            anomaly_count = df['Prediction'].value_counts().get('Anomaly', 0)
            st.write(f"Ensemble Predictions:")
            st.write(f"Total Normal Points: {normal_count}")
            st.write(f"Total Anomalies: {anomaly_count}")

            # LSTM
            daily_quantity = preprocess_data_for_lstm(df)
            if daily_quantity is not None:
                # Normalize data
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(daily_quantity)
                
                # Create sequences for prediction
                window_size = 30  # The same window size as used in training
                X_new = create_sequences(scaled_data, window_size)
                
                # Predict using the model
                predictions = lstm_model.predict(X_new)

                # Calculate MSE (Reconstruction Error)
                mse = np.mean(np.square(predictions - scaled_data[window_size:]), axis=1)
                
                # Set anomaly threshold (95th percentile of MSE)
                threshold = np.percentile(mse, 95)
                anomalies = mse > threshold
                
                # Create results dataframe
                results = daily_quantity.iloc[window_size:].copy()
                results['Predicted'] = scaler.inverse_transform(predictions)
                results['MSE'] = mse
                results['Anomaly'] = anomalies
                # Plot actual vs predicted values with anomalies
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(results.index, results['Quantity'], label='Actual Quantity', color='blue')
                ax.plot(results.index, results['Predicted'], label='Predicted', color='orange', linestyle='--')
                ax.scatter(results[results['Anomaly']].index,
                        results[results['Anomaly']]['Quantity'],
                        color='red', label='Anomaly')
                ax.set_title('LSTM Anomaly Detection')
                ax.set_xlabel('Date')
                ax.set_ylabel('Quantity')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                st.write("LSTM Predictions:")
                # Display results
                st.write("Anomalies Detected:", sum(anomalies))

            # ARIMA
            arima = models["ARIMA"]
            df_arima = preprocess_for_arima(df)
            df_arima = predict_anomalies_arima(df_arima, arima)
            normal_df = df_arima[df_arima['Prediction'] == "Normal"]
            anomaly_df = df_arima[df_arima['Prediction'] == "Anomaly"]
            plt.figure(figsize=(10, 5))
            plt.scatter(normal_df.index, normal_df['Quantity'], color='green', label="Normal", alpha=0.5)
            plt.scatter(anomaly_df.index, anomaly_df['Quantity'], color='red', label="Anomaly", alpha=0.5)
            plt.xlabel("Transaction Index")
            plt.ylabel("Quantity")
            plt.title("Anomalies detected by ARIMA")
            plt.legend()
            st.pyplot(plt)
            normal_count = len(normal_df)
            anomaly_count = len(anomaly_df)
            st.write(f"ARIMA Predictions:")
            st.write(f"Total Normal Points: {normal_count}")
            st.write(f"Total Anomalies: {anomaly_count}")
                    
        elif model_choice == "ARIMA":
              # Preprocess data for ARIMA
              # Run the selected model
            model = models.get(model_choice)
            df_arima = preprocess_for_arima(df)
            df_arima = predict_anomalies_arima(df_arima, model)
            # Display results
            st.write("Anomaly Detection Results using ARIMA:")
            st.write(df_arima[['Quantity', 'Prediction']].head(10))  # Display top rows
            # Plot anomalies
            normal_df = df_arima[df_arima['Prediction'] == "Normal"]
            anomaly_df = df_arima[df_arima['Prediction'] == "Anomaly"]
            plt.figure(figsize=(10, 5))
            plt.scatter(normal_df.index, normal_df['Quantity'], color='green', label="Normal", alpha=0.5)
            plt.scatter(anomaly_df.index, anomaly_df['Quantity'], color='red', label="Anomaly", alpha=0.5)
            plt.xlabel("Transaction Index")
            plt.ylabel("Quantity")
            plt.title("Anomalies detected by ARIMA")
            plt.legend()
            st.pyplot(plt)
            csv = df_arima.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {model_choice} Predictions", csv, f"{model_choice}_predictions.csv", "text/csv", key=f'download-{model_choice}')
        
        elif model_choice== "Isolation Forest":
            # Run the selected model
            model = models.get(model_choice)
            if model:
                pred = model.predict(processed_data)
                predictions[model_choice] = ["Normal" if p == -1 else "Anomaly" for p in pred]

                result_df = df.copy()
                result_df["Prediction"] = predictions[model_choice]
                st.write(f"Predictions from {model_choice} model:")
                st.write(result_df)

                plot_anomalies_iso(result_df, model_choice)

                normal_count = result_df['Prediction'].value_counts().get('Normal', 0)
                anomaly_count = result_df['Prediction'].value_counts().get('Anomaly', 0)

                st.write(f"Total Normal Points: {normal_count}")
                st.write(f"Total Anomalies: {anomaly_count}")

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {model_choice} Predictions", csv, f"{model_choice}_predictions.csv", "text/csv", key=f'download-{model_choice}')

            elif model_choice=="LSTM":
                # Preprocess the new data
                daily_quantity = preprocess_data_for_lstm(df)
                if daily_quantity is not None:
                    # Normalize data
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(daily_quantity)
                    
                    # Create sequences for prediction
                    window_size = 30  # The same window size as used in training
                    X_new = create_sequences(scaled_data, window_size)
                    
                    # Predict using the model
                    predictions = lstm_model.predict(X_new)

                    # Calculate MSE (Reconstruction Error)
                    mse = np.mean(np.square(predictions - scaled_data[window_size:]), axis=1)
                    
                    # Set anomaly threshold (95th percentile of MSE)
                    threshold = np.percentile(mse, 95)
                    anomalies = mse > threshold
                    
                    # Create results dataframe
                    results = daily_quantity.iloc[window_size:].copy()
                    results['Predicted'] = scaler.inverse_transform(predictions)
                    results['MSE'] = mse
                    results['Anomaly'] = anomalies

                    # Display results
                    st.write("Anomalies Detected:", sum(anomalies))
                    st.write("Threshold MSE:", threshold)
                    
                    # Plot actual vs predicted values with anomalies
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(results.index, results['Quantity'], label='Actual Quantity', color='blue')
                    ax.plot(results.index, results['Predicted'], label='Predicted', color='orange', linestyle='--')
                    ax.scatter(results[results['Anomaly']].index,
                            results[results['Anomaly']]['Quantity'],
                            color='red', label='Anomaly')
                    ax.set_title('LSTM Anomaly Detection')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Quantity')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # Display anomalies
                    st.write(results[results['Anomaly']][['Quantity', 'MSE']])

        else:
            # Run the selected model
            model = models.get(model_choice)
            if model:
                pred = model.predict(processed_data)
                predictions[model_choice] = ["Normal" if p == -1 else "Anomaly" for p in pred]

                result_df = df.copy()
                result_df["Prediction"] = predictions[model_choice]
                st.write(f"Predictions from {model_choice} model:")
                st.write(result_df)

                if 'NEAR_token_transaction_data' in uploaded_file.name:
                    plot_anomalies(result_df, model_choice)
                else:
                    plot_anomalies_iso(result_df, model_choice)

                normal_count = result_df['Prediction'].value_counts().get('Normal', 0)
                anomaly_count = result_df['Prediction'].value_counts().get('Anomaly', 0)

                st.write(f"Total Normal Points: {normal_count}")
                st.write(f"Total Anomalies: {anomaly_count}")

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {model_choice} Predictions", csv, f"{model_choice}_predictions.csv", "text/csv", key=f'download-{model_choice}')
    
        
    else:
        st.error("Please select a model for prediction.")
