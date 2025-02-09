Project Name: Wash Trading Detection and Anomaly Modeling

See it in action: https://wash-trading-case-study-for-erc20-token.streamlit.app/

The project contributers
2025

Description
This project aims to detect wash trading activities and other anomalies in cryptocurrency transaction data. It employs various machine learning models such as Isolation Forest, LOF, ARIMA, LSTM, and SVM for anomaly detection, along with preprocessing techniques to handle and normalize data from ERC20 and NEAR token transactions.

The project contains multiple trained models and a Jupyter notebook for analyzing wash trading patterns. The dataset includes transaction data that is preprocessed and normalized to detect fraudulent activities effectively.

Features
Anomaly Detection Models: Multiple models trained for detecting anomalies in transaction data:

LOF Model: Local Outlier Factor for detecting anomalies.
Isolation Forest: Isolation-based anomaly detection.
ARIMA Model: Time-series forecasting and anomaly detection.
LSTM Model: Long Short-Term Memory model for anomaly detection in time-series data.
SVM Model: Support Vector Machine for classification tasks.
Wash Trading Analysis: Detection of wash trading activities through statistical analysis in the Wash_Trading_Final_Colab.ipynb notebook.

Data Preprocessing: Tools and scripts for cleaning and normalizing cryptocurrency transaction data.

Installation
Follow these steps to set up the project locally:

1. Clone the repository:
git clone https://github.com/SruthiMangu133/Wash-Trading-Case-Study-for-ERC20-Token.git

2. Navigate into the project directory:
cd Wash-Trading-Case-Study-for-ERC20-Token

3. Create a virtual environment (optional but recommended):
python -m venv venv

4. Activate the virtual environment:
On Windows:
venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

5. Install the required dependencies:
pip install -r requirements.txt

6. Verify the installation:
python --version
pip list

Usage
1. Open Google Colab.
2. Upload Wash_Trading_Final_Colab.ipynb.
3. Run the notebook step by step to analyze the wash trading detection models.
4. To use a specific model for anomaly detection, load the corresponding .pkl file and apply it to the dataset.

Files in the Repository
1. LICENSE: License information for the project.
2. LOF_model.pkl: Pre-trained LOF model for anomaly detection.
3. NEAR_token_transaction_data.xlsx: Raw transaction data for NEAR tokens.
4. Pre_processed_ERC20_token_transaction.csv: Preprocessed transaction data for ERC20 tokens.
5. README.md: Project documentation.
6. Wash_Trading_Final_Colab.ipynb: Jupyter Notebook containing wash trading detection analysis.
7. arima_model.pkl: Pre-trained ARIMA model for anomaly detection.
8. ensemble_model.pkl: An ensemble of multiple models for better anomaly detection.
9. iso_forest_model.pkl: Isolation Forest model for detecting anomalies.
10. lstm_anomaly_detector.h5: LSTM model for anomaly detection in time-series data.
11. preprocessed_normalized_dataset.csv: Preprocessed and normalized dataset used for model training.
12. svm_model.pkl: Support Vector Machine model for classification tasks.

Contributing
If you would like to contribute to this project:
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m "Added a new feature").
4. Push to the branch (git push origin feature-branch).
5. Open a Pull Request.

License: 
This project is licensed under the MIT License - see the LICENSE file for details.

Contact: 
For any questions or issues, please reach out to the contributors through GitHub or open an issue in the repository.
