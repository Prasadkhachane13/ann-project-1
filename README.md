# Post-Harvest Storage Loss Prediction Using Artificial Neural Network (ANN)

<p align="center">
  <b>An Artificial Intelligence-Based Approach for Predicting Post-Harvest Storage Losses</b>
</p>

---

## 📌 Project Overview

Post-harvest losses are a major challenge in agriculture, affecting food availability, quality, farmer income, and supply chain efficiency.

The **Post-Harvest Storage Loss Prediction Using Artificial Neural Network (ANN)** project focuses on applying Artificial Neural Networks to analyze agricultural storage-related parameters and predict potential losses during the post-harvest storage process.

The system aims to identify patterns and relationships between storage conditions and post-harvest losses using neural network-based learning techniques.

By learning from historical agricultural and storage data, the ANN model can help estimate storage losses and support better decision-making in agricultural storage management.

## 🎯 Objectives

- Develop an Artificial Neural Network model for predicting post-harvest storage losses.
- Analyze important agricultural and storage-related parameters.
- Understand the relationship between storage conditions and product deterioration.
- Apply data preprocessing and feature engineering techniques.
- Train and evaluate an ANN-based prediction model.
- Predict potential post-harvest losses using input parameters.
- Explore how machine learning can contribute to reducing agricultural wastage.

## 💡 Key Features

- **Data Preprocessing:** Prepare and clean agricultural storage data.
- **Exploratory Data Analysis:** Understand patterns and relationships within the dataset.
- **Feature Selection:** Identify relevant parameters influencing storage losses.
- **ANN Model Development:** Build a neural network for prediction.
- **Model Training:** Train the ANN using historical data.
- **Loss Prediction:** Estimate post-harvest storage losses.
- **Model Evaluation:** Evaluate prediction performance using appropriate regression metrics.
- **Data Visualization:** Visualize model performance and prediction results.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow / Keras | Artificial Neural Network implementation |
| Jupyter Notebook | Model development and experimentation |

## ⚙️ Project Methodology

The project follows these major stages:

### 1. Data Collection

Collect agricultural and post-harvest storage-related information required for predicting storage losses.

### 2. Data Preprocessing

- Handle missing values.
- Remove duplicate records, where applicable.
- Identify and handle outliers.
- Convert categorical variables into numerical representations, if required.
- Normalize or standardize numerical features.

### 3. Exploratory Data Analysis (EDA)

Perform exploratory analysis to understand:

- Distribution of storage loss values.
- Relationships between input features.
- Correlation among storage parameters.
- Patterns that may influence post-harvest losses.

### 4. Feature Engineering

Select and prepare relevant input variables for ANN training.

### 5. Dataset Splitting

Divide the dataset into training and testing subsets to evaluate the model on unseen observations.

### 6. ANN Model Development

Develop a feedforward Artificial Neural Network consisting of:

- Input Layer
- Hidden Layer(s)
- Output Layer

The hidden layers learn complex relationships between storage-related input parameters and the target variable.

### 7. Model Training

Train the neural network using an appropriate optimizer and loss function.

### 8. Model Evaluation

Evaluate the model using suitable regression evaluation metrics.

### 9. Prediction

Use the trained ANN model to estimate post-harvest storage losses based on input parameters.

## 🏗️ ANN Architecture

The proposed Artificial Neural Network follows a feedforward neural network architecture.

**Input Layer:** Receives processed storage-related features.

**Hidden Layers:** Learn nonlinear relationships between the storage parameters and storage losses.

**Output Layer:** Produces the predicted storage loss value.

Possible model configurations include:

- Activation Function: ReLU for hidden layers.
- Output Activation: Linear for continuous-value regression.
- Optimizer: Adam.
- Loss Function: Mean Squared Error (MSE).

The final architecture and hyperparameters depend on the implemented model.

## 📊 Model Evaluation Metrics

The following metrics are suitable for evaluating the regression model:

| Metric | Description |
|---|---|
| MAE | Measures the average absolute prediction error. |
| MSE | Measures the average squared prediction error. |
| RMSE | Represents prediction error in the target variable's units. |
| R² Score | Measures how much variation in the target is explained by the model. |

These metrics help assess the accuracy and reliability of the ANN model.

## 📁 Project Structure

```text
Post_Harvest_Storage_Loss/
│
├── dataset/
│   └── dataset.csv
│
├── notebooks/
│   └── ANN_Post_Harvest_Storage_Loss.ipynb
│
├── models/
│   └── trained_model
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   └── prediction.py
│
├── requirements.txt
├── README.md
└── main.py
```

*Note: This is an illustrative structure. Update the filenames and folder names to match the actual repository contents.*

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Prasadkhachane13/ann-project-1.git
```

### 2. Navigate to the Project Directory

```bash
cd ann-project-1/Post_Harvest_Storage_Loss
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Project

If the project includes a Python execution script:

```bash
python main.py
```

Alternatively, if the implementation is provided as a Jupyter Notebook:

```bash
jupyter notebook
```

Open the ANN project notebook and execute its cells sequentially.

## 🌾 Applications

- Agricultural storage management.
- Post-harvest loss estimation.
- Food supply chain optimization.
- Agricultural decision-support systems.
- Reduction of food wastage.
- Improved storage planning.
- Data-driven agricultural management.

## 🔮 Future Enhancements

- Integrate real-time temperature and humidity sensor data.
- Develop IoT-based agricultural storage monitoring.
- Explore deep learning models for improved prediction.
- Implement explainable AI to identify major factors contributing to storage losses.
- Develop a web-based interface for interactive predictions.
- Extend the model to different agricultural commodities.
- Integrate predictive analytics into agricultural supply chain management.

## 👨‍💻 Repository

**GitHub Repository:**  
[Prasadkhachane13/ann-project-1](https://github.com/Prasadkhachane13/ann-project-1)

**Project:** Post-Harvest Storage Loss Prediction Using ANN

---

## 📜 License

This project is intended for educational and academic purposes.

---

<p align="center">
  <b>Artificial Neural Networks for Smarter Agricultural Storage</b>
  <br>
  Applying AI to understand and reduce post-harvest losses.
</p>
