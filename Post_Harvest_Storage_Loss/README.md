# Post-Harvest Crop Storage Loss Prediction Using ANN

## 1. Project overview

This project demonstrates a supervised Artificial Neural Network (ANN) regression system that estimates the percentage of net crop weight loss during storage. It uses crop characteristics and storage conditions as input features and a measured or demonstration weight-loss percentage as the target.

The project is designed as a moderate-level college ANN subject project. It includes a CSV dataset, a Jupyter notebook for preprocessing/training/evaluation, a saved-model workflow, and a Streamlit prediction interface.

> **Dataset integrity notice:** The included `dataset/storage_data.csv` is synthetic demonstration data generated for software development and learning. It is not collected from farms, warehouses or laboratory experiments. Its metrics must not be presented as evidence of real-world predictive performance. Replace it with measured data before making agricultural claims.

## 2. Objectives

- Understand the factors associated with crop weight change during storage.
- Prepare a structured dataset in CSV format.
- Build a feedforward ANN (MLP) regression model using TensorFlow/Keras.
- Evaluate the model using MAE, RMSE and R².
- Deploy the trained model through a simple Streamlit interface.
- Compare predicted loss across crops and storage scenarios.

## 3. Project folder structure

```text
Post_Harvest_Storage_Loss/
│
├── dataset/
│   └── storage_data.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── models/
│   ├── storage_ann.keras       # created after running the notebook
│   ├── preprocessor.pkl        # created after running the notebook
│   └── metadata.json           # created after running the notebook
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

## 4. Dataset description

The CSV contains one row per demonstration storage observation. The included dataset has 720 synthetic records.

| Column | Meaning | Type / unit |
|---|---|---|
| `sample_id` | Unique row identifier | Text |
| `crop_type` | Stored crop | Category |
| `storage_duration_days` | Duration represented by the observation | Days |
| `average_temperature_c` | Average temperature over the storage interval | °C |
| `average_relative_humidity_pct` | Average relative humidity over the interval | % |
| `initial_moisture_content_pct` | Moisture content at the start | % |
| `storage_method` | Storage arrangement | Category |
| `pest_level` | Observed pest severity category | Category |
| `ventilation` | Ventilation condition | Category |
| `initial_net_weight_kg` | Initial net crop mass, excluding container | kg |
| `final_net_weight_kg` | Final net crop mass, excluding container | kg |
| `weight_loss_pct` | Target calculated from initial and final net mass | % |
| `data_origin` | Indicates synthetic demonstration origin | Text |

### Target calculation

\[
\text{Weight Loss (\%)} =
100 \times \frac{W_i-W_f}{W_i}
\]

where \(W_i\) is initial net crop weight and \(W_f\) is final net crop weight.

The notebook excludes initial and final mass from the ANN predictors because those values are used to calculate the target. Including them as inputs would cause target leakage. Initial mass is instead used after prediction to estimate remaining quantity:

\[
\text{Estimated remaining mass} =
W_i \times (1-\text{predicted loss}/100)
\]

## 5. Recommended real-world data collection

For a credible study, replace the synthetic CSV with measurements collected under a documented protocol.

1. Select a manageable set of crops (for example wheat, rice and maize) based on local availability.
2. Obtain clean, representative grain samples and use labelled, independent containers or storage units.
3. Record the initial net grain mass, excluding the container.
4. Measure initial moisture content using a calibrated grain moisture meter where available.
5. Record storage type, ventilation and pest severity using consistent definitions.
6. Monitor temperature and relative humidity throughout storage. Use a data logger if available; otherwise take readings at a consistent schedule and document gaps.
7. Weigh each sample at fixed intervals (for example weekly). Avoid removing grain during the experiment unless the removed quantity is recorded.
8. At the end of each interval, record final net mass and, ideally, final moisture content.
9. Calculate the target consistently. Keep units and measurement procedures identical across samples.
10. Use multiple independent replicates per condition. A repeated weekly reading from the same container is not an independent sample.

### Important experimental-design considerations

- Environmental features should summarize the full storage interval, not a single temperature/humidity reading at the end.
- A practical pilot might use several crop types, storage methods and replicates, but the number of samples should be based on available time and equipment. A small dataset is a pilot, not proof of generalization.
- Keep a sample/container ID in the data. If the same container contributes multiple rows, split training and testing by container or experimental batch (group split) to avoid leakage.
- Record missing readings and instrument limitations rather than silently inventing values.
- Do not combine data collected using different measurement procedures without documenting the difference.

### Moisture-related interpretation

Net weight loss includes moisture evaporation. It is not automatically equivalent to dry-matter loss, pest damage, spoilage or quality deterioration. If initial and final moisture fractions are measured, a moisture-corrected dry-matter-loss estimate can be calculated:

\[
DML(\%) =
100 \times
\frac{W_i(1-M_i)-W_f(1-M_f)}
{W_i(1-M_i)}
\]

Here \(M_i\) and \(M_f\) are moisture fractions, e.g. 0.12 rather than 12. This correction is only as reliable as the mass and moisture measurements. Decide whether the project predicts net weight loss or dry-matter loss and label the target accordingly.

## 6. ANN model and architecture

The notebook uses a feedforward Multilayer Perceptron for regression:

```text
Input features (encoded and scaled)
              |
       Dense: 32 neurons
         ReLU activation
              |
       Dense: 16 neurons
         ReLU activation
              |
        Dense: 1 neuron
        Linear activation
              |
 Predicted weight-loss percentage
```

### Input features

- Crop type
- Storage duration
- Average temperature
- Average relative humidity
- Initial moisture content
- Storage method
- Pest level
- Ventilation

### Training configuration

- Framework: TensorFlow/Keras
- Network: fully connected feedforward MLP
- Hidden layers: 32 and 16 neurons
- Hidden activation: ReLU
- Output activation: Linear
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Metric during training: Mean Absolute Error (MAE)
- Maximum epochs: 150 with early stopping
- Batch size: 16
- Data split: 70% training, 15% validation, 15% test (random split for the synthetic independent rows)

For real repeated-measurement data, replace the random split with a group-based split by physical sample/container or experiment.

## 7. Evaluation metrics

- **MAE:** Average absolute difference between predicted and observed loss, expressed in percentage points.
- **RMSE:** Square root of mean squared prediction error; it penalizes larger errors more strongly.
- **R²:** Measures how much target variation is explained relative to a mean-prediction baseline. It can be negative.

Always report metrics on a held-out test set. Do not tune hyperparameters against the test set. Metrics produced from the bundled synthetic dataset only demonstrate the workflow and do not establish field accuracy.

## 8. Installation

Use Python 3.10 or 3.11 in a virtual environment where possible.

```bash
python -m venv .venv
```

Activate it:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 9. How to train the model

From the project root:

```bash
jupyter notebook
```

Open `notebooks/model_training.ipynb` and run all cells in order. The notebook reads `../dataset/storage_data.csv` and creates:

- `models/storage_ann.keras`
- `models/preprocessor.pkl`
- `models/metadata.json`

Do not move or rename these files unless you also update the paths in the app.

## 10. How to run the Streamlit application

After training and saving the model:

```bash
streamlit run app/app.py
```

Enter the crop and storage details, then click **Predict storage loss**. The app displays predicted loss percentage and estimated remaining mass for the entered initial mass.

## 11. Additional notebook prediction examples

At the bottom of the notebook, additional runnable examples demonstrate:

- Predicting loss for one detailed storage case.
- Comparing multiple crop types under identical conditions.
- Comparing storage methods for one crop.
- Loading a CSV of new cases and exporting batch predictions.
- Performing a what-if analysis across temperature and humidity combinations.

These are scenario comparisons, not validated storage recommendations. The model may behave unreliably for inputs outside the range and categories represented in the training data.

## 12. Suggested team division

For a four-person team:

1. **Agricultural data and literature:** define the target, protocol, crop/storage categories and collect measurements.
2. **Data engineering:** clean CSV data, document variables, perform exploratory analysis and preprocessing.
3. **ANN development:** implement the MLP, train it, tune a small number of configurations and evaluate results.
4. **Application and documentation:** build the Streamlit interface, test integration and prepare the report/demo.

All team members should be able to explain the target definition, preprocessing, network architecture, backpropagation, evaluation and limitations.

## 13. Suggested project workflow

1. Confirm the target definition and measurement protocol.
2. Collect and validate sample observations.
3. Explore distributions, missing values and outliers.
4. Split the data without sample/container leakage.
5. Fit preprocessing on training data only.
6. Train the ANN and monitor validation loss.
7. Evaluate once on the held-out test set.
8. Compare against a simple baseline (such as predicting the training-set mean) and optionally linear regression.
9. Save the model and preprocessing pipeline.
10. Connect them to Streamlit and test realistic inputs.
11. Document limitations and avoid unsupported claims.

## 14. How the model works (plain-language explanation)

The model learns a mathematical relationship between storage conditions and measured crop weight loss from examples.

**Step 1 — Input:** A sample is described using crop type, storage duration, temperature, humidity, initial moisture, storage method, pest level and ventilation.

**Step 2 — Preprocessing:** Numeric features are standardized so their scales are comparable. Categorical features are one-hot encoded into numerical indicators. The exact transformations are learned from the training data and reused for future predictions.

**Step 3 — Forward propagation:** The processed values enter the first dense layer (32 neurons) and then the second dense layer (16 neurons). Each neuron calculates a weighted sum plus a bias. ReLU activation introduces nonlinearity. The final neuron produces one continuous estimate: weight-loss percentage.

**Step 4 — Calculate error:** During training, the predicted value is compared with the observed target. MSE averages the squared differences between predictions and observations.

**Step 5 — Backpropagation and weight updates:** Backpropagation calculates how each network weight contributed to the error using the chain rule. Adam uses these gradients to update weights and biases. Repeated updates aim to reduce training loss while validation monitoring checks whether performance generalizes to unseen data.

**Step 6 — Test evaluation:** The held-out test observations are transformed using the already-fitted preprocessor. The ANN predicts their loss percentages, which are compared with measured targets using MAE, RMSE and R².

**Step 7 — Application prediction:** Streamlit loads the saved ANN and preprocessing pipeline. A user enters storage details, the app applies the same transformations, and the ANN estimates loss. If initial mass is supplied, the app calculates estimated remaining mass from the predicted percentage.

### Final limitation

An ANN learns patterns in its training data; it does not establish causation. Its reliability depends on representative measurements, consistent experimental procedures, sufficient independent samples and appropriate validation. It should be presented as a student decision-support prototype, not as a certified grain-storage safety or quality assessment system.
