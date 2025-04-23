# Sepsis Prediction Model

This project focuses on building a machine learning model to predict the likelihood of sepsis in patients based on clinical data. The workflow includes data preprocessing, exploratory data analysis, model training, evaluation, and result visualization.

## Project Workflow

1. **Data Preparation**:
    - Load and preprocess clinical data from multiple datasets (`training_setA` and `training_setB`).
    - Handle missing values and concatenate data into unified datasets for training and testing.

2. **Exploratory Data Analysis (EDA)**:
    - Analyze the distribution of key features such as `Age` and `SepsisLabel`.
    - Visualize data using count plots and distribution plots to understand patterns and correlations.

3. **Model Training**:
    - Train a logistic regression model to classify patients as septic or non-septic.
    - Split the data into training and testing sets for evaluation.

4. **Model Evaluation**:
    - Evaluate the model using metrics such as accuracy, confusion matrix, classification report, and AUROC score.
    - Analyze the impact of different prediction thresholds on model performance.

5. **Result Visualization**:
    - Plot the ROC curve to visualize the trade-off between sensitivity and specificity.
    - Save the predicted probabilities and labels for further analysis.

## Key Features

- **Data Handling**: Efficiently processes large clinical datasets with missing values.
- **Visualization**: Provides insightful visualizations to understand data distribution and model performance.
- **Threshold Analysis**: Allows fine-tuning of prediction thresholds to optimize model performance.
- **Export Results**: Saves predictions and labels in a structured format for downstream tasks.

## Requirements

- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

## Output

- `train_a.csv` and `train_b.csv`: Preprocessed datasets.
- `scoring_labels.psv`: Ground truth labels for the test set.
- `scoring_predictions.psv`: Predicted probabilities and labels for the test set.

## Acknowledgments

This project demonstrates the application of machine learning in healthcare, aiming to assist in early detection of sepsis and improve patient outcomes. The dataset is restricted and cannot be used publicly anymore.