## Dashboard Preview

The Streamlit dashboard provides an interactive interface for exploring customer data, evaluating churn predictions, and identifying important features.

**Screenshots to add:**
- Dataset overview and data-quality checks
- Model evaluation metrics and confusion matrix
- Feature importance chart
- Customer predictions and CSV export

To add screenshots, create a `screenshots/` folder in the repository and save your images there.

Example:

```markdown
![Model Evaluation](screenshots/model-evaluation.png)
![Feature Importance](screenshots/feature-importance.png)
```

Replace these example image paths with your actual screenshot filenames.

## How to Use the Dashboard

1. Start the Streamlit application.
2. Upload a customer dataset in CSV or Excel format, if using the universal version.
3. Review the dataset preview and missing-value summary.
4. Select the target column that represents churn.
5. Choose the label that indicates a churned customer.
6. Select the input features for training.
7. Set the test-data percentage and train the model.
8. Review evaluation metrics, the confusion matrix, and feature importance.
9. Inspect test-set predictions and download the results as a CSV file.

The original manufacturing version uses its generated dataset and saved model. Follow the setup instructions for the version you are running.

## Model Details

The project uses a Random Forest classifier, an ensemble learning algorithm that combines predictions from multiple decision trees.

A preprocessing pipeline handles missing values and converts categorical features into numerical representations. Model performance is evaluated using a held-out test set.

The reported metrics include:

- **Precision:** The proportion of predicted churners who are actual churners.
- **Recall:** The proportion of actual churners correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** How well the model ranks churners above non-churners across classification thresholds.
- **Confusion matrix:** A breakdown of correct and incorrect predictions.

Feature importance indicates which transformed input features the trained Random Forest relied on most. It does not establish causation.

## Reproducibility and Data Privacy

- The synthetic dataset generator uses a fixed random seed for reproducible data generation.
- Results may vary when the input dataset, selected features, or train/test split changes.
- Do not upload confidential customer records, personal information, API keys, or credentials to a public repository.
- Predictions should be reviewed by business users before retention actions are taken.
- Validate the model on representative, time-separated real-world data before using it for operational decisions.

## Future Enhancements

- Add configurable churn thresholds and risk categories.
- Compare Random Forest with Logistic Regression and Gradient Boosting.
- Add precision-recall curves and ROC curves.
- Improve probability calibration and threshold selection.
- Validate performance on future-period customer data.
- Deploy the dashboard using Streamlit Community Cloud or another suitable hosting service.
- Add model monitoring and a controlled retention experiment.

## Project Status

**Status:** Working prototype for learning and demonstration.

The project demonstrates a machine-learning workflow and an interactive analytics dashboard. Business effectiveness has not been established using real manufacturing customer data.
