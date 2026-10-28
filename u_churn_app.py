
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Universal Churn Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction & Retention Analytics")
st.write("Upload a customer CSV dataset to train and evaluate a churn model.")

uploaded_file = st.file_uploader(
    "Upload your customer dataset (CSV)",
    type=["csv","xlsx","xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()

    if df.empty or len(df.columns) < 2:
        st.error("The dataset needs rows and at least two columns.")
        st.stop()

    st.subheader("1. Explore the dataset")
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.write("Missing values per column")
    st.dataframe(
        df.isnull().sum().rename("Missing values")
    )

    st.subheader("2. Select the prediction target")

    target = st.selectbox(
        "Which column contains the churn label?",
        options=df.columns
    )

    # Remove rows without a known target label.
    data = df.dropna(subset=[target]).copy()

    labels = data[target].astype(str).unique().tolist()

    if len(labels) != 2:
        st.warning(
            "This version supports binary churn classification only. "
            "Your selected target must have exactly two distinct labels "
            "after rows with missing target values are removed."
        )
        st.stop()

    positive_label = st.selectbox(
        "Which label means the customer churned?",
        options=labels
    )

    y = (
        data[target].astype(str) == positive_label
    ).astype(int)

    if y.value_counts().min() < 2:
        st.error(
            "Each class needs at least two records for a stratified split."
        )
        st.stop()

    st.subheader("3. Select input features")

    candidates = [
        c for c in data.columns if c != target
    ]

    features = st.multiselect(
        "Choose columns the model can use",
        options=candidates,
        default=candidates
    )

    if not features:
        st.warning("Select at least one input feature.")
        st.stop()

    X = data[features].copy()

    # Convert dates to numeric timestamps where possible.
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64").astype(float)

    # Convert remaining unsupported data types to strings.
    for col in X.columns:
        if (
            pd.api.types.is_object_dtype(X[col])
            or isinstance(X[col].dtype, pd.CategoricalDtype)
        ):
            X[col] = X[col].where(
                X[col].isna(), X[col].astype(str)
            )

    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    test_size = st.slider(
        "Test data percentage",
        min_value=20,
        max_value=40,
        value=20,
        step=5
    ) / 100

    if st.button("Train and evaluate model", type="primary"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test, predictions, zero_division=0
            )
            recall = recall_score(
                y_test, predictions, zero_division=0
            )
            f1 = f1_score(
                y_test, predictions, zero_division=0
            )

            auc = roc_auc_score(y_test, probabilities)

            st.subheader("4. Model evaluation")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{accuracy:.1%}")
            c2.metric("Precision", f"{precision:.1%}")
            c3.metric("Recall", f"{recall:.1%}")
            c4.metric("F1-score", f"{f1:.3f}")
            c5.metric("ROC-AUC", f"{auc:.3f}")

            st.write(
                "Confusion matrix: rows are actual classes; "
                "columns are predicted classes. Class 1 means churn."
            )

            cm = confusion_matrix(y_test, predictions, labels=[0, 1])
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"],
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

            st.text(
                classification_report(
                    y_test,
                    predictions,
                    target_names=["Retained", "Churned"],
                    zero_division=0
                )
            )

            st.subheader("5. Feature importance")

            fitted_preprocessor = model.named_steps["preprocess"]
            rf = model.named_steps["classifier"]

            names = fitted_preprocessor.get_feature_names_out()
            importances = rf.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": names,
                "Importance": importances
            }).sort_values(
                "Importance", ascending=False
            ).head(15)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=importance_df,
                x="Importance",
                y="Feature",
                ax=ax
            )
            ax.set_title("Top Features Used by the Model")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("6. Test-set predictions")

            results = X_test.copy()
            results["Actual churn"] = y_test
            results["Predicted churn"] = predictions
            results["Churn probability"] = probabilities

            st.dataframe(
                results.head(50),
                use_container_width=True
            )

            st.download_button(
                "Download test predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.info(
                "Check the selected features, class counts, missing "
                "values, and data types."
            )

st.caption(
    "Use valid historical labels and exclude information that would "
    "not be available at prediction time. Validate on future-period "
    "data before using predictions for real business decisions."
)
