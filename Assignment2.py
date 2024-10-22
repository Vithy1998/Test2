import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data from the provided URL
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Vithy1998/Test2/refs/heads/main/kickstarter_2016.csv'
    df = pd.read_csv(url)
    df['Launched'] = pd.to_datetime(df['Launched'])
    df['Deadline'] = pd.to_datetime(df['Deadline'])
    return df

df = load_data()

# Q1: Target variable creation and feature engineering
df['success'] = df['State'].apply(lambda x: 1 if x.lower() == 'successful' else 0)

# Feature engineering
df = df[df['Goal'] > 0].copy()  # Remove rows with zero or negative Goal
df['log_goal'] = np.log(df['Goal'])
df['campaign_duration'] = (df['Deadline'] - df['Launched']).dt.days
df['name_length'] = df['Name'].apply(lambda x: len(str(x).split()))

# Sidebar: Classifier selection
st.sidebar.header("Classifier Selection")
classifier_name = st.sidebar.selectbox(
    'Select Classifier',
    ('Logistic Regression', 'Random Forest', 'Gradient Boosting')
)

# Sidebar: Feature selection
st.sidebar.header("Feature Selection")
all_features = ['log_goal', 'campaign_duration', 'name_length', 'Category', 'Country']
selected_features = st.sidebar.multiselect(
    'Select features for the model:',
    all_features,
    default=all_features
)

if not selected_features:
    st.error("Please select at least one feature.")
    st.stop()

# Prepare data for model training
X = df[selected_features]
y = df['success']

# Dynamically determine numeric and categorical features based on selection
numeric_features = [f for f in selected_features if f in ['log_goal', 'campaign_duration', 'name_length']]
categorical_features = [f for f in selected_features if f in ['Category', 'Country']]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(max_samples=0.1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Sidebar: Cross-validation parameters
st.sidebar.header("Model Evaluation")
cv_folds = st.sidebar.slider("Number of Cross-Validation Folds", min_value=3, max_value=10, value=5)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline with selected classifier
model = classifiers[classifier_name]
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Train and evaluate the model using cross-validation
st.write(f"### {classifier_name} Cross-Validation Results")

# Define scoring metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Perform cross-validation
cv_results = cross_validate(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)

# Display cross-validation results
for metric in scoring:
    mean_score = cv_results[f'test_{metric}'].mean() * 100  # Convert to percentage
    st.write(f'**{metric.capitalize()}:** {mean_score:.2f}%')

# Train the model on the entire training set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Model evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, zero_division=0) * 100
recall = recall_score(y_test, y_pred, zero_division=0) * 100
f1 = f1_score(y_test, y_pred, zero_division=0) * 100

st.write(f"### {classifier_name} Evaluation on Test Set")
st.write(f'**Accuracy:** {accuracy:.2f}%')
st.write(f'**Precision:** {precision:.2f}%')
st.write(f'**Recall:** {recall:.2f}%')
st.write(f'**F1 Score:** {f1:.2f}%')

# Q4: Feature impact analysis (Backward Elimination)
st.write("### Feature Impact Analysis")
if len(selected_features) > 1:
    st.write("Use the sidebar to select features and observe their impact on model performance.")
else:
    st.write("Select multiple features to perform feature impact analysis.")
