# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from google.colab import files
import io

# Upload your CSV
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Features and target
features = ['category','supplier','region','marketing_channel','price','quantity','delivery_days','is_damaged']
target = 'is_returned'

X = df[features]
y = df[target]

# Identify categorical and numerical columns
categorical_features = ['category','supplier','region','marketing_channel']
numerical_features = ['price','quantity','delivery_days','is_damaged']

# Preprocessing pipeline with imputer
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create Logistic Regression pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict probabilities
df['pred_return_prob'] = model.predict_proba(X)[:,1]

# Save updated CSV
output_filename = 'orders_with_predicted_returns.csv'
df.to_csv(output_filename, index=False)

print(f"Updated CSV saved as {output_filename}")
