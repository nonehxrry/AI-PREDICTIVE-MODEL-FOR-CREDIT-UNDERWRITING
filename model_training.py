










import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'credit_underwriting1.csv'  # Update the file path as needed
data = pd.read_csv(file_path)

s
#Preprocessing
#Encode categorical variables
categorical_columns = ['gender', 'marital_status', 'employee_status', 'residence_type', 'loan_purpose']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode the target variable
label_encoder_status = LabelEncoder()
data['loan_status'] = label_encoder_status.fit_transform(data['loan_status'])

# Define features and target
X = data.drop(columns=['loan_id', 'loan_status'])  # Exclude ID and target
y = data['loan_status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model with feature names explicitly added
model.feature_names_in_ = X.columns.tolist()
joblib.dump(model, 'best_features_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report_result)
