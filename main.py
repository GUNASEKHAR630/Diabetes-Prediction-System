import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

print("Loading diabetes dataset...")
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
df = pd.read_csv(url)

print(f"Dataset Sahpe: Pdf.shape")
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget Distribution (Outcome):")
print(df['Outcome'].value_counts())
print(f"Diabetes Positive Raate: {df['Outcome'].mean()*100:.2f}%")

print("\nBasic Statistics:")
print(df.describe())

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                            labels=['Underweight', 'Normal', 'Overweight','Obese'])
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100],
                         labels=['Young', 'Middle', 'Senior', 'Elser'])

df['BMI_Category'] = df['BMI_Category'].astype(str)
df['Age_Group'] = df['Age_Group'].astype(str)

le_dict = {}
for col in ['BMI_Category', 'Age_Group']:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"\nTraining samples: {x_train.shape[0]}")
print(f"Testing samples : {x_test.shape[0]}")

print("\n Training models.....")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(x_train_scaled, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(x_train_scaled, y_train)

print("Models trained successfully")

def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    print(f"\n {model_name} Performance:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC : {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

print("="*50)
evaluate_model(lr_model, x_test_scaled, y_test, "Logistic Regression")
evaluate_model(rf_model, x_test_scaled, y_test, "Random Forest")

joblib.dump(rf_model, 'diabetes_rf_model.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')
joblib.dump(le_dict, 'Diabetes_label_encoders.pkl')
print("\n Best modl (Random Forest) and prprocessors saved")

def predict_diabetes(patient_data):
    model = joblib.load('diabetes_rf_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    encoders = joblib.load('diabetes_label_encoders.pkl')

    df_pred = pd.DataFrame([patient_data])

    for col, le in encoders.items():
        if col in df_pred.columns:
            df_pred[col] = le.transform(df_pred[col])
    
    features_scaled = scaler.transform(df_pred)

    probability = model.predict_proba(features_scaled)[0][1]
    prediction = "Diabetic" if probability >=0.5 else "Non_Diabetic"
    return {
        "Prediction": prediction,
        "Diabetes_Probability": round(probability*100, 2)
    }

if __name__ == "__main__":
    print("\n"+ "="*65)
    print(" DIABETES PREDICTION SYSTM READY")
    print("="*65)

    example_patient = {
        'Pregnancies':6,
        'Glucose':148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin':0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age' : 50,
        'BMI_Category': 'Obese',
        'Age_Group': 'Middle'
    }

    result = predict_diabetes(example_patient)
    print("\nExample Patient Prediction:")
    print(f"Result : {result['Prediction']}")
    print(f"Probablity :{result['Diabetes_Probability']}%")

    if result['Diabetes_Probability'] > 70:
        print("High Risk - Recommend immediate medical consultation")
    elif result['Diabetes_Probability']>40:
        print("Moderate Risk - Regualr Monitoring suggested")
    else:
        print("Low Risk")
