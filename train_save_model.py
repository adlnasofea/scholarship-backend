import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import os for directory creation

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# === 1: Load Data ===
print("1. Loading the dataset...")
df = pd.read_csv("train_test.csv")

# # Split into 90% train_test and 10% validation
# train_test_df, validation_df = train_test_split(df, test_size=0.10, random_state=42, shuffle=True)

# # Save both files
# train_test_df.to_csv("train_test.csv", index=False)
# validation_df.to_csv("validation.csv", index=False)

# print("Saved to 'train_test.csv' and 'validation.csv'")
    
# === 2: Clean column names and check dataset shape ===
print("2. Cleaning column names...")
df.columns = df.columns.str.strip()  # Remove extra spaces
df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
print(f"Cleaned Columns: {df.columns.tolist()}")
print(f"Dataset shape: {df.shape}")

# === 3: Rename SPM columns and convert grades to numeric values ===
print("3. Renaming SPM columns and converting grades to numeric values...")
df.rename(columns={
    '6._Grades_for_each_core_SPM_subjects__[Bahasa_Melayu]': 'SPM_BM',
    '6._Grades_for_each_core_SPM_subjects__[Bahasa_Inggeris]': 'SPM_English',
    '6._Grades_for_each_core_SPM_subjects__[Sejarah]': 'SPM_Sejarah',
    '6._Grades_for_each_core_SPM_subjects__[Matematik]': 'SPM_Math',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Pendidikan_Islam]': 'SPM_PI',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Pendidikan_Moral]': 'SPM_Moral',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Matematik_Tambahan]': 'SPM_AddMath',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Fizik]': 'SPM_Physics',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Kimia]': 'SPM_Chemistry',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Biologi]': 'SPM_Biology',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Prinsip_Perakaunan]': 'SPM_Accounting',
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Ekonomi]': 'SPM_Economy'
}, inplace=True)

# Define grade-to-number mapping
grade_map = {
    'A+': 10, 'A': 9, 'A-': 8,
    'B+': 7, 'B': 6,
    'C+': 5, 'C': 4,
    'D': 3, 'E': 2, 'G': 1, 'None': 0
}

# List of cleaned SPM column names
spm_columns = [
    'SPM_BM', 'SPM_English', 'SPM_Sejarah', 'SPM_Math',
    'SPM_PI', 'SPM_Moral', 'SPM_AddMath', 'SPM_Physics', 'SPM_Chemistry',
    'SPM_Biology', 'SPM_Accounting', 'SPM_Economy'
]

# Apply the mapping to each subject
for col in spm_columns:
    df[col] = df[col].map(grade_map)

# === 4: Preprocess other requirements except top5achievements ===
print("4. Preprocessing other numerical and categorical features...")

# === Rename long columns to simple names ===
df.rename(columns={
    '8._PAJSK_Score_(range:_0.00_–_10.00)': 'PAJSK_Score',
    '10._What_is_your_CGPA_for_your_pre-university_program?\n(e.g.,_Diploma,_Foundation,_Matrikulasi,_or_STPM)\n\nExample:_\n3.96': 'PreUni_CGPA',
    '11._What_is_your_CGPA_for_your_first_semester_in_degree_study?_\n\nExample:_\n3.96': 'Degree_CGPA',
    '1._Age_(in_January_2025)': 'Age',
    '4._What_is_your_household\'s_monthly_income_range?': 'Household_Income',
    '3._Citizenship_Status': 'Citizenship_Status',
    '12._What_is_your_current_field_of_study?': 'Field_of_Study',
    '13._Are_you_currently_receiving_a_merit-based_scholarship_(not_a_loan_or_PTPTN)?\n(e.g.,_JPA,_university_excellence_scholarship,_state_foundation_scholarships,_etc)': 'Scholarship_Status',
    '9._Top_5_Co-curricular_Highest_Achievements\n\nExample:\n-_Gold_Medalist_in_the_International_Mathematics_Olympiad_(IMO)_2023\n-_President_of_the_Science_&_Innovation_Club,_Organized_National_STEM_Fair_2022\n-_Captain_of_the_State_Football_Team,_Represented_Malaysia_in_ASEAN_Youth_Championship\n(Please_continue_in_this_format_until_you_have_listed_5_achievements.)':
    'Top5Achievements'
}, inplace=True)

# *** MODIFIED STEP: Relabel 'Yes'/'No' to 'eligible'/'notEligible' ***
print("Relabeling 'Scholarship_Status' from 'Yes'/'No' to 'eligible'/'notEligible'...")
status_relabel_map = {
    'Yes': 'eligible',
    'No': 'notEligible'
}
# Use .replace() for robustness and fillna for any missing values
df['Scholarship_Status'] = df['Scholarship_Status'].replace(status_relabel_map).fillna('notEligible')
print(f"Unique labels in Scholarship_Status after relabeling: {df['Scholarship_Status'].unique()}")


# === Handle numeric features: Convert + Fill missing ===
num_cols = ['PAJSK_Score', 'PreUni_CGPA', 'Degree_CGPA', 'Age']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# === Ordinal encode for Household Income ===
income_map = {
    'Below RM4,850': 0,
    'RM4,851 – RM7,100': 1,
    'RM7,101 – RM10,970': 2,
    'RM10,971 – RM15,040': 3,
    'Above RM15,040': 4
}
df['Household_Income'] = df['Household_Income'].map(income_map).fillna(2)

# === Label encode for categorical features ===
# Now Scholarship_Status will contain 'eligible'/'notEligible'
label_cols = ['Field_of_Study', 'Citizenship_Status', 'Scholarship_Status']
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    # Fit and transform, then store the fitted encoder
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le # Save the encoder for future use
    print(f"LabelEncoder for {col} fitted on classes: {le.classes_}") # Debugging: Check learned classes

# === Scale all numerical columns including income AND SPM grades ===
# THIS IS THE KEY CHANGE: Added spm_columns to the list to be scaled by main_scaler
all_scaled_cols = num_cols + ['Household_Income'] + spm_columns
scaler = MinMaxScaler()
df[all_scaled_cols] = scaler.fit_transform(df[all_scaled_cols])

# === 5: Convert top 5 achievements into BERT-based score ===
print("5. Generating BERT-based achievement scores...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function: Clean, split, score lines, reward longer entries
def score_achievements(text):
    if not isinstance(text, str):
        return 0.0

    # Split into bullet points (each line)
    lines = re.split(r'[\n•]\s*|\;\s+', text)
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    if len(lines) == 0:
        return 0.0

    # Get BERT embeddings for each line
    embeddings = bert_model.encode(lines)
    scores = [np.mean(emb) for emb in embeddings]

    # Mean of scores + small bonus for length (e.g. 5 lines = +5%)
    base_score = np.mean(scores)
    bonus = min(len(lines), 5) * 0.01  # 1% per line, max 5%
    return base_score + bonus

# Apply scoring to the column
df['pajskAchievementScore'] = df['Top5Achievements'].apply(score_achievements)

# Normalize after scoring using its dedicated scaler
pajsk_scaler = MinMaxScaler() # Ensure this is a new instance
df[['pajskAchievementScore']] = pajsk_scaler.fit_transform(df[['pajskAchievementScore']])

# === 6: Prepare data for model training ===
print("6. Preparing data for model training...")
# Define the final list of input features (X)
feature_cols = [
    'PAJSK_Score',
    'PreUni_CGPA',
    'Degree_CGPA',
    'Field_of_Study',
    'Age',
    'Household_Income',
    'Citizenship_Status',
    'pajskAchievementScore'
] + spm_columns  # include all SPM grades

# Define input X and target y
X = df[feature_cols]
y = df['Scholarship_Status'] # This column is already label encoded

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fix: Ensure no missing values by filling with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("y_train distribution (encoded values):\n", y_train.value_counts(normalize=True))


# === 7: Train Hybrid SVM -> DT model with evaluation metrics ===
print("7. Training the Hybrid SVM and Decision Tree models...")
# Train SVM with probability enabled
svm_model = SVC(C=3, gamma=0.1, kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
print("Models trained successfully.")

# # === Evaluation ===
# print("\n=== Model Evaluation ===")

# # 1. Training Accuracy
# train_preds = svm_model.predict(X_train)
# train_accuracy = accuracy_score(y_train, train_preds)
# print(f"Training Accuracy: {round(train_accuracy * 100, 2)} %")

# # Predict on test set using SVM
# svm_probs = svm_model.predict_proba(X_test)
# svm_preds = svm_model.predict(X_test)

# # Hybrid fallback: use DT if SVM confidence is between 0.4 and 0.6
# uncertain_mask = (svm_probs.max(axis=1) >= 0.4) & (svm_probs.max(axis=1) <= 0.6)
# final_preds = np.array(svm_preds)
# if np.any(uncertain_mask): # Only predict with DT if there are uncertain cases
#     dt_preds = dt_model.predict(X_test[uncertain_mask])
#     final_preds[uncertain_mask] = dt_preds

# # 2. Testing Accuracy (final_preds includes DT fallback)
# test_accuracy = accuracy_score(y_test, final_preds)
# print(f"Testing Accuracy (Hybrid SVM → DT): {round(test_accuracy * 100, 2)} %")

# # 3. Classification Report
# print("\nClassification Report:\n", classification_report(y_test, final_preds))

# # 4. Confusion Matrix (visualized)
# cm = confusion_matrix(y_test, final_preds)
# labels = sorted(y_test.unique())  # ensure labels match actual values

# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=labels, yticklabels=labels)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.savefig('confusion_matrix.png') # Save the plot instead of showing in a script
# print("Confusion Matrix saved as 'confusion_matrix.png'")

# === 8: Save models and tools ===
print("\n8. Saving trained models and preprocessing tools...")
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(dt_model, 'models/dt_model.pkl')
joblib.dump(scaler, 'models/minmax_scaler.pkl') # Use a more descriptive name for the main scaler
joblib.dump(pajsk_scaler, 'models/pajsk_minmax_scaler.pkl') # Save the scaler specifically for PAJSK scores
joblib.dump(encoders, 'models/label_encoders.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl') # Save the list of feature columns
joblib.dump(spm_columns, 'models/spm_columns.pkl') # Save SPM columns for reusability
joblib.dump(grade_map, 'models/spm_grade_map.pkl') # Save the grade map
joblib.dump(income_map, 'models/household_income_map.pkl') # Save the income map

print("\nAll models and preprocessing tools have been saved in the 'models' directory.")
print("Script execution finished.")