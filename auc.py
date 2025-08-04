import pandas as pd
import numpy as np
import joblib
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# === Load pretrained components ===
svm_model = joblib.load("models/svm_model.pkl")
dt_model = joblib.load("models/dt_model.pkl")
main_scaler = joblib.load("models/minmax_scaler.pkl")
pajsk_scaler = joblib.load("models/pajsk_minmax_scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Load and clean validation data ===
df = pd.read_csv("validation.csv")

df.columns = df.columns.str.strip().str.replace(' ', '_')

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
    '7._Grades_for_each_additional_SPM_subject_(_If_you_did_not_take_the_subject,_please_select_"None"_)_[Ekonomi]': 'SPM_Economy',
    '8._PAJSK_Score_(range:_0.00_–_10.00)': 'PAJSK_Score',
    '10._What_is_your_CGPA_for_your_pre-university_program?\n(e.g.,_Diploma,_Foundation,_Matrikulasi,_or_STPM)\n\nExample:_\n3.96': 'PreUni_CGPA',
    '11._What_is_your_CGPA_for_your_first_semester_in_degree_study?_\n\nExample:_\n3.96': 'Degree_CGPA',
    '1._Age_(in_January_2025)': 'Age',
    '4._What_is_your_household\'s_monthly_income_range?': 'Household_Income',
    '3._Citizenship_Status': 'Citizenship_Status',
    '12._What_is_your_current_field_of_study?': 'Field_of_Study',
    '13._Are_you_currently_receiving_a_merit-based_scholarship_(not_a_loan_or_PTPTN)?\n(e.g.,_JPA,_university_excellence_scholarship,_state_foundation_scholarships,_etc)': 'Scholarship_Status',
    '9._Top_5_Co-curricular_Highest_Achievements\n\nExample:\n-_Gold_Medalist_in_the_International_Mathematics_Olympiad_(IMO)_2023\n-_President_of_the_Science_&_Innovation_Club,_Organized_National_STEM_Fair_2022\n-_Captain_of_the_State_Football_Team,_Represented_Malaysia_in_ASEAN_Youth_Championship\n(Please_continue_in_this_format_until_you_have_listed_5_achievements.)': 'Top5Achievements'
}, inplace=True)

# === Map SPM grades ===
grade_map = {'A+': 10, 'A': 9, 'A-': 8, 'B+': 7, 'B': 6, 'C+': 5, 'C': 4, 'D': 3, 'E': 2, 'G': 1, 'None': 0}
spm_columns = [
    'SPM_BM', 'SPM_English', 'SPM_Sejarah', 'SPM_Math', 'SPM_PI',
    'SPM_Moral', 'SPM_AddMath', 'SPM_Physics', 'SPM_Chemistry',
    'SPM_Biology', 'SPM_Accounting', 'SPM_Economy'
]
for col in spm_columns:
    df[col] = df[col].map(grade_map)

# === Numeric conversion ===
num_cols = ['PAJSK_Score', 'PreUni_CGPA', 'Degree_CGPA', 'Age']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Household income ===
income_map = {
    'Below RM4,850': 0,
    'RM4,851 – RM7,100': 1,
    'RM7,101 – RM10,970': 2,
    'RM10,971 – RM15,040': 3,
    'Above RM15,040': 4
}
df['Household_Income'] = df['Household_Income'].map(income_map).fillna(2)

df['Scholarship_Status'] = df['Scholarship_Status'].replace({'Yes': 'eligible', 'No': 'notEligible'})

# === Label encoding ===
for col in ['Field_of_Study', 'Citizenship_Status', 'Scholarship_Status']:
    df[col] = encoders[col].transform(df[col].astype(str))

# === Scale numerical features ===
all_scaled_cols = num_cols + ['Household_Income'] + spm_columns
df[all_scaled_cols] = main_scaler.transform(df[all_scaled_cols])

# === Compute BERT scores ===
def score_achievements(text):
    if not isinstance(text, str): return 0.0
    lines = re.split(r'[\n•]\s*|\;\s+', text)
    lines = [line.strip() for line in lines if line.strip()]
    if not lines: return 0.0
    embeddings = bert_model.encode(lines)
    scores = [np.mean(e) for e in embeddings]
    bonus = min(len(lines), 5) * 0.01
    return np.mean(scores) + bonus

df['pajskAchievementScore'] = df['Top5Achievements'].apply(score_achievements)
df[['pajskAchievementScore']] = pajsk_scaler.transform(df[['pajskAchievementScore']])

# === Prepare input and ground truth ===
feature_cols = [
    'PAJSK_Score', 'PreUni_CGPA', 'Degree_CGPA',
    'Field_of_Study', 'Age', 'Household_Income',
    'Citizenship_Status', 'pajskAchievementScore'
] + spm_columns
X_val = df[feature_cols].fillna(0)
y_val = df['Scholarship_Status']

# === Hybrid prediction logic ===
svm_probs = svm_model.predict_proba(X_val)
svm_confidence = np.max(svm_probs, axis=1)
svm_preds = svm_model.predict(X_val)

final_preds = []
final_probs = []

for i in range(len(X_val)):
    conf = svm_confidence[i]
    if 0.4 <= conf <= 0.6:
        final_pred = dt_model.predict(X_val.iloc[[i]])[0]
        final_prob = dt_model.predict_proba(X_val.iloc[[i]])[0][1]  # Probability of eligible
    else:
        final_pred = svm_preds[i]
        final_prob = svm_probs[i][1]  # Probability of eligible
    final_preds.append(final_pred)
    final_probs.append(final_prob)

# === AUC Score ===
auc_score = roc_auc_score(y_val, final_probs)
print(f" ")
print(f"AUC Score: {auc_score:.4f}")
print(f" ")

# === Plot ROC curve ===
fpr, tpr, thresholds = roc_curve(y_val, final_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Hybrid Model (AUC = {auc_score:.4f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_validation.png")
print("ROC curve saved as 'roc_curve_validation.png'")
