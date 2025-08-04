from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mysqldb import MySQL
from werkzeug.security import check_password_hash
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import io
import joblib
import os
import re

from sentence_transformers import SentenceTransformer

print("🔁 Loaded THIS app.py!", flush=True)

# === Flask Setup ===
app = Flask(__name__)
CORS(app)

load_dotenv()
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')

mysql = MySQL(app)

# === Load Models and Tools ===
try:
    # Load all models and preprocessing tools from the 'models/' directory
    svm_model = joblib.load("models/svm_model.pkl")
    dt_model = joblib.load("models/dt_model.pkl")
    main_scaler = joblib.load("models/minmax_scaler.pkl") # Renamed from 'scaler' for clarity
    pajsk_scaler = joblib.load("models/pajsk_minmax_scaler.pkl") # New dedicated scaler for PAJSK
    encoders = joblib.load("models/label_encoders.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl") # The exact list of features expected by the model
    spm_columns_model_names = joblib.load("models/spm_columns.pkl") # SPM names as used in the model (e.g., SPM_BM)
    spm_grade_map = joblib.load("models/spm_grade_map.pkl") # The grade to number mapping
    household_income_map = joblib.load("models/household_income_map.pkl") # The income mapping

    bert_model = SentenceTransformer('all-MiniLM-L6-v2') # BERT model needs to be loaded directly

    # Reverse map for scholarship status for output
    scholarship_status_encoder = encoders.get('Scholarship_Status')
    if scholarship_status_encoder:
        print(f"DEBUG: scholarship_status_encoder classes: {scholarship_status_encoder.classes_}") # Add this line for verification
        scholarship_reverse_map = {idx: label for idx, label in enumerate(scholarship_status_encoder.classes_)}
    else:
        # Fallback if encoder not found (shouldn't happen if train_save_model.py ran correctly)
        # *** MODIFIED FALLBACK MAP ***
        scholarship_reverse_map = {0: "eligible", 1: "notEligible"} # Default based on common encoding
        print("DEBUG: Scholarship_Status encoder not found in loaded encoders. Using default map for 'notEligible'/'eligible'.")

    print("All models and preprocessing tools loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model assets: {e}. Ensure all .pkl files are in the 'models' directory.")
    exit() # Exit if models can't be loaded, app won't function

# --- Comprehensive Column Renaming Map (MUST MATCH train_save_model.py's renames) ---
# This maps the long, original CSV column names to the short names used in the model
column_rename_map = {
    # SPM Subjects
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
    
    # Other features
    '8._PAJSK_Score_(range:_0.00_–_10.00)': 'PAJSK_Score',
    '10._What_is_your_CGPA_for_your_pre-university_program?\n(e.g.,_Diploma,_Foundation,_Matrikulasi,_or_STPM)\n\nExample:_\n3.96': 'PreUni_CGPA',
    '11._What_is_your_CGPA_for_your_first_semester_in_degree_study?_\n\nExample:_\n3.96': 'Degree_CGPA',
    '1._Age_(in_January_2025)': 'Age',
    '2._Gender': 'Gender',
    '3._Citizenship_Status': 'Citizenship_Status',
    '4._What_is_your_household\'s_monthly_income_range?': 'Household_Income',
    '5._Number_of_SPM_subjects': "Num_SPM",
    '12._What_is_your_current_field_of_study?': 'Field_of_Study',
    '13._Are_you_currently_receiving_a_merit-based_scholarship_(not_a_loan_or_PTPTN)?\n(e.g.,_JPA,_university_excellence_scholarship,_state_foundation_scholarships,_etc)': 'Scholarship_Status',
    '9._Top_5_Co-curricular_Highest_Achievements\n\nExample:\n-_Gold_Medalist_in_the_International_Mathematics_Olympiad_(IMO)_2023\n-_President_of_the_Science_&_Innovation_Club,_Organized_National_STEM_Fair_2022\n-_Captain_of_the_State_Football_Team,_Represented_Malaysia_in_ASEAN_Youth_Championship\n(Please_continue_in_this_format_until_you_have_listed_5_achievements.)': 'Top5Achievements'
}

# Reverse map for SPM grades for display in predictions
spm_grade_reverse_map = {v: k for k, v in spm_grade_map.items()}

def sanitize(obj):
    """Recursively converts NaN to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

def compute_bert_score(text):
    """Computes a BERT-based achievement score."""
    if not isinstance(text, str):
        return 0.0
    lines = re.split(r'[\n•;]\s*|\s*;\s*|\s*•\s*', text)
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) == 0:
        return 0.0
    embeddings = bert_model.encode(lines)
    scores = [np.mean(emb) for emb in embeddings]
    base_score = np.mean(scores)
    bonus = min(len(lines), 5) * 0.01
    return base_score + bonus

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))

        # --- Initial Cleaning and Renaming (MUST MATCH TRAIN_SAVE_MODEL.PY's START) ---
        df.columns = df.columns.str.strip() # Remove extra spaces from original headers
        df.columns = df.columns.str.replace(' ', '_') # Replace spaces with underscores
        
        # Rename ALL relevant column 
        df.rename(columns=column_rename_map, inplace=True)
        
        print("DEBUG: DataFrame Columns after rename:", df.columns.tolist(), flush=True)
        print("DEBUG: Expected SPM model columns:", spm_columns_model_names, flush=True)

        # Assign Applicant ID after CSV is read and renamed
        df['applicantId'] = [f'{i:04d}' for i in range(1, len(df) + 1)] 

        df_original = df.copy() # Keep a copy of original data for display
        
        # --- PREPROCESSING STEPS (MUST EXACTLY MATCH TRAIN_SAVE_MODEL.PY) ---

        # 1. Convert SPM grades to numeric values using the loaded map
        for col_model_name in spm_columns_model_names:
            df[col_model_name] = df[col_model_name].astype(str).str.extract(r'(A\+|A-|A|B\+|B|C\+|C|D|E|G|None)', expand=False) # Added 'None' to regex
            df[col_model_name] = df[col_model_name].replace({'None': np.nan}) # Replace 'None' string with actual NaN
            df[col_model_name] = df[col_model_name].map(spm_grade_map)
            df[col_model_name] = df[col_model_name].fillna(0) # Fill NaN after mapping with 0
            df_original[col_model_name] = df[col_model_name].copy()

            print(f"DEBUG: Processed {col_model_name} (numerical): {df_original[col_model_name].iloc[0]}", flush=True)
            print(f"DEBUG: Original input for {col_model_name} (check manually): {df_original[col_model_name].map(spm_grade_reverse_map).iloc[0]}", flush=True)

        # 2. Compute BERT score for achievements
        df['pajskAchievementScore'] = df['Top5Achievements'].apply(compute_bert_score)

        # 3. Ordinal encode Household Income using the loaded map
        df['Household_Income'] = df['Household_Income'].map(household_income_map).fillna(
            household_income_map.get(df['Household_Income'].mode()[0], 2) if not df['Household_Income'].mode().empty else 2 # Use mode or default 2 if mode is empty
        )

        # 4. Handle numerical features: Convert to numeric, fill missing (for the model)
        # This list includes PAJSK, CGPA, Age, Household_Income, and all SPM grades
        all_numerical_cols_for_main_scaler_process = [
            'PAJSK_Score', 'PreUni_CGPA', 'Degree_CGPA', 'Age', 'Household_Income'
        ] + spm_columns_model_names

        for col in all_numerical_cols_for_main_scaler_process:
            # Convert to numeric, coercing errors (e.g., 'Others' in Age will become NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with 0. This is the fix for the TypeError.
            # Using 0 is a robust default that MinMaxScaler can handle.
            df[col] = df[col].fillna(0) 

        # 5. Label Encode categorical features using loaded encoders
        categorical_cols_for_encoding = ['Citizenship_Status', 'Field_of_Study']
        for col in categorical_cols_for_encoding:
            if col in df.columns:
                encoder = encoders.get(col)
                if encoder:
                    df[col] = df[col].astype(str)
                    # Handle unseen labels: convert to 'missing' if not in encoder classes
                    df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'missing')
                    # This transformation will succeed if 'missing' was in training data
                    df[col] = encoder.transform(df[col])
                else:
                    print(f"Warning: Encoder for {col} not found. Skipping encoding for this column.")
                    df[col] = 0 # Default to 0 or handle as error more robustly

        # 6. Apply Main Scaler to numerical features (including income and SPM grades)
        # This list must precisely match what main_scaler was fitted on in train_save_model.py
        cols_to_scale_by_main_scaler_final = [
            'PAJSK_Score', 'PreUni_CGPA', 'Degree_CGPA', 'Age', 'Household_Income'
        ] + spm_columns_model_names
        
        # Filter for only existing columns to prevent errors if a column is missing
        existing_cols_for_main_scaler = [col for col in cols_to_scale_by_main_scaler_final if col in df.columns]
        if existing_cols_for_main_scaler:
            df[existing_cols_for_main_scaler] = main_scaler.transform(df[existing_cols_for_main_scaler])
        else:
            print("Warning: No numerical columns found for main scaler transformation. Check feature names.")


        # 7. Apply PAJSK Scaler to pajskAchievementScore
        if 'pajskAchievementScore' in df.columns:
            df[['pajskAchievementScore']] = pajsk_scaler.transform(df[['pajskAchievementScore']])
        else:
            print("Warning: 'pajskAchievementScore' column not found for PAJSK scaler transformation.")


        # --- Prepare final DataFrame for model prediction ---
        # Ensure the order of columns matches feature_columns.pkl
        X_pred = df[feature_columns]
        X_pred = X_pred.fillna(0) # Final fillna just in case

        # --- Make Hybrid Prediction ---
        svm_probs = svm_model.predict_proba(X_pred)
        svm_preds = svm_model.predict(X_pred)

        # Hybrid fallback logic
        uncertain_mask = (svm_probs.max(axis=1) >= 0.4) & (svm_probs.max(axis=1) <= 0.6)
        final_preds = np.array(svm_preds)
        if np.any(uncertain_mask):
            dt_preds = dt_model.predict(X_pred[uncertain_mask])
            final_preds[uncertain_mask] = dt_preds

        # Add predictions to the original DataFrame for output
        df_original['prediction_encoded'] = final_preds
        df_original['prediction'] = df_original['prediction_encoded'].map(scholarship_reverse_map)
        df_original['confidence'] = svm_probs.max(axis=1)
        df_original['pajskAchievementScore'] = df['pajskAchievementScore']

        
        # --- Analysis Data (for grouped_avg) ---
        def grouped_avg(col_name_in_df_original):
            # This function needs to operate on numerical columns
            # Ensure the column exists and is numeric for averaging
            col_data = pd.to_numeric(df_original[col_name_in_df_original], errors='coerce')

            # Assuming 'eligible' maps to 1, 'notEligible' maps to 0
            eligible_values = col_data[df_original["prediction_encoded"] == 1]
            not_eligible_values = col_data[df_original["prediction_encoded"] == 0]

            return {
                "eligible": eligible_values.mean() if not eligible_values.empty else None,
                "notEligible": not_eligible_values.mean() if not not_eligible_values.empty else None
            }

        # Use the consistent, renamed column names for analysis
        spm_analysis = [{"subject": subject, **grouped_avg(subject)} for subject in spm_columns_model_names]
        pajsk_analysis = grouped_avg("PAJSK_Score")
        preuni_analysis = grouped_avg("PreUni_CGPA")
        degree_analysis = grouped_avg("Degree_CGPA")

        response = {
            "summary": {
                "total": len(final_preds),
                # *** MODIFIED SUMMARY COUNTS HERE ***
                "eligible": int((final_preds == scholarship_status_encoder.transform(["eligible".strip()])[0]).sum()),
                "notEligible": int((final_preds == scholarship_status_encoder.transform(["notEligible".strip()])[0]).sum())
            },
            "predictions": df_original.to_dict(orient="records"), # Send original data + new prediction
            "spm_analysis": spm_analysis,
            "pajsk_analysis": pajsk_analysis,
            "preuni_analysis": preuni_analysis,
            "degree_analysis": degree_analysis
        }

        print("DEBUG: First row of SPM numerical grades sent to frontend:", df_original[spm_columns_model_names].iloc[0].to_dict(), flush=True)

        return jsonify(sanitize(response))

    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({'error': str(e)}), 500

@app.route('/predict-single', methods=['POST'])
def predict_single():
    try:

        print("✅ predict_single triggered")

        data = request.get_json()

        print("📥 Incoming JSON data:", data, flush=True)
        
        # Prepare input dictionary with model-expected column names
        input_row = {}

        # 1. Direct Numerical Features (from JSON payload)
        input_row['PAJSK_Score'] = float(data.get("PAJSK_Score", 0))
        input_row['PreUni_CGPA'] = float(data.get("PreUni_CGPA", 0))
        input_row['Degree_CGPA'] = float(data.get("Degree_CGPA", 0))
        input_row['Age'] = float(data.get("Age", 0)) # Assuming 'age' is always numeric in single prediction
        
        # 2. Household Income (from JSON payload)
        input_row['Household_Income'] = data.get("Household_Income", "missing")
        input_row['Household_Income'] = household_income_map.get(input_row['Household_Income'], 2) # Map to numerical value, default to 2 if unseen

        # 3. Categorical Features (from JSON payload, for LabelEncoder)
        input_row['Citizenship_Status'] = data.get("Citizenship_Status", "missing")
        input_row['Field_of_Study'] = data.get("Field_of_Study", "missing")

        # 4. SPM Grades (from JSON payload)
        spm_grades_raw_input = {}

        incoming_spm_to_model_map = {
            'Bahasa Melayu': 'SPM_BM',
            'Bahasa Inggeris': 'SPM_English',
            'Sejarah': 'SPM_Sejarah',
            'Matematik': 'SPM_Math',
            'Pendidikan Islam': 'SPM_PI',
            'Pendidikan Moral': 'SPM_Moral',
            'Matematik Tambahan': 'SPM_AddMath',
            'Fizik': 'SPM_Physics',
            'Kimia': 'SPM_Chemistry',
            'Biologi': 'SPM_Biology',
            'Prinsip Perakaunan': 'SPM_Accounting',
            'Ekonomi': 'SPM_Economy'
        }

        # Prioritize 'spmGradeString' if present (as in your original logic for single)
        if "spmresult" in data:
            spm_str = data.get("spmresult", "")
            
            # Strictly match only valid SPM grades in your grade_map
            parsed_grades = re.findall(r"([\w\s]+?):\s*(A\+|A\-|A|B\+|B|C\+|C|D|E|G)", spm_str)
            
            for subj_name_camel, grade_char in parsed_grades:
                spm_grades_raw_input[subj_name_camel.strip()] = grade_char.strip()
        else:
            for incoming_json_key, model_spm_name in incoming_spm_to_model_map.items():
                if incoming_json_key in data:
                    spm_grades_raw_input[incoming_json_key] = data.get(incoming_json_key)

        # Now, convert to model's numerical format using spm_grade_map
        for col_model_name in spm_columns_model_names:
            grade_char = None
            # Check if the model's expected name is directly in the raw input (e.g. from /predict)
            if col_model_name in spm_grades_raw_input:
                grade_char = spm_grades_raw_input.get(col_model_name)
            else: # Otherwise, try to map from the simplified frontend keys
                for incoming_json_key, model_mapped_name in incoming_spm_to_model_map.items():
                    if model_mapped_name == col_model_name and incoming_json_key in spm_grades_raw_input:
                        grade_char = spm_grades_raw_input.get(incoming_json_key)
                        break

            input_row[col_model_name] = spm_grade_map.get(grade_char, 0) # Convert to number, default 0


        # 5. pajskAchievementScore
        input_row['pajskAchievementScore'] = compute_bert_score(data.get("Top5Achievements", ""))

        df_single = pd.DataFrame([input_row])

        print("Parsed SPM:", input_row)
        print("BERT score:", input_row['pajskAchievementScore'])
        print("Final DataFrame:", df_single.head())


        # --- PREPROCESSING STEPS FOR SINGLE PREDICTION (MIRRORING /PREDICT) ---

        # 1. Apply Label Encoders
        categorical_cols_for_encoding = ['Citizenship_Status', 'Field_of_Study']
        for col in categorical_cols_for_encoding:
            encoder = encoders.get(col)
            if encoder:
                df_single[col] = df_single[col].astype(str)
                df_single[col] = df_single[col].apply(lambda x: x if x in encoder.classes_ else 'missing')
                df_single[col] = encoder.transform(df_single[col])
            else:
                df_single[col] = 0 # Default to 0 if encoder or column not found

        # 2. Apply Main Scaler to numerical features (including income and SPM grades)
        # This list must precisely match what main_scaler was fitted on in train_save_model.py
        cols_to_scale_by_main_scaler_final = [
            'PAJSK_Score', 'PreUni_CGPA', 'Degree_CGPA', 'Age', 'Household_Income'
        ] + spm_columns_model_names
        
        existing_cols_for_main_scaler = [col for col in cols_to_scale_by_main_scaler_final if col in df_single.columns]
        if existing_cols_for_main_scaler:
            df_single[existing_cols_for_main_scaler] = main_scaler.transform(df_single[existing_cols_for_main_scaler])

        # 3. Apply PAJSK Scaler
        if 'pajskAchievementScore' in df_single.columns:
            df_single[['pajskAchievementScore']] = pajsk_scaler.transform(df_single[['pajskAchievementScore']])
        
        # Prepare final DataFrame for model prediction
        X_pred_single = df_single[feature_columns]
        X_pred_single = X_pred_single.fillna(0) # Final fillna

        # Make Hybrid Prediction
        svm_probs = svm_model.predict_proba(X_pred_single)
        svm_pred = svm_model.predict(X_pred_single)[0]
        confidence = float(np.max(svm_probs[0])) # Max probability for the predicted class

        # Hybrid fallback logic for single prediction
        if (confidence >= 0.4) and (confidence <= 0.6):
            final_pred_encoded = dt_model.predict(X_pred_single)[0]
        else:
            final_pred_encoded = svm_pred

        # Inverse transform the prediction to original label
        # This uses scholarship_reverse_map, which is updated to 'eligible'/'notEligible'
        final_pred_label = scholarship_reverse_map.get(final_pred_encoded, "Unknown")

        print("✅ predict_single triggered", flush=True)
        print("Parsed SPM:", input_row, flush=True)
        print("BERT score (raw):", input_row['pajskAchievementScore'], flush=True)

        print("🚨 Final DataFrame sent to model:", flush=True)
        print(df_single[feature_columns].head(), flush=True)

        print("✅ SVM prediction:", svm_pred, flush=True)
        print("✅ SVM confidence:", confidence, flush=True)
        print("✅ Was fallback used (0.4–0.6)?", 0.4 <= confidence <= 0.6, flush=True)
        print("✅ Final prediction (after hybrid):", final_pred_encoded, flush=True)
        print("✅ Final label (mapped):", final_pred_label, flush=True)
        print("⚠️ DEBUG LOG END ⚠️", flush=True)



        return jsonify({
            "prediction": final_pred_label, # "eligible" or "notEligible"
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print(f"Error in /predict-single endpoint: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)