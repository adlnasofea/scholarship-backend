from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import joblib
import re

app = Flask(__name__)
CORS(app)

# Load once to avoid reloading on every request
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")))

        df.columns = [
            'timestamp', 'email', 'age', 'gender', 'citizenshipStatus',
            'householdIncome', 'numSPMSubjects', 'spmBahasaMelayu', 'spmEnglish',
            'spmHistory', 'spmMath', 'spmIslamicStudies', 'spmMoralStudies',
            'spmAddMath', 'spmPhysics', 'spmChemistry', 'spmBiology',
            'spmAccounting', 'spmEconomics', 'pajskScore', 'top5Achievements',
            'preUniCGPA', 'degreeCGPA', 'fieldOfStudy'
        ]

        df_original = df.copy()
        df = df.drop(columns=['timestamp', 'email', 'top5Achievements'])
        df = df.replace(['None', 'Others', 'other', '', 'null', 'N/A', 'NaN', 'nan'], np.nan)

        grade_map = {
            'A+': 9, 'A': 8, 'A-': 7, 'B+': 6, 'B': 5, 'C+': 4,
            'C': 3, 'D': 2, 'E': 1, 'G': 0
        }

        spm_subjects = [
            'spmBahasaMelayu', 'spmEnglish', 'spmHistory', 'spmMath',
            'spmIslamicStudies', 'spmMoralStudies', 'spmAddMath',
            'spmPhysics', 'spmChemistry', 'spmBiology',
            'spmAccounting', 'spmEconomics'
        ]

        for col in spm_subjects:
            df[col] = df[col].astype(str).str.extract(r'(A\+|A-|A|B\+|B|C\+|C|D|E|G)', expand=False)
            df[col] = df[col].map(grade_map)

        numeric_cols = [
            'age', 'numSPMSubjects', 'pajskScore', 'preUniCGPA', 'degreeCGPA', *spm_subjects
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

        for col, encoder in encoders.items():
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'None', 'null', 'N/A', ''], 'missing')
            df[col] = df[col].fillna('missing')
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'missing')
            df[col] = encoder.transform(df[col])

        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        df_original['prediction'] = predictions

        def compute_spm_string(row):
            display_order = ['A+', 'A', 'A-', 'B+', 'B', 'C+', 'C', 'D', 'E', 'G']
            grade_counts = {grade: 0 for grade in display_order}
            for subject in spm_subjects:
                original_val = str(row.get(subject, '')).strip()
                match = re.match(r'(A\+|A-|A|B\+|B|C\+|C|D|E|G)', original_val, re.IGNORECASE)
                if match:
                    grade_label = match.group(1).upper()
                    if grade_label in grade_counts:
                        grade_counts[grade_label] += 1
            result_parts = [f"{count}{grade}" for grade, count in grade_counts.items() if count > 0]
            return ' '.join(result_parts) if result_parts else "N/A"

        df_original['spmGradeString'] = df_original.apply(compute_spm_string, axis=1)

        def grouped_avg(col):
            return {
                "eligible": pd.to_numeric(df_original[df_original["prediction"] == 1][col], errors='coerce').mean(),
                "notEligible": pd.to_numeric(df_original[df_original["prediction"] == 0][col], errors='coerce').mean()
            }

        spm_analysis = [{"subject": subject, **grouped_avg(subject)} for subject in spm_subjects]

        response = {
            "summary": {
                "total": len(predictions),
                "eligible": int((predictions == 1).sum()),
                "notEligible": int((predictions == 0).sum())
            },
            "predictions": df_original.to_dict(orient="records"),
            "spm_analysis": spm_analysis,
            "pajsk_analysis": grouped_avg("pajskScore"),
            "preuni_analysis": grouped_avg("preUniCGPA"),
            "degree_analysis": grouped_avg("degreeCGPA")
        }

        return jsonify(sanitize(response))

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/predict-single', methods=['POST'])
def predict_single():
    try:
        data = request.get_json()
        spm_str = data.get("spmGradeString", "")
        grades = dict(re.findall(r"([^:]+):\s*([A\+A\-B\+B\-C\+CDEFG])", spm_str))

        spm_subjects = [
            'spmBahasaMelayu', 'spmEnglish', 'spmHistory', 'spmMath',
            'spmIslamicStudies', 'spmMoralStudies', 'spmAddMath',
            'spmPhysics', 'spmChemistry', 'spmBiology',
            'spmAccounting', 'spmEconomics'
        ]

        grade_map = {
            'A+': 9, 'A': 8, 'A-': 7, 'B+': 6, 'B': 5, 'C+': 4,
            'C': 3, 'D': 2, 'E': 1, 'G': 0
        }

        input_row = {
            'age': float(data.get("age", 0)),
            'gender': data.get("gender", "missing"),
            'citizenshipStatus': data.get("citizenshipStatus", "missing"),
            'householdIncome': data.get("householdIncome", "missing"),
            'pajskScore': float(data.get("pajskScore", 0)),
            'preUniCGPA': float(data.get("preUniCGPA", 0)),
            'degreeCGPA': float(data.get("degreeCGPA", 0)),
            'fieldOfStudy': data.get("fieldOfStudy", "missing"),
            'numSPMSubjects': len(grades)
        }

        for subject in spm_subjects:
            input_row[subject] = grade_map.get(grades.get(subject, ''), np.nan)

        df = pd.DataFrame([input_row])

        for col, encoder in encoders.items():
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'None', 'null', 'N/A', ''], 'missing')
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'missing')
            df[col] = encoder.transform(df[col])

        numeric_cols = [
            'age', 'numSPMSubjects', 'pajskScore', 'preUniCGPA', 'degreeCGPA', *spm_subjects
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled)[0][1]  # probability of class 1

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4)  # round to 4 decimals
        })

    except Exception as e:
        print("PREDICT-SINGLE ERROR:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
