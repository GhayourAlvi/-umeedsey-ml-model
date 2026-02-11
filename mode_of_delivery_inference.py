"""
Mode of Delivery Inference Script

This script loads trained mode of delivery models and makes predictions on CSV files.
It determines the appropriate trimester group (1+2 or 3) based on gestational age
and uses the corresponding models for predictions.

Usage:
    python mode_of_delivery_inference.py input.csv output.csv
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_value(value: str) -> str:
    """Clean string values by removing extra quotes, brackets, and spaces."""
    if pd.isna(value) or value == "":
        return ""
    return str(value).strip().strip("'").strip('"').strip("]").strip("[").lower()


def safe_process_field(field_name: str, field_value: Any, conditions: List[str]) -> Dict[str, str]:
    """Safely process a field with error handling for problematic field names."""
    try:
        if field_value in (None, ""):
            field_value = "no"
        field_value = str(field_value).lower()
        row_list = [x.strip().lower() for x in str(field_value).split(",")]
        return {cond: ("yes" if cond.lower() in row_list else "no") for cond in conditions}
    except Exception:
        return {cond: "no" for cond in conditions}


def creating_features(data: List[str], record_lower: Dict[str, Any], encoder_model: Any, dic_json: Dict[str, List[str]]):
    """Transform record into encoded feature space."""
    if dic_json is None:
        raise ValueError("dic_json not provided")

    # Build the feature dictionary
    filtered = {}
    expanded_columns = []
    
    # Process each column in the data list
    for col in data:
        if col not in dic_json:
            # Regular column (not comma-separated) - keep as-is
            expanded_columns.append(col)
            if col in record_lower:
                filtered[col] = record_lower[col]
            else:
                filtered[col] = ""
        else:
            # Comma-separated column - expand into binary features
            conditions = dic_json[col]
            row_value = record_lower.get(col, "")
            
            # Process comma-separated values into binary features
            encoded_row = safe_process_field(col, row_value, conditions)
            
            # Add expanded binary features (column:value format)
            for k in sorted(conditions):
                expanded_col_name = f"{col}:{k}"
                expanded_columns.append(expanded_col_name)
                filtered[expanded_col_name] = encoded_row.get(k, "no")

    # Create DataFrame
    feature_df = pd.DataFrame([filtered])
    
    # Get expected feature names from encoder
    expected_features = None
    try:
        if hasattr(encoder_model, 'feature_names_in_') and encoder_model.feature_names_in_ is not None:
            expected_features = list(encoder_model.feature_names_in_)
    except Exception:
        pass

    # Reorder columns to match expected order if available
    if expected_features is not None:
        ordered_data = {}
        for col in expected_features:
            if col in filtered:
                ordered_data[col] = filtered[col]
            else:
                ordered_data[col] = ""
        feature_df = pd.DataFrame([ordered_data])
        feature_df = feature_df[expected_features]
    else:
        feature_df = feature_df[[col for col in expanded_columns if col in feature_df.columns]]
    
    # Ensure all string values are lowercase
    feature_df = feature_df.map(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Fill NaN values with empty string
    feature_df = feature_df.fillna("")
    
    # Transform using encoder
    return encoder_model.transform(feature_df)


def get_top_k_labels(col_list: List[str], probs_list: List[float], k: int = 3) -> List[str]:
    """Return top-k label names by probability (descending)."""
    if not probs_list:
        return []

    try:
        arr = np.array(probs_list, dtype=float)
    except Exception:
        try:
            arr = np.array([float(x) for x in probs_list], dtype=float)
        except Exception:
            return []

    idx = list(np.argsort(-arr)[:k])
    return [col_list[i] for i in idx if i < len(col_list)]


def predict_labels_for_model(
    model: Any,
    X_new: Any,
    col_list: List[str],
    threshold: float = 0.1,
) -> Tuple[List[str], List[float]]:
    """Convert model outputs to label names and keep raw probabilities."""
    if model is None:
        return ["Model not loaded"], []

    probs_for_sample = None

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_new)
        except Exception as e:
            print(f"WARNING: predict_proba failed: {e}")
            probs = None

        if isinstance(probs, list):
            extracted = []
            for arr in probs:
                try:
                    arr_np = np.array(arr)
                    if arr_np.ndim == 2 and arr_np.shape[1] >= 2:
                        extracted.append(float(arr_np[0, 1]))
                    else:
                        extracted.append(float(np.ravel(arr_np)[0]))
                except Exception:
                    extracted.append(0.0)
            probs_for_sample = extracted

        elif isinstance(probs, np.ndarray):
            if probs.ndim == 2 and probs.shape[1] == len(col_list):
                probs_for_sample = probs[0].tolist()
            elif probs.ndim == 2 and probs.shape[1] == 2:
                probs_for_sample = [float(probs[0, 1])]
            else:
                probs_for_sample = probs.ravel().tolist()

    if probs_for_sample is None and hasattr(model, "predict"):
        try:
            pred = model.predict(X_new)
            arr = np.array(pred)
            if arr.ndim == 2:
                indices = np.where(arr[0] == 1)[0].tolist()
                labels = [col_list[i] for i in indices if i < len(col_list)]
                return labels if labels else ["Model failed to predict"], []
            else:
                idx = int(arr[0])
                return (
                    [col_list[idx]] if idx < len(col_list) else ["Model failed to predict"],
                    [],
                )
        except Exception as e:
            print(f"WARNING: predict failed: {e}")
            return ["Model prediction failed"], []

    if probs_for_sample is None:
        return ["Model failed to produce probabilities"], []

    try:
        float_probs = [float(p) for p in probs_for_sample]
    except Exception:
        float_probs = []

    selected_indices = [i for i, p in enumerate(float_probs) if p >= float(threshold)]
    if not selected_indices and float_probs:
        selected_indices = [int(np.argmax(float_probs))]

    labels = [col_list[i] for i in selected_indices if i < len(col_list)]
    if not labels:
        labels = ["Model failed to predict"]
    return labels, float_probs


def load_models():
    """Load all trained mode of delivery models."""
    models = {}
    save_dir = "./save_models"
    
    # Load models for each trimester group
    model_files = {
        "trimester_1_2_history": "mode_of_delivery_trimester1_2_history.pkl",
        "trimester_1_2_condition": "mode_of_delivery_trimester1_2_condition.pkl",
        "trimester_3_history": "mode_of_delivery_trimester3_history.pkl",
        "trimester_3_condition": "mode_of_delivery_trimester3_condition.pkl",
    }
    
    for key, filename in model_files.items():
        filepath = os.path.join(save_dir, filename)
        if os.path.exists(filepath):
            try:
                model_data = joblib.load(filepath)
                models[key] = {
                    "model": model_data.get("mode_of_delivery_model"),
                    "encoder": model_data.get("encoder"),
                    "col_list": model_data.get("col_list", []),
                    "orignal_features": model_data.get("orignal_features", {}),
                }
                print(f"✅ Loaded {key} from {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                models[key] = None
        else:
            print(f"⚠️  {filename} not found")
            models[key] = None
    
    return models


def load_history_condition_features():
    """Load history and condition features from CSV."""
    try:
        df_features = pd.read_csv("./dataset/Divided History Features.csv")
        history = df_features.iloc[0, :].dropna().tolist()
        condition = df_features.iloc[1, :].dropna().tolist()
        
        # Features to exclude from mode of delivery model
        features_to_exclude = [
            "PatientAge",
            "PatientHasPosGp",
            "PartenerHasPosGp",
            "First degree relative with following medical conditions",
            "Mental Health Symptoms",
            "Previous history of PPH (Postpartum Hemorrhage(PPH)) Exploration",
            "Any Anesthesia Complication previously",
            "Pallor",
        ]
        
        # Remove excluded features from history and condition lists
        history = [f for f in history if f not in features_to_exclude]
        condition = [f for f in condition if f not in features_to_exclude]
        
        return history, condition
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return [], []


def convert_ga_to_days(ga_value):
    """
    Convert gestational age from "weeks.days" format to total days.
    Example: 23.5 (23 weeks and 5 days) -> 23 * 7 + 5 = 166 days
    
    Args:
        ga_value: Gestational age value (can be float, int, or string)
        
    Returns:
        Total days as float or None if conversion fails
    """
    if pd.isna(ga_value):
        return None
    
    try:
        # Convert to string first to handle decimal properly
        ga_str = str(ga_value)
        
        # Split by decimal point
        if '.' in ga_str:
            parts = ga_str.split('.')
            weeks = int(float(parts[0]))  # Before decimal point
            days = int(float('0.' + parts[1]) * 10) if len(parts) > 1 else 0  # After decimal point
            total_days = weeks * 7 + days
        else:
            # If no decimal point, treat as whole weeks
            weeks = int(float(ga_str))
            total_days = weeks * 7
        
        return float(total_days)
    except (ValueError, AttributeError):
        # If conversion fails, try direct float conversion
        try:
            return float(ga_value) * 7  # Assume it's already in weeks
        except:
            return None


def determine_trimester_group(ga_value: float) -> str:
    """Determine trimester group based on gestational age (in days)."""
    # Convert gestational age to days first
    ga_days = convert_ga_to_days(ga_value)
    
    if ga_days is None:
        return "trimester_1_2"  # Default to 1+2 if missing
    
    try:
        # Trimester ranges in days:
        # Combined Trimester 1+2: 0-27 weeks = 0-189 days (27 * 7 = 189)
        # Trimester 3: 28+ weeks = 196+ days (28 * 7 = 196)
        if ga_days >= 0 and ga_days <= 189:
            return "trimester_1_2"
        elif ga_days >= 196:
            return "trimester_3"
        else:
            return "trimester_1_2"  # Default
    except Exception:
        return "trimester_1_2"  # Default


def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess a single record for prediction."""
    # Fill missing values in examination columns
    if "Per speculum examination (If Yes, specify findings)" not in record or pd.isna(record.get("Per speculum examination (If Yes, specify findings)")):
        record["Per speculum examination (If Yes, specify findings)"] = "no"
    if "Per vaginal examination (If Yes, specify findings)" not in record or pd.isna(record.get("Per vaginal examination (If Yes, specify findings)")):
        record["Per vaginal examination (If Yes, specify findings)"] = "no"
    
    # Convert to lowercase strings
    record_lower = {}
    for k, v in record.items():
        if pd.isna(v):
            record_lower[k] = ""
        elif isinstance(v, str):
            record_lower[k] = v.strip().lower()
        else:
            record_lower[k] = str(v).strip().lower()
    
    return record_lower


def predict_mode_of_delivery(record: Dict[str, Any], models: Dict, history_data: List[str], condition_data: List[str]) -> Dict[str, Any]:
    """Make mode of delivery prediction for a single record.
    
    Returns:
        Dictionary with separate predictions for history and condition models,
        each with top 2 predictions and their probabilities.
    """
    # Preprocess record
    record_lower = preprocess_record(record)
    
    # Determine trimester group
    ga_value = record.get("Gestational Age (GA) (in weeks)", None)
    trimester_group = determine_trimester_group(ga_value)
    
    # Get appropriate models
    history_key = f"{trimester_group}_history"
    condition_key = f"{trimester_group}_condition"
    
    result = {
        "history_prediction": "",
        "history_probability": "",
        "condition_prediction": "",
        "condition_probability": "",
        "combined_prediction": "",
        "trimester_group": trimester_group,
    }
    
    if models.get(history_key) is None or models.get(condition_key) is None:
        result["history_prediction"] = "Models not loaded for this trimester group"
        result["condition_prediction"] = "Models not loaded for this trimester group"
        return result
    
    history_model_data = models[history_key]
    condition_model_data = models[condition_key]
    
    # Initialize dictionaries to store label-probability mappings for combined prediction
    history_label_probs = {}
    condition_label_probs = {}
    
    # Remove keys that shouldn't be in features
    temp_record = record_lower.copy()
    keys_to_remove = [
        "Postnatal_Symptoms",
        "Postnatal_Examination",
        "Mode_of_delivery2",
        "Antenatal_Peripartum_Maternal_Complications",
        "Neonatal__Fetal_Complications",
        "Postnatal_Maternal_Complications",
        "Neonatal Apgar Score:(1 Minute)",
        "Neonatal Apgar Score:(5 Minute)",
        # Features excluded from mode of delivery model
        "PatientAge",
        "PatientHasPosGp",
        "PartenerHasPosGp",
        "First degree relative with following medical conditions",
        "Mental Health Symptoms",
        "Previous history of PPH (Postpartum Hemorrhage(PPH)) Exploration",
        "Any Anesthesia Complication previously",
        "Pallor",
    ]
    for k in keys_to_remove:
        temp_record.pop(k, None)
    
    # Create features for history model
    try:
        history_features = creating_features(
            history_data,
            temp_record,
            history_model_data["encoder"],
            history_model_data["orignal_features"]
        )
        
        # Make predictions
        _, probs_history = predict_labels_for_model(
            history_model_data["model"],
            history_features,
            history_model_data["col_list"],
        )
        history_top = get_top_k_labels(history_model_data["col_list"], probs_history, k=2)
        
        # Get probabilities for top 2
        if probs_history and len(probs_history) > 0:
            arr = np.array(probs_history, dtype=float)
            top_indices = list(np.argsort(-arr)[:2])
            top_probs = [float(probs_history[i]) for i in top_indices if i < len(probs_history)]
            top_labels = [history_model_data["col_list"][i] for i in top_indices if i < len(history_model_data["col_list"])]
            
            # Store label-probability pairs for combined prediction
            for label, prob in zip(top_labels, top_probs):
                history_label_probs[label] = prob
            
            result["history_prediction"] = ",".join(str(x) for x in top_labels)
            result["history_probability"] = ",".join([f"{p:.4f}" for p in top_probs])
        else:
            result["history_prediction"] = ",".join(str(x) for x in history_top) if history_top else "Model failed to predict"
            result["history_probability"] = ""
    except Exception as e:
        print(f"Error in history prediction: {e}")
        result["history_prediction"] = f"Error: {str(e)}"
        result["history_probability"] = ""
    
    # Create features for condition model
    try:
        condition_features = creating_features(
            condition_data,
            temp_record,
            condition_model_data["encoder"],
            condition_model_data["orignal_features"]
        )
        
        # Make predictions
        _, probs_condition = predict_labels_for_model(
            condition_model_data["model"],
            condition_features,
            condition_model_data["col_list"],
        )
        condition_top = get_top_k_labels(condition_model_data["col_list"], probs_condition, k=2)
        
        # Get probabilities for top 2
        if probs_condition and len(probs_condition) > 0:
            arr = np.array(probs_condition, dtype=float)
            top_indices = list(np.argsort(-arr)[:2])
            top_probs = [float(probs_condition[i]) for i in top_indices if i < len(probs_condition)]
            top_labels = [condition_model_data["col_list"][i] for i in top_indices if i < len(condition_model_data["col_list"])]
            
            # Store label-probability pairs for combined prediction
            for label, prob in zip(top_labels, top_probs):
                condition_label_probs[label] = prob
            
            result["condition_prediction"] = ",".join(str(x) for x in top_labels)
            result["condition_probability"] = ",".join([f"{p:.4f}" for p in top_probs])
        else:
            result["condition_prediction"] = ",".join(str(x) for x in condition_top) if condition_top else "Model failed to predict"
            result["condition_probability"] = ""
    except Exception as e:
        print(f"Error in condition prediction: {e}")
        result["condition_prediction"] = f"Error: {str(e)}"
        result["condition_probability"] = ""
    
    # Combine predictions and sort by probability (highest to lowest, left to right)
    # Merge probabilities from both models - if a label appears in both, take the maximum probability
    combined_label_probs = {}
    
    # Add history model probabilities
    for label, prob in history_label_probs.items():
        combined_label_probs[label] = max(combined_label_probs.get(label, 0), prob)
    
    # Add condition model probabilities (take max if label already exists)
    for label, prob in condition_label_probs.items():
        combined_label_probs[label] = max(combined_label_probs.get(label, 0), prob)
    
    # Sort by probability (descending - highest first, left to right)
    if combined_label_probs:
        sorted_labels = sorted(combined_label_probs.items(), key=lambda x: x[1], reverse=True)
        combined = [label for label, prob in sorted_labels]
    else:
        # Fallback: use simple union if no probabilities available
        history_labels = result["history_prediction"].split(",") if result["history_prediction"] else []
        condition_labels = result["condition_prediction"].split(",") if result["condition_prediction"] else []
        combined = list(set(history_labels + condition_labels))
    
    if not combined or combined == [""]:
        combined = ["Model failed to predict"]
    result["combined_prediction"] = ",".join(str(x) for x in combined)
    
    return result


def main():
    """Main inference function."""
    if len(sys.argv) < 3:
        print("Usage: python inference.py <input_csv> <output_csv>")
        print("Example: python inference.py input.csv output.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file '{input_csv}' not found")
        sys.exit(1)
    
    print("=" * 60)
    print("MODE OF DELIVERY INFERENCE SCRIPT")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    models = load_models()
    
    # Check if models are loaded
    if all(v is None for v in models.values()):
        print("❌ Error: No models loaded. Please ensure model files exist in save_models/")
        sys.exit(1)
    
    # Load feature lists
    print("\nLoading feature lists...")
    history_data, condition_data = load_history_condition_features()
    if not history_data or not condition_data:
        print("❌ Error: Could not load feature lists from dataset/Divided History Features.csv")
        sys.exit(1)
    
    # Load input CSV
    print(f"\nLoading input CSV: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
        print(f"✅ Loaded {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)
    
    # Make predictions
    print("\nMaking predictions...")
    history_predictions = []
    history_probabilities = []
    condition_predictions = []
    condition_probabilities = []
    combined_predictions = []
    trimester_groups = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} records...")
        
        record = row.to_dict()
        try:
            result = predict_mode_of_delivery(record, models, history_data, condition_data)
            history_predictions.append(result["history_prediction"])
            history_probabilities.append(result["history_probability"])
            condition_predictions.append(result["condition_prediction"])
            condition_probabilities.append(result["condition_probability"])
            combined_predictions.append(result["combined_prediction"])
            trimester_groups.append(result["trimester_group"])
        except Exception as e:
            print(f"  ⚠️  Error predicting record {idx + 1}: {e}")
            history_predictions.append("Error in prediction")
            history_probabilities.append("")
            condition_predictions.append("Error in prediction")
            condition_probabilities.append("")
            combined_predictions.append("Error in prediction")
            trimester_groups.append("unknown")
    
    # Add separate prediction columns to dataframe
    df["Mode_of_delivery2_History_Model_Prediction_Top2"] = history_predictions
    df["Mode_of_delivery2_History_Model_Probability_Top2"] = history_probabilities
    df["Mode_of_delivery2_Condition_Model_Prediction_Top2"] = condition_predictions
    df["Mode_of_delivery2_Condition_Model_Probability_Top2"] = condition_probabilities
    df["Mode_of_delivery2_Combined_Prediction"] = combined_predictions
    df["Trimester_Group_Used"] = trimester_groups
    
    # Save output CSV
    print(f"\nSaving predictions to: {output_csv}")
    try:
        df.to_csv(output_csv, index=False)
        print(f"✅ Successfully saved {len(df)} predictions to {output_csv}")
        print(f"\nOutput columns:")
        print(f"  - Mode_of_delivery2_History_Model_Prediction_Top2: Top 2 predictions from history model")
        print(f"  - Mode_of_delivery2_History_Model_Probability_Top2: Probabilities for top 2 history predictions")
        print(f"  - Mode_of_delivery2_Condition_Model_Prediction_Top2: Top 2 predictions from condition model")
        print(f"  - Mode_of_delivery2_Condition_Model_Probability_Top2: Probabilities for top 2 condition predictions")
        print(f"  - Mode_of_delivery2_Combined_Prediction: Combined predictions from both models")
        print(f"  - Trimester_Group_Used: Trimester group (trimester_1_2 or trimester_3) used for prediction")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

