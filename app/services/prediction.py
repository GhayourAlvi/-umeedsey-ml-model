from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.state import state


DEFAULT_PROB_THRESHOLD = 0.1


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


def map_predictions_to_labels(predictions: Any, col_list: List[str]) -> List[str]:
    """Map binary predictions to class labels."""
    preds = np.array(predictions)
    if preds.ndim == 2:
        preds = preds[0]

    if np.issubdtype(preds.dtype, np.floating):
        preds = (preds >= 0.5).astype(int)
    else:
        preds = preds.astype(int)

    labels = [col_list[i] for i, pred in enumerate(preds) if int(pred) == 1 and i < len(col_list)]
    return labels if labels else ["Model failed to predict"]


def predict_labels_for_model(
    model: Any,
    X_new: Any,
    col_list: List[str],
    mode: str,
    threshold: float = DEFAULT_PROB_THRESHOLD,
    ensure_one: bool = True,
) -> Tuple[List[str], List[float]]:
    """Convert model outputs to label names and keep raw probabilities."""
    if model is None:
        return ["Model not loaded"], []

    probs_for_sample: Optional[List[float]] = None

    if mode == "multilabel":
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_new)
            except Exception as e:
                print(f"WARNING: predict_proba failed: {e}")
                import traceback
                traceback.print_exc()
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
                import traceback
                traceback.print_exc()
                return ["Model prediction failed"], []

        if probs_for_sample is None:
            return ["Model failed to produce probabilities"], []

        try:
            float_probs = [float(p) for p in probs_for_sample]
        except Exception:
            float_probs = []

        selected_indices = [i for i, p in enumerate(float_probs) if p >= float(threshold)]
        if not selected_indices and ensure_one and float_probs:
            selected_indices = [int(np.argmax(float_probs))]

        labels = [col_list[i] for i in selected_indices if i < len(col_list)]
        if not labels:
            labels = ["Model failed to predict"]
        return labels, float_probs

    # single label fallback
    try:
        pred = model.predict(X_new)
        arr = np.array(pred)
        if arr.ndim == 1:
            idx = int(arr[0])
            if idx < len(col_list):
                return [col_list[idx]], []
    except Exception:
        return ["Model prediction failed"], []

    return ["Model failed to produce probabilities"], []


def safe_process_field(field_name: str, field_value: Any, conditions: List[str]) -> Dict[str, str]:
    """Safely process a field with error handling for problematic field names."""
    try:
        if field_value in (None, ""):
            field_value = "no"
        field_value = str(field_value).lower()
        row_list = [x.strip().lower() for x in field_value.split(",")]
        return {cond: ("yes" if cond.lower() in row_list else "no") for cond in conditions}
    except Exception:
        return {cond: "no" for cond in conditions}


def creating_features(data: List[str], record_lower: Dict[str, Any], encoder_model: Any, dic_json: Optional[Dict[str, List[str]]] = None):
    """Transform record into encoded feature space.
    
    This function replicates the expand_comma_columns logic from training:
    1. Expands comma-separated columns into binary features (column:value format)
    2. Keeps regular columns as-is
    3. Ensures all columns match training format and order
    """
    # Use provided dic_json or fall back to state.dic_json
    feature_mapping = dic_json if dic_json is not None else state.dic_json
    if feature_mapping is None:
        raise ValueError("dic_json not loaded and not provided")

    # Get expected feature names from encoder (in the order they were fitted)
    expected_features = None
    try:
        # Try to get feature names from encoder (sklearn >= 1.0)
        if hasattr(encoder_model, 'feature_names_in_') and encoder_model.feature_names_in_ is not None:
            expected_features = list(encoder_model.feature_names_in_)
        elif hasattr(encoder_model, 'get_feature_names_out'):
            # This requires input feature names, which we'll get from the data list
            # But first, let's build the expanded feature list
            pass
    except Exception:
        pass

    # Build the feature dictionary, processing columns in the same order as data list
    # This ensures we match the training order
    filtered = {}
    expanded_columns = []  # Track column order
    
    # Process each column in the data list (maintains order)
    for col in data:
        if col not in feature_mapping:
            # Regular column (not comma-separated) - keep as-is
            expanded_columns.append(col)
            if col in record_lower:
                filtered[col] = record_lower[col]
            else:
                # Missing column - set to empty string
                filtered[col] = ""
        else:
            # Comma-separated column - expand into binary features
            conditions = feature_mapping[col]
            row_value = record_lower.get(col, "")
            
            # Process comma-separated values into binary features
            encoded_row = safe_process_field(col, row_value, conditions)
            
            # Add expanded binary features (column:value format) in sorted order for consistency
            # Note: original column is NOT added (matching training behavior)
            for k in sorted(conditions):  # Sort for consistent order
                expanded_col_name = f"{col}:{k}"
                expanded_columns.append(expanded_col_name)
                filtered[expanded_col_name] = encoded_row.get(k, "no")

    # Create DataFrame with features in the order they were processed
    # This should match the training order
    feature_df = pd.DataFrame([filtered])
    
    # Reorder columns to match expected order if available, otherwise use processed order
    if expected_features is not None:
        # Create ordered dictionary with all expected features
        ordered_data = {}
        for col in expected_features:
            if col in filtered:
                ordered_data[col] = filtered[col]
            else:
                # Missing column - set to empty string
                ordered_data[col] = ""
        feature_df = pd.DataFrame([ordered_data])
        # Ensure column order matches exactly
        feature_df = feature_df[expected_features]
    else:
        # Use the order we processed columns
        feature_df = feature_df[[col for col in expanded_columns if col in feature_df.columns]]
    
    # Ensure all string values are lowercase (matching training format)
    feature_df = feature_df.map(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Fill NaN values with empty string (matching training behavior)
    feature_df = feature_df.fillna("")
    
    # Transform using encoder (encoder will handle missing/extra columns with handle_unknown="ignore")
    return encoder_model.transform(feature_df)


def history_based_features():
    """Load history and condition features from CSV."""
    df_features = pd.read_csv("./dataset/Divided History Features.csv")
    history = df_features.iloc[0, :].dropna().tolist()
    condition = df_features.iloc[1, :].dropna().tolist()
    return history, condition


def get_neonatal_predictions(record_data: Dict[str, Any], record_lower: Dict[str, Any]) -> List[str]:
    if (
        state.neonatal_historymodel_withapgar is None
        or state.neonatal_conditionmodel_withapgar is None
        or state.neonatal_col_list is None
        or state.antenatal_encoder_history is None
        or state.neonatal_encoder_condition is None
    ):
        return ["Neonatal Model Not Loaded"]

    temp_record = record_data.copy()
    keys_to_remove = [
        "Postnatal_Symptoms",
        "Postnatal_Examination",
        "Mode_of_delivery2",
        "Antenatal_Peripartum_Maternal_Complications",
        "Neonatal__Fetal_Complications",
    ]
    for k in keys_to_remove:
        temp_record.pop(k, None)

    history_data, condition_data = history_based_features()
    neonatal_history_data = creating_features(history_data, record_lower, state.antenatal_encoder_history)

    condition_data.extend(
        ["Neonatal Apgar Score:(1 Minute)", "Neonatal Apgar Score:(5 Minute)"]
    )
    neonatal_condition_data = creating_features(
        condition_data, record_lower, state.neonatal_encoder_condition
    )

    _, probs_history = predict_labels_for_model(
        state.neonatal_historymodel_withapgar,
        neonatal_history_data,
        state.neonatal_col_list[0],
        "multilabel",
    )
    neonatal_history_top3 = get_top_k_labels(state.neonatal_col_list[0], probs_history, k=2)

    _, probs_condition = predict_labels_for_model(
        state.neonatal_conditionmodel_withapgar,
        neonatal_condition_data,
        state.neonatal_col_list[0],
        "multilabel",
    )
    neonatal_condition_top3 = get_top_k_labels(state.neonatal_col_list[0], probs_condition, k=2)

    # _ = neonatal_labels_history, neonatal_labels_condition  # satisfy lint
    return list(set(neonatal_history_top3 + neonatal_condition_top3)) or ["Model failed to predict"]


def get_postnatal_predictions(record_data: Dict[str, Any], record_lower: Dict[str, Any]) -> List[str]:
    if (
        state.postnatal_model is None
        or state.postnatal_encoder is None
        or state.postnatal_dic_json is None
        or state.postnatal_col_list is None
    ):
        return ["Postnatal Model Not Loaded"]

    temp_record = record_data.copy()
    rename_map = {"Mode_of_Delivery2": "Mode_of_delivery2"}
    temp_record = {(rename_map.get(k, k)): v for k, v in temp_record.items()}

    processed_record = temp_record.copy()
    for value, conditions in state.postnatal_dic_json.items():
        row_value = record_lower.get(value, "")
        encoded_row = safe_process_field(value, row_value, conditions)
        processed_record.pop(value, None)
        for k, v in encoded_row.items():
            processed_record[f"{value}:{k}"] = v

    new_row = pd.DataFrame([processed_record])
    X_new = state.postnatal_encoder.transform(new_row)
    postnatal_labels, _ = predict_labels_for_model(
        state.postnatal_model, X_new, state.postnatal_col_list[0], "multilabel"
    )
    return postnatal_labels


def multilabel_predict(models: List[Any], X_new: Any, threshold: float = DEFAULT_PROB_THRESHOLD, ensure_one: bool = True):
    """Legacy helper retained for compatibility/testing."""
    representative_probs = []
    for model in models:
        prob_val = 0.0
        try:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_new)
                if isinstance(p, list):
                    vals = []
                    for arr in p:
                        try:
                            vals.append(float(arr[0, 1]))
                        except Exception:
                            vals.append(0.0)
                    prob_val = float(max(vals)) if vals else 0.0
                elif isinstance(p, np.ndarray):
                    if p.ndim == 2 and p.shape[1] == 2:
                        prob_val = float(p[0, 1])
                    else:
                        prob_val = float(np.max(p[0]))
                else:
                    prob_val = float(p)
            elif hasattr(model, "predict"):
                pred = model.predict(X_new)
                arr = np.array(pred)
                prob_val = float(np.max(arr[0])) if arr.ndim == 2 else float(arr[0])
        except Exception:
            prob_val = 0.0
        representative_probs.append(prob_val)

    probs = np.array(representative_probs).reshape(1, -1)
    preds = (probs >= threshold).astype(int)
    if ensure_one:
        for i in range(preds.shape[0]):
            if np.all(preds[i] == 0):
                preds[i, int(np.argmax(probs[i]))] = 1
    return preds, probs

