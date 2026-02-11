from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.state import state
from app.schemas import AntenatalResponse, PatientRecord
from app.services.prediction import (
    creating_features,
    get_top_k_labels,
    history_based_features,
    predict_labels_for_model,
)

router = APIRouter()


def convert_ga_to_days(ga_value: Any) -> Any:
    """
    Convert gestational age from "weeks.days" format to total days.
    Example: 23.5 (23 weeks and 5 days) -> 23 * 7 + 5 = 166 days

    Args:
        ga_value: Gestational age value (can be float, int, or string)

    Returns:
        Total days as float, or original value if conversion fails.
    """
    if ga_value is None or ga_value == "":
        return ga_value

    try:
        ga_str = str(ga_value)
        if "." in ga_str:
            parts = ga_str.split(".", 1)
            weeks = int(float(parts[0]))  # before decimal
            days = int(float("0." + parts[1]) * 10) if parts[1] != "" else 0
            return float(weeks * 7 + days)
        else:
            weeks = int(float(ga_str))
            return float(weeks * 7)
    except Exception:
        # Fallback: if it's a plain week number, multiply by 7; otherwise return as-is
        try:
            return float(ga_value) * 7.0
        except Exception:
            return ga_value


def apply_fixed_antenatal_labels(record: dict, labels: list) -> list:
    """
    Apply hard-fixed antenatal complications based on specific feature values.

    If certain features are "yes", force-add corresponding complications
    into the antenatal prediction list.
    """
    def is_yes(value: Any) -> bool:
        return str(value).strip().lower() in {"yes", "y", "true", "1"}

    # Define fixed antenatal complications (in desired priority order)
    fixed_order = [
        "depression",
        "postpartum hemorrhage(pph)",
        "anesthesia complication",
        "anemia",
    ]

    fixed_labels: List[str] = []
    # Add fixed labels (in order) if corresponding input feature is answered as yes
    feature_to_label = {
        "Mental Health Symptoms": "depression",
        "Previous history of PPH (Postpartum Hemorrhage(PPH)) Exploration": "postpartum hemorrhage(pPH)",
        "Any Anesthesia Complication previously": "anesthesia complication",
        "Pallor": "anemia",
    }
    for feature_name, label in feature_to_label.items():
        if is_yes(record.get(feature_name, "")) and label not in fixed_labels:
            fixed_labels.append(label)

    # Keep original model predictions order but remove any that are already in fixed_labels
    remaining = [lbl for lbl in (labels or []) if lbl not in fixed_labels]

    # Final combined list: fixed labels first (priority), then remaining model predictions
    return fixed_labels + remaining


@router.post("/predict-antenatal", response_model=AntenatalResponse)
async def predict_antenatal(patient_record: PatientRecord):
    # Check if new structure (history + condition) is available, otherwise fall back to old structure
    use_new_structure = (
        state.antenatal_history_model is not None
        and state.antenatal_condition_model is not None
        and state.antenatal_encoder_history is not None
        and state.antenatal_encoder_condition is not None
    )

    if use_new_structure:
        # Use new structure with history and condition models
        if (
            state.antenatal_dic_json_history is None
            or state.antenatal_dic_json_condition is None
            or state.antenatal_col_list_history is None
            or state.antenatal_col_list_condition is None
        ):
            raise HTTPException(status_code=500, detail="Antenatal model not fully loaded")

        try:
            record = patient_record.model_dump()["patientData"]

            # Convert gestational age from "weeks.days" to total days (e.g., 23.5 -> 23*7 + 5)
            ga_key = "Gestational Age (GA) (in weeks)"
            if ga_key in record and record[ga_key] not in (None, ""):
                record[ga_key] = convert_ga_to_days(record[ga_key])

            record_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()}

            # Get history and condition features
            history_data, condition_data = history_based_features()

            # Create features for history model
            antenatal_history_features = creating_features(
                history_data, record_lower, state.antenatal_encoder_history,
                dic_json=state.antenatal_dic_json_history
            )

            # Create features for condition model
            antenatal_condition_features = creating_features(
                condition_data, record_lower, state.antenatal_encoder_condition,
                dic_json=state.antenatal_dic_json_condition
            )

            # Get predictions from both models
            _, history_probs = predict_labels_for_model(
                state.antenatal_history_model,
                antenatal_history_features,
                state.antenatal_col_list_history,
                "multilabel",
            )

            _, condition_probs = predict_labels_for_model(
                state.antenatal_condition_model,
                antenatal_condition_features,
                state.antenatal_col_list_condition,
                "multilabel",
            )

            # Combine predictions from both models
            history_top3 = get_top_k_labels(state.antenatal_col_list_history, history_probs, k=3)
            condition_top3 = get_top_k_labels(state.antenatal_col_list_condition, condition_probs, k=3)

            # Combine and deduplicate
            antenatal_labels = list(set(history_top3 + condition_top3))
            if not antenatal_labels:
                antenatal_labels = history_top3 if history_top3 else condition_top3

            # Apply hard-fixed antenatal complications based on input features
            antenatal_labels = apply_fixed_antenatal_labels(record, antenatal_labels)

            payload = {"prediction": ", ".join(antenatal_labels)}
            return JSONResponse(content=payload)

        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Antenatal prediction failed: {exc}"
            ) from exc
    else:
        # Fall back to old structure
        if (
            state.antenatal_model is None
            or state.delivery_encoder is None
            or state.delivery_dic_json is None
            or state.delivery_col_list is None
        ):
            raise HTTPException(status_code=500, detail="Antenatal model not loaded")

        try:
            import pandas as pd
            from app.services.prediction import safe_process_field

            record = patient_record.model_dump()["patientData"]

            # Convert gestational age from "weeks.days" to total days (e.g., 23.5 -> 23*7 + 5)
            ga_key = "Gestational Age (GA) (in weeks)"
            if ga_key in record and record[ga_key] not in (None, ""):
                record[ga_key] = convert_ga_to_days(record[ga_key])

            record_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()}

            for value, conditions in state.delivery_dic_json.items():
                row_value = record_lower.get(value, "")
                encoded_row = safe_process_field(value, row_value, conditions)
                record.pop(value, None)
                for k, v in encoded_row.items():
                    record[f"{value}:{k}"] = v

            new_row = pd.DataFrame([record])
            X_new = state.delivery_encoder.transform(new_row)
            antenatal_labels, _ = predict_labels_for_model(
                state.antenatal_model, X_new, state.delivery_col_list[1], "multilabel"
            )

            # Apply hard-fixed antenatal complications based on input features
            antenatal_labels = apply_fixed_antenatal_labels(record, antenatal_labels)

            payload = {"prediction": ", ".join(antenatal_labels)}
            return JSONResponse(content=payload)

        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Antenatal prediction failed: {exc}"
            ) from exc

