from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.state import state
from app.schemas import ModeDeliveryResponse, PatientRecord
from app.services.prediction import (
    creating_features,
    get_top_k_labels,
    history_based_features,
    predict_labels_for_model,
)

router = APIRouter()


@router.post("/predict-mode-delivery", response_model=ModeDeliveryResponse)
async def predict_mode_delivery(patient_record: PatientRecord):
    # Check if new structure (history + condition) is available, otherwise fall back to old structure
    use_new_structure = (
        state.mode_delivery_history_model is not None
        and state.mode_delivery_condition_model is not None
        and state.mode_delivery_encoder_history is not None
        and state.mode_delivery_encoder_condition is not None
    )

    if use_new_structure:
        # Use new structure with history and condition models
        if (
            state.mode_delivery_dic_json_history is None
            or state.mode_delivery_dic_json_condition is None
            or state.mode_delivery_col_list_history is None
            or state.mode_delivery_col_list_condition is None
        ):
            raise HTTPException(status_code=500, detail="Mode delivery model not fully loaded")

        try:
            record = patient_record.model_dump()["patientData"]
            record_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()}

            # Get history and condition features
            history_data, condition_data = history_based_features()

            # Create features for history model
            mode_delivery_history_features = creating_features(
                history_data, record_lower, state.mode_delivery_encoder_history,
                dic_json=state.mode_delivery_dic_json_history
            )

            # Create features for condition model
            mode_delivery_condition_features = creating_features(
                condition_data, record_lower, state.mode_delivery_encoder_condition,
                dic_json=state.mode_delivery_dic_json_condition
            )

            # Get predictions from both models
            _, history_probs = predict_labels_for_model(
                state.mode_delivery_history_model,
                mode_delivery_history_features,
                state.mode_delivery_col_list_history,
                "multilabel",
            )

            _, condition_probs = predict_labels_for_model(
                state.mode_delivery_condition_model,
                mode_delivery_condition_features,
                state.mode_delivery_col_list_condition,
                "multilabel",
            )

            # Combine predictions from both models
            history_top2 = get_top_k_labels(state.mode_delivery_col_list_history, history_probs, k=2)
            condition_top2 = get_top_k_labels(state.mode_delivery_col_list_condition, condition_probs, k=2)

            # Combine and deduplicate
            mode_delivery_labels = list(set(history_top2 + condition_top2))
            if not mode_delivery_labels:
                mode_delivery_labels = history_top2 if history_top2 else condition_top2

            payload = {"prediction": ", ".join(mode_delivery_labels)}
            return JSONResponse(content=payload)

        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Mode delivery prediction failed: {exc}"
            ) from exc
    else:
        # Fall back to old structure
        if (
            state.mode_delivery_model is None
            or state.delivery_encoder is None
            or state.delivery_dic_json is None
            or state.delivery_col_list is None
        ):
            raise HTTPException(status_code=500, detail="Mode delivery model not loaded")

        try:
            import pandas as pd
            from app.services.prediction import safe_process_field

            record = patient_record.model_dump()["patientData"]
            record_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()}

            for value, conditions in state.delivery_dic_json.items():
                row_value = record_lower.get(value, "")
                encoded_row = safe_process_field(value, row_value, conditions)
                record.pop(value, None)
                for k, v in encoded_row.items():
                    record[f"{value}:{k}"] = v

            new_row = pd.DataFrame([record])
            X_new = state.delivery_encoder.transform(new_row)
            mode_delivery_labels, _ = predict_labels_for_model(
                state.mode_delivery_model, X_new, state.delivery_col_list[0], "multilabel"
            )

            payload = {"prediction": ", ".join(mode_delivery_labels)}
            return JSONResponse(content=payload)

        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Mode delivery prediction failed: {exc}"
            ) from exc

