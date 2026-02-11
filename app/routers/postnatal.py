from typing import Any, List, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.state import state
from app.schemas import PatientRecord, PostnatalResponse
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


def apply_fixed_postnatal_labels(record: dict, labels: list) -> list:
    """
    Apply hard-fixed postnatal complications based on specific feature values.

    If certain features are "yes", force-add corresponding complications
    into the postnatal prediction list, with fixed labels first.
    """
    def is_yes(value: Any) -> bool:
        return str(value).strip().lower() in {"yes", "y", "true", "1"}

    # Fixed postnatal complications in priority order
    fixed_order = [
        "depression",
        "postpartum hemorrhage(pph)",
        "anemia",
    ]

    fixed_labels: List[str] = []
    feature_to_label = {
        "Mental Health Symptoms": "depression",
        "Previous history of PPH (Postpartum Hemorrhage(PPH)) Exploration": "postpartum hemorrhage(pph)",
        "Pallor": "anemia",
    }

    # Add fixed labels in defined order if corresponding feature is "yes"
    for feature_name, label in feature_to_label.items():
        if is_yes(record.get(feature_name, "")) and label not in fixed_labels:
            fixed_labels.append(label)

    # Keep original model predictions order but remove any duplicates of fixed labels
    remaining = [str(lbl) for lbl in (labels or []) if str(lbl) not in fixed_labels]

    return fixed_labels + remaining

def determine_postnatal_risk_level(postnatal_complications: list) -> Tuple[str, str, str]:
    """
    Determine risk level for postnatal complications based on predictions.
    Priority: Critical > High Risk > Low Risk
    """
    # Critical postnatal complications
    critical_postnatal = [
        "cardiac failure",
        "shock",
        "icu admission",
        "acute hepatic failure",
        "acute renal failure",
        "sepsis",
        "deep vein thrombosis and pulmonary embolism",
        "deep vein thrombosis and sepsis",
        "eclampsia and icu admission",
        "infected wound and sepsis",
        "intracranial hemorrhage and icu admission",
        "laparotomy and shock and acute renal failure and icu admission",
        "laparotomy and shock and icu admission",
        "peripartum cardiomyopathy",
        "postpartum hemorrhage and sepsis",
        "postpartum hemorrhage and icu admission",
        "postpartum hemorrhage and shock and icu admission",
        "preeclampsia",
        "sudden maternal collapse",
    ]

    # High risk postnatal complications
    high_risk_postnatal = [
        "postpartum depression",
        "uterine infection",
        "wound hematoma",
        "acute hepatitis e",
        "acute mastitis",
        "anemia",
        "hypertension",
        "infected surgical/episiotomy wound",
        "postpartum eclampsia",
        "rpocs",
        "urinary tract infection",
    ]

    # Check for CRITICAL conditions
    for complication in postnatal_complications:
        if complication.lower() in critical_postnatal:
            instruction = (
                "These prediction values need urgent referral to nearest healthcare facility. "
                "The patient shows critical signs or symptoms indicating a high-risk condition. "
                "Do not delay.\n"
                "- Stop all routine assessments immediately.\n"
                "- Refer the patient to the nearest health facility or hospital without delay.\n"
                "- Ensure safe transportation and inform the referral center before arrival.\n"
                "- Continue providing emotional support and basic first aid as needed.\n"
                "- Record the referral in the mobile application and notify your supervisor.\n\n"
                "Your quick action can save the mother's and baby's life."
            )
            return instruction, "Critical", "Refer immediately"

    # Check for HIGH RISK conditions
    high_risk_postnatal_lower = [item.lower() for item in high_risk_postnatal]
    for complication in postnatal_complications:
        if complication.lower() in high_risk_postnatal_lower:
            instruction = (
                "These prediction values indicate a high risk pregnancy which needs to be managed closely "
                "in liaison with Expert Gynecologist.\n"
                "The patient has high-risk indicators that need special attention and regular follow-up.\n\n"
                "- Inform the patient and family about the need for care under a gynecologist's supervision.\n"
                "- Arrange an early consultation with the nearest gynecologist or health facility.\n"
                "- Continue routine care and monitoring as per guidelines.\n"
                "- Report any new symptoms or changes in the patient's condition promptly.\n"
                "- Record and update all findings in the mobile application.\n\n"
                "Early supervision and timely follow-up can prevent complications."
            )
            return (
                instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )

    # Default: Low Risk
    instruction = (
        "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits. "
        "Continue providing routine antenatal, delivery, and postnatal care according to standard guidelines."
    )
    return instruction, "Low Risk", "Follow routine antenatal care protocol."


@router.post("/predict/postnatal-complications")
async def predict_postnatal(patient_record: PatientRecord):
    # Check if new structure (history + condition) is available, otherwise fall back to old structure
    use_new_structure = (
        state.postnatal_history_model is not None
        and state.postnatal_condition_model is not None
        and state.postnatal_encoder_history is not None
        and state.postnatal_encoder_condition is not None
    )

    if use_new_structure:
        # Use new structure with history and condition models
        if (
            state.postnatal_dic_json_history is None
            or state.postnatal_dic_json_condition is None
            or state.postnatal_col_list_history is None
            or state.postnatal_col_list_condition is None
        ):
            raise HTTPException(status_code=500, detail="Postnatal model not fully loaded")

        try:
            record = patient_record.model_dump()["patientData"]
            rename_map = {"Mode_of_Delivery2": "Mode_of_delivery2"}
            record = {(rename_map.get(k, k)): v for k, v in record.items()}

            # Convert gestational age from "weeks.days" to total days (e.g., 23.5 -> 23*7 + 5 = 166 days)
            ga_key = "Gestational Age (GA) (in weeks)"
            if ga_key in record and record[ga_key] not in (None, ""):
                record[ga_key] = convert_ga_to_days(record[ga_key])

            record_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()}

            # Get history and condition features
            history_data, condition_data = history_based_features()

            # Create features for history model
            postnatal_history_features = creating_features(
                history_data, record_lower, state.postnatal_encoder_history,
                dic_json=state.postnatal_dic_json_history
            )

            # Create features for condition model
            postnatal_condition_features = creating_features(
                condition_data, record_lower, state.postnatal_encoder_condition,
                dic_json=state.postnatal_dic_json_condition
            )

            # Get predictions from both models
            _, history_probs = predict_labels_for_model(
                state.postnatal_history_model,
                postnatal_history_features,
                state.postnatal_col_list_history,
                "multilabel",
            )

            _, condition_probs = predict_labels_for_model(
                state.postnatal_condition_model,
                postnatal_condition_features,
                state.postnatal_col_list_condition,
                "multilabel",
            )

            # Combine predictions from both models
            history_top3 = get_top_k_labels(state.postnatal_col_list_history, history_probs, k=3)
            condition_top3 = get_top_k_labels(state.postnatal_col_list_condition, condition_probs, k=3)

            # Combine and deduplicate
            postnatal_labels = list(set(history_top3 + condition_top3))
            if not postnatal_labels:
                postnatal_labels = history_top3 if history_top3 else condition_top3

            # Apply hard-fixed postnatal complications based on input features
            postnatal_labels = apply_fixed_postnatal_labels(record, postnatal_labels)

            # Determine risk level using code-based logic
            remarks_text, risk_level, referral_text = determine_postnatal_risk_level(
                postnatal_labels
            )
            postnatal_labels = ",".join(str(x) for x in postnatal_labels)
            response = {
                "message": "prediction completed for postnatal complications",
                "predictions": postnatal_labels,
                "Remarks": remarks_text,
                "risk_Level": risk_level,
                "referral": referral_text,
            }
            return JSONResponse(content=response)

        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Postnatal prediction failed: {exc}"
            ) from exc
    else:
        # Fall back to old structure
        if (
            state.postnatal_model is None
            or state.postnatal_encoder is None
            or state.postnatal_dic_json is None
            or state.postnatal_col_list is None
        ):
            raise HTTPException(status_code=500, detail="Postnatal model not loaded")

        try:
            import pandas as pd
            from app.services.prediction import safe_process_field

            record = patient_record.model_dump()["patientData"]
            rename_map = {"Mode_of_Delivery2": "Mode_of_delivery2"}
            record = {(rename_map.get(k, k)): v for k, v in record.items()}
            # Convert gestational age from "weeks.days" to total days (e.g., 23.5 -> 23*7 + 5 = 166 days)
            ga_key = "Gestational Age (GA) (in weeks)"
            if ga_key in record and record[ga_key] not in (None, ""):
                record[ga_key] = convert_ga_to_days(record[ga_key])
            record_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()}

            for value, conditions in state.postnatal_dic_json.items():
                row_value = record_lower.get(value, "")
                encoded_row = safe_process_field(value, row_value, conditions)
                record.pop(value, None)
                for k, v in encoded_row.items():
                    record[f"{value}:{k}"] = v

            new_row = pd.DataFrame([record])
            X_new = state.postnatal_encoder.transform(new_row)

            postnatal_labels, _ = predict_labels_for_model(
                state.postnatal_model, X_new, state.postnatal_col_list[0], "multilabel"
            )

            # Apply hard-fixed postnatal complications based on input features
            postnatal_labels = apply_fixed_postnatal_labels(record, postnatal_labels)

            # Determine risk level using code-based logic
            remarks_text, risk_level, referral_text = determine_postnatal_risk_level(
                postnatal_labels
            )
            postnatal_labels = ",".join(str(x) for x in postnatal_labels)
            response = {
                "message": "prediction completed for postnatal complications",
                "predictions": postnatal_labels,
                "Remarks": remarks_text,
                "risk_Level": risk_level,
                "referral": referral_text,
            }
            return JSONResponse(content=response)

        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Postnatal prediction failed: {exc}"
            ) from exc

