from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from app.routers.outcome import check_rh_incompatibility
from app.core.state import state
from app.routers.outcome import determine_risk_level_and_referral_with_postnatal
from app.services.prediction import (
    creating_features,
    get_neonatal_predictions,
    get_postnatal_predictions,
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


def apply_fixed_antenatal_labels(record: Dict[str, Any], labels: list) -> list:
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
    # Map specific input features to fixed complication labels
    feature_to_label = {
        "Mental Health Symptoms": "depression",
        "Previous history of PPH (Postpartum Hemorrhage(PPH)) Exploration": "postpartum hemorrhage(pph)",
        "Any Anesthesia Complication previously": "anesthesia complication",
        "Pallor": "anemia",
    }

    # Add fixed labels in defined order if corresponding input feature is answered as yes
    for feature_name, label in feature_to_label.items():
        if is_yes(record.get(feature_name, "")) and label not in fixed_labels:
            fixed_labels.append(label)

    # Keep original model predictions order but remove any that are already in fixed_labels
    remaining = [str(lbl) for lbl in (labels or []) if str(lbl) not in fixed_labels]

    # Final combined list: fixed labels first (priority), then remaining model predictions
    return fixed_labels + remaining


def apply_mode_delivery_hard_rules(record: Dict[str, Any], labels: list) -> list:
    """
    Apply hard-coded business rules for mode of delivery predictions.

    Rules (only affect 'vaginal'):
    1. If "Any Uterine Surgery previously (Other than Cesarean Section)" == Yes
       AND gestational age > 16 weeks -> do not predict 'vaginal'.
    2. If "Number of Cesarean Sections (C-sections)" >= 2
       AND gestational age > 28 weeks -> do not predict 'vaginal'.
    """
    if not labels:
        return labels

    def is_yes(value: Any) -> bool:
        return str(value).strip().lower() in {"yes", "y", "true", "1"}

    ga_key = "Gestational Age (GA) (in weeks)"
    ga_val = record.get(ga_key)
    ga_days: Optional[float] = None
    try:
        ga_days = float(ga_val) if ga_val not in (None, "") else None
    except Exception:
        ga_days = None

    block_vaginal = False

    # Rule 1: any uterine surgery (other than C-section) and GA > 16 weeks
    if ga_days is not None and ga_days > 16 * 7:
        if is_yes(record.get("Any Uterine Surgery previously (Other than Cesarean Section)", "")):
            block_vaginal = True

    # Rule 2: >= 2 C-sections and GA > 28 weeks
    csec_val = record.get("Number of Cesarean Sections (C-sections)")
    try:
        csec_num = float(csec_val) if csec_val not in (None, "") else 0.0
    except Exception:
        csec_num = 0.0

    if ga_days is not None and ga_days > 28 * 7 and csec_num >= 2:
        block_vaginal = True

    if not block_vaginal:
        return labels

    # Filter out 'vaginal' (case-insensitive), preserving original order
    filtered = [lbl for lbl in labels if str(lbl).strip().lower() != "vaginal"]
    return filtered or labels  # if everything was 'vaginal', keep original list as fallback


@router.post("/predict_from_csv")
async def predict_outcome_from_csv(file: UploadFile = File(...)):
    # Check if new structure is available
    use_new_structure = (
        state.mode_delivery_history_model is not None
        and state.mode_delivery_condition_model is not None
        and state.antenatal_history_model is not None
        and state.antenatal_condition_model is not None
        and state.neonatal_historymodel_withoutapgar is not None
        and state.neonatal_conditionmodel_withoutapgar is not None
    )

    # # Fallback to old structure check
    # use_old_structure = (

    #     state.mode_delivery_history_model is not None
    #     and state.mode_delivery_condition_model is not None
    #     and state.antenatal_history_model is not None
    #     and state.antenatal_condition_model is not None
    #     and state.neonatal_historymodel_withoutapgar is not None
    #     and state.neonatal_conditionmodel_withoutapgar is not None
    #     and state.encoder_history is not None
    #     and state.encoder_condition is not None
    #     and state.dic_json is not None
    #     and state.col_list_combined is not None
    #     and state.single_encoder is not None
    # )

    # if not use_new_structure and not use_old_structure:
    #     raise HTTPException(status_code=500, detail="Models not loaded properly")

    try:
        contents = await file.read()
        s = StringIO(contents.decode("utf-8"))
        df = pd.read_csv(s)
        # Replace NaN values with empty string to avoid isnan errors
        df = df.fillna("")
        # print(df.dtypes)

        # df = df.apply(pd.to_numeric, errors='coerce')
        all_predictions: List[Dict[str, Any]] = []

        for index, record_dict in df.iterrows():
            record = record_dict.to_dict()
            keys_to_remove = [
                "Neonatal Apgar Score:(1 Minute)",
                "Neonatal Apgar Score:(5 Minute)",
                "Postnatal_Symptoms",
                "Postnatal_Examination",
                "Mode_of_delivery2",
                "Antenatal_Peripartum_Maternal_Complications",
                "Neonatal__Fetal_Complications",
            ]
            for k in keys_to_remove:
                record.pop(k, None)

            convert_to_float = [
                "PatientAge",
                "bmi",
                "Number of Abortions (Births before 28 Weeks)",
                "Gestational Age (GA) (in weeks)",
                "Number of Cesarean Sections (C-sections)",
                "Number of ERPC (Evacuation of Retained Products of Conception)",
                "Systolic B.P",
                "Diastolic B.P",
                "Pulse in full 1 minute",
                "Respiratory rate in full 1 minute",
                "Temp. ( armpit for 2 ) in *F",
                "Number of Previous Childbirths after 28 Weeks",
            ]
            for key in convert_to_float:
                if key in record and record[key] not in (None, ""):
                    # Check if value is NaN-safe before converting
                    try:
                        # Handle both string and numeric types
                        val = record[key]
                        if isinstance(val, str) and val.strip() == "":
                            continue
                        # Use pd.isna for safe NaN checking
                        if not (isinstance(val, float) and pd.isna(val)):
                            record[key] = float(val)
                    except (ValueError, TypeError):
                        # If conversion fails, keep original value or set to None
                        pass

            # Convert gestational age from "weeks.days" to total days (e.g., 23.5 -> 23*7 + 5)
            ga_key = "Gestational Age (GA) (in weeks)"
            if ga_key in record and record[ga_key] not in (None, ""):
                record[ga_key] = convert_ga_to_days(record[ga_key])

            # Convert record to lowercase strings, handling None and NaN values
            record_lower = {}
            for k, v in record.items():
                if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                    record_lower[k] = ""
                elif isinstance(v, str):
                    record_lower[k] = v.lower()
                else:
                    record_lower[k] = v

            history_data, condition_data = history_based_features()

            if use_new_structure:
                # Use new structure with history + condition models
                
                # Mode of Delivery: history + condition
                mode_delivery_history_features = creating_features(
                    history_data,
                    record_lower,
                    state.mode_delivery_encoder_history,
                    dic_json=state.mode_delivery_dic_json_history,
                )
                mode_delivery_condition_features = creating_features(
                    condition_data,
                    record_lower,
                    state.mode_delivery_encoder_condition,
                    dic_json=state.mode_delivery_dic_json_condition,
                )

                _, mode_probs_history = predict_labels_for_model(
                    state.mode_delivery_history_model,
                    mode_delivery_history_features,
                    state.mode_delivery_col_list_history,
                    "multilabel",
                )
                _, mode_probs_condition = predict_labels_for_model(
                    state.mode_delivery_condition_model,
                    mode_delivery_condition_features,
                    state.mode_delivery_col_list_condition,
                    "multilabel",
                )

                mode_history_top2 = get_top_k_labels(
                    state.mode_delivery_col_list_history, mode_probs_history, k=2
                )
                mode_condition_top2 = get_top_k_labels(
                    state.mode_delivery_col_list_condition, mode_probs_condition, k=2
                )
                mode_delivery_top3 = list(set(mode_history_top2 + mode_condition_top2))
                if not mode_delivery_top3:
                    mode_delivery_top3 = (
                        mode_history_top2 if mode_history_top2 else mode_condition_top2
                    )

                # Apply hard-coded mode of delivery rules (may remove 'vaginal')
                mode_delivery_top3 = apply_mode_delivery_hard_rules(record, mode_delivery_top3)

                # Antenatal: history + condition
                antenatal_history_features = creating_features(
                    history_data,
                    record_lower,
                    state.antenatal_encoder_history,
                    dic_json=state.antenatal_dic_json_history,
                )
                antenatal_condition_features = creating_features(
                    condition_data,
                    record_lower,
                    state.antenatal_encoder_condition,
                    dic_json=state.antenatal_dic_json_condition,
                )

                _, antenatal_probs_history = predict_labels_for_model(
                    state.antenatal_history_model,
                    antenatal_history_features,
                    state.antenatal_col_list_history,
                    "multilabel",
                )
                _, antenatal_probs_condition = predict_labels_for_model(
                    state.antenatal_condition_model,
                    antenatal_condition_features,
                    state.antenatal_col_list_condition,
                    "multilabel",
                )

                antenatal_history_top3 = get_top_k_labels(
                    state.antenatal_col_list_history, antenatal_probs_history, k=2
                )
                antenatal_condition_top3 = get_top_k_labels(
                    state.antenatal_col_list_condition, antenatal_probs_condition, k=2
                )
                antenatal_top3 = list(
                    set(antenatal_history_top3 + antenatal_condition_top3)
                )
                if not antenatal_top3:
                    antenatal_top3 = (
                        antenatal_history_top3
                        if antenatal_history_top3
                        else antenatal_condition_top3
                    )

                # Apply hard-fixed antenatal complications based on input features
                antenatal_top3 = apply_fixed_antenatal_labels(record_dict.to_dict(), antenatal_top3)

                # Neonatal: history + condition (using without-apgar models)
                neonatal_history_features = creating_features(
                    history_data,
                    record_lower,
                    state.antenatal_encoder_history
                )
                neonatal_condition_features = creating_features(
                    condition_data,
                    record_lower,
                    state.antenatal_encoder_condition,
                )

                neonatal_col_list = (
                    state.col_list_combined[4]
                    if (
                        state.col_list_combined is not None
                        and isinstance(state.col_list_combined, list)
                        and len(state.col_list_combined) > 2
                    )
                    else []
                )

                _, neonatal_probs_history = predict_labels_for_model(
                    state.neonatal_historymodel_withoutapgar,
                    neonatal_history_features,
                    neonatal_col_list,
                    "multilabel",
                )
                _, neonatal_probs_condition = predict_labels_for_model(
                    state.neonatal_conditionmodel_withoutapgar,
                    neonatal_condition_features,
                    neonatal_col_list,
                    "multilabel",
                )

                neonatal_top3_history = get_top_k_labels(
                    neonatal_col_list, neonatal_probs_history, k=2
                )
                neonatal_top3_condition = get_top_k_labels(
                    neonatal_col_list, neonatal_probs_condition, k=2
                )
        
            if "not delivered yet" in mode_delivery_top3:
                mode_delivery_top3.remove("not delivered yet")
            
            neonatal_top3 = list(set(neonatal_top3_history + neonatal_top3_condition))
            if not neonatal_top3:
                neonatal_top3 = (
                    neonatal_top3_history if neonatal_top3_history else neonatal_top3_condition
                )

            record["Neonatal Apgar Score:(1 Minute)"] = record_dict.get("Neonatal Apgar Score:(1 Minute)", "")
            record["Neonatal Apgar Score:(5 Minute)"] = record_dict.get("Neonatal Apgar Score:(5 Minute)", "")
            # Convert record to lowercase strings, handling None and NaN values
            record_lower = {}
            for k, v in record.items():
                if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                    record_lower[k] = ""
                elif isinstance(v, str):
                    record_lower[k] = v.lower()
                else:
                    record_lower[k] = v

            neonatal_complications = get_neonatal_predictions(record_dict.to_dict(), record_lower)
            postnatal_complications = get_postnatal_predictions(record_dict.to_dict(), record_lower)

            # Apply hard-fixed postnatal complications based on input features
            postnatal_complications = apply_fixed_postnatal_labels(record_dict.to_dict(), postnatal_complications)
            original_record = record_dict.to_dict()
            patient_bg = original_record.get("PatientHasPosGp", "")
            partner_bg = original_record.get("PartenerHasPosGp", "")
            
            # Check for Rh incompatibility based on blood groups (matching outcome.py logic)
            if "antid prophylaxis due to rh incompatibility" in antenatal_top3:
                antenatal_top3.remove("antid prophylaxis due to rh incompatibility")
            if "rh incompatibility" in antenatal_top3:
                antenatal_top3.remove("rh incompatibility")
            if "antid prophylaxis rh incompatibility due to rh incompatibility" in antenatal_top3:
                antenatal_top3.remove("antid prophylaxis rh incompatibility due to rh incompatibility")
            if check_rh_incompatibility(patient_bg, partner_bg):
                rh_incompatibility = "Rh Incompatibility"
                # Add to antenatal results if not already present
                antenatal_top3_lower = [item.lower() for item in antenatal_top3]
                
                if rh_incompatibility.lower() not in antenatal_top3_lower:
                    antenatal_top3.append(rh_incompatibility)
            
            results: Dict[str, Any] = {
                "Mode_of_delivery": mode_delivery_top3,
                "Antenatal_complication": antenatal_top3,
                "Neonatal_Complications": neonatal_top3,
                "Apgar_Neonatal_Complications": neonatal_complications,
                "Postnatal_Complications": postnatal_complications,
            }
            
            # Prepare comma-separated strings for risk determination
            mode_of_delivery_str = ", ".join(str(x) for x in mode_delivery_top3) if mode_delivery_top3 else ""
            antenatal_str = ", ".join(str(x) for x in antenatal_top3) if antenatal_top3 else ""
            neonatal_str = ", ".join(str(x) for x in neonatal_top3) if neonatal_top3 else ""
            postnatal_str = ", ".join(str(x) for x in postnatal_complications) if postnatal_complications else ""
            print('mode_of_delivery: ', mode_of_delivery_str)
            print('antenatal: ', antenatal_str)
            print('neonatal: ', neonatal_str)
            print('postnatal: ', postnatal_str)
            print('--------------------------------')
            # Determine risk level, remarks, and referral using code-based logic
            remarks_text, risk_level, referral_text = determine_risk_level_and_referral_with_postnatal(
                mode_of_delivery=mode_of_delivery_str,
                antenatal_complications=antenatal_str,
                neonatal_complications=neonatal_str,
                postnatal_complications=postnatal_str,
            )

            response_for_record = {
                "message": f"Prediction completed for record {index}",
                "predictions": results,
                "Remarks": remarks_text,
                "risk_Level": risk_level,
                "referral": referral_text,
            }
            all_predictions.append(response_for_record)

        predictions_df = pd.DataFrame(all_predictions)
        if "predictions" in predictions_df.columns:
            predictions_df = pd.json_normalize(
                predictions_df.to_dict(orient="records"), sep="_"
            )

        output_buffer = StringIO()
        predictions_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="predictions.csv"'}
        return StreamingResponse(output_buffer, media_type="text/csv", headers=headers)

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction from CSV failed: {exc}") from exc

