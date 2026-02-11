from typing import Any, Optional, Tuple

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.state import state
from app.schemas import PatientRecord
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


def determine_neonatal_risk_level(neonatal_complications: list) -> Tuple[str, str, str]:
    """
    Determine risk level for neonatal complications based on predictions.
    Priority: Critical > High Risk > Low Risk
    """
    # Critical neonatal complications
    critical_neonatal = [
        "neonatal death",
        "birth asphyxia",
        "neonatal sepsis",
        "congenital anomalies",
        "congenital heart defects",
        "neural tube defects",
        "chromosomal abnormalities",
        "neonatal meningitis",
        "neonatal pneumonia",
        "severe respiratory distress syndrome",
        "severe jaundice requiring exchange transfusion",
        "neonatal seizures",
        "neonatal hypoglycemia",
    ]

    # High risk neonatal complications
    high_risk_neonatal = [
        "miscarriage",
        "respiratory distress syndrome",
        "jaundice",
        "low birth weight",
        "fetal distress",
        "intrauterine growth restriction, preterm birth complications, neonatal jaundice",
        "respiratory distress",
        "neonatal infection",
        "birth trauma",
        "premature birth",
        "premature birth and respiratory distress syndrome",
        "neonatal complications requiring nicu",
    ]

    # Check for CRITICAL conditions
    for complication in neonatal_complications:
        if complication.lower() in critical_neonatal:
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
    high_risk_neonatal_lower = [item.lower() for item in high_risk_neonatal]
    for complication in neonatal_complications:
        if complication.lower() in high_risk_neonatal_lower:
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


@router.post("/predict/neonatal-complications")
async def predict_neonatal(patient_record: PatientRecord):
    if (
        state.neonatal_historymodel_withapgar is None
        or state.neonatal_conditionmodel_withapgar is None
        or state.neonatal_col_list is None
        or state.encoder_history is None
        or state.neonatal_encoder_condition is None
    ):
        raise HTTPException(status_code=500, detail="Neonatal with Apgar model not loaded")

    try:
        record = patient_record.model_dump()["patientData"]

        keys_to_remove = [
            "Postnatal_Symptoms",
            "Postnatal_Examination",
            "Mode_of_delivery2",
            "Antenatal_Peripartum_Maternal_Complications",
            "Neonatal__Fetal_Complications",
        ]
        for k in keys_to_remove:
            record.pop(k, None)

        # Convert gestational age from "weeks.days" to total days (e.g., 23.5 -> 23*7 + 5)
        ga_key = "Gestational Age (GA) (in weeks)"
        if ga_key in record and record[ga_key] not in (None, ""):
            record[ga_key] = convert_ga_to_days(record[ga_key])

        record_lower = {k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()}

        history_data, condition_data = history_based_features()
        neonatal_history_data = creating_features(history_data, record_lower, state.neonatal_encoder_history)

        condition_data.extend(["Neonatal Apgar Score:(1 Minute)", "Neonatal Apgar Score:(5 Minute)"])
        neonatal_condition_data = creating_features(
            condition_data, record_lower, state.neonatal_encoder_condition
        )

        _, neonatal_history_probs = predict_labels_for_model(
            state.neonatal_historymodel_withapgar,
            neonatal_history_data,
            state.neonatal_col_list[0],
            "multilabel",
        )
        _, neonatal_condition_probs = predict_labels_for_model(
            state.neonatal_conditionmodel_withapgar,
            neonatal_condition_data,
            state.neonatal_col_list[0],
            "multilabel",
        )

        neonatal_history_top3 = get_top_k_labels(state.neonatal_col_list[0], neonatal_history_probs, k=2)
        neonatal_condition_top3 = get_top_k_labels(
            state.neonatal_col_list[0], neonatal_condition_probs, k=2
        )

        prediction = list(set(neonatal_history_top3 + neonatal_condition_top3))

        # Determine risk level using high_risk_neonatal variable
        remarks_text, risk_level, referral_text = determine_neonatal_risk_level(prediction)

        # Fallback to external service if available (optional)
        # remarks_payload = {
        #     "predictions": {"Neonatal_Fetal_Complications": prediction},
        #     "endpoint": "neonatal",
        # }

        # try:
        #     resp = requests.post(
        #         "http://127.0.0.1:8003/generate-remarks",
        #         json=remarks_payload,
        #         timeout=10,
        #     )
        #     if resp.ok:
        #         result_response = resp.json()
        #         external_remarks = result_response.get("instruction")
        #         # Use external remarks if available, otherwise use determined remarks
        #         if external_remarks:
        #             remarks_text = external_remarks
        # except Exception:
        #     pass
        prediction=",".join(
            str(x) for x in  prediction  )
        response = {
            "message": "prediction completed for neonatal complications",
            "predictions": prediction,
            "Remarks": remarks_text,
            "risk_Level": risk_level,
            "referral": referral_text,
        }
        return JSONResponse(content=response)

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Neonatal prediction failed: {exc}") from exc

