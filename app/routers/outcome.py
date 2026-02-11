from typing import Any, Dict, List, Optional, Tuple

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
    # Add fixed labels (in order) if corresponding input feature is answered as yes
    feature_to_label = {
        "Mental Health Symptoms": "depression",
        "Previous history of PPH (Postpartum Hemorrhage(PPH)) Exploration": "postpartum hemorrhage(pph)",
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

def determine_risk_level_and_referral(
    mode_of_delivery_history: str = "",
    antenatal_complications_history: str = "",
    neonatal_complications_history: str = "",
    mode_of_delivery_condition: str = "",
    antenatal_complications_condition: str = "",
    neonatal_complications_condition: str = "",
) -> Tuple[str, str, str]:
    """
    Determine risk level, remarks, and referral based on predictions.
    Priority: Critical > High Risk > Low Risk
    
    CRITICAL: Only checked from condition model predictions
    HIGH RISK and LOW RISK: Checked from both history and condition model predictions
    
    If ANY one item in condition predictions matches Critical, return Critical.
    If ANY one item in history OR condition predictions matches High Risk, return High Risk.
    """
    # Split history predictions
    mode_list_history = [item.strip() for item in mode_of_delivery_history.split(",") if item.strip()] if mode_of_delivery_history else []
    antenatal_list_history = [
        item.strip() for item in antenatal_complications_history.split(",") if item.strip()
    ] if antenatal_complications_history else []
    neonatal_list_history = [
        item.strip() for item in neonatal_complications_history.split(",") if item.strip()
    ] if neonatal_complications_history else []
    
    # Split condition predictions
    mode_list_condition = [item.strip() for item in mode_of_delivery_condition.split(",") if item.strip()] if mode_of_delivery_condition else []
    antenatal_list_condition = [
        item.strip() for item in antenatal_complications_condition.split(",") if item.strip()
    ] if antenatal_complications_condition else []
    neonatal_list_condition = [
        item.strip() for item in neonatal_complications_condition.split(",") if item.strip()
    ] if neonatal_complications_condition else []

    # Critical maternal complications
    critical_maternal = [
        "massive blood transfusion",
        "postpartum hemorrhage",
        "preeclampsia",
        "preterm labour",
        "acute pyelonephritis",
        "acute renal failure",
        "icu admission",
        "eclampsia",
        "sudden maternal collapse",
        "placental abruption",
        "cardiac failure",
        "pulmonary edema",
        "ectopic pregnancy",
        "hepatitis e and termination of pregnancy and postpartum hemorrhage",
        "hepatitis e and termination of pregnancy and postpartum hemorrhage and shock",
        "imminent eclampsia",
        "neglected transverse lie",
        "pulmonary embolism",
        "shock",
        "peripartum hysterectomy",
        "ruptured uterus",
        "scar dehiscence",
        "sepsis",
    ]

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
    # Critical mode of delivery
    critical_delivery_modes = [
        "induced abortion",
        "laparotomy",
        "laparotomy and peripartum hysterectomy and icu admission",
        "laparotomy and repair of uterus",
    ]

    # High risk maternal complications
    high_risk_maternal = [
        "anemia",
        "rh incompatibility and antid prophylaxis",
        "gestational diabetes mellitus",
        "obstetric cholestasis",
        "urinary tract infection",
        "cervical incompetence",
        "chorioamnionitis",
        "cord prolapse",
        "deep vein thrombosis",
        "diabetes mellitus",
        "obstructed labour",
        "perineal tears",
        "acute hepatitis e",
        "hypertension",
        "hypertension and pprom",
        "intrauterine growth restriction",
        "low lying placenta",
        "molar pregnancy",
        "puerperal pyrexia",
        "ecv",
        "placenta accreta spectrum",
        "pprom",
        "recurrent urinary tract infection",
        "shoulder dystocia",
        "thalasemia minor",
        "threatened miscarriage",
        "threatened preterm labour",
        "thyroid disease",
        "uterine perforation",
        "rh incompatibility",
    ]

    # High risk mode of delivery
    high_risk_delivery_modes = [
        "cesarean section",
        "assisted vaginal delivery",
        "erpc",
        "ecv and vaginal",
        "hysterotomy",
        "termination of pregnancy",
        "induction of labour",
        "induction of labour and vaginal",
        "laparoscopy",
        "medical termination of pregnancy and vaginal",
        "methotrexate",
        "suction evacuation",
        "not delivered yet and cervical cerclage",
        "not delivered yet and ecv",
    ]

    # Check for CRITICAL conditions (highest priority) - ONLY from condition models
    critical_instruction = (
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
    
    print("\n" + "="*80)
    print("RISK LEVEL DETERMINATION - Checking for CRITICAL (Condition Model Only)")
    print("="*80)
    print(f"Mode of Delivery (Condition): {mode_list_condition}")
    print(f"Antenatal Complications (Condition): {antenatal_list_condition}")
    print(f"Neonatal Complications (Condition): {neonatal_list_condition}")
    print("-"*80)
    
    # Check mode of delivery (condition only)
    for mode in mode_list_condition:
        if mode.lower() in critical_delivery_modes:
            print(f"🚨 CRITICAL DETECTED: Mode of Delivery = '{mode}' (from condition model)")
            print(f"   Matched critical delivery mode: {mode.lower()}")
            print("="*80 + "\n")
            return critical_instruction, "Critical", "Refer immediately"

    # Check antenatal complications (condition only)
    for complication in antenatal_list_condition:
        if complication.lower() in critical_maternal:
            print(f"🚨 CRITICAL DETECTED: Antenatal Complication = '{complication}' (from condition model)")
            print(f"   Matched critical maternal complication: {complication.lower()}")
            print("="*80 + "\n")
            return critical_instruction, "Critical", "Refer immediately"

    # Check neonatal complications (condition only)
    for complication in neonatal_list_condition:
        if complication.lower() in critical_neonatal:
            print(f"🚨 CRITICAL DETECTED: Neonatal Complication = '{complication}' (from condition model)")
            print(f"   Matched critical neonatal complication: {complication.lower()}")
            print("="*80 + "\n")
            return critical_instruction, "Critical", "Refer immediately"
    
    print("No CRITICAL conditions found in condition model predictions.")

    # Check for HIGH RISK conditions - from BOTH history and condition models
    high_risk_instruction = (
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
    
    # Combine history and condition lists for high risk checking
    antenatal_combined = list(set(antenatal_list_history + antenatal_list_condition))
    mode_combined = list(set(mode_list_history + mode_list_condition))
    neonatal_combined = list(set(neonatal_list_history + neonatal_list_condition))
    
    print("\n" + "="*80)
    print("RISK LEVEL DETERMINATION - Checking for HIGH RISK (History + Condition Models)")
    print("="*80)
    print(f"Mode of Delivery - History: {mode_list_history}, Condition: {mode_list_condition}")
    print(f"Mode of Delivery Combined: {mode_combined}")
    print(f"Antenatal Complications - History: {antenatal_list_history}, Condition: {antenatal_list_condition}")
    print(f"Antenatal Complications Combined: {antenatal_combined}")
    print(f"Neonatal Complications - History: {neonatal_list_history}, Condition: {neonatal_list_condition}")
    print(f"Neonatal Complications Combined: {neonatal_combined}")
    print("-"*80)
    
    # Check antenatal complications (from both history and condition)
    for complication in antenatal_combined:
        if complication.lower() in high_risk_maternal:
            source = "history" if complication in antenatal_list_history else "condition"
            print(f"⚠️  HIGH RISK DETECTED: Antenatal Complication = '{complication}' (from {source} model)")
            print(f"   Matched high risk maternal complication: {complication.lower()}")
            print("="*80 + "\n")
            return (
                high_risk_instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )

    # Check mode of delivery (from both history and condition)
    for mode in mode_combined:
        if mode.lower() in high_risk_delivery_modes:
            source = "history" if mode in mode_list_history else "condition"
            print(f"⚠️  HIGH RISK DETECTED: Mode of Delivery = '{mode}' (from {source} model)")
            print(f"   Matched high risk delivery mode: {mode.lower()}")
            print("="*80 + "\n")
            return (
                high_risk_instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )

    # Check neonatal complications (from both history and condition)
    high_risk_neonatal_lower = [item.lower() for item in high_risk_neonatal]
    for complication in neonatal_combined:
        if complication.lower() in high_risk_neonatal_lower:
            source = "history" if complication in neonatal_list_history else "condition"
            print(f"⚠️  HIGH RISK DETECTED: Neonatal Complication = '{complication}' (from {source} model)")
            print(f"   Matched high risk neonatal complication: {complication.lower()}")
            print("="*80 + "\n")
            return (
                high_risk_instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )
    
    print("No HIGH RISK conditions found in history or condition model predictions.")

    # Check for LOW RISK conditions - from BOTH history and condition models
    # For low risk, ALL values from both history and condition should match low risk criteria
    antenatal_combined_lower = [c.lower() for c in antenatal_combined]
    mode_combined_lower = [m.lower() for m in mode_combined]
    neonatal_combined_lower = [n.lower() for n in neonatal_combined]

    print("\n" + "="*80)
    print("RISK LEVEL DETERMINATION - Checking for LOW RISK (History + Condition Models)")
    print("="*80)
    print(f"All predictions checked: Mode={mode_combined}, Antenatal={antenatal_combined}, Neonatal={neonatal_combined}")
    print("-"*80)

    # Check if all antenatal complications are low risk (from both history and condition)
    all_antenatal_low_risk = (
        all(c in ["no complication", "none", "not specified"] for c in antenatal_combined_lower)
        if antenatal_combined
        else True
    )
    print(f"All antenatal complications are low risk: {all_antenatal_low_risk}")

    # Check if all modes of delivery are low risk (from both history and condition)
    all_mode_low_risk = (
        all(m in ["vaginal", "normal delivery"] for m in mode_combined_lower)
        if mode_combined
        else True
    )
    print(f"All modes of delivery are low risk: {all_mode_low_risk}")

    # Check if all neonatal complications are low risk (from both history and condition)
    all_neonatal_low_risk = (
        all(
            n in ["no complication", "normal", "none", "not specified"]
            for n in neonatal_combined_lower
        )
        if neonatal_combined
        else True
    )
    print(f"All neonatal complications are low risk: {all_neonatal_low_risk}")

    is_low_risk = all_antenatal_low_risk and all_mode_low_risk and all_neonatal_low_risk

    if is_low_risk:
        print("✅ LOW RISK DETECTED: All conditions from both history and condition models are low risk")
        print("="*80 + "\n")
        instruction = "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits."
        return instruction, "Low Risk", "Follow routine antenatal care protocol."

    # Default: Low Risk for other cases
    print("✅ LOW RISK (Default): No critical or high risk conditions found, defaulting to low risk")
    print("="*80 + "\n")
    instruction = (
        "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits. "
        "Continue providing routine antenatal, delivery, and postnatal care according to standard guidelines."
    )
    return instruction, "Low Risk", "Follow routine antenatal care protocol."


def determine_risk_level_and_referral_with_postnatal(
    mode_of_delivery: str,
    antenatal_complications: str,
    neonatal_complications: str,
    postnatal_complications: str = "",
) -> Tuple[str, str, str]:
    """
    Determine risk level, remarks, and referral based on predictions including postnatal complications.
    Priority: Critical > High Risk > Low Risk
    Checks ALL fields (mode_of_delivery, antenatal, neonatal, postnatal) together.
    If ANY one item in ANY field matches Critical, return Critical.
    If ANY one item in ANY field matches High Risk, return High Risk.
    """
    # Split comma-separated values into lists
    mode_list = (
        [item.strip() for item in mode_of_delivery.split(",") if item.strip()]
        if mode_of_delivery
        else []
    )
    antenatal_list = (
        [item.strip() for item in antenatal_complications.split(",") if item.strip()]
        if antenatal_complications
        else []
    )
    neonatal_list = (
        [item.strip() for item in neonatal_complications.split(",") if item.strip()]
        if neonatal_complications
        else []
    )
    postnatal_list = (
        [item.strip() for item in postnatal_complications.split(",") if item.strip()]
        if postnatal_complications
        else []
    )

    # Critical maternal complications
    critical_maternal = [
        "Massive Blood Transfusion",
        "Postpartum Hemorrhage",
        "Preeclampsia",
        "Preterm Labour",
        "Acute Pyelonephritis",
        "Acute Renal Failure",
        "ICU Admission",
        "Eclampsia",
        "Sudden Maternal Collapse",
        "Placental Abruption",
        "Cardiac Failure",
        "Pulmonary Edema",
        "Ectopic Pregnancy",
        "Hepatitis E and Termination of Pregnancy and Postpartum Hemorrhage",
        "Hepatitis E and Termination of Pregnancy and Postpartum Hemorrhage and Shock",
        "Imminent Eclampsia",
        "Neglected Transverse lie",
        "Pulmonary Embolism",
        "Shock",
        "Peripartum Hysterectomy",
        "Ruptured Uterus",
        "Scar Dehiscence",
        "Sepsis",
    ]

    # Critical neonatal complications
    critical_neonatal = [
        "miscarriage",
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

    # Critical mode of delivery
    critical_delivery_modes = [
        "induced abortion",
        "laparotomy",
        "laparotomy and peripartum hysterectomy and icu admission",
        "laparotomy and repair of uterus",
    ]

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

    # High risk maternal complications
    high_risk_maternal = [
        "anemia",
        "rh incompatibility and antid prophylaxis",
        "gestational diabetes mellitus",
        "obstetric cholestasis",
        "urinary tract infection",
        "cervical incompetence",
        "chorioamnionitis",
        "cord prolapse",
        "deep vein thrombosis",
        "diabetes mellitus",
        "obstructed labour",
        "perineal tears",
        "acute hepatitis e",
        "hypertension",
        "hypertension and pprom",
        "intrauterine growth restriction",
        "low lying placenta",
        "molar pregnancy",
        "puerperal pyrexia",
        "ecv",
        "placenta accreta spectrum",
        "pprom",
        "recurrent urinary tract infection",
        "shoulder dystocia",
        "thalasemia minor",
        "threatened miscarriage",
        "threatened preterm labour",
        "thyroid disease",
        "uterine perforation",
        "rh incompatibility",
    ]

    # High risk mode of delivery
    high_risk_delivery_modes = [
        "cesarean section",
        "assisted vaginal delivery",
        "erpc",
        "ecv and vaginal",
        "hysterotomy",
        "termination of pregnancy",
        "induction of labour",
        "induction of labour and vaginal",
        "laparoscopy",
        "medical termination of pregnancy and vaginal",
        "methotrexate",
        "suction evacuation",
        "not delivered yet and cervical cerclage",
        "not delivered yet and ecv",
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

    # CRITICAL INSTRUCTION (same for all critical conditions)
    critical_instruction = (
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

    # HIGH RISK INSTRUCTION (same for all high risk conditions)
    high_risk_instruction = (
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

    # PRIORITY 1: Check for CRITICAL conditions in ALL fields
    # Check mode of delivery
    for mode in mode_list:
        if mode.lower() in critical_delivery_modes:
            return critical_instruction, "Critical", "Refer immediately"

    # Check antenatal complications
    for complication in antenatal_list:
        if complication.lower() in critical_maternal:
            return critical_instruction, "Critical", "Refer immediately"

    # Check neonatal complications
    for complication in neonatal_list:
        if complication.lower() in critical_neonatal:
            return critical_instruction, "Critical", "Refer immediately"

    # Check postnatal complications
    for complication in postnatal_list:
        if complication.lower() in critical_postnatal:
            return critical_instruction, "Critical", "Refer immediately"

    # PRIORITY 2: Check for HIGH RISK conditions in ALL fields
    # Check antenatal complications
    for complication in antenatal_list:
        if complication.lower() in high_risk_maternal:
            return (
                high_risk_instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )

    # Check mode of delivery
    for mode in mode_list:
        if mode.lower() in high_risk_delivery_modes:
            return (
                high_risk_instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )

    # Check postnatal complications
    for complication in postnatal_list:
        if complication.lower() in high_risk_postnatal:
            return (
                high_risk_instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )

    # Check neonatal complications
    high_risk_neonatal_lower = [item.lower() for item in high_risk_neonatal]
    for complication in neonatal_list:
        if complication.lower() in high_risk_neonatal_lower:
            return (
                high_risk_instruction,
                "High Risk",
                "No need for immediate referral however patient should be cared under supervision of expert Gynecologist.",
            )

    # PRIORITY 3: Check for LOW RISK conditions
    antenatal_lower = [c.lower() for c in antenatal_list]
    mode_lower = [m.lower() for m in mode_list]
    neonatal_lower = [n.lower() for n in neonatal_list]
    postnatal_lower = [p.lower() for p in postnatal_list]

    # Check if all antenatal complications are low risk
    all_antenatal_low_risk = (
        all(c in ["no complication", "none", "not specified"] for c in antenatal_lower)
        if antenatal_list
        else True
    )

    # Check if all modes of delivery are low risk
    all_mode_low_risk = (
        all(m in ["vaginal", "normal delivery"] for m in mode_lower)
        if mode_list
        else True
    )

    # Check if all neonatal complications are low risk
    all_neonatal_low_risk = (
        all(
            n in ["no complication", "normal", "none", "not specified"]
            for n in neonatal_lower
        )
        if neonatal_list
        else True
    )

    # Check if all postnatal complications are low risk
    all_postnatal_low_risk = (
        all(
            p in ["no complication", "normal", "none", "not specified"]
            for p in postnatal_lower
        )
        if postnatal_list
        else True
    )

    is_low_risk = (
        all_antenatal_low_risk
        and all_mode_low_risk
        and all_neonatal_low_risk
        and all_postnatal_low_risk
    )

    if is_low_risk:
        instruction = "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits."
        return instruction, "Low Risk", "Follow routine antenatal care protocol."

    # Default: Low Risk for other cases
    instruction = (
        "These prediction values indicate currently low risk pregnancy to be followed for routine followup visits. "
        "Continue providing routine antenatal, delivery, and postnatal care according to standard guidelines."
    )
    return instruction, "Low Risk", "Follow routine antenatal care protocol."


def check_rh_incompatibility(patient_bg: str, partner_bg: str) -> bool:
    """
    Check if there is Rh incompatibility between patient and partner blood groups.
    Rh incompatibility occurs when:
    - Patient has negative blood group (A-, B-, AB-, O-)
    - Partner has positive blood group (A+, B+, AB+, O+)
    """
    if not patient_bg or not partner_bg:
        return False

    patient_bg_str = str(patient_bg).upper().strip()
    partner_bg_str = str(partner_bg).upper().strip()

    # Check if patient has negative blood group (ends with '-' or contains 'negative')
    patient_is_negative = (
        patient_bg_str.endswith("-")
        or "negative" in patient_bg_str
        or "neg" in patient_bg_str
    )

    # Check if partner has positive blood group (ends with '+' or contains 'positive')
    partner_is_positive = (
        partner_bg_str.endswith("+")
        or "positive" in partner_bg_str
        or "pos" in partner_bg_str
    )

    return patient_is_negative and partner_is_positive


@router.post("/outcomeModel")
async def predict_outcome(patient_record: PatientRecord):
    # Check if new structure is available
    use_new_structure = (
        state.mode_delivery_history_model is not None
        and state.mode_delivery_condition_model is not None
        and state.antenatal_history_model is not None
        and state.antenatal_condition_model is not None
        and state.neonatal_historymodel_withoutapgar is not None
        and state.neonatal_conditionmodel_withoutapgar is not None
    )

    # Fallback to old structure check
    use_old_structure = (
        state.modeofdeliverywithout is not None
        and state.antinatalwithout is not None
        and state.neonatalwithouthistory is not None
        and state.neonatalwithoutcondition is not None
        and state.encoder_history is not None
        and state.encoder_condition is not None
        and state.dic_json is not None
        and state.col_list_combined is not None
        and state.single_encoder is not None
    )

    if not use_new_structure and not use_old_structure:
        raise HTTPException(status_code=500, detail="Models not loaded properly")

    try:
        payload = patient_record.model_dump()
        record = payload.get("patientData", {})
        print(record)
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
                try:
                    record[key] = float(record[key])
                except Exception:
                    pass

        # Convert gestational age from "weeks.days" to total days (e.g., 23.5 -> 23*7 + 5)
        ga_key = "Gestational Age (GA) (in weeks)"
        if ga_key in record and record[ga_key] not in (None, ""):
            record[ga_key] = convert_ga_to_days(record[ga_key])

        record_lower = {
            k: (v.lower() if isinstance(v, str) else v) for k, v in record.items()
        }

        history_data, condition_data = history_based_features()

        # Initialize variables for history and condition predictions
        mode_history_top2 = []
        mode_condition_top2 = []
        antenatal_history_top3 = []
        antenatal_condition_top3 = []
        neonatal_top3_history = []
        neonatal_top3_condition = []

        if use_new_structure:
            # Use new structure with history + condition models for all three predictions

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
            print("Mode of Delivery Top 2 from History:")
            print("History: ", mode_history_top2)
        
            mode_condition_top2 = get_top_k_labels(
                state.mode_delivery_col_list_condition, mode_probs_condition, k=2
            )
            print("Mode of Delivery Top 2 from Condition:")
            print("Condition: ", mode_condition_top2)
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
            print("Antenatal Top 3:")
            print("History: ", antenatal_history_top3)
            print("Condition: ", antenatal_condition_top3)
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
            antenatal_top3 = apply_fixed_antenatal_labels(payload.get("patientData", {}), antenatal_top3)

            # Neonatal: history + condition (already using new structure)
            # condition_data_with_apgar = condition_data + [
            #     "Neonatal Apgar Score:(1 Minute)",
            #     "Neonatal Apgar Score:(5 Minute)",
            # ]

            # Get neonatal col_list - same pattern as neonatal router
            neonatal_col_list = (
                state.neonatal_col_list[0]
                if (
                    isinstance(state.neonatal_col_list, list)
                    and len(state.neonatal_col_list) > 0
                    and isinstance(state.neonatal_col_list[0], list)
                )
                else (state.neonatal_col_list if state.neonatal_col_list else [])
            )

            neonatal_history_features = creating_features(
                history_data,
                record_lower,
                state.antenatal_encoder_history,
                dic_json=state.dic_json,
            )
            neonatal_condition_features = creating_features(
                condition_data,
                record_lower,
                state.antenatal_encoder_condition,
                dic_json=state.dic_json,
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
        else:
            # Use old structure
            history_filtered = creating_features(
                history_data, record_lower, state.encoder_history
            )

            condition_data.extend(
                ["Neonatal Apgar Score:(1 Minute)", "Neonatal Apgar Score:(5 Minute)"]
            )
            condition_filtered = creating_features(
                condition_data, record_lower, state.encoder_condition
            )
            mode_of_delivery_data = creating_features(
                list(record.keys()), record_lower, state.single_encoder
            )

            models = [
                state.modeofdeliverywithout,
                state.antinatalwithout,
                state.neonatalwithouthistory,
                state.neonatalwithoutcondition,
            ]

            _, mode_probs = predict_labels_for_model(
                models[0],
                mode_of_delivery_data,
                state.col_list_combined[0],
                "multilabel",
            )
            _, antenatal_probs = predict_labels_for_model(
                models[1],
                mode_of_delivery_data,
                state.col_list_combined[1],
                "multilabel",
            )
            _, neonatal_probs_history = predict_labels_for_model(
                models[2], history_filtered, state.col_list_combined[2], "multilabel"
            )
            _, neonatal_probs_condition = predict_labels_for_model(
                models[3], condition_filtered, state.col_list_combined[2], "multilabel"
            )

            mode_delivery_top3 = get_top_k_labels(
                state.col_list_combined[0], mode_probs, k=2
            )
            antenatal_top3 = get_top_k_labels(
                state.col_list_combined[1], antenatal_probs, k=3
            )
            neonatal_top3_history = get_top_k_labels(
                state.col_list_combined[2], neonatal_probs_history, k=2
            )
            neonatal_top3_condition = get_top_k_labels(
                state.col_list_combined[2], neonatal_probs_condition, k=2
            )

        if "not delivered yet" in mode_delivery_top3:
            mode_delivery_top3.remove("not delivered yet")

        def determine_rh_incompatibility(antenatal_pred):
                    # Check for Rh incompatibility based on blood groups
            original_record = payload.get("patientData", {})
            patient_bg = original_record.get("PatientHasPosGp", "")
            partner_bg = original_record.get("PartenerHasPosGp", "")
            if "antid prophylaxis due to rh incompatibility" in antenatal_pred:
                antenatal_pred.remove("antid prophylaxis due to rh incompatibility")
            if "rh incompatibility" in antenatal_pred:
                antenatal_pred.remove("rh incompatibility")
            if (
                "antid prophylaxis rh incompatibility due to rh incompatibility"
                in antenatal_pred
            ):
                antenatal_pred.remove(
                    "antid prophylaxis rh incompatibility due to rh incompatibility"
                )
            if check_rh_incompatibility(patient_bg, partner_bg):
                rh_incompatibility = "Rh Incompatibility"
                # Add to antenatal results if not already present
                antenatal_pred_lower = [item.lower() for item in antenatal_pred]

                if rh_incompatibility.lower() not in antenatal_pred_lower:
                    antenatal_pred.append(rh_incompatibility)
            return antenatal_pred
        neonatal_top3_history = determine_rh_incompatibility(neonatal_top3_history)
        neonatal_top3_condition = determine_rh_incompatibility(neonatal_top3_condition)
        print("Neonatal Top 3:")
        print("History: ", neonatal_top3_history)
        print("Condition: ", neonatal_top3_condition)
        
        # Prepare separate history and condition predictions for risk determination
        if use_new_structure:
            # For new structure, we have separate history and condition predictions
            mode_delivery_history_str = ",".join(str(x) for x in mode_history_top2) if mode_history_top2 else ""
            mode_delivery_condition_str = ",".join(str(x) for x in mode_condition_top2) if mode_condition_top2 else ""
            antenatal_history_str = ",".join(str(x) for x in antenatal_history_top3) if antenatal_history_top3 else ""
            antenatal_condition_str = ",".join(str(x) for x in antenatal_condition_top3) if antenatal_condition_top3 else ""
            neonatal_history_str = ",".join(str(x) for x in neonatal_top3_history) if neonatal_top3_history else ""
            neonatal_condition_str = ",".join(str(x) for x in neonatal_top3_condition) if neonatal_top3_condition else ""
        else:
            # For old structure, we only have separate predictions for neonatal
            # For mode and antenatal, use combined as both history and condition
            mode_delivery_history_str = ",".join(str(x) for x in mode_delivery_top3) if mode_delivery_top3 else ""
            mode_delivery_condition_str = ",".join(str(x) for x in mode_delivery_top3) if mode_delivery_top3 else ""
            antenatal_history_str = ",".join(str(x) for x in antenatal_top3) if antenatal_top3 else ""
            antenatal_condition_str = ",".join(str(x) for x in antenatal_top3) if antenatal_top3 else ""
            neonatal_history_str = ",".join(str(x) for x in neonatal_top3_history) if neonatal_top3_history else ""
            neonatal_condition_str = ",".join(str(x) for x in neonatal_top3_condition) if neonatal_top3_condition else ""
        
        results: Dict[str, Any] = {
            "Mode_of_delivery2": mode_delivery_top3,
            "Antenatal_Peripartum_Maternal_Complications": antenatal_top3,
            "Neonatal_Complications": list(
                set(neonatal_top3_history + neonatal_top3_condition)
            ),
        }
        results["Mode_of_delivery2"] = ",".join(
            str(x) for x in results["Mode_of_delivery2"]
        )
        results["Antenatal_Peripartum_Maternal_Complications"] = ",".join(
            str(x) for x in results["Antenatal_Peripartum_Maternal_Complications"]
        )
        results["Neonatal_Complications"] = ",".join(
            str(x) for x in results["Neonatal_Complications"]
        )

        # Determine risk level, remarks, and referral using code-based logic
        # Critical: only from condition models
        # High/Low Risk: from both history and condition models
        remarks_text, risk_level, referral_text = determine_risk_level_and_referral(
            mode_of_delivery_history=mode_delivery_history_str,
            antenatal_complications_history=antenatal_history_str,
            neonatal_complications_history=neonatal_history_str,
            mode_of_delivery_condition=mode_delivery_condition_str,
            antenatal_complications_condition=antenatal_condition_str,
            neonatal_complications_condition=neonatal_condition_str,
        )

        response = {
            "message": "Sequential prediction completed successfully",
            "predictions": results,
            "Remarks": remarks_text,
            "risk_Level": risk_level,
            "referral": referral_text,
        }
        return JSONResponse(content=response)

    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {exc}"
        ) from exc
