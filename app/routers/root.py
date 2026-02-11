from fastapi import APIRouter

from app.core.state import state

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Patient Record Prediction API", "status": "running"}


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "all_models_loaded": state.modeofdeliverywithout is not None
        and state.antinatalwithout is not None
        and state.neonatalwithouthistory is not None,
        "individual_models_loaded": state.mode_delivery_model is not None
        and state.antenatal_model is not None,
        "neonatal_withapgar_loaded": state.neonatal_model_withapgar is not None,
        "postnatal_loaded": state.postnatal_model is not None,
        "encoders_loaded": state.encoder is not None
        and state.delivery_encoder is not None
        and state.neonatal_encoder is not None
        and state.postnatal_encoder is not None,
        "features_loaded": state.dic_json is not None
        and state.delivery_dic_json is not None
        and state.neonatal_dic_json is not None
        and state.postnatal_dic_json is not None,
        "column_lists_loaded": state.col_list_combined is not None
        and state.delivery_col_list is not None
        and state.neonatal_col_list is not None
        and state.postnatal_col_list is not None,
    }


@router.get("/debug-models")
async def debug_models():
    """Debug endpoint to check model loading status."""
    return {
        "model_status": {
            "modeofdeliverywithout": state.modeofdeliverywithout is not None,
            "antinatalwithout": state.antinatalwithout is not None,
            "neonatalwithouthistory": state.neonatalwithouthistory is not None,
            "encoder": state.encoder is not None,
            "dic_json": state.dic_json is not None,
            "col_list_combined": state.col_list_combined is not None,
            "mode_delivery_model": state.mode_delivery_model is not None,
            "antenatal_model": state.antenatal_model is not None,
            "delivery_encoder": state.delivery_encoder is not None,
            "delivery_dic_json": state.delivery_dic_json is not None,
            "delivery_col_list": state.delivery_col_list is not None,
            "neonatal_model_withapgar": state.neonatal_model_withapgar is not None,
            "neonatal_encoder": state.neonatal_encoder is not None,
            "neonatal_dic_json": state.neonatal_dic_json is not None,
            "neonatal_col_list": state.neonatal_col_list is not None,
            "postnatal_model": state.postnatal_model is not None,
            "postnatal_encoder": state.postnatal_encoder is not None,
            "postnatal_dic_json": state.postnatal_dic_json is not None,
            "postnatal_col_list": state.postnatal_col_list is not None,
        }
    }


@router.get("/debug-fields")
async def debug_fields():
    """Debug endpoint to check field names in the model."""
    if state.dic_json is None:
        return {"error": "dic_json not loaded"}

    field_names = list(state.dic_json.keys())
    problematic_fields = [
        name
        for name in field_names
        if "Any other Antenatal" in name
        or "Postnatal" in name
        or "Delivery Complications" in name
    ]

    return {
        "total_fields": len(field_names),
        "problematic_fields": problematic_fields,
        "all_fields": field_names[:10],
        "field_count": len(field_names),
    }

