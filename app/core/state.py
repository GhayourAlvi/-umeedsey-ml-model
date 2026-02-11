from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelState:
    modeofdeliverywithout: Optional[Any] = None
    antinatalwithout: Optional[Any] = None
    neonatalwithouthistory: Optional[Any] = None
    neonatalwithoutcondition: Optional[Any] = None

    encoder_history: Optional[Any] = None
    encoder_condition: Optional[Any] = None
    encoder: Optional[Any] = None

    dic_json: Optional[Dict[str, List[str]]] = None
    col_list_combined: Optional[List[List[str]]] = None

    single_encoder: Optional[Any] = None

    mode_delivery_model: Optional[Any] = None
    antenatal_model: Optional[Any] = None
    delivery_encoder: Optional[Any] = None
    delivery_dic_json: Optional[Dict[str, List[str]]] = None
    delivery_col_list: Optional[List[List[str]]] = None
    
    # New structure: Mode of Delivery with history and condition models
    mode_delivery_history_model: Optional[Any] = None
    mode_delivery_condition_model: Optional[Any] = None
    mode_delivery_encoder_history: Optional[Any] = None
    mode_delivery_encoder_condition: Optional[Any] = None
    mode_delivery_dic_json_history: Optional[Dict[str, List[str]]] = None
    mode_delivery_dic_json_condition: Optional[Dict[str, List[str]]] = None
    mode_delivery_col_list_history: Optional[List[str]] = None
    mode_delivery_col_list_condition: Optional[List[str]] = None
    
    # New structure: Antenatal with history and condition models
    antenatal_history_model: Optional[Any] = None
    antenatal_condition_model: Optional[Any] = None
    antenatal_encoder_history: Optional[Any] = None
    antenatal_encoder_condition: Optional[Any] = None
    antenatal_dic_json_history: Optional[Dict[str, List[str]]] = None
    antenatal_dic_json_condition: Optional[Dict[str, List[str]]] = None
    antenatal_col_list_history: Optional[List[str]] = None
    antenatal_col_list_condition: Optional[List[str]] = None

    neonatal_model_withapgar: Optional[Any] = None
    neonatal_encoder: Optional[Any] = None
    neonatal_dic_json: Optional[Dict[str, List[str]]] = None
    neonatal_col_list: Optional[List[List[str]]] = None
    neonatal_single_model: Optional[Any] = None
    neonatal_historymodel_withapgar: Optional[Any] = None
    neonatal_encoder_history: Optional[Any] = None
    neonatal_conditionmodel_withapgar: Optional[Any] = None
    neonatal_encoder_condition: Optional[Any] = None

    postnatal_model: Optional[Any] = None
    postnatal_encoder: Optional[Any] = None
    postnatal_dic_json: Optional[Dict[str, List[str]]] = None
    postnatal_col_list: Optional[List[List[str]]] = None
    
    # New structure: Postnatal with history and condition models
    postnatal_history_model: Optional[Any] = None
    postnatal_condition_model: Optional[Any] = None
    postnatal_encoder_history: Optional[Any] = None
    postnatal_encoder_condition: Optional[Any] = None
    postnatal_dic_json_history: Optional[Dict[str, List[str]]] = None
    postnatal_dic_json_condition: Optional[Dict[str, List[str]]] = None
    postnatal_col_list_history: Optional[List[str]] = None
    postnatal_col_list_condition: Optional[List[str]] = None


state = ModelState()

