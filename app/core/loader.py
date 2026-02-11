import joblib

from .state import state


def load_models() -> bool:
    """
    Load all persisted models and encoders into the shared state object.
    Returns True when every artifact loads successfully, False otherwise.
    """
    try:
        print("Loading models...")

        print("Loading delivery_antinatal_neonatal.pkl...")
        try:
            artifacts = joblib.load("save_models/delivery_antinatal_neonatal.pkl")
            print(f"Keys in delivery_antinatal_neonatal.pkl: {list(artifacts.keys())}")

            # Try to load old structure keys (for backward compatibility)
            state.mode_delivery_history_model = artifacts.get("Modeofdeliverwithoutapgrarhistory")
            state.mode_delivery_condition_model = artifacts.get("Modeofdeliverwithoutapgrarcondition")      
            state.antenatal_history_model = artifacts.get("antnatalwithoutapgrarhistory")
            state.antenatal_condition_model = artifacts.get("antnatalwithoutapgrarcondition")
            state.neonatal_historymodel_withoutapgar = artifacts.get("neonatalwithoutapgrarhistory")
            state.neonatal_conditionmodel_withoutapgar = artifacts.get("neonatalwithoutapgrarcondition")
            
            
            state.dic_json = artifacts.get("orignal_features") or artifacts.get("orignal_features_list")
            state.col_list_combined = artifacts.get("col_list")
            
            print("✅ delivery_antinatal_neonatal.pkl loaded")
        except FileNotFoundError:
            print("⚠️ delivery_antinatal_neonatal.pkl not found, skipping...")
        except KeyError as e:
            print(f"⚠️ Key {e} not found in delivery_antinatal_neonatal.pkl, continuing with available keys...")

        # # Try to load old single model files (for backward compatibility)
        # try:
        #     state.modeofdeliverywithout = joblib.load("save_models/modeofdelivery_without.pkl")
        # except FileNotFoundError:
        #     print("⚠️ modeofdelivery_without.pkl not found, skipping...")
        
        # try:
        #     delivery_single_model = joblib.load("save_models/delivery_single_model.pkl")
        #     state.encoder = delivery_single_model.get("encoder")
        #     state.single_encoder = delivery_single_model.get("encoder")
        #     state.neonatal_single_model = delivery_single_model.get("neonatalwithoutapgrar")
        # except FileNotFoundError:
        #     print("⚠️ delivery_single_model.pkl not found, skipping...")

        # Try to load old encoder files (for backward compatibility)
        try:
            state.encoder_history = joblib.load("save_models/historyEncoded.pkl")
        except FileNotFoundError:
            print("⚠️ historyEncoded.pkl not found, skipping...")
        
        try:
            state.encoder_condition = joblib.load("save_models/conditionEncoded.pkl")
        except FileNotFoundError:
            print("⚠️ conditionEncoded.pkl not found, skipping...")

        print("Loading delivery_antinatal.pkl...")
        try:
            delivery_antinatal = joblib.load("save_models/delivery_antinatal.pkl")
            print(f"Keys in delivery_antinatal.pkl: {list(delivery_antinatal.keys())}")
            state.mode_delivery_model = delivery_antinatal.get("mode_of_delivery_model")
            state.antenatal_model = delivery_antinatal.get("antenatal_complications_model")
            state.delivery_encoder = delivery_antinatal.get("encoder")
            state.delivery_dic_json = delivery_antinatal.get("orignal_features")
            state.delivery_col_list = delivery_antinatal.get("col_list")
            print("✅ delivery_antinatal.pkl loaded")
        except FileNotFoundError:
            print("⚠️ delivery_antinatal.pkl not found, skipping...")

        # Load new structure: Mode of Delivery with history and condition models
        print("Loading mode_of_delivery.pkl...")
        try:
            mode_delivery = joblib.load("save_models/mode_of_delivery.pkl")
            print(f"Keys in mode_of_delivery.pkl: {list(mode_delivery.keys())}")
            state.mode_delivery_history_model = mode_delivery.get("mode_of_delivery_history")
            state.mode_delivery_condition_model = mode_delivery.get("mode_of_delivery_condition")
            state.mode_delivery_encoder_history = mode_delivery.get("encoder_history")
            state.mode_delivery_encoder_condition = mode_delivery.get("encoder_condition")
            state.mode_delivery_dic_json_history = mode_delivery.get("orignal_features_history")
            state.mode_delivery_dic_json_condition = mode_delivery.get("orignal_features_condition")
            col_list = mode_delivery.get("col_list", [])
            if isinstance(col_list, list) and len(col_list) >= 2:
                state.mode_delivery_col_list_history = col_list[0] if isinstance(col_list[0], list) else col_list
                state.mode_delivery_col_list_condition = col_list[1] if isinstance(col_list[1], list) else col_list
            elif isinstance(col_list, list) and len(col_list) == 1:
                # If only one col_list, use it for both
                state.mode_delivery_col_list_history = col_list[0] if isinstance(col_list[0], list) else col_list
                state.mode_delivery_col_list_condition = col_list[0] if isinstance(col_list[0], list) else col_list
            print("✅ mode_of_delivery.pkl loaded")
        except FileNotFoundError:
            print("⚠️ mode_of_delivery.pkl not found, skipping...")

        # Load new structure: Antenatal with history and condition models
        print("Loading antenatal.pkl...")
        try:
            antenatal = joblib.load("save_models/antenatal.pkl")
            print(f"Keys in antenatal.pkl: {list(antenatal.keys())}")
            state.antenatal_history_model = antenatal.get("antenatal_history")
            state.antenatal_condition_model = antenatal.get("antenatal_condition")
            state.antenatal_encoder_history = antenatal.get("encoder_history")
            state.antenatal_encoder_condition = antenatal.get("encoder_condition")
            state.antenatal_dic_json_history = antenatal.get("orignal_features_history")
            state.antenatal_dic_json_condition = antenatal.get("orignal_features_condition")
            col_list = antenatal.get("col_list", [])
            if isinstance(col_list, list) and len(col_list) >= 2:
                state.antenatal_col_list_history = col_list[0] if isinstance(col_list[0], list) else col_list
                state.antenatal_col_list_condition = col_list[1] if isinstance(col_list[1], list) else col_list
            elif isinstance(col_list, list) and len(col_list) == 1:
                # If only one col_list, use it for both
                state.antenatal_col_list_history = col_list[0] if isinstance(col_list[0], list) else col_list
                state.antenatal_col_list_condition = col_list[0] if isinstance(col_list[0], list) else col_list
            print("✅ antenatal.pkl loaded")
        except FileNotFoundError:
            print("⚠️ antenatal.pkl not found, skipping...")

        print("Loading neonatal.pkl...")
        try:
            neonatal_withapgar = joblib.load("save_models/neonatal.pkl")
            print(f"Keys in neonatal.pkl: {list(neonatal_withapgar.keys())}")
            state.neonatal_model_withapgar = neonatal_withapgar.get("neonatal_model")
            state.neonatal_encoder = neonatal_withapgar.get("encoder")
            state.neonatal_dic_json = neonatal_withapgar.get("orignal_features")
            state.neonatal_col_list = neonatal_withapgar.get("col_list")
            print("✅ neonatal.pkl loaded")
        except FileNotFoundError:
            print("⚠️ neonatal.pkl not found, skipping...")

        try:
            neonatal_single = joblib.load("save_models/neonatal_single.pkl")
            state.neonatal_single_model = neonatal_single.get("neonatal_model")
        except FileNotFoundError:
            print("⚠️ neonatal_single.pkl not found, skipping...")

        try:
            neonatal_history = joblib.load("save_models/neonatalapgarhistory.pkl")
            state.neonatal_historymodel_withapgar = neonatal_history.get("neonatal_model")
            state.neonatal_encoder_history = neonatal_history.get("encoder")
            print("✅ neonatalapgarhistory.pkl loaded")
        except FileNotFoundError:
            print("⚠️ neonatalapgarhistory.pkl not found, skipping...")

        # try:
        #     state.neonatal_encoder_history = joblib.load("save_models/neonatal_encoder_history.pkl")
        # except FileNotFoundError:
        #     print("⚠️ neonatal_encoder_history.pkl not found, skipping...")

        try:
            neonatal_condition = joblib.load("save_models/neonatalapgarcondition.pkl")
            state.neonatal_conditionmodel_withapgar = neonatal_condition.get("neonatal_model")
            state.neonatal_encoder_condition = neonatal_condition.get("encoder")
            print("✅ neonatalapgarcondition.pkl loaded")
        except FileNotFoundError:
            print("⚠️ neonatalapgarcondition.pkl not found, skipping...")

        # try:
        #     state.neonatal_encoder_condition = joblib.load("save_models/neonatal_encoder_encoder.pkl")
        # except FileNotFoundError:
        #     # Try alternative name
        #     try:
        #         state.neonatal_encoder_condition = joblib.load("save_models/neonatal_encoder_condition.pkl")
        #     except FileNotFoundError:
        #         print("⚠️ neonatal encoder condition file not found, skipping...")

        print("Loading postnatal.pkl...")
        try:
            postnatal_model_file = joblib.load("save_models/postnatal.pkl")
            print(f"Keys in postnatal.pkl: {list(postnatal_model_file.keys())}")
            state.postnatal_model = postnatal_model_file.get("postnatal_model")
            state.postnatal_encoder = postnatal_model_file.get("encoder")
            state.postnatal_dic_json = postnatal_model_file.get("orignal_features")
            state.postnatal_col_list = postnatal_model_file.get("col_list")
            print("✅ postnatal.pkl loaded")
        except FileNotFoundError:
            print("⚠️ postnatal.pkl not found, skipping...")

        # Load new structure: Postnatal with history and condition models
        print("Loading postnatalhistory.pkl...")
        try:
            postnatal_history = joblib.load("save_models/postnatalhistory.pkl")
            print(f"Keys in postnatalhistory.pkl: {list(postnatal_history.keys())}")
            state.postnatal_history_model = postnatal_history.get("postnatal_model")
            state.postnatal_encoder_history = postnatal_history.get("encoder")
            state.postnatal_dic_json_history = postnatal_history.get("orignal_features")
            col_list = postnatal_history.get("col_list")
            if isinstance(col_list, list):
                state.postnatal_col_list_history = col_list[0] if isinstance(col_list[0], list) else col_list
            else:
                state.postnatal_col_list_history = col_list
            print("✅ postnatalhistory.pkl loaded")
        except FileNotFoundError:
            print("⚠️ postnatalhistory.pkl not found, skipping...")

        print("Loading postnatalcondition.pkl...")
        try:
            postnatal_condition = joblib.load("save_models/postnatalcondition.pkl")
            print(f"Keys in postnatalcondition.pkl: {list(postnatal_condition.keys())}")
            state.postnatal_condition_model = postnatal_condition.get("postnatal_model")
            state.postnatal_encoder_condition = postnatal_condition.get("encoder")
            state.postnatal_dic_json_condition = postnatal_condition.get("orignal_features")
            col_list = postnatal_condition.get("col_list")
            if isinstance(col_list, list):
                state.postnatal_col_list_condition = col_list[0] if isinstance(col_list[0], list) else col_list
            else:
                state.postnatal_col_list_condition = col_list
            print("✅ postnatalcondition.pkl loaded")
        except FileNotFoundError:
            print("⚠️ postnatalcondition.pkl not found, skipping...")

        print("✅ All models loaded successfully")
        return True

    except Exception as exc:
        print(f"Error loading models: {exc}")
        return False

