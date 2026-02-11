"""
Model Training Script for Medical Prediction Models

This script trains three separate models:
1. Mode of Delivery prediction
2. Antenatal complications prediction
3. Neonatal complications prediction

The script processes comma-separated values in the dataset and creates
binary features for each condition.
"""

import numpy as np
import pandas as pd
import json
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import MultiOutputClassifier


PRF_Report = "Classfication_report"
modelTarget_Labels = "ModelTarget_Labels"
Testing_Data_Report = "Testing_Data_Report"
os.makedirs(PRF_Report, exist_ok=True)
os.makedirs(modelTarget_Labels, exist_ok=True)
os.makedirs(Testing_Data_Report, exist_ok=True)


def clean_value(value: str) -> str:
    """
    Clean string values by removing extra quotes, brackets, and spaces.

    Args:
        value: String value to clean

    Returns:
        Cleaned lowercase string
    """
    return value.strip().strip("'").strip('"').strip("]").strip("[").lower()


def convert_ga_to_days(ga_value):
    """
    Convert gestational age from "weeks.days" format to total days.
    Example: 23.5 (23 weeks and 5 days) -> 23 * 7 + 5 = 166 days
    
    Args:
        ga_value: Gestational age value (can be float, int, or string)
        
    Returns:
        Total days as float
    """
    if pd.isna(ga_value):
        return None
    
    try:
        # Convert to string first to handle decimal properly
        ga_str = str(ga_value)
        
        # Split by decimal point
        if '.' in ga_str:
            parts = ga_str.split('.')
            weeks = int(float(parts[0]))  # Before decimal point
            days = int(float('0.' + parts[1]) * 10) if len(parts) > 1 else 0  # After decimal point
            total_days = weeks * 7 + days
        else:
            # If no decimal point, treat as whole weeks
            weeks = int(float(ga_str))
            total_days = weeks * 7
        
        return float(total_days)
    except (ValueError, AttributeError):
        # If conversion fails, try direct float conversion
        try:
            return float(ga_value) * 7  # Assume it's already in weeks
        except:
            return None


def expand_comma_columns(df):
    """
    Expand comma-separated values into binary features.

    Args:
        df: DataFrame with comma-separated values

    Returns:
        tuple: (expanded_df, specific_columns, specific_values)
    """
    new_df = df.copy()
    specific_col = []
    specific_val = []

    for col in df.columns:
        # Skip non-string columns
        if not pd.api.types.is_string_dtype(df[col]):
            continue

        # Check if column contains comma-separated values
        if df[col].dropna().astype(str).str.contains(",").any():
            specific_col.append(col)

            # Split and clean values
            new_df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .apply(
                    lambda x: (
                        [
                            clean_value(v)
                            for v in x.split(",")
                            if clean_value(v) and len(clean_value(v)) > 1
                        ]
                        if x
                        else []
                    )
                )
            )

            # Expand with MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            expanded = pd.DataFrame(
                mlb.fit_transform(new_df[col]),
                columns=[f"{col}:{cls}" for cls in mlb.classes_],
                index=df.index,
            )
            specific_val.append([f"{cls}" for cls in mlb.classes_])

            # Convert 0/1 to Yes/No
            expanded = expanded.replace({0: "no", 1: "yes"})

            # Drop original column and join expanded
            new_df = new_df.drop(columns=[col]).join(expanded)
            new_df = new_df.map(lambda x: x.lower() if isinstance(x, str) else x)

    return new_df, specific_col, specific_val


def load_and_preprocess_data():
    """
    Load and preprocess the dataset.

    Returns:
        tuple: (features_df, target_df, apgar_df, original_df)
    """
    print("Loading dataset...")
    df = pd.read_csv("./dataset/latest data 3 feb.csv")
    original_df = df.copy()

    # Fill missing values in examination columns
    df["Per speculum examination (If Yes, specify findings)"] = df[
        "Per speculum examination (If Yes, specify findings)"
    ].fillna("no")
    df["Per vaginal examination (If Yes, specify findings)"] = df[
        "Per vaginal examination (If Yes, specify findings)"
    ].fillna("no")

    # Clean string values and remove rows with missing data
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna()

    # Extract Apgar score features
    apgar_features = [
        "Neonatal Apgar Score:(1 Minute)",
        "Neonatal Apgar Score:(5 Minute)",
    ]
    apgar_df = df[apgar_features].copy()
    df = df.drop(columns=apgar_features)

    # Extract postnatal symptoms
    postnatal_symptoms = ["Postnatal_Symptoms", "Postnatal_Examination"]
    postnatal_df = df[postnatal_symptoms].copy()
    df = df.drop(columns=postnatal_symptoms)
    postnatal_df = postnatal_df.map(lambda x: x.strip() if isinstance(x, str) else x)
    postnatal_df = postnatal_df.dropna()

    # Remove rows where Mode_of_delivery2 is not in ['Single', 'Multiple']
    # Convert gestational age from "weeks.days" format to total days
    # Example: 23.5 (23 weeks and 5 days) -> 23 * 7 + 5 = 166 days
    df["Gestational Age (GA) (in weeks)"] = df[
        "Gestational Age (GA) (in weeks)"
    ].apply(convert_ga_to_days)
    
    df["Systolic B.P"] = df["Systolic B.P"].astype(float)
    df["Temp. ( armpit for 2 ) in *F"] = df["Temp. ( armpit for 2 ) in *F"].astype(
        float
    )
    apgar_df["Neonatal Apgar Score:(5 Minute)"] = apgar_df[
        "Neonatal Apgar Score:(5 Minute)"
    ].astype(float)

    # Extract target variables
    target_columns = [
        "Mode_of_delivery2",
        "Antenatal_Peripartum_Maternal_Complications",
        "Neonatal__Fetal_Complications",
        "Postnatal_Maternal_Complications",
    ]

    target_df = df[target_columns].copy()
    target_df = target_df.map(lambda s: s.lower() if isinstance(s, str) else s)
    df = df.drop(columns=target_columns)

    df_features = pd.read_csv("./dataset/Divided History Features.csv")
    history = df_features.iloc[0, :].dropna().tolist()
    condition = df_features.iloc[1, :].dropna().tolist()
    df_history = df[history]
    df_condition = df[condition]
    condition.extend(
        ["Neonatal Apgar Score:(1 Minute)", "Neonatal Apgar Score:(5 Minute)"]
    )

    return (
        df,
        target_df,
        apgar_df,
        original_df,
        postnatal_df,
        df_history,
        df_condition,
    )


def train_models(features_df, target_df, df_history, df_condition, mode):
    """
    Train the three prediction models.

    Args:
        features_df: Features DataFrame
        target_df: Target variables DataFrame

    Returns:
        tuple: (models_list, encoder, feature_mapping)
    """
    print("Expanding comma-separated features...")
    # print(features_df.columns.tolist())
    # Create a copy of target_df to avoid SettingWithCopyWarning
    target_df = target_df.copy()

    # Step 1: Expand features
    df_expanded, specific_col, specific_val = expand_comma_columns(features_df)

    # Step 2: Create feature mapping dictionary
    feature_mapping = dict(zip(specific_col, specific_val))
    target_columns = target_df.columns.tolist()

    print("Training models...")
    models_list = []
    encoder_list = []
    col_list = []
    feature_mapping_list = []

    # Initialize encoder and X_encoded for backward compatibility (used in else branch)
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_encoded = encoder.fit_transform(df_expanded)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf = MultiOutputClassifier(rf)

    # Step 4: Loop through each target column
    if mode == "neonatal":
        for i, target_col in enumerate(target_columns):
            # Clean target text
            target_df[target_col] = (
                target_df[target_col].str.replace(r"\s+", " ", regex=True).str.strip()
            )

            # One-hot encode target labels
            target_encoded = target_df[target_col].str.get_dummies(sep=",")
            target_encoded.columns = target_encoded.columns.str.strip()
            target_encoded = (
                target_encoded.T.groupby(level=0).max().T
            )  # remove duplicates
            df_expanded, specific_col, specific_val = expand_comma_columns(df_history)
            feature_mapping_history = dict(zip(specific_col, specific_val))
            encoder = OneHotEncoder(handle_unknown="ignore")
            # Step 3: Encode features once for all models
            X_encoded = encoder.fit_transform(df_expanded)
            encoder_list.append(encoder)
            feature_mapping_list.append(feature_mapping_history)
            # Step 5: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, target_encoded, test_size=0.2, random_state=42
            )
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf = MultiOutputClassifier(rf)
            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"✅ Accuracy: {accuracy:.4f}")
            print(
                f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
            )
            print("history_results")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoded.columns.tolist(),
                output_dict=True,
                zero_division=0,
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                PRF_Report
                + "/"
                + "historywithapgar"
                + str(i + 1)
                + "_classification_report.csv"
            )
            df_expanded, specific_col, specific_val = expand_comma_columns(
                pd.concat([df_condition, features_df.iloc[:, -2:]], axis=1)
            )
            feature_mapping_condition = dict(zip(specific_col, specific_val))
            encoder = OneHotEncoder(handle_unknown="ignore")
            # Step 3: Encode features once for all models
            X_encoded = encoder.fit_transform(df_expanded)
            encoder_list.append(encoder)
            feature_mapping_list.append(feature_mapping_condition)
            # Step 5: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, target_encoded, test_size=0.2, random_state=42
            )
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf = MultiOutputClassifier(rf)
            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"✅ Accuracy: {accuracy:.4f}")
            print(
                f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
            )
            print("condition_resutls")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoded.columns.tolist(),
                output_dict=True,
                zero_division=0,
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                PRF_Report
                + "/"
                + "conditionwithapgar"
                + str(i + 1)
                + "_classification_report.csv"
            )
    elif mode == "delivery_antinatal_neonatal" or mode == "postnatal":
        # Train 2 models (history and condition) for each target column
        for i, target_col in enumerate(target_columns):
            print(f"\nTraining model {i+1}/{len(target_columns)}: {target_col}")
            # Clean target text
            target_df[target_col] = (
                target_df[target_col].str.replace(r"\s+", " ", regex=True).str.strip()
            )

            # One-hot encode target labels
            target_encoded = target_df[target_col].str.get_dummies(sep=",")
            target_encoded.columns = target_encoded.columns.str.strip()
            target_encoded = (
                target_encoded.T.groupby(level=0).max().T
            )  # remove duplicates

            # Features to exclude from mode of delivery model (first model only)
            features_to_exclude = [
                "PatientAge",
                "PatientHasPosGp",
                "PartenerHasPosGp",
                "First degree relative with following medical conditions",
                "Mental Health Symptoms",
                "Previous history of PPH (Postpartum Hemorrhage(PPH)) Exploration",
                "Any Anesthesia Complication previously",
                "Pallor",
            ]

            # For delivery_antinatal_neonatal:
            # - target 1 (Mode_of_delivery2): exclude the 8 features from history/condition (already handled below)
            # - target 2 & 3 (Antenatal & Neonatal): keep all features, but drop specific labels from targets
            if mode == "delivery_antinatal_neonatal":
                # Exclude selected labels from antenatal & neonatal target columns
                labels_to_exclude = [
                    "depression",
                    "postpartum hemorrhage(pph)",
                    "anesthesia complication",
                    "anemia",
                    ''
                ]
                if target_col in [
                    "Antenatal_Peripartum_Maternal_Complications"    
                ]:
                    # Drop these label-columns so model is not trained on them
                    drop_cols = [
                        c for c in target_encoded.columns if c in labels_to_exclude
                    ]
                    if drop_cols:
                        target_encoded = target_encoded.drop(columns=drop_cols)
            
            # For postnatal mode: exclude specific labels from postnatal target column
            if mode == "postnatal" or (mode == "delivery_antinatal_neonatal" and target_col == "Postnatal_Maternal_Complications"):
                # Exclude selected labels from postnatal target column
                postnatal_labels_to_exclude = [
                    "depression",
                    "postpartum hemorrhage(pph)",
                    "anemia",
                    ''
                ]
                # Drop these label-columns so model is not trained on them
                drop_cols = [
                    c for c in target_encoded.columns if c in postnatal_labels_to_exclude
                ]
                if drop_cols:
                    target_encoded = target_encoded.drop(columns=drop_cols)

            # Filter features only for Mode_of_delivery2 (first target column)
            if mode == "delivery_antinatal_neonatal" and target_col == "Mode_of_delivery2":
                # Exclude features from history and condition DataFrames
                history_filtered = df_history.drop(
                    columns=[f for f in features_to_exclude if f in df_history.columns]
                )
                condition_filtered = df_condition.drop(
                    columns=[f for f in features_to_exclude if f in df_condition.columns]
                )
            else:
                # Use original DataFrames for other models
                history_filtered = df_history
                condition_filtered = df_condition

            # Train History-based model
            print(f"Training history-based model for {target_col}...")
            df_expanded, specific_col, specific_val = expand_comma_columns(history_filtered)
            feature_mapping_history = dict(zip(specific_col, specific_val))
            encoder = OneHotEncoder(handle_unknown="ignore")
            X_encoded = encoder.fit_transform(df_expanded)
            encoder_list.append(encoder)
            feature_mapping_list.append(feature_mapping_history)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, target_encoded, test_size=0.2, random_state=42
            )

            # Recreate test indices for saving test data
            all_indices = np.arange(len(df_expanded))
            train_idx, test_idx = train_test_split(
                all_indices, test_size=0.2, random_state=42
            )

            # Get original test rows + labels for history model
            original_test_rows = history_filtered.iloc[test_idx].reset_index(drop=True)
            original_test_labels = target_encoded.iloc[test_idx].reset_index(drop=True)
            merged_test_data = pd.concat(
                [original_test_rows, original_test_labels], axis=1
            )
            merged_filename = f"{target_col}_history_merged_test_data.csv"
            merged_test_data.to_csv(
                Testing_Data_Report + "/" + merged_filename, index=False
            )
            print(f"✅ Saved merged test data: {merged_filename}")

            # Get original train rows + labels for history model
            original_train_rows = history_filtered.iloc[train_idx].reset_index(drop=True)
            original_train_labels = target_encoded.iloc[train_idx].reset_index(
                drop=True
            )
            merged_train_data = pd.concat(
                [original_train_rows, original_train_labels], axis=1
            )
            merged_filename_train = f"{target_col}_history_merged_train_data.csv"
            merged_train_data.to_csv(
                Testing_Data_Report + "/" + merged_filename_train, index=False
            )
            print(f"✅ Saved merged train data: {merged_filename_train}")

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf = MultiOutputClassifier(rf)
            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"✅ History Model Accuracy: {accuracy:.4f}")
            print(
                f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
            )
            print("history_results")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoded.columns.tolist(),
                output_dict=True,
                zero_division=0,
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                PRF_Report
                + "/"
                + mode
                + "_history_"
                + str(i + 1)
                + "_classification_report.csv"
            )

            # Train Condition-based model
            print(f"Training condition-based model for {target_col}...")
            df_expanded, specific_col, specific_val = expand_comma_columns(condition_filtered)
            feature_mapping_condition = dict(zip(specific_col, specific_val))
            encoder = OneHotEncoder(handle_unknown="ignore")
            X_encoded = encoder.fit_transform(df_expanded)
            encoder_list.append(encoder)
            feature_mapping_list.append(feature_mapping_condition)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, target_encoded, test_size=0.2, random_state=42
            )

            # Recreate test indices for saving test data
            all_indices = np.arange(len(df_expanded))
            train_idx, test_idx = train_test_split(
                all_indices, test_size=0.2, random_state=42
            )

            # Get original test rows + labels for condition model
            original_test_rows = condition_filtered.iloc[test_idx].reset_index(drop=True)
            original_test_labels = target_encoded.iloc[test_idx].reset_index(drop=True)
            merged_test_data = pd.concat(
                [original_test_rows, original_test_labels], axis=1
            )
            merged_filename = f"{target_col}_condition_merged_test_data.csv"
            merged_test_data.to_csv(
                Testing_Data_Report + "/" + merged_filename, index=False
            )
            print(f"✅ Saved merged test data: {merged_filename}")

            # Get original train rows + labels for condition model
            original_train_rows = condition_filtered.iloc[train_idx].reset_index(drop=True)
            original_train_labels = target_encoded.iloc[train_idx].reset_index(
                drop=True
            )
            merged_train_data = pd.concat(
                [original_train_rows, original_train_labels], axis=1
            )
            merged_filename_train = f"{target_col}_condition_merged_train_data.csv"
            merged_train_data.to_csv(
                Testing_Data_Report + "/" + merged_filename_train, index=False
            )
            print(f"✅ Saved merged train data: {merged_filename_train}")

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf = MultiOutputClassifier(rf)
            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"✅ Condition Model Accuracy: {accuracy:.4f}")
            print(
                f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
            )
            print("condition_results")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoded.columns.tolist(),
                output_dict=True,
                zero_division=0,
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                PRF_Report
                + "/"
                + mode
                + "_condition_"
                + str(i + 1)
                + "_classification_report.csv"
            )
    else:
        for i, target_col in enumerate(target_columns):
            # Clean target text
            target_df[target_col] = (
                target_df[target_col].str.replace(r"\s+", " ", regex=True).str.strip()
            )

            # One-hot encode target labels
            target_encoded = target_df[target_col].str.get_dummies(sep=",")
            target_encoded.columns = target_encoded.columns.str.strip()
            target_encoded = (
                target_encoded.T.groupby(level=0).max().T
            )  # remove duplicates

            print(i)
            print(f"\nTraining model {i+1}/{len(target_columns)}: {target_col}")

            # Step 5: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, target_encoded, test_size=0.2, random_state=42
            )

            # 🔹 Step 6: Recreate test indices (to match df_expanded)
            all_indices = np.arange(len(df_expanded))
            train_idx, test_idx = train_test_split(
                all_indices, test_size=0.2, random_state=42
            )

            # 🔹 Step 7: Get original test rows + labels
            original_test_rows = features_df.iloc[test_idx].reset_index(drop=True)
            original_test_labels = target_encoded.iloc[test_idx].reset_index(drop=True)

            # 🔹 Step 8: Merge horizontally
            merged_test_data = pd.concat(
                [original_test_rows, original_test_labels], axis=1
            )

            # 🔹 Step 9: Save merged test data
            merged_filename = f"{target_col}_merged_test_data.csv"
            merged_test_data.to_csv(
                Testing_Data_Report + "/" + merged_filename, index=False
            )
            print(f"✅ Saved merged test data: {merged_filename}")

            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"✅ Accuracy: {accuracy:.4f}")
            print(
                f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
            )

            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoded.columns.tolist(),
                output_dict=True,
                zero_division=0,
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                PRF_Report + "/" + mode + str(i + 1) + "_classification_report.csv"
            )
            # pd.DataFrame({"LabelName": target_encoded.columns.tolist()}).to_csv(mode+str(i+1)+".csv", index=False)

    # Return encoder_list and feature_mapping_list if they were populated, otherwise return single encoder and feature_mapping
    if encoder_list and feature_mapping_list:
        return models_list, encoder_list, feature_mapping_list, col_list
    else:
        return models_list, encoder, feature_mapping, col_list


def save_models(models_list, encoder_or_list, feature_mapping_or_list, mode, col_list):
    """
    Save trained models and artifacts to save_models folder.

    Args:
        models_list: List of trained models
        encoder_or_list: Either a single encoder or list of encoders
        feature_mapping_or_list: Either a single feature_mapping or list of feature_mappings
        mode: Training mode
        col_list: List of column labels for each model
    """
    print("\nSaving models and artifacts...")

    # Create save_models directory if it doesn't exist
    save_dir = "save_models"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created directory: {save_dir}")

    if mode == "delivery_antinatal_neonatal":
        # Check if we have lists (new structure) or single values (old structure)
        if isinstance(encoder_or_list, list) and isinstance(
            feature_mapping_or_list, list
        ):
            # New structure: 6 models (2 for each target: history + condition)
            # models_list[0] = mode_of_delivery_history
            # models_list[1] = mode_of_delivery_condition
            # models_list[2] = antenatal_history
            # models_list[3] = antenatal_condition
            # models_list[4] = neonatal_history
            # models_list[5] = neonatal_condition

            del_antnatal_neonatal = {
                "Modeofdeliverwithoutapgrarhistory": models_list[0],
                "Modeofdeliverwithoutapgrarcondition": models_list[1],
                "antnatalwithoutapgrarhistory": models_list[2],
                "antnatalwithoutapgrarcondition": models_list[3],
                "neonatalwithoutapgrarhistory": models_list[4],
                "neonatalwithoutapgrarcondition": models_list[5],
                "col_list": col_list,
                "encoder_list": encoder_or_list,
                "orignal_features_list": feature_mapping_or_list,
            }

            # Save to save_models folder
            model_path = os.path.join(save_dir, "delivery_antinatal_neonatal.pkl")
            joblib.dump(del_antnatal_neonatal, model_path)
            print(f"✅ Models and artifacts saved to {model_path}")

            # Save mode_of_delivery models (history + condition)
            model_delivery = {
                "mode_of_delivery_history": models_list[0],
                "mode_of_delivery_condition": models_list[1],
                "col_list": col_list[0:2] if len(col_list) >= 2 else col_list,
                "encoder_history": encoder_or_list[0],
                "encoder_condition": encoder_or_list[1],
                "orignal_features_history": feature_mapping_or_list[0],
                "orignal_features_condition": feature_mapping_or_list[1],
            }
            delivery_path = os.path.join(save_dir, "mode_of_delivery.pkl")
            joblib.dump(model_delivery, delivery_path)
            print(f"✅ Mode of delivery models saved to {delivery_path}")

            # Save antenatal models (history + condition)
            model_antenatal = {
                "antenatal_history": models_list[2],
                "antenatal_condition": models_list[3],
                "col_list": col_list[2:4] if len(col_list) >= 4 else col_list,
                "encoder_history": encoder_or_list[2],
                "encoder_condition": encoder_or_list[3],
                "orignal_features_history": feature_mapping_or_list[2],
                "orignal_features_condition": feature_mapping_or_list[3],
            }
            antenatal_path = os.path.join(save_dir, "antenatal.pkl")
            joblib.dump(model_antenatal, antenatal_path)
            print(f"✅ Antenatal models saved to {antenatal_path}")

            # Save neonatal models (history + condition)
            model_neonatal = {
                "neonatal_history": models_list[4],
                "neonatal_condition": models_list[5],
                "col_list": col_list[4:6] if len(col_list) >= 6 else col_list,
                "encoder_history": encoder_or_list[4],
                "encoder_condition": encoder_or_list[5],
                "orignal_features_history": feature_mapping_or_list[4],
                "orignal_features_condition": feature_mapping_or_list[5],
            }
            neonatal_path = os.path.join(save_dir, "neonatal.pkl")
            joblib.dump(model_neonatal, neonatal_path)
            print(f"✅ Neonatal models saved to {neonatal_path}")

        else:
            # Old structure (backward compatibility)
            del_antnatal_neonatal = {
                "Modeofdeliverwithoutapgrar": models_list[0],
                "antnatalwithoutapgrar": models_list[1],
                "neonatalwithoutapgrarhistory": models_list[2],
                "neonatalwithoutapgrarcondition": models_list[3],
                "col_list": col_list,
                "encoder": encoder_or_list,
                "orignal_features": feature_mapping_or_list,
            }
            model_path = os.path.join(save_dir, "delivery_antinatal_neonatal.pkl")
            joblib.dump(del_antnatal_neonatal, model_path)
            print(f"✅ Models and artifacts saved to {model_path}")

    elif mode == "neonatal":
        if isinstance(encoder_or_list, list) and isinstance(
            feature_mapping_or_list, list
        ):
            # New structure: 2 models (history + condition)
            neonatal_model_history = {
                "neonatal_model": models_list[0],
                "col_list": col_list[0] if isinstance(col_list[0], list) else col_list,
                "encoder": encoder_or_list[0],
                "orignal_features": feature_mapping_or_list[0],
            }
            neonatal_path_history = os.path.join(save_dir, "neonatalapgarhistory.pkl")
            joblib.dump(neonatal_model_history, neonatal_path_history)
            print(f"✅ Neonatal history model saved to {neonatal_path_history}")

            neonatal_model_condition = {
                "neonatal_model": models_list[1],
                "col_list": col_list[1] if isinstance(col_list[1], list) else col_list,
                "encoder": encoder_or_list[1],
                "orignal_features": feature_mapping_or_list[1],
            }
            neonatal_path_condition = os.path.join(
                save_dir, "neonatalapgarcondition.pkl"
            )
            joblib.dump(neonatal_model_condition, neonatal_path_condition)
            print(f"✅ Neonatal condition model saved to {neonatal_path_condition}")
        else:
            # Old structure (backward compatibility)
            neonatal_model = {
                "neonatal_model": models_list[0],
                "col_list": col_list,
                "encoder": encoder_or_list,
                "orignal_features": feature_mapping_or_list,
            }
            neonatal_path = os.path.join(save_dir, "neonatalapgarhistory.pkl")
            joblib.dump(neonatal_model, neonatal_path)
            neonatal_model = {
                "neonatal_model": models_list[1],
                "col_list": col_list,
                "encoder": encoder_or_list,
                "orignal_features": feature_mapping_or_list,
            }
            neonatal_path = os.path.join(save_dir, "neonatalapgarcondition.pkl")
            joblib.dump(neonatal_model, neonatal_path)
            print(f"✅ Neonatal model saved to {neonatal_path}")

    elif mode == "postnatal":
        if isinstance(encoder_or_list, list) and isinstance(
            feature_mapping_or_list, list
        ):
            # New structure: 2 models (history + condition)
            postnatal_model_history = {
                "postnatal_model": models_list[0],
                "col_list": col_list[0] if isinstance(col_list[0], list) else col_list,
                "encoder": encoder_or_list[0],
                "orignal_features": feature_mapping_or_list[0],
            }
            postnatal_path_history = os.path.join(save_dir, "postnatalhistory.pkl")
            joblib.dump(postnatal_model_history, postnatal_path_history)
            print(f"✅ Postnatal history model saved to {postnatal_path_history}")

            postnatal_model_condition = {
                "postnatal_model": models_list[1],
                "col_list": col_list[1] if isinstance(col_list[1], list) else col_list,
                "encoder": encoder_or_list[1],
                "orignal_features": feature_mapping_or_list[1],
            }
            postnatal_path_condition = os.path.join(save_dir, "postnatalcondition.pkl")
            joblib.dump(postnatal_model_condition, postnatal_path_condition)
            print(f"✅ Postnatal condition model saved to {postnatal_path_condition}")
        else:
            # Old structure (backward compatibility)
            postnatal_model = {
                "postnatal_model": models_list[0],
                "col_list": col_list,
                "encoder": encoder_or_list,
                "orignal_features": feature_mapping_or_list,
            }
            postnatal_path = os.path.join(save_dir, "postnatal.pkl")
            joblib.dump(postnatal_model, postnatal_path)
            print(f"✅ Postnatal model saved to {postnatal_path}")


def main():
    """Main training pipeline."""
    print("=" * 50)
    print("MEDICAL PREDICTION MODELS TRAINING")
    print("=" * 50)

    try:
        # Load and preprocess data
        (
            features_df,
            target_df,
            apgar_df,
            original_df,
            postnatal_df,
            df_history,
            df_condition,
        ) = load_and_preprocess_data()
        print(
            f"Dataset loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features"
        )

        # Train models
        models_list, encoder, feature_mapping, col_list = train_models(
            features_df,
            target_df.iloc[:, 0:3],
            df_history,
            df_condition,
            mode="delivery_antinatal_neonatal",
        )

        # # Save models
        save_models(
            models_list,
            encoder,
            feature_mapping,
            "delivery_antinatal_neonatal",
            col_list,
        )

        models_list, encoder, feature_mapping, col_list = train_models(
            pd.concat([features_df, apgar_df], axis=1),
            target_df.iloc[:, 2].to_frame(),
            df_history,
            df_condition,
            mode="neonatal",
        )
        save_models(models_list, encoder, feature_mapping, "neonatal", col_list)
        # Train single postnatal model (not history/condition split, not trimester-wise)
        print("\n" + "=" * 50)
        print("TRAINING POSTNATAL MODEL (SINGLE MODEL)")
        print("=" * 50)

        postnatal_target = target_df.iloc[:, 3].to_frame()
        postnatal_features = pd.concat([features_df, postnatal_df, apgar_df], axis=1)

        # Clean target text
        postnatal_target_col = postnatal_target.columns[0]
        postnatal_target[postnatal_target_col] = (
            postnatal_target[postnatal_target_col]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # One-hot encode target labels
        target_encoded = postnatal_target[postnatal_target_col].str.get_dummies(sep=",")
        target_encoded.columns = target_encoded.columns.str.strip()
        target_encoded = target_encoded.T.groupby(level=0).max().T  # remove duplicates

        # Exclude selected labels from postnatal target column
        postnatal_labels_to_exclude = [
            "depression",
            "postpartum hemorrhage(pph)",
            "anemia",
            ''
        ]
        # Drop these label-columns so model is not trained on them
        drop_cols = [
            c for c in target_encoded.columns if c in postnatal_labels_to_exclude
        ]
        if drop_cols:
            target_encoded = target_encoded.drop(columns=drop_cols)
            print(f"✅ Excluded postnatal labels from training: {drop_cols}")

        # Expand features
        df_expanded, specific_col, specific_val = expand_comma_columns(
            postnatal_features
        )
        feature_mapping = dict(zip(specific_col, specific_val))

        # Encode features
        encoder = OneHotEncoder(handle_unknown="ignore")
        X_encoded = encoder.fit_transform(df_expanded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, target_encoded, test_size=0.2, random_state=42
        )

        # Recreate test indices for saving test data
        all_indices = np.arange(len(df_expanded))
        train_idx, test_idx = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )

        # Get original test rows + labels
        original_test_rows = postnatal_features.iloc[test_idx].reset_index(drop=True)
        original_test_labels = target_encoded.iloc[test_idx].reset_index(drop=True)
        merged_test_data = pd.concat([original_test_rows, original_test_labels], axis=1)
        merged_filename = f"{postnatal_target_col}_merged_test_data.csv"
        merged_test_data.to_csv(
            Testing_Data_Report + "/" + merged_filename, index=False
        )
        print(f"✅ Saved merged test data: {merged_filename}")

        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf = MultiOutputClassifier(rf)
        clf.fit(X_train, y_train)

        # Evaluate model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ Postnatal Model Accuracy: {accuracy:.4f}")
        print(
            f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
        )

        # Save classification report
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=target_encoded.columns.tolist(),
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report_dict).transpose().to_csv(
            PRF_Report + "/postnatal_single_classification_report.csv"
        )

        # Save model
        postnatal_model = {
            "postnatal_model": clf,
            "col_list": target_encoded.columns.tolist(),
            "encoder": encoder,
            "orignal_features": feature_mapping,
        }
        postnatal_path = os.path.join("save_models", "postnatal_single.pkl")
        joblib.dump(postnatal_model, postnatal_path)
        print(f"✅ Postnatal single model saved to {postnatal_path}")
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
