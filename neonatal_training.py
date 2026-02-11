"""
Neonatal Model Training Script

This script trains neonatal complications prediction models based on gestational age:
- Combined Trimester 1+2 (0-27 weeks): History and Condition models
- Trimester 3 (28+ weeks): History and Condition models

Total: 4 models (2 models for combined Trimester 1+2, 2 models for Trimester 3)

The script processes comma-separated values in the dataset and creates
binary features for each condition.
"""

import numpy as np
import pandas as pd
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
    Load and preprocess the dataset for neonatal training.

    Returns:
        tuple: (features_df, target_df, apgar_df, df_history, df_condition, ga_series, original_features_df)
    """
    print("Loading dataset...")
    df = pd.read_csv("./dataset/Latest data 23rd January 2026.csv")
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

    # Convert numeric columns
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

    # Extract gestational age series for trimester splitting (now in days)
    ga_series = df["Gestational Age (GA) (in weeks)"].copy()
    
    # Store original features_df with gestational age column (before dropping target)
    # This will be used to save complete CSV files with all features + target
    original_features_df = df.copy()

    # Extract only neonatal target variable
    target_column = "Neonatal__Fetal_Complications"
    target_df = df[[target_column]].copy()
    target_df = target_df.map(lambda s: s.lower() if isinstance(s, str) else s)
    df = df.drop(columns=[target_column])

    # Load history and condition features
    df_features = pd.read_csv("./dataset/Divided History Features.csv")
    history = df_features.iloc[0, :].dropna().tolist()
    condition = df_features.iloc[1, :].dropna().tolist()
    df_history = df[history]
    df_condition = df[condition]
    
    # Add Apgar scores to condition features
    condition.extend(
        ["Neonatal Apgar Score:(1 Minute)", "Neonatal Apgar Score:(5 Minute)"]
    )

    return (
        df,
        target_df,
        apgar_df,
        df_history,
        df_condition,
        ga_series,
        original_features_df,
    )


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


def get_trimester_split(ga_series):
    """
    Split data into trimesters based on gestational age.
    
    Args:
        ga_series: Series with gestational age in days (converted from weeks.days format)
        
    Returns:
        dict: Dictionary with trimester masks
    """
    # Define trimester ranges in days
    # Combined Trimester 1+2: 0-27 weeks = 0-189 days (27 * 7 = 189)
    # Trimester 3: 28+ weeks = 196+ days (28 * 7 = 196)
    trimester_1_2_mask = (ga_series >= 0) & (ga_series <= 189)
    trimester_3_mask = ga_series >= 196
    
    return {
        "trimester_1_2": trimester_1_2_mask,
        "trimester_3": trimester_3_mask,
    }


def train_neonatal_models(features_df, target_df, df_history, df_condition, apgar_df, ga_series, original_features_df):
    """
    Train neonatal prediction models based on gestational age.
    Combined Trimester 1+2 and Trimester 3 each have 2 models: history-based and condition-based.

    Args:
        features_df: Features DataFrame
        target_df: Target variables DataFrame (neonatal complications)
        df_history: History features DataFrame
        df_condition: Condition features DataFrame
        apgar_df: Apgar score features DataFrame
        ga_series: Gestational age series for splitting
        original_features_df: Original features DataFrame with gestational age column

    Returns:
        tuple: (models_list, encoder_list, feature_mapping_list, col_list, trimester_info)
    """
    print("Training neonatal models by trimester groups...")
    
    # Create a directory for trimester CSV files
    trimester_csv_dir = "Trimester_Data"
    os.makedirs(trimester_csv_dir, exist_ok=True)
    
    # Create a copy of target_df to avoid SettingWithCopyWarning
    target_df = target_df.copy()
    target_columns = target_df.columns.tolist()

    models_list = []
    encoder_list = []
    col_list = []
    feature_mapping_list = []
    trimester_info = []

    # Get trimester splits
    trimester_masks = get_trimester_split(ga_series)
    
    # Define trimester groups: Combined 1+2 and separate 3
    trimester_groups = [
        ("trimester_1_2", "Combined Trimester 1+2 (0-27 weeks)", 0, 27),
        ("trimester_3", "Trimester 3 (28+ weeks)", 28, None),
    ]

    # Loop through each trimester group
    for trimester_key, trimester_name, min_weeks, max_weeks in trimester_groups:
        mask = trimester_masks[trimester_key]
        trimester_indices = mask[mask].index
        
        print(f"\n{'='*60}")
        print(f"Training models for {trimester_name}")
        print(f"Number of samples: {len(trimester_indices)}")
        print(f"{'='*60}")
        
        if len(trimester_indices) == 0:
            print(f"⚠️  Warning: No samples found for {trimester_name}. Skipping...")
            continue
        
        # Filter data for this trimester
        trimester_history = df_history.loc[trimester_indices].reset_index(drop=True)
        trimester_condition = df_condition.loc[trimester_indices].reset_index(drop=True)
        trimester_apgar = apgar_df.loc[trimester_indices].reset_index(drop=True)
        trimester_target = target_df.loc[trimester_indices].reset_index(drop=True)
        
        # Save complete CSV file for this trimester: features + gestational age + target labels
        # Get original features with gestational age column for this trimester
        trimester_features_complete = original_features_df.loc[trimester_indices].reset_index(drop=True)
        trimester_target_for_csv = target_df.loc[trimester_indices].reset_index(drop=True)
        
        # Combine features + target labels (gestational age is already in features in its original position)
        # The original_features_df already has all columns in their original order including gestational age
        trimester_complete_data = pd.concat([trimester_features_complete, trimester_target_for_csv], axis=1)
        
        # Save CSV file for this trimester
        csv_filename = f"{trimester_key}_complete_data.csv"
        csv_path = os.path.join(trimester_csv_dir, csv_filename)
        trimester_complete_data.to_csv(csv_path, index=False)
        print(f"✅ Saved complete trimester data: {csv_filename} ({len(trimester_complete_data)} rows)")
        print(f"   Columns: {len(trimester_complete_data.columns)} (including Gestational Age in original position)")
        
        # Loop through each target column (should be just one for neonatal)
        for i, target_col in enumerate(target_columns):
            # Clean target text
            trimester_target[target_col] = (
                trimester_target[target_col].str.replace(r"\s+", " ", regex=True).str.strip()
            )

            # One-hot encode target labels
            target_encoded = trimester_target[target_col].str.get_dummies(sep=",")
            target_encoded.columns = target_encoded.columns.str.strip()
            target_encoded = (
                target_encoded.T.groupby(level=0).max().T
            )  # remove duplicates

            # Train History-based model
            print(f"\nTraining history-based model for {trimester_name}...")
            df_expanded, specific_col, specific_val = expand_comma_columns(trimester_history)
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
            original_test_rows = trimester_history.iloc[test_idx].reset_index(drop=True)
            original_test_labels = target_encoded.iloc[test_idx].reset_index(drop=True)
            merged_test_data = pd.concat([original_test_rows, original_test_labels], axis=1)
            merged_filename = f"{target_col}_{trimester_key}_history_merged_test_data.csv"
            merged_test_data.to_csv(Testing_Data_Report + "/" + merged_filename, index=False)
            print(f"✅ Saved merged test data: {merged_filename}")
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf = MultiOutputClassifier(rf)
            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"✅ {trimester_name} History Model Accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            trimester_info.append(f"{trimester_key}_history")
            
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
                + f"{trimester_key}_history_classification_report.csv"
            )

            # Train Condition-based model (with Apgar scores)
            print(f"\nTraining condition-based model for {trimester_name}...")
            df_expanded, specific_col, specific_val = expand_comma_columns(
                pd.concat([trimester_condition, trimester_apgar], axis=1)
            )
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
            original_test_rows = pd.concat([trimester_condition, trimester_apgar], axis=1).iloc[test_idx].reset_index(drop=True)
            original_test_labels = target_encoded.iloc[test_idx].reset_index(drop=True)
            merged_test_data = pd.concat([original_test_rows, original_test_labels], axis=1)
            merged_filename = f"{target_col}_{trimester_key}_condition_merged_test_data.csv"
            merged_test_data.to_csv(Testing_Data_Report + "/" + merged_filename, index=False)
            print(f"✅ Saved merged test data: {merged_filename}")
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf = MultiOutputClassifier(rf)
            clf.fit(X_train, y_train)

            # Evaluate model
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"✅ {trimester_name} Condition Model Accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            trimester_info.append(f"{trimester_key}_condition")
            
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
                + f"{trimester_key}_condition_classification_report.csv"
            )

    return models_list, encoder_list, feature_mapping_list, col_list, trimester_info


def save_neonatal_models(models_list, encoder_list, feature_mapping_list, col_list, trimester_info):
    """
    Save trained neonatal models and artifacts to save_models folder.
    Saves 4 models: Combined Trimester 1+2 (2 models) + Trimester 3 (2 models)

    Args:
        models_list: List of trained models (4 models total)
        encoder_list: List of encoders (4 encoders total)
        feature_mapping_list: List of feature mappings (4 mappings total)
        col_list: List of column labels for each model
        trimester_info: List of trimester identifiers for each model
    """
    print("\nSaving neonatal models and artifacts...")

    # Create save_models directory if it doesn't exist
    save_dir = "save_models"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created directory: {save_dir}")
    
    # Save all 4 models (Combined Trimester 1+2: 2 models, Trimester 3: 2 models)
    model_names = {
        "trimester_1_2_history": "neonatal_trimester1_2_history.pkl",
        "trimester_1_2_condition": "neonatal_trimester1_2_condition.pkl",
        "trimester_3_history": "neonatal_trimester3_history.pkl",
        "trimester_3_condition": "neonatal_trimester3_condition.pkl",
    }
    
    for idx, trimester_key in enumerate(trimester_info):
        if idx < len(models_list):
            model_data = {
                "neonatal_model": models_list[idx],
                "col_list": col_list[idx] if isinstance(col_list[idx], list) else col_list[idx],
                "encoder": encoder_list[idx],
                "orignal_features": feature_mapping_list[idx],
                "trimester": trimester_key,
            }
            
            model_filename = model_names.get(trimester_key, f"neonatal_{trimester_key}.pkl")
            model_path = os.path.join(save_dir, model_filename)
            joblib.dump(model_data, model_path)
            print(f"✅ Saved {trimester_key} model to {model_path}")
    
    # Also save a combined model file for easy access
    combined_model = {
        "models": models_list,
        "encoders": encoder_list,
        "feature_mappings": feature_mapping_list,
        "col_lists": col_list,
        "trimester_info": trimester_info,
    }
    combined_path = os.path.join(save_dir, "neonatal_all_trimesters.pkl")
    joblib.dump(combined_model, combined_path)
    print(f"✅ Saved combined model file to {combined_path}")


def main():
    """Main training pipeline for neonatal models by trimester."""
    print("=" * 50)
    print("NEONATAL PREDICTION MODELS TRAINING (BY TRIMESTER)")
    print("=" * 50)

    try:
        # Load and preprocess data
        (
            features_df,
            target_df,
            apgar_df,
            df_history,
            df_condition,
            ga_series,
            original_features_df,
        ) = load_and_preprocess_data()
        print(
            f"Dataset loaded: {features_df.shape[0]} samples, {features_df.shape[1]} features"
        )

        # Print trimester distribution
        trimester_masks = get_trimester_split(ga_series)
        print("\nTrimester Distribution:")
        print(f"  Combined Trimester 1+2 (0-189 days, 0-27 weeks): {trimester_masks['trimester_1_2'].sum()} samples")
        print(f"  Trimester 3 (196+ days, 28+ weeks): {trimester_masks['trimester_3'].sum()} samples")

        # Train neonatal models by trimester
        models_list, encoder_list, feature_mapping_list, col_list, trimester_info = train_neonatal_models(
            features_df,
            target_df,
            df_history,
            df_condition,
            apgar_df,
            ga_series,
            original_features_df,
        )

        # Save models
        save_neonatal_models(
            models_list,
            encoder_list,
            feature_mapping_list,
            col_list,
            trimester_info,
        )
        
        print("\n" + "=" * 50)
        print(f"NEONATAL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Total models trained: {len(models_list)} (Combined Trimester 1+2: 2 models, Trimester 3: 2 models)")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
