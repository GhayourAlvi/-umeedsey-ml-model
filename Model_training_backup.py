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
    df = pd.read_csv("./dataset/Final_data_29_September.csv")
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
    # Convert numeric columns
    df["Gestational Age (GA) (in weeks)"] = df[
        "Gestational Age (GA) (in weeks)"
    ].astype(float)
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
    # Step 1: Expand features
    df_expanded, specific_col, specific_val = expand_comma_columns(features_df)

    # Step 2: Create feature mapping dictionary
    feature_mapping = dict(zip(specific_col, specific_val))
    target_columns = target_df.columns.tolist()

    print("Training models...")
    models_list = []
    encoder_list = []
    col_list = []

    encoder = OneHotEncoder(handle_unknown="ignore")
    # Step 3: Encode features once for all models
    X_encoded = encoder.fit_transform(df_expanded)
    encoder_list.append(encoder)
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
            encoder = OneHotEncoder(handle_unknown="ignore")
            # Step 3: Encode features once for all models
            X_encoded = encoder.fit_transform(df_expanded)
            encoder_list.append(encoder)
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
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            print("history_results")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoded.columns.tolist(),
                output_dict=True,
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
            encoder = OneHotEncoder(handle_unknown="ignore")
            # Step 3: Encode features once for all models
            X_encoded = encoder.fit_transform(df_expanded)
            encoder_list.append(encoder)
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
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            print("condition_resutls")
            models_list.append(clf)
            col_list.append(target_encoded.columns.tolist())
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoded.columns.tolist(),
                output_dict=True,
            )
            pd.DataFrame(report_dict).transpose().to_csv(
                PRF_Report
                + "/"
                + "conditionwithapgar"
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

            if i == 2:
                df_expanded, specific_col, specific_val = expand_comma_columns(
                    df_history
                )
                encoder = OneHotEncoder(handle_unknown="ignore")
                # Step 3: Encode features once for all models
                X_encoded = encoder.fit_transform(df_expanded)
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
                    f"Classification Report:\n{classification_report(y_test, y_pred)}"
                )
                print("history_results")
                models_list.append(clf)
                col_list.append(target_encoded.columns.tolist())
                report_dict = classification_report(
                    y_test,
                    y_pred,
                    target_names=target_encoded.columns.tolist(),
                    output_dict=True,
                )
                pd.DataFrame(report_dict).transpose().to_csv(
                    PRF_Report
                    + "/"
                    + "history"
                    + str(i + 1)
                    + "_classification_report.csv"
                )
                df_expanded, specific_col, specific_val = expand_comma_columns(
                    df_condition
                )
                encoder = OneHotEncoder(handle_unknown="ignore")
                # Step 3: Encode features once for all models
                X_encoded = encoder.fit_transform(df_expanded)
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
                    f"Classification Report:\n{classification_report(y_test, y_pred)}"
                )
                print("condition_resutls")
                models_list.append(clf)
                col_list.append(target_encoded.columns.tolist())
                report_dict = classification_report(
                    y_test,
                    y_pred,
                    target_names=target_encoded.columns.tolist(),
                    output_dict=True,
                )
                pd.DataFrame(report_dict).transpose().to_csv(
                    PRF_Report
                    + "/"
                    + "condition"
                    + str(i + 1)
                    + "_classification_report.csv"
                )
            else:
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
                original_test_labels = target_encoded.iloc[test_idx].reset_index(
                    drop=True
                )

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
                    f"Classification Report:\n{classification_report(y_test, y_pred)}"
                )

                models_list.append(clf)
                col_list.append(target_encoded.columns.tolist())
                report_dict = classification_report(
                    y_test,
                    y_pred,
                    target_names=target_encoded.columns.tolist(),
                    output_dict=True,
                )
                pd.DataFrame(report_dict).transpose().to_csv(
                    PRF_Report + "/" + mode + str(i + 1) + "_classification_report.csv"
                )
                # pd.DataFrame({"LabelName": target_encoded.columns.tolist()}).to_csv(mode+str(i+1)+".csv", index=False)

    return models_list, encoder, feature_mapping, col_list


def save_models(models_list, encoder, feature_mapping, mode, col_list):
    """
    Save trained models and artifacts to save_models folder.

    Args:
        models_list: List of trained models
        encoder: Fitted OneHotEncoder
        feature_mapping: Dictionary mapping original features to expanded features
    """
    print("\nSaving models and artifacts...")

    # Create save_models directory if it doesn't exist
    save_dir = "save_models"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created directory: {save_dir}")
    if mode == "delivery_antinatal_neonatal":
        del_antnatal_neonatal = {
            "Modeofdeliverwithoutapgrar": models_list[0],
            "antnatalwithoutapgrar": models_list[1],
            "neonatalwithoutapgrarhistory": models_list[2],
            "neonatalwithoutapgrarcondition": models_list[3],
            "col_list": col_list,
            "encoder": encoder,
            "orignal_features": feature_mapping,
        }

        # Save to save_models folder
        model_path = os.path.join(save_dir, "delivery_antinatal_neonatal.pkl")
        joblib.dump(del_antnatal_neonatal, model_path)
        print(f"✅ Models and artifacts saved to {model_path}")

        # Save two models in a single combined file
        model_delivery_antinatal = {
            "mode_of_delivery_model": models_list[0],
            "antenatal_complications_model": models_list[1],
            "col_list": col_list,
            "encoder": encoder,
            "orignal_features": feature_mapping,
        }

        # Save the two-model combination
        two_models_path = os.path.join(save_dir, "delivery_antinatal.pkl")
        joblib.dump(model_delivery_antinatal, two_models_path)
        print(f"✅ Two models saved to {two_models_path}")

    elif mode == "neonatal":
        neonatal_model = {
            "neonatal_model": models_list[0],
            "col_list": col_list,
            "encoder": encoder,
            "orignal_features": feature_mapping,
        }
        neonatal_path = os.path.join(save_dir, "neonatalapgarhistory.pkl")
        joblib.dump(neonatal_model, neonatal_path)
        neonatal_model = {
            "neonatal_model": models_list[1],
            "col_list": col_list,
            "encoder": encoder,
            "orignal_features": feature_mapping,
        }
        neonatal_path = os.path.join(save_dir, "neonatalapgarcondition.pkl")
        joblib.dump(neonatal_model, neonatal_path)
        print(f"✅ Neonatal model saved to {neonatal_path}")
    elif mode == "postnatal":
        postnatal_model = {
            "postnatal_model": models_list[0],
            "col_list": col_list,
            "encoder": encoder,
            "orignal_features": feature_mapping,
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

        # # Train models
        # models_list, encoder, feature_mapping, col_list = train_models(
        #     features_df,
        #     target_df.iloc[:, 0:3],
        #     df_history,
        #     df_condition,
        #     mode="delivery_antinatal_neonatal",
        # )

        # # # Save models
        # save_models(
        #     models_list,
        #     encoder,
        #     feature_mapping,
        #     "delivery_antinatal_neonatal",
        #     col_list,
        # )

        models_list, encoder, feature_mapping, col_list = train_models(
            pd.concat([features_df, apgar_df], axis=1),
            target_df.iloc[:, 2].to_frame(),
            df_history,
            df_condition,
            mode="neonatal",
        )
        save_models(models_list, encoder, feature_mapping, "neonatal", col_list)
        models_list, encoder, feature_mapping, col_list = train_models(
            pd.concat(
                [features_df, postnatal_df, apgar_df, target_df.iloc[:, 0:3]], axis=1
            ),
            target_df.iloc[:, 3].to_frame(),
            df_history,
            df_condition,
            mode="postnatal",
        )
        save_models(models_list, encoder, feature_mapping, "postnatal", col_list)
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
