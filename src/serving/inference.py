"""
PIPELINE D'INFÉRENCE - Déploiement en production du modèle ML avec cohérence des caractéristiques
=========================================================================
Ce module fournit les fonctionnalités d'inférence essentielles du modèle de prédiction du taux de désabonnement des opérateurs télécoms.
Il garantit que les transformations des caractéristiques lors du déploiement correspondent exactement à celles effectuées lors de l'entraînement,
ce qui est ESSENTIEL pour la précision du modèle en production.
Responsabilités principales :
1. Charger les métadonnées du modèle et des caractéristiques enregistrées par MLflow lors de l’entraînement.
2. Appliquer les mêmes transformations de caractéristiques que celles utilisées lors de l’entraînement.
3. Garantir l’ordre correct des caractéristiques en entrée du modèle.
4. Convertir les prédictions du modèle en un format de sortie convivial.
PRINCIPE CRITIQUE : Cohérence entraînement/production
- Utilise une BINARY_MAP fixe pour un encodage binaire déterministe.
- Applique le même encodage one-hot avec drop_first=True.
- Conserve l’ordre exact des colonnes de caractéristiques utilisé lors de l’entraînement.
- Gère correctement les valeurs catégorielles manquantes ou nouvelles.
Déploiement en production :
- MODEL_DIR pointe vers les artefacts du modèle conteneurisés.
- Schéma des caractéristiques chargé à partir des artefacts d’entraînement.
- Optimisé pour l’inférence sur une seule ligne (production en temps réel).
"""

import os
import pandas as pd
import mlflow
import xgboost as xgb  # Ajouté pour le chargement direct
import joblib       # Ajouté pour charger les métadonnées
from pathlib import Path

# === CONFIGURATION DE CHARGEMENT DU MODÈLE ===
# On définit le chemin racine
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# 1. Définition des chemins prioritaires
# En Docker, le modèle sera à la racine /app/
DOCKER_MODEL = Path("/app/model.json")
# En local, on utilise le dossier artifacts à la racine du projet
LOCAL_MODEL = ARTIFACTS_DIR / "model.json"

# 2. Logique de sélection du chemin
if DOCKER_MODEL.exists():
    MODEL_PATH = str(DOCKER_MODEL)
    print(f"🐳 Mode Docker détecté. Chargement depuis : {MODEL_PATH}")
elif LOCAL_MODEL.exists():
    MODEL_PATH = str(LOCAL_MODEL)
    print(f"💻 Mode Local détecté. Chargement depuis : {MODEL_PATH}")
else:
    # Fallback ultime vers MLflow si le fichier direct n'existe pas encore
    RUN_ID = "df05d68dc6114077990d8aa17bb75f85"
    MODEL_PATH = (BASE_DIR / "mlruns" / "157377834127683807" / RUN_ID / "artifacts" / "model").as_uri()
    print(f"⚠️ Aucun fichier direct trouvé. Tentative via MLflow : {MODEL_PATH}")

# 3. Chargement effectif du modèle
try:
    if "mlruns" in str(MODEL_PATH) or "file://" in str(MODEL_PATH):
        # Chargement via MLflow (Ancienne méthode/Fallback)
        model = mlflow.pyfunc.load_model(MODEL_PATH)
    else:
        # Chargement DIRECT via XGBoost (Recommandé, évite les bugs Windows)
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
    print(f"✅ Modèle chargé avec succès !")
except Exception as e:
    raise Exception(f"❌ Erreur critique de chargement du modèle : {e}")

# === FEATURE SCHEMA LOADING ===
# On charge la liste des colonnes depuis le preprocessing.pkl (plus fiable)
try:
    if 'FEATURE_COLS' not in globals() or not FEATURE_COLS:
        # Fallback au cas où le chargement précédent via joblib aurait échoué
        import json
        feature_json_path = ARTIFACTS_DIR / "feature_columns.json"
        if feature_json_path.exists():
            with open(feature_json_path, 'r') as f:
                FEATURE_COLS = json.load(f)
        else:
            # Si vraiment rien n'est trouvé
            raise Exception("Aucun fichier de colonnes trouvé (ni pkl, ni json)")
    
    print(f"✅ Validation finale : {len(FEATURE_COLS)} colonnes prêtes pour l'inférence.")
except Exception as e:
    raise Exception(f"❌ Erreur critique : impossible de finaliser le schéma des colonnes : {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    df_enc = _serve_transform(df)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service