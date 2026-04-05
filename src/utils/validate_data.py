import great_expectations as ge
from great_expectations.dataset.pandas_dataset import PandasDataset
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validation complète des données pour le jeu de données de désabonnement des clients Telco à l'aide de Great Expectations.
    
    Cette fonction implémente des contrôles critiques de qualité des données qui doivent être réussis avant l'entraînement du modèle.
    Elle valide l'intégrité des données, les contraintes de logique métier et les propriétés statistiques
    que le modèle d'apprentissage automatique attend.
    
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = PandasDataset(df)
    
    # === VALIDATION DU SCHÉMA - COLONNES ESSENTIELLES ===
    print("   📋 Validating schema and required columns...")
    
    # L'identifiant du client doit exister (requis pour les opérations commerciales)  
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    
    # Caractéristiques démographiques principales
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    
    # Caractéristiques de service (critiques pour l'analyse du désabonnement)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    # Caractéristiques financières (principaux prédicteurs de désabonnement)
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    
    # === VALIDATION DE LA LOGIQUE MÉTIER ===
    print("   💼 Validating business logic constraints...")
    
    # Le genre doit être parmi les valeurs attendues (intégrité des données)
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    
    # Les champs Yes/No doivent avoir des valeurs valides
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    # Les types de contrat doivent être valides (contrainte commerciale)
    ge_df.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )
    
    # Les types de services Internet (contrainte commerciale)
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )
    
    # === VALIDATION DES PLAGES NUMÉRIQUES ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Tenure must be non-negative (business logic - can't have negative tenure)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
    # Monthly charges must be positive (business logic - no free service)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
    # Total charges should be non-negative (business logic)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
    # === VALIDATION STATISTIQUE ===
    print("   📈 Validating statistical properties...")
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    # Les frais mensuels doivent être dans une plage commerciale raisonnable
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    
    # Aucune valeur manquante dans les caractéristiques numériques critiques  
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
    # === VÉRIFICATIONS DE COHÉRENCE DES DONNÉES ===
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exceptions for edge cases
    )
    
    # === EXECUTION DE LA SUITE DE VALIDATION ===
    print("   ⚙️  Running complete validation suite...")
    results = ge_df.validate()
    
    # === TRAITEMENT DES RÉSULTATS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # Afficher le résumé de la validation
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations