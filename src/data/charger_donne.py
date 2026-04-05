import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge des données CSV dans un DataFrame pandas.

    Arguments :

    file_path (str) : Chemin d'accès au fichier CSV.

    Retourne :

    pd.DataFrame : Jeu de données chargé.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)