import pandas as pd

def select_features(score_data: pd.DataFrame, nfeatures: int, target_name: str) -> list[str]:
    """
    Select N best features based on their scores.

    Parameters
    ----------
    score_data : pd.DataFrame
        DataFrame containing the scores for each feature.
    nfeatures : int
        Number of features to select.
    target_name : str
        Name of the target contained in score_data.

    Returns
    -------
    selected_features : list[str]
        List of the selected feature names.
    """
    scores = score_data[target_name].drop(target_name, axis=0)  # select column containing scores of feature pairs that contain the target
    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature in sorted_features[:nfeatures]]

    return selected_features
