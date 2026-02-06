import pandas as pd
import numpy as np


def filter_large_structures(dataframe, quantile=0.95):
    """
    Filter out the extremely large protein structures (=large gyration radius).

    Gyration radius is a measure of the size of a protein structure.
    It is calculated as the square root of the sum of the squared distances of each atom from the center of mass.
    Args:
        dataframe: pandas dataframe with the protein structures
        quantile: the quantile of the gyration radius to filter the protein structures

    Returns:
        filtered_dataframe: pandas dataframe with the protein structures filtered by gyration radius
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    y_quant = pd.pivot_table(
        dataframe,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = dataframe.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])

    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = dataframe.modeled_seq_len.map(lambda x: pred_y[x - 1])
    return dataframe[dataframe.radius_gyration < row_rog_cutoffs]


def filter_with_sequence_length(dataframe, min_seq_len, max_seq_len):
    return dataframe[
        (dataframe.modeled_seq_len >= min_seq_len)
        & (dataframe.modeled_seq_len <= max_seq_len)
    ]


def filter_with_plddt(dataframe, min_plddt_percent=90):
    """
    Filter out the protein structures with plddt score below a certain threshold.

    PLDDT score is a measure of the confidence of the predicted structure.
    It is calculated as the average of the pLDDT scores of the predicted residues.

    Args:
        dataframe: pandas dataframe with the protein structures
        min_plddt_percent: the minimum percentage of confident plddt to filter the protein structures

    Returns:
        filtered_dataframe: pandas dataframe with the protein structures filtered by plddt
    """

    return dataframe[dataframe.num_confident_plddt > min_plddt_percent]


def filter_with_max_coil_percent(dataframe, max_coil_percent=90):
    """
    Filter out the protein structures with a maximum percentage of coil residues.
    Coil residues are residues that are not part of the secondary structure (alpha-helices or beta-sheets).

    Args:
        dataframe: pandas dataframe with the protein structures
        max_coil_percent: the maximum percentage of coil residues to filter the protein structures

    Returns:
        filtered_dataframe: pandas dataframe with the protein structures filtered by coil percentage
    """

    return dataframe[dataframe.coil_percent <= max_coil_percent]
