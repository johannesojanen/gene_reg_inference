import pandas as pd

# load data
data = {
    'time': [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190],
    'CBF1': [0.0419,0.0365,0.0514,0.0473,0.0482,0.0546,0.0648,0.0552,0.0497,0.0352,0.0358,0.0338,0.0309,0.0232,0.0191,0.019,0.0176,0.0105,0.0081,0.0072],
    'GAL4': [0.0207,0.0122,0.0073,0.0079,0.0084,0.01,0.0096,0.0107,0.0113,0.0116,0.0073,0.0075,0.0082,0.0078,0.0089,0.0104,0.0114,0.01,0.0086,0.0078],
    'SWI5': [0.076, 0.0186, 0.009, 0.0117, 0.0088, 0.0095, 0.0075, 0.007, 0.0081, 0.0057, 0.0052, 0.0093, 0.0055, 0.006,0.0069, 0.0093, 0.009, 0.0129, 0.0022, 0.0018],
    'GAL80': [0.0225,0.0175,0.0165,0.0147,0.0145,0.0144,0.0106,0.0119,0.0104,0.0142,0.0084,0.0097,0.0088,0.0087,0.0086,0.011,0.0124,0.0093,0.0079,0.0103],
    'ASH1': [0.1033,0.0462,0.0439,0.0371,0.0475,0.0468,0.0347,0.0247,0.0269,0.019,0.0134,0.0148,0.0101,0.0088,0.008,0.009,0.0113,0.0154,0.003,0.0012]
}

time_data = pd.DataFrame(data)

# Define the true binary matrix from the original study as a DataFrame
# Columns are regulators and rows are target genes
true_structure = pd.DataFrame(
    {
        'CBF1': [0, 1, 0, 0, 0],    # CBF1 regulates these genes
        'GAL4': [0, 0, 1, 0, 0],   #
        'SWI5': [1, 0, 0, 1, 1],    #
        'GAL80': [0, 0, 0, 0, 0],   #
        'ASH1': [1, 0, 0, 0, 0],    #
    },
    index=['CBF1', 'GAL4', 'SWI5', 'GAL80', 'ASH1']  # Target Genes
)

