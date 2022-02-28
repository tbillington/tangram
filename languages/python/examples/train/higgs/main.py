import pandas as pd
import numpy as np

dataset_url = "https://datasets.tangram.dev/"

# Load the data.
path_train = dataset_url + 'higgs_train.csv'
path_test = dataset_url + 'higgs_test.csv'
target_column_name = "signal"
dtype = {
	'signal': bool,
	'lepton_pt': np.float64,
	'lepton_eta': np.float64,
	'lepton_phi': np.float64,
	'missing_energy_magnitude': np.float64,
	'missing_energy_phi': np.float64,
	'jet_1_pt': np.float64,
	'jet_1_eta': np.float64,
	'jet_1_phi': np.float64,
	'jet_1_b_tag': np.float64,
	'jet_2_pt': np.float64,
	'jet_2_eta': np.float64,
	'jet_2_phi': np.float64,
	'jet_2_b_tag': np.float64,
	'jet_3_pt': np.float64,
	'jet_3_eta': np.float64,
	'jet_3_phi': np.float64,
	'jet_3_b_tag': np.float64,
	'jet_4_pt': np.float64,
	'jet_4_eta': np.float64,
	'jet_4_phi': np.float64,
	'jet_4_b_tag': np.float64,
	'm_jj': np.float64,
	'm_jjj': np.float64,
	'm_lv': np.float64,
	'm_jlv': np.float64,
	'm_bb': np.float64,
	'm_wbb': np.float64,
	'm_wwbb': np.float64,
}
data_train = pd.read_csv(path_train, dtype=dtype)
data_test = pd.read_csv(path_test, dtype=dtype)

model = tangram.train(
	data_train,
	'signal',
	data_test,
	grid=[
		{
			"type": "tree",
			"binned_features_layout": "row_major",
			"learning_rate": 0.1,
			"max_rounds": 100,
			"max_leaf_nodes": 255
		}
	],
)

accuracy = model.test_metrics().default_threshold.accuracy

print("accuracy: ", accuracy)
