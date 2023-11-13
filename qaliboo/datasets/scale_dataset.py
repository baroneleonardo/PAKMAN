import pandas as pd
from sklearn.preprocessing import StandardScaler

original_dataset = pd.read_csv("ligen_synth_table.csv")

def scale_and_save(dataset, output_filename):
    features = dataset.iloc[:, :-2].values
    
    scaler = StandardScaler()
    
    scaled_features = scaler.fit_transform(features)
    
    scaled_dataset = pd.DataFrame(data=scaled_features, columns=dataset.columns[:-2])
    
    scaled_dataset[dataset.columns[-2:]] = dataset[dataset.columns[-2:]]
    
    scaled_dataset.to_csv(output_filename, index=False)

scale_and_save(original_dataset, "scaledligentot.csv")