import pandas as pd
import matplotlib.pyplot as plt

# --- Load CSV ---
csv_file = "accuracies.csv"  # replace with your CSV path
df = pd.read_csv(csv_file)

model_colours = {
    "DEC": "red",
    "DEC w/o backprop": "yellow",
    "KMeans": "lime",
    "Spectral Clustering": "purple",
}

# --- Loop through datasets ---
for dataset_name, dataset_df in df.groupby('dataset'):
    plt.figure(figsize=(8,5))
    
    # Loop through models in this dataset
    for model_name, model_df in dataset_df.groupby('model'):
        # Sort by hyperparameter
        model_df_sorted = model_df.sort_values('hyperparameter')
        plt.plot(
            model_df_sorted['hyperparameter'],
            model_df_sorted['test_acc'],
            color=model_colours.get(model_name, "black"),
            marker=None,
            label=model_name,
            linewidth=3,
        )
    

    plt.title(f"{dataset_name}")
    plt.xlabel("Parameter Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()