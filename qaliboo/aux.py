import pandas as pd
import json
from qaliboo import machine_learning_models

def create_csv_init(file1, result_folder):
    dataset_csv_path = '/home/lbarone/QALIBOO/qaliboo/datasets/ligen_synth_table.csv'
    df = pd.read_csv(dataset_csv_path)

    with open(file1) as f:
        json_data = json.load(f)
    
    dat_indices = [entry['dat_index'] for entry in json_data]
    new_df = pd.DataFrame({'dat_index': dat_indices})
    new_df = df.iloc[dat_indices, :]
    
    output_csv_path = f'{result_folder}/init.csv'
    new_df.to_csv(output_csv_path, index=False)

def create_csv_history(file_path, result_folder):
    dataset_csv_path = '/home/lbarone/QALIBOO/qaliboo/datasets/ligen_synth_table.csv'
    df = pd.read_csv(dataset_csv_path)

    with open(file_path) as f:
        json_data = json.load(f)
    
    iter_values = [entry['iter'] for entry in json_data]
    dat_indices = [entry['dat_index'] for entry in json_data]

    selected_rows_df = df.loc[dat_indices, :]
    selected_rows_df.insert(0, 'index', iter_values)
    output_csv_path = f'{result_folder}/history.csv'
    selected_rows_df.to_csv(output_csv_path, index=False)

def create_csv_info(file_path, result_folder):
    with open(file_path) as f:
        json_data = json.load(f)

    # Estrai i dati dai file JSON
    iter_values = [entry['iteration'] for entry in json_data]
    minimum_cost_evaluated = [entry['minimum_cost_evaluated'] for entry in json_data]
    n_evaluations = [entry['n_evaluations'] for entry in json_data]
    error = [entry['error'] for entry in json_data]
    optimizer_time = [entry['optimizer_time'] for entry in json_data]

    # Crea un DataFrame con i dati estratti
    data = {
        'iteration': iter_values,
        'minimum_cost_evaluated': minimum_cost_evaluated,
        'n_evaluations': n_evaluations,
        'error': error,
        'optimizer_time': optimizer_time
    }
    info_df = pd.DataFrame(data)

    # Salva il DataFrame in un file CSV
    output_csv_path = f'{result_folder}/info.csv'
    info_df.to_csv(output_csv_path, index=False)
