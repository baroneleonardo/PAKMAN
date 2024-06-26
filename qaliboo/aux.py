import pandas as pd
import json
from qaliboo import machine_learning_models
import os
import datetime
#dat = '/home/lbarone/QALIBOO/qaliboo/datasets/query26_vm_ram.csv' 
#dat = '/home/lbarone/QALIBOO/qaliboo/datasets/ligen_synth_table.csv'
#dat = '/home/lbarone/QALIBOO/qaliboo/datasets/stereomatch10.csv'

def define_dat(problem):
    if problem == 'ScaledLiGenTot':
        dat = '/home/lbarone/PAKMAN/qaliboo/datasets/ligen_synth_table.csv'
    elif problem == 'ScaledStereoMatch10':
        dat = '/home/lbarone/PAKMAN/qaliboo/datasets/stereomatch10.csv'
    elif problem == 'ScaledQuery26':
        dat = '/home/lbarone/PAKMAN/qaliboo/datasets/query26_vm_ram.csv'
    return dat

def create_result_folder(sub_folder):
    main_folder = './results/'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    folder_path_sub = os.path.join(main_folder, sub_folder)
    if not os.path.exists(folder_path_sub):
        os.makedirs(folder_path_sub)
    now_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    folder_path_now = os.path.join(folder_path_sub, now_dir)
    if not os.path.exists(folder_path_now):
        os.makedirs(folder_path_now)

    result_folder = folder_path_now
    return result_folder

def csv_init(result_folder, dat_indices, dat):
    dataset_csv_path = dat
    df = pd.read_csv(dataset_csv_path)
    new_df = pd.DataFrame({'dat_index': dat_indices})
    new_df = df.iloc[dat_indices, :]
    output_csv_path = f'{result_folder}/init.csv'
    new_df.to_csv(output_csv_path, index=False)

def csv_history(result_folder, iter_values, dat_indices, dat):
    dataset_csv_path = dat
    df = pd.read_csv(dataset_csv_path)
    output_csv_path = os.path.join(result_folder, 'history.csv')
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
    else:
        existing_df = pd.DataFrame()
    selected_rows_df = df.loc[dat_indices, :]
    selected_rows_df.insert(0, 'index', iter_values)
    updated_df = pd.concat([existing_df, selected_rows_df], ignore_index=True)
    updated_df.to_csv(output_csv_path, index=False)

def csv_info(iteration, q, evaluation_count, global_time, unfeasible_points, mape, result_folder, error):
    # Creare un DataFrame con i nuovi dati
    data = {
        'iteration': [iteration],
        'points_evaluated':[q],
        'n_evaluations': [evaluation_count],
        'unfeasible_points':[unfeasible_points],
        'optimizer_time': [global_time],
        'mape': [mape],
        'error': [error]
    }
    new_data_df = pd.DataFrame(data)

    # Percorso del file CSV
    output_csv_path = os.path.join(result_folder, 'info.csv')

    # Se il file CSV esiste già, leggi i dati esistenti
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
    else:
        existing_df = pd.DataFrame()  # Se il file non esiste, crea un DataFrame vuoto

    # Concatena i dati esistenti con i nuovi dati
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)

    # Salva il DataFrame aggiornato come file CSV
    updated_df.to_csv(output_csv_path, index=False)


def csv_result_XGB(iteration, q, min_evaluated, evaluation_count, global_time, unfeasible_points, best_point, result_file):
    # Creare un DataFrame con i nuovi dati
    data = {
        'iteration': [iteration],
        'points_evaluated':[q],
        'minimum_cost_evaluated': [min_evaluated],
        'n_evaluations': [evaluation_count],
        'unfeasible_points':[unfeasible_points],
        'optimizer_time': [global_time],
        'best target': best_point
    }
    new_data_df = pd.DataFrame(data)

    # Percorso del file CSV
    # Se il file CSV esiste già, leggi i dati esistenti
    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file)
    else:
        existing_df = pd.DataFrame()  # Se il file non esiste, crea un DataFrame vuoto

    # Concatena i dati esistenti con i nuovi dati
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)

    # Salva il DataFrame aggiornato come file CSV
    updated_df.to_csv(result_file, index=False)

def save_execution_time(times, result_folder):
    # Creare un DataFrame con i nuovi dati
    data = {
        'time': times
    }
    new_data_df = pd.DataFrame(data)

    # Percorso del file CSV
    output_csv_path = os.path.join(result_folder, 'execution_times.csv')

    # Se il file CSV esiste già, leggi i dati esistenti
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
    else:
        existing_df = pd.DataFrame()  # Se il file non esiste, crea un DataFrame vuoto

    # Concatena i dati esistenti con i nuovi dati
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)

    # Salva il DataFrame aggiornato come file CSV
    updated_df.to_csv(output_csv_path, index=False)

def csv_testfunction(iteration, q, objective_func_name, minimum_evaluated,n_evaluations,unfeasible_point, result_file):
    # Creare un DataFrame con i nuovi dati
    data = {
        'iteration': [iteration],
        'points_evaluated':[q],
        'objective_func_name': [objective_func_name],
        'minimum_cost_evaluated': [minimum_evaluated],
        'n_evaluations': [n_evaluations],
        'unfeasible_points':[unfeasible_point]
    }
    new_data_df = pd.DataFrame(data)

    # Se il file CSV esiste già, leggi i dati esistenti
    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file)
    else:
        existing_df = pd.DataFrame()

    # Concatena i dati esistenti con i nuovi dati
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)

    # Salva il DataFrame aggiornato come file CSV

    updated_df.to_csv(result_file, index=False)






'''
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

'''