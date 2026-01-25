import pandas as pd
import os

def load_raw_data(raw_data_folder_path, target_file_path):
    '''
    Loads the data from raw_data_folder_path and puts it into target_file_path.
    Note that the structure inside the raw_data_folder_path and output folder are harcoded in the function, as making it more dynamic would add unnecessary complexity.

    :param raw_data_folder_path: Path to the folder containing raw data files
    :param target_file_path: Path to the target file containing target data
    '''

    raw_data_paths = [
        ('/depression', '/targets/DPQ_L_Target_Depression.xpt'),
        ('/insomnia', '/targets/SLQ_L_Target_Insomnia.xpt'),
    ]

    for i in range(len(raw_data_paths)):

        data = pd.read_sas(raw_data_folder_path + raw_data_paths[i][1], format='xport')

        # Remove rows with any missing values in the targets
        data = data[data[list(data.columns)].notnull().all(1)]

        for file_path in os.listdir(raw_data_folder_path + raw_data_paths[i][0]):
            if not file_path.endswith('.xpt'):
                raise ValueError("Unsupported file format")

            file_path = os.path.join(raw_data_folder_path + raw_data_paths[i][0], file_path)
            new_data = pd.read_sas(file_path, format='xport') 

            if 'SEQN' not in new_data.columns:
                print("Missing SEQN column in file: " + file_path)
            else:     
                data = pd.merge(data, new_data, how="left", left_on="SEQN", right_on="SEQN")

        numeric_cols = data.select_dtypes(include=['float64']).columns
        for col in numeric_cols:
            data[col] = data[col].round().astype("Int64")

        os.makedirs(target_file_path, exist_ok=True)
        data.to_csv(target_file_path + '/' + raw_data_paths[i][0] + '_data.csv', index=False)