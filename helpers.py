import requests
import os
import zipfile
import time
import pycountry
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score

class GetData:
    
    def __init__(self, max_retries=3, delay=2):
        self.main_url = 'http://api.worldbank.org/v2/country/all/indicator/{}?downloadformat=csv'
        self.max_retries = max_retries
        self.delay = delay
    
    def get_query_url(self, indicator_code):
        
        query_url = self.main_url.format(indicator_code)
        print(query_url)
        return query_url
    
    def download_data(self, indicator_ids, download_dir):
        os.makedirs(download_dir, exist_ok=True)

        for id in indicator_ids:
            url = self.get_query_url(id)
            success = False

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = requests.get(url)

                    if response.status_code == 200:
                        # Save the file to the specified directory
                        filename = os.path.join(download_dir, f"{id}.zip")
                        with open(filename, 'wb') as file:
                            file.write(response.content)
                        
                        print(f"Downloaded {filename}")
                        success = True
                        break
                    else:
                        print(f"Error {response.status_code}")

                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt}: Request failed with exception: {e}")

                if not success and attempt < self.max_retries:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                
            if not success:
                print(f"Failed to download data for {id} after {self.max_retries} attempts.")
        
        print("Download process finished.")
    
    def unzip_downloaded_files(self, download_dir='raw_data/zip_files'):
        # Unzip all files in the specified directory
        csv_files_dir = 'raw_data/csv_files'
        os.makedirs(csv_files_dir, exist_ok=True)

        for file_name in os.listdir(download_dir):
            if file_name.endswith('.zip'):
                zip_path = os.path.join(download_dir, file_name)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(csv_files_dir)
                print(f"Unzipped {file_name} to {csv_files_dir}")

        print("All files unzipped successfully.")
    
    def rename_csv_files(self, indicator_ids, csv_dir='raw_data/csv_files'):
        '''
        Renames the files that start with API_{indicator_id} to {indicator_id}.csv
        to make it easier to open the files.
        '''
        for indicator_id in indicator_ids:
            # Find all files that start with "API_{indicator_id}" in the csv_dir
            for filename in os.listdir(csv_dir):
                if filename.startswith(f'API_{indicator_id}'):
                    old_file_path = os.path.join(csv_dir, filename)
                    new_filename = f'{indicator_id}.csv'
                    new_file_path = os.path.join(csv_dir, new_filename)

                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed {filename} to {new_filename}')
                    break  # Exit the loop after renaming the first match

        print("Renaming completed.")

class DataFrameAnalyzer:
    def __init__(self):
        pass

    def print_info(self, dataframes):
        
        for name, df in dataframes.items():
            print(f"Info of {name}:")
            print(df.info())
            print("\n" + "="*50 + "\n")

    def print_shape(self, dataframes):
        
        for name, df in dataframes.items():
            print(f"Shape of {name}: {df.shape}")

    def print_describe(self, dataframes):
        
        for name, df in dataframes.items():
            print(f"Description of {name}:")
            print(df.describe())
            print("\n" + "="*50 + "\n")

    def print_head(self, dataframes, n=5):
        
        for name, df in dataframes.items():
            print(f"First {n} rows of {name}:")
            print(df.head(n))
            print("\n" + "="*50 + "\n")

    def print_missing_values(self, dataframes):
        
        for name, df in dataframes.items():
            print(f"Missing values in {name}:")
            print(df.isnull().sum())
            print("\n" + "="*50 + "\n")

    def print_unique_values(self, dataframes, col_name):
       
        for df in dataframes.values():
            print(f"{col_name} unique values:")
            print(df[col_name].unique())
            print("\n" + "="*50 + "\n")

    def print_column_names(self, dataframes):
        
        for name, df, in dataframes.items():
            print(f'{name} column names: {df.columns}')
            print("\n" + "="*50 + "\n")

    def check_column_names_equal(self, dataframes, target_dataframe_name='cotwo_emi'):
        
        target_dataframe = dataframes[target_dataframe_name]
        for name, df, in dataframes.items():
            if not np.array_equal(target_dataframe.columns, df.columns):
                print(f'The dataframe {name} has different column names')
            else:
                print("- All columns are equal -")


class ManipulateData:
    def __init__(self) -> None:
        self.valid_countries = [country.name for country in pycountry.countries]
        self.oecd_countries = [
                        "Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", 
                        "Costa Rica", "Czechia", "Denmark", "Estonia", "Finland", "France", 
                        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", 
                        "Italy", "Japan", "Korea, Rep.", "Latvia", "Lithuania", "Luxembourg", 
                        "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", 
                        "Portugal", "Slovak Republic", "Slovenia", "Spain", "Sweden", 
                        "Switzerland", "Turkiye", "United Kingdom", "United States", "OECD members"
                        ]

        
    def create_oecd_country_subset(self, dataframe_dict):
        
        for indicator_name, indicator_df in dataframe_dict.items():
            dataframe_dict[indicator_name] = indicator_df[indicator_df['Country Name'].isin(self.oecd_countries)].reset_index(drop=True)

        return dataframe_dict


    def eliminate_non_country_data(self, dataframe_dict):
        '''
        Returns a dataframe dictionary with only valid countries data
        '''

        for indicator_name, indicator_df in dataframe_dict.items():
            dataframe_dict[indicator_name] = indicator_df[(indicator_df['Country Name'].isin(self.valid_countries)) | (indicator_df['Country Name'] == 'World')]

        return dataframe_dict
    
    def modify_dataframes_based_on_a_target_dataframe(self, dataframe_dict, target_dataframe_name ='cotwo_emi'):
        '''
        Returns a dataframe dictionary where each dataframe is modified to meet the shape and convey the same information as the
        target_dataframe
        '''
        
        target_dataframe = dataframe_dict[target_dataframe_name]
        target_dataframe_columns = list(target_dataframe.columns)
        target_dataframe_countries = list(target_dataframe['Country Name'].unique())
        
        for indicator_name, indicator_df in dataframe_dict.items():

            if indicator_name != target_dataframe_name:
                dataframe_dict[indicator_name] = indicator_df[target_dataframe_columns][indicator_df['Country Name'].isin(target_dataframe_countries)]
            else:
                continue
        
        return dataframe_dict
    
    def impute_data(self, dataframe_dict):

        for indicator_name, indicator_df in dataframe_dict.items():
            

            indicator_df = indicator_df.infer_objects(copy=False)
            
            # Apply interpolation only to numeric columns and then forward/backward fill
            numeric_cols = indicator_df.select_dtypes(include=[np.number])
            indicator_df[numeric_cols.columns] = numeric_cols.interpolate(method='linear', axis=0).ffill().bfill()
            
            dataframe_dict[indicator_name] = indicator_df

        return dataframe_dict
    
    def create_oecd_dataframe(self, dataframe_dict):

        oecd_df = pd.DataFrame()

        for indicator_name, indicator_df in dataframe_dict.items():

            oecd_row = indicator_df[indicator_df['Country Name'] == 'OECD members']
            
            oecd_row = oecd_row.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'])

            # Transpose the row to have years as the index and the indicator name as the column name
            oecd_row = oecd_row.transpose()
            oecd_row.columns = [indicator_name]

            oecd_df = pd.concat([oecd_df, oecd_row], axis=1)

        return oecd_df
    
    def drop_oecd_members_rows(self, dataframe_dict):

        for indicator_name, indicator_df in dataframe_dict.items():
            dataframe_dict[indicator_name] = indicator_df[indicator_df['Country Name'] != 'OECD members'].reset_index(drop=True)

        return dataframe_dict
    

    
    def df_to_csv_all(self, dataframe_dict, storage_folder = 'clean_data'):

        for indicator_name, indicator_df in dataframe_dict.items():
            file_path = f'{storage_folder}/{indicator_name}.csv'
            indicator_df.to_csv(file_path, index=False)
    
    def drop_indicator_data_cols(self, dataframe_dict):
        """
        Drops the Indicator Name and Indicator Code columns from the dataframes
        """
        for indicator_name, indicator_df in dataframe_dict.items():
            dataframe_dict[indicator_name] = indicator_df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])
        
        return dataframe_dict
    
    def add_suffix_to_cols(self, dataframe_dict):

        for indicator_name, indicator_df in dataframe_dict.items():
            dataframe_dict[indicator_name] = indicator_df.rename(columns=lambda x: x + '_' + indicator_name if x != "Country Name" else x)

        return dataframe_dict
    
    def merge_all_indicators(self, dataframe_dict):

        dfs = list(dataframe_dict.values())
        combined_df = pd.merge(dfs[0], dfs[1], on="Country Name")
        for df in dfs[2:]:
            combined_df = pd.merge(combined_df, df, on="Country Name")
        
        return combined_df
    
    def q4_create_target_variable(self, cotwo_df, start='1997', end='2006', threshold = -10):
        
        cotwo_df_copy = cotwo_df.copy()
        # Calculate the percentage change between the start and end year
        cotwo_df_copy['Percentage Change'] = ((cotwo_df_copy[end] - cotwo_df_copy[start]) / cotwo_df_copy[start]) * 100
        
        # Define the binary target variable
        cotwo_df_copy['Target'] = cotwo_df_copy['Percentage Change'].apply(lambda x: 1 if x <= threshold else 0)

        return cotwo_df_copy

class ModelEvaluation:

    def __init__(self) -> None:
        pass

    def run_q2_regression_experiment(self, model, X, y, poly_degree=None, scaling=True, cv_folds=5):
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            

            if poly_degree:
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X_train = poly.fit_transform(X_train)
                X_test = poly.transform(X_test)
            

            if scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
        
        return {
            "Model": model.__class__.__name__,
            "Poly Degree": poly_degree,
            "Scaling": scaling,
            "Average RMSE": np.mean(rmse_scores),
            "Average R²": np.mean(r2_scores)
        }

