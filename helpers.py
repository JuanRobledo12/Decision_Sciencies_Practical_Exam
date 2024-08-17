import requests
import os
import zipfile
import time

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
