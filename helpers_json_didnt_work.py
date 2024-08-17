import requests
import os
import time

class GetData:

    def __init__(self, max_retries=3, delay=2) -> None:
        self.main_url = 'http://api.worldbank.org/v2/country/all/indicator/{}?format=json'
        self.max_retries = max_retries
        self.delay = delay
    
    def get_query_url(self, indicator_code):
        query_url = self.main_url.format(indicator_code)
        print(f"Generated URL: {query_url}")
        return query_url
    
    def download_data(self, indicator_ids, download_dir):

        os.makedirs(download_dir, exist_ok=True)

        for indicator_id in indicator_ids:
            url = self.get_query_url(indicator_id)
            success = False
            attempt = 0

            while not success and attempt < self.max_retries:
                response = requests.get(url)
                attempt += 1

                if response.status_code == 200:
                    # Save JSON response to a file
                    filename = os.path.join(download_dir, f"{indicator_id}.json")
                    with open(filename, 'w') as file:
                        file.write(response.text)
                    
                    print(f"Downloaded {filename}")
                    success = True
                else:
                    print(f"Failed to download {indicator_id} (Status code: {response.status_code}). Attempt {attempt} of {self.max_retries}")
                    time.sleep(self.delay)
            
            if not success:
                print(f"Failed to download {indicator_id} after {self.max_retries} attempts.")

        print("Download Finished")