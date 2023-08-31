# File management
import os
import requests, zipfile
from tqdm import tqdm

# ---------------------------------------------------
# Functions relating to download and extracting zip 
# and loading the dataset
# ----------------------------------------------------
class FetchData:
    def download_data_from_url(url: str, save_path: str):
        '''Download zip file from url to the data dir
        Args:
            url (str): url to the zip to be downloaded
            save_path (str):  Where to save and name the zip file'''

        # Check whether the zip file have been downloaded
        if not os.path.exists(save_path):   
            print(f'Commencing downloading from {url}...')
            
             # Stream the response content
            response = requests.get(url, stream=True)

            # Get the total file size
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024  # Adjust the chunk size according to your preference

            if response.status_code == 200:
                with open(save_path, 'wb') as file, tqdm(desc=save_path,total=total_size,unit='B',unit_scale=True,unit_divisor=1024) as bar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        file.write(data)
                        bar.update(len(data))
                print(f"Downloaded {url} and saved it to {save_path} ...")
                print('Finished downloading ...')
            else:
                print(f"Failed to download {url}. Status code: {response.status_code}")
        else:
            print(f'The compressed file {save_path} already exists, no need to download the zip file')


    def extract_compressed_file(filepath:str, save_path:str):
        """Extracted a compressed zip file to save_path
        Args:
            filepath (str): path to the compressed zip file
            save_path (str): path where the compressed file should be extracted
        """
        # If image dir does not exists
        filepath_without_extension = filepath.replace('.zip', '')
        if not os.path.exists(filepath_without_extension):
            # Commencing extracting zip
            print(f'Extracting data compressed file: {filepath} to {save_path}/...')
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(save_path)
            print(f'Finished extracting: {filepath} ...')
        else:
            print(f'The folder {filepath_without_extension} already exists, no need to extract')

