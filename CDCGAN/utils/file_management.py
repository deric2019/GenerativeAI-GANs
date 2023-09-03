# File management 
import os
import re
import glob
import shutil
import zipfile

class FileManagement():
    '''File management functions such as removing, copying, zip files'''
    def check_existent_dir(folder_list: list):
        '''Check whether a folder exists in a project, else remove it from the list 
        Args:
            folder_list: list of folders to be checked

        '''
        # Create a new list to store the valid folders
        valid_folders = []
        print(f'Checking folders: {folder_list} if they exists ...')
        # Iterate through the folder_list
        for folder in folder_list:
            # Check if the folder exists
            if os.path.exists(folder):
                valid_folders.append(os.path.basename(folder))
                print(f'The folder: {folder} exists')

            else:
                print(f'The folder: {folder} does not exist')
        
        print(f'Folders that exists: {valid_folders}')
        return valid_folders

    def recursively_remove_directory(directory: str):
        '''Recursively remove all dirs and files in a dir, including the dir
        Args: 
            directory: path to the directory'''
        if os.path.exists(directory):
            try:
                for root, dirs, files in os.walk(directory, topdown=False):
                #  Generate the file names in a directory tree
                    for file in sorted(files):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        print(f'Deleting file in: {file_path} ...')
                                    
                    # Remove parent directories 
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        os.rmdir(dir_path)
                        print(f'Deleting directory: {dir} ...')

                print(f"All files and folders in '{directory}' and its subdirectories have been deleted ...")
                
                print(f'Finally removed the folder: {directory} ...')
                os.rmdir(directory)

            except Exception as e:
                print(f"Error deleting files and folders in: '{directory}': {e}!")
        else:
            print(f'Directory: {directory} does not exists!')

    def copy_folder_with_exclusion(src_dir: str, dest_dir: str, ignored_items: list):
        '''Copying a folder to another path taking account to ignored patterns or files
        Args:
            ignored_items: list of ignored patterns or folder names
        '''
        
        # Check if the destination already exists
        if not os.path.exists(dest_dir):
            try:
                # Copy the folder and its contents while excluding specific folders
                shutil.copytree(src_dir, dest_dir, ignore=shutil.ignore_patterns(*ignored_items))
                
                print('Copying ...')
                for item in ignored_items:
                    print(f'Ignoring: {item}')
                print(f"Folder '{src_dir}' has been successfully copied to '{dest_dir}'.")
            
            except shutil.Error as e:
                print(f"Error copying folder: {e}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f'Cant copy, the folder: {dest_dir} already exists')

    def remove_files_in_dir_with_matching_pattern(dir: str, pattern: str):
        '''Remove files in a directory that matches the patter
        Args:
            dir (str): directory
            pattern (str): pattern files to be removed
        '''
        # Find files that match the pattern
        files_to_remove = glob.glob(os.path.join(dir, pattern))

        # Remove the matched files
        for file_path in files_to_remove:
            print(f'Removing: {file_path} ... ')
            os.remove(file_path)
            

    def write_zip(folder_paths: list , output_zipfile: str):
        '''
        Zip multiple folders into a single zip archive.
        Args:
            folder_paths: A list of folder paths to be zipped.
            output_zipfile: The path to the output zip archive.
        '''
        # Check if the folders exists before ziping them
        folder_paths = FileManagement.check_existent_dir(folder_paths)

        with zipfile.ZipFile(output_zipfile, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for folder_path in folder_paths:
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder '{folder_path}' does not exist. Skipping...")
                    continue

                print(f'Compressing folder: {folder_path}')

                # Calculate the base folder name to be used in the zip archive
                base_folder_name = os.path.basename(folder_path)

                # Walk through the directory and add all files and subdirectories to the zip archive
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate the relative path to preserve folder structure in the zip archive
                        relative_path = os.path.relpath(file_path, folder_path)
                        zipf.write(file_path, arcname=os.path.join(base_folder_name, relative_path))

        print(f"Successfully zipped {len(folder_paths)} folders into '{output_zipfile}'.")

    def extract_zip(zip_file_path: str, extract_to_path: str):
        '''Extract a compressed file into a folder

        Args:
            zip_file_path : path to zip file to be extracted
            extract_to_path : extract zip file to the pat
        '''
        print(f'Extracting folder: {zip_file_path} to: {extract_to_path}...')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f'Finished extracting: {zip_file_path} ...')

    def sort_files_by_number(filenames: list):
        '''Sort a list of filenames by its number 
        independent on the text before or after it'''
        def extract_numeric(filename):
            return int(re.search(r'\d+', filename).group())
        sorted_filenames = sorted(filenames, key=extract_numeric)
        return sorted_filenames
    
    def create_folder_it_not_already_exists(dir_path):
        """Create a folder if it is not already exists
        Args:
            path: path to create the folder
        """
        if not os.path.exists(dir_path):
            # Create a checkpoint directory if it does not already exists
            print(f'Creating the directory: {dir_path} ...')
            os.mkdir(dir_path)
    
    def load_trained_generator(model_ckpt_dir: str, n_classes: int, latent_dim: int, model_name: str):
        '''Load trained generator
        Args:
            model_ckpt_dir: model checkpoint
            n_classes (int): number of class labels
            latent_dim (int):  latent dim
            model_name: model_name is the name of the directory of saved models '''
        # Create the generator
        import models.networks as networks

        generator_o = networks.define_generator(n_classes, latent_dim)

        # Find filepath
        model_dir = os.path.join(model_ckpt_dir, model_name)

        generator_files = glob.glob(os.path.join(model_dir, '*'))
        generator_files = FileManagement.sort_files_by_number(generator_files)

        # If not generator list not empty
        if generator_files:
            # Extract the latest one
            generator_file = generator_files[-1]
            print('Loading the latest generator ...')

            # Load weights into the generator 
            generator_o.load_weights(generator_file)
            print(f'Loading generator: {generator_file} ...')

            return generator_o
        else:
            print('No generator loaded, no generator file was found ...')