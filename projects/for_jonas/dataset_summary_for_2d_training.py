import sys
import pathlib
import pandas as pd
import torch
import SimpleITK as sitk

def dataset_summary_for_2d_training(dataset_path, *extensions):
    """
    Creates a summary that can be used to randomly select a slice from the dataset of 3D images.
    
    The returned dataframe will be used by the dataloader use as follows:
    (1) Randomly select an row in the dataframe
    (2) Use the filename specified in the row to load the volume
    (3) The key 'slice' in the row is used to select the appropriate slice from the volume
    (4) The slice is normalized using the mean and std specified in the row
    
    Parameters
    ----------
    dataset_path : str
        The root of the dataset that will be used during the training as well.
    *extension : str 
    
    Returns
    -------
    pandas.Dataframe : Saves to csv and returns dataset's summary dataframe that is used by the dataloader.
    """

    dataset_path = pathlib.Path(dataset_path)
    df = pd.DataFrame(columns=['slice', 'volume_filename', 'volume_mean', 'volume_std', 'volume_num_slices'])
    
    # Fetch all filepaths for each specified extension
    file_list = []
    print(extensions)
    for ext in extensions:
        if not ext.startswith("."):
            ext = "." + ext
        partial_file_list = list(dataset_path.glob(f"**/*{ext}"))
        file_list.extend(partial_file_list)
    
    # Go through all files and store the information into the dataframe
    for file in file_list:
        volume = sitk_load(str(file))
        num_slices = volume.GetSize()[2]
        volume = sitk_get_tensor(volume) # because torch is used in dataloader to perform normalization
        mean = float(volume.mean())
        std = float(volume.std())
        
        # Add as many rows to dataframe as there are slices in the volume, along with the useful data
        rows = []
        for i in range(num_slices):
            rows.append(
                {
                    'slice': i, 
                     'volume_filename': str(file.relative_to(dataset_path)), 
                     'volume_mean': mean, 
                     'volume_std': std, 
                     'volume_num_slices': num_slices
                }
            )
            
        df = df.append(rows, ignore_index=True)

    csv_path = str(dataset_path).replace('/', '_') + '.csv'
    df.to_csv(csv_path, index=False) 
    print(f'Dataset summary saved as { pathlib.Path(csv_path).absolute() }')
    return df

def sitk_load(file_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    sitk_image = reader.Execute()
    return sitk_image

def sitk_get_tensor(sitk_image):
    return torch.Tensor(sitk.GetArrayFromImage(sitk_image))

if __name__ == '__main__':
    data_path = sys.argv[1]
    extensions = sys.argv[2:]
    dataset_summary_for_2d_training(data_path, *extensions)
