# @dataclass
# class CBCTtoCTInferenceDatasetConfig(configs.base.BaseDatasetConfig):
#     name:                    str = "CBCTtoCTInferenceDataset"
#     hounsfield_units_range:  Tuple[int, int] = field(default_factory=lambda: (-1000, 2000)) #TODO: what should be the default range

# class CBCTtoCTInferenceDataset(Dataset):
#     def __init__(self, conf):
#         self.paths = io.make_dataset_of_directories(conf.train.dataset.root, EXTENSIONS)
#         self.num_datapoints = len(self.paths)
#         # Min and max HU values for clipping and normalization
#         self.hu_min, self.hu_max = conf.train.dataset.hounsfield_units_range

#     def __getitem__(self, index):
#         path = str(Path(self.paths[index]) / 'CT.nrrd')
#         # load nrrd as SimpleITK objects
#         volume = sitk_utils.load(path)
#         metadata = (path,
#                     volume.GetOrigin(),
#                     volume.GetSpacing(),
#                     volume.GetDirection(),
#                     sitk_utils.get_npy_dtype(volume))

#         volume = sitk_utils.get_tensor(volume)
#         # Limits the lowest and highest HU unit
#         volume = torch.clamp(volume, self.hu_min, self.hu_max)
#         # Normalize Hounsfield units to range [-1,1]
#         volume = min_max_normalize(volume, self.hu_min, self.hu_max)
#         # Add channel dimension (1 = grayscale)
#         volume = volume.unsqueeze(0)

#         return volume, metadata

#     def __len__(self):
#         return self.num_datapoints

#     def save(self, tensor, save_dir, metadata):
#         tensor = tensor.squeeze()
#         tensor = min_max_denormalize(tensor, self.hu_min, self.hu_max)

#         datapoint_path, origin, spacing, direction, dtype = metadata
#         sitk_image = sitk_utils.tensor_to_sitk_image(tensor, origin, spacing, direction, dtype)

#         # Dataset used has a directory per each datapoint, the name of each datapoint's dir is used to save the output
#         datapoint_name = Path(str(datapoint_path)).parent.name
#         save_path = Path(save_dir) / Path(datapoint_name).with_suffix('.nrrd')

#         sitk_utils.write(sitk_image, save_path)
