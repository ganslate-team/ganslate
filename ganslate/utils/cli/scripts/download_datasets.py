import wget
import zipfile
import os
from pathlib import Path
import shutil

AVAILABLE_DATASETS = ["ae_photos", "apple2orange", "summer2winter_yosemite", "horse2zebra",  \
                    "monet2photo", "cezanne2photo","ukiyoe2photo", "vangogh2photo", "maps", \
                    "cityscapes", "facades", "iphone2dslr_flower", "mini", "mini_pix2pix", "mini_colorization"] 

def download(name, path):
    if name not in AVAILABLE_DATASETS:
        print(""".Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos
        
                facades: 400 images from the CMP Facades dataset. [Citation]
                cityscapes: 2975 images from the Cityscapes training set. [Citation]. Note: Due to license issue, we cannot directly provide the Cityscapes dataset. Please download the Cityscapes dataset from https://cityscapes-dataset.com
                maps: 1096 training images scraped from Google Maps.
                horse2zebra: 939 horse images and 1177 zebra images downloaded from ImageNet using keywords wild horse and zebra
                apple2orange: 996 apple images and 1020 orange images downloaded from ImageNet using keywords apple and navel orange.
                summer2winter_yosemite: 1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API. See more details in our paper.
                monet2photo, vangogh2photo, ukiyoe2photo, cezanne2photo: The art images were downloaded from Wikiart. The real photos are downloaded from Flickr using the combination of the tags landscape and landscapephotography. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.
                iphone2dslr_flower: both classes of images were downlaoded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316. See more details in our paper.
                        
                Refer link: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md
        """)

    else:

        assert Path(path).is_dir(), f"{path} provided is not a directory"
        url = f"https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{name}.zip"


        path_to_zip_file = f"{path}/{name}.zip"

        # Remove if file already exists to handle corrupt files.
        if os.path.isfile(path_to_zip_file):
            os.remove(path_to_zip_file)

        print(f"Fetching {name} datasets from {url}:")
        wget.download(url, out=path_to_zip_file)

        if Path(f"{path}/{name}").is_dir():
            shutil.rmtree(Path(f"{path}/{name}"))

        print(f"Extracting zip file to {path}")
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(path)

        os.remove(path_to_zip_file)

        print("Reorganizing folder structure for ganslate")
        # Make folders for train and test
        train_path = Path(f"{path}/{name}/train")
        test_path = Path(f"{path}/{name}/test")
        
        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)

        # Copy contents of download to path structure required by ganslate
        shutil.move(f"{path}/{name}/trainA", str(train_path / "A"))
        shutil.move(f"{path}/{name}/trainB", str(train_path / "B"))
        shutil.move(f"{path}/{name}/testA", str(test_path / "A"))
        shutil.move(f"{path}/{name}/testB", str(test_path / "B"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("name")
    parser.add_argument("path")

    args = parser.parse_args()

    download(args.name, args.path)
