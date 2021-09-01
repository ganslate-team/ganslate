FILE=$1
FOLDER=$2

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "mini" && $FILE != "mini_pix2pix" && $FILE != "mini_colorization" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

if [[ $FILE == "cityscapes" ]]; then
    echo "Due to license issue, we cannot provide the Cityscapes dataset from our repository. Please download the Cityscapes dataset from https://cityscapes-dataset.com, and use the script ./datasets/prepare_cityscapes_dataset.py."
    echo "You need to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip. For further instruction, please read ./datasets/prepare_cityscapes_dataset.py"
    exit 1
fi

echo "Specified [$FILE]"
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=./$FILE.zip
TARGET_DIR=./$FILE/
wget -N $URL -O $ZIP_FILE --no-check-certificate
mkdir $TARGET_DIR
unzip $ZIP_FILE -d $FOLDER/datasets/
rm $ZIP_FILE

# Restructure dataset for into train and test with subfolders A and B
mkdir $FOLDER/datasets/$FILE/train
mkdir $FOLDER/datasets/$FILE/test

mv $FOLDER/datasets/$FILE/trainA $FOLDER/datasets/$FILE/train/A
mv $FOLDER/datasets/$FILE/trainB $FOLDER/datasets/$FILE/train/B
mv $FOLDER/datasets/$FILE/testA $FOLDER/datasets/$FILE/test/A
mv $FOLDER/datasets/$FILE/testB $FOLDER/datasets/$FILE/test/B
