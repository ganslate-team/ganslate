cd data-pipelines
echo "Converting dicom data to nrrd ..."
conda run python projects/mirada_wbx/process_dcm_data.py /data --output_dir /data/processed
cd ..
cd midaGAN
echo "Running inference .."
conda run python tools/infer.py config=/model/config.yaml infer.checkpointing.load_iter=50000

cd ..
cd clinical-evaluation
echo "Converting translated nrrd to dicom"
conda run python projects/mirada_wbx/generate_dicoms.py /model/infer/saved --output_dir /data/out