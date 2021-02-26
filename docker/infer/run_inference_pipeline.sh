cd data-pipelines
echo "Converting dicom data to nrrd ..."
conda run python projects/mirada_wbx/process_dcm_data.py /data --output_dir /data/processed
cd ..
cd midaGAN
echo "Running inference .."
conda run python tools/infer.py config=/model/config.yaml infer.checkpointing.load_iter=50000