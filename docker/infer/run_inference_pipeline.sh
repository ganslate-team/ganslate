cd data-pipelines
echo "Converting dicom data to nrrd ..."
conda run python projects/mirada_wbx/process_dcm_data.py /data --output_dir /data/processed