# Use the container (docker ≥ 19.03 required)

To build training docker:
```
cd docker/train
docker build -t midagan:latest .
```

To run using all GPUs:
```
docker run --gpus all -it \
	--shm-size=24gb --volume=<source_to_data>:/data --volume=<source_to_results>:/output\
	--name=midagan midagan:latest /bin/bash
```
