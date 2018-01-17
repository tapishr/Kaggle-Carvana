export JOB_NAME="test_unet1024_$(date +%Y%m%d_%H%M%S)"
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://crack-muse-182419/jetson-kaggle-carvana-t303/test_data \
  --runtime-version 1.2 \
  --module-name tester.test_submit_multithreaded \
  --package-path ./tester \
  --region $REGION \
  --config=tester/cloudml-gpu.yaml \
  -- \
  --weights-dir gs://crack-muse-182419/jetson-kaggle-carvana-t303/weights/unet_1024_best_weights.hdf5 \
  --output-dir gs://crack-muse-182419/jetson-kaggle-carvana-t303/logs/$JOB_NAME