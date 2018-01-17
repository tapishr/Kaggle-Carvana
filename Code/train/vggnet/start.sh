export JOB_NAME="train_unet128_$(date +%Y%m%d_%H%M%S)"
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir gs://crack-muse-182419/jetson-kaggle-carvana-t303/logs/$JOB_NAME \
  --runtime-version 1.2 \
  --module-name trainer.train_cv2 \
  --package-path ./trainer \
  --region $REGION \
  --config=trainer/cloudml-gpu.yaml \
  -- \
  --train-file gs://crack-muse-182419/jetson-kaggle-carvana-t303/tmp