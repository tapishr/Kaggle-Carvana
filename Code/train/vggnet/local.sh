gcloud ml-engine local train \
--module-name trainer.train_cv2 \
--package-path ./trainer \
-- \
--train-file ./tmp \
--job-dir ./tmp