gcloud ml-engine local train \
--module-name tester.test_submit_multithreaded \
--package-path ./tester \
-- \
--job-dir ./tmp/test_data \
--weights-dir ./tmp/unet_1024_best_weights.hdf5 \
--output-dir ./tmp