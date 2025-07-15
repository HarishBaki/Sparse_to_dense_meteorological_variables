#rclone copy --progress --transfers 24 hb533188_dgxhead:/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables/checkpoints ./checkpoints
#rclone copy --progress --transfers 24 ./checkpoints hb533188_dgxhead:/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables/checkpoints
  #--include "**/NYSM_test_metrics.zarr/**" \
  #--include "**/RTMA_test_metrics.zarr/**" \
rclone copy --progress --transfers 24 \
  --include "**/RTMA_test_metrics.nc" \
  --include "**/NYSM_test_metrics.nc" \
  ./checkpoints \
  hb533188_dgxhead:/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables/Predictions
 