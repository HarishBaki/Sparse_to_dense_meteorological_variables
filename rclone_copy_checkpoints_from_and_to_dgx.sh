#rclone copy --progress --transfers 24 hb533188_dgxhead:/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables/checkpoints ./checkpoints
#rclone copy --progress --transfers 24 ./checkpoints hb533188_dgxhead:/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables/checkpoints
rclone copy --progress --transfers 24 \
  --include "best_model.pt" \
  --include "latest.pt" \
  ./checkpoints \
  hb533188_dgxhead:/network/rit/dgx/dgx_basulab/Harish/Sparse_to_dense_meteorological_variables/checkpoints
 