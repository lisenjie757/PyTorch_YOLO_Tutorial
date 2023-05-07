# 8 GPUs
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -d voc \
                                                    --root ../data/ \
                                                    -m yolov2 \
                                                    -bs 16 \
                                                    -size 640 \
                                                    --num_workers 15 \
                                                    --wp_epoch 1 \
                                                    --max_epoch 150 \
                                                    --eval_epoch 10 \
                                                    --ema \
                                                    --fp16 \
                                                    --multi_scale \
                                                    --distributed \
                                                    --sybn \
