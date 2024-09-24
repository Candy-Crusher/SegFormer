export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,2
./tools/dist_test.sh \
    local_configs/segformer/B2/segformer.b2.440x640.dsec.40k.py \
    work_dirs/segformer.b2.440x640.dsec.160k/iter_4000.pth \
    2 \
