task=realworld_ml1
algo=bc_transformer
exp_name=test

CUDA_VISIBLE_DEVICES=$1 python eval.py \
    task=$task \
    algo=$algo \
    exp_name=$exp_name \
    variant_name=block_32_ds_4 \
    stage=1 \
    training.use_tqdm=true \
    checkpoint_path=experiments/realworld/ML4/bc_transformer_policy/flow-decomposer/block_10/0/run_002/multitask_model_epoch_0100.pth \
    seed=0 \
    make_unique_experiment_dir=true \
    fps=20 \
    compress=false \
    save_verbose=true \
    overwrite=false \
    # algo.action_horizon=8 \
    # rollout.num_parallel_envs=15 \
    # rollout.rollouts_per_env=15 \
    # rollout.n_video=3 \
    # +f2a_path=experiments/metaworld/ML11/quest/flow2action_12demos_nf_corner/run_001/flow2action_model_epoch_0050.pth \

#  checkpoint_path=/tmp2/hcfang/flow-decomposer/QueST/experiments/metaworld/ML11/quest/flow_image_30_nf_corner_v2/block_32_ds_4/0/run_022/multitask_model_epoch_0100.pth \

# Note1: this will automatically load the latest checkpoint as per your exp_name, variant_name, algo, and stage.
#        Else you can specify the checkpoint_path to load a specific checkpoint.