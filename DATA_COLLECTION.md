## Collecting

- Record episodes

```bash
# Add --debug to skip input, allowing breakpoint to work normally
# Tasks are moved to config/tasks.yaml
python3 record_episodes_with_cli.py --log warn --task_name <TASK_NAME> -n 10
```

- Visualize episodes

```bash
# No episode_idx means process all episodes
python3 visualize_episodes.py --dataset_dir ~/aloha_data/flow_decomp/<TASK_NAME>
```

## Processing

- Gather episodes

Under `~/flow-decomposer`

```bash
# Check utils/aloha/gather_task_data.py for details
sh scripts/create_aloha_data.sh <TASK_NAME>
```

- Preprocess data

<!-- Under `~/Projects/flow-decomposer` of `ssh tomchen@140.112.194.91 -p 1009` -->
Under `~/flow-decomposer`

Copy the gathered data from `~/flow-decomposer/data/realworld/ML4/train` to `~/Projects/flow-decomposer/data/realworld/ML4/train` before processing

```bash
conda activate flow-decomposer
# Check process_data.sh for tweaking max_len, demos_per_task, etc.
sh ./scripts/process_data.sh 0
```

## TODOs

- [ ] Modify the implementation of compression in `visualize_episodes.py` and `gather_task_data.py`