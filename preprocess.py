import argparse
import tqdm
from utils import seed_everything, make_map_fn
from datasets import load_dataset
from datasets import Dataset, DatasetDict

if __name__ == "__main__":
    seed_everything(seed=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--split")
    parser.add_argument("--make_crossover", action="store_true", default=False)
    parser.add_argument("--save_dir", default="../EvoGRPO-datasets")
    args = parser.parse_args()

    if args.make_crossover:
        import json
        with open(f"../EvoGRPO-datasets/{args.task}/{args.split}_evo.json", "r") as f:
            dataset = json.load(f)
        dataset = Dataset.from_list(dataset)
        dataset = DatasetDict({args.split: dataset})
    else:
        data_source = "lhkhiem28/EvoGRPO-datasets"
        print(f"Loading the {data_source}/{args.task} dataset from huggingface...", flush=True)
        dataset = load_dataset(
            data_source, args.task
        )

    dataset = dataset[args.split]
    dataset = dataset.map(function=make_map_fn(args.split, args.make_crossover), with_indices=True)

    import os
    local_dir = os.path.join(os.path.expanduser(args.save_dir), args.task)
    dataset.to_parquet(os.path.join(local_dir, f"{args.split}.parquet" if not args.make_crossover else f"{args.split}_evo.parquet"))

    import json
    example = dataset[0]
    with open(os.path.join(local_dir, f"{args.split}_example.json" if not args.make_crossover else f"{args.split}_evo_example.json"), "w") as f:
        json.dump(example, f, indent=2)