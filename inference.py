import argparse
import tqdm
from utils import *
from datasets import load_dataset
from vllm import LLM, SamplingParams
from verl.utils.reward_score import math_reward

if __name__ == "__main__":
    seed_everything(seed=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--split", default="test")
    parser.add_argument("--make_crossover", action="store_true", default=False)
    parser.add_argument("--repo_id")
    args = parser.parse_args()

    if args.make_crossover:
        data_source = "lhkhiem28/EvoGRPO-datasets"
        print(f"Loading the {data_source}/{args.task} dataset from huggingface...", flush=True)
        dataset = load_dataset(
            data_source, args.task
        )
        dataset = dataset[args.split]
        dataset = dataset.map(function=make_map_fn(args.split), with_indices=True)

        generator = LLM(model=args.repo_id, tensor_parallel_size=1)

        dataset_evo = []
        for i in tqdm.tqdm(range(0, len(dataset), 128)):
            batch = dataset[i:i+128]
            batch_prompts = [item[0]["content"] for item in batch["prompt"]]

            batch_outputs = generator.generate(batch_prompts, SamplingParams(
                max_tokens=3072, n=2, 
                temperature=0.6, top_p=0.95, 
            ))
            batch_outputs = [[output.text for output in outputs.outputs] for outputs in batch_outputs]
            for prompt, outputs, ground_truth in zip(batch_prompts, batch_outputs, [item["ground_truth"] for item in batch["reward_model"]]):
                question = prompt.replace("You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{}. Please stop generating immediately after outputting the box.\n", "")
                question += "\n\nBelow are 2 candidate solutions:"
                for i, output in enumerate(outputs):
                    question += f"\n\nThe candidate solution {i+1}: {output}"
                dataset_evo.append({
                    "question": question,
                    "answer": ground_truth,
                })

        import json
        with open(f"../EvoGRPO-datasets/{args.task}/{args.split}_evo.json", "w") as f:
            json.dump(dataset_evo, f, indent=4)