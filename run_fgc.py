print("--- RUNNING SELF-CONTAINED FGC EXPERIMENT SCRIPT ---")

import logging
import re
import torch
import os
import json
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Tuple

# --- Step 1: Import all necessary components from the OAT library ---
from oat.algorithms.ppo import PPOArgs, PPOLearner
from oat.actors.reward import RewardActor
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.types import TrajectoryData, Metric
from oat.oracles.base import RewardOracleBase
from oat.utils.data import PromptDataset # We only need PromptDataset from here
from datasets import load_dataset


# --- Step 2: Define our custom arguments ---
@dataclass
class FGCArgs(PPOArgs):
    cal_model_name: str = field(default="gemini-1.5-flash-latest")
    cal_few_shot_path: str = field(default="cal_few_shot_examples.json")
    negative_reward: float = field(default=-1.0)
    prompt_data_name: str = field(default="main")


# --- Step 3: Define our CALOracle ---
class CALOracle(RewardOracleBase):
    # (Your full CALOracle logic will go here. For now, using placeholders)
    def __init__(self, cal_model_name: str, few_shot_path: str, api_key_env: str = "GOOGLE_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key: raise ValueError(f"'{api_key_env}' not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(cal_model_name)
    def get_reward(self, inputs: List[str], responses: List[str], references: List[str], **kwargs) -> Tuple[torch.Tensor, Metric]:
        return torch.zeros(len(inputs)), {"cal_outputs": [{"error_segment": ""}] * len(inputs)}


# --- 4. DEFINE OUR OWN FIXED VERSION OF get_datasets ---
def get_datasets_fixed(args, tokenizer, strategy) -> Tuple[PromptDataset, PromptDataset | None]:
    """This is our custom version that correctly handles prompt_data_name."""
    logging.info(f"Using our fixed get_datasets to load '{args.prompt_data}' config '{args.prompt_data_name}'")
    
    # This is the fixed data loading call
    prompt_dataset_from_hub = load_dataset(args.prompt_data, name=args.prompt_data_name)
    
    train_dataset = PromptDataset(
        prompt_dataset_from_hub[args.train_split],
        tokenizer,
        strategy,
        input_key=args.input_key,
        output_key=args.output_key,
        apply_chat_template=args.apply_chat_template,
    )
    
    eval_dataset = None
    # This logic for handling eval split is copied from the original library for completeness
    if args.eval_split and args.eval_split in prompt_dataset_from_hub:
        eval_dataset = PromptDataset(
            prompt_dataset_from_hub[args.eval_split],
            tokenizer,
            strategy,
            input_key=args.input_key,
            output_key=args.output_key,
            apply_chat_template=args.apply_chat_template,
        )
    return train_dataset, eval_dataset


# --- Step 5: Define our custom Actor ---
class FGCActor(RewardActor):
    def __init__(self, ipc_server, vllm_args, args: FGCArgs):
        super().__init__(args=args, ipc_server=ipc_server, vllm_args=vllm_args)
        logging.info(f"FGCActor: Initializing CALOracle")
        self.cal_oracle = CALOracle(args.cal_model_name, args.cal_few_shot_path)
    def step(self, prompts: List[str], formatted_prompts: List[str], references: List[str] = None) -> List[TrajectoryData]:
        logging.info(f"FGCActor received {len(prompts)} prompts. Launch successful.")
        # Your real research logic will replace this placeholder
        return self.ipc_client.serialize_ipc([])


# --- Step 6: Define our custom Learner ---
class CustomPPOLearner(PPOLearner):
    def prepare_data(self, strategy, tokenizer):
        """Overrides the base method to call our fixed data loading function."""
        # This is the fix: call our own get_datasets_fixed function
        self.prompts_dataset, self.eval_prompts_dataset = get_datasets_fixed(
            self.args, tokenizer, strategy
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, self.args.rollout_batch_size_per_device
        )


# --- Step 7: Define the main run function ---
def run_fgc(args: FGCArgs):
    program, local_resources = get_program(
        args, learner_cls=CustomPPOLearner, actor_cls=FGCActor
    )
    lp.launch(program, launch_type=args.launch_type, local_resources=local_resources)


if __name__ == "__main__":
    args: FGCArgs = get_default_args(FGCArgs)
    args.algo = "FGC_PPO"
    run_fgc(args)