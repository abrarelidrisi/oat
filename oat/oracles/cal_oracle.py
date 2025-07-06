# In oat/oracle/cal_oracle.py

import os
import json
import torch
from typing import List, Tuple # <--- CORRECTED: Added 'List' here
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from .base import RewardOracleBase
from oat.types import Metric


class CALOracle(RewardOracleBase):
    def __init__(self, cal_model_name: str, few_shot_path: str, api_key_env: str = "GOOGLE_API_KEY"):
        """
        Initializes the Credit Assignment Oracle using Gemini. This CAL Oracle is responsible for evaluating (preparing the error segment that will be evaluated)

        Args:

        - cal_model_name: The name of the Gemini model to use (e.g., "gemini-1.5-flash").
        - few_shot_path: Path to a JSON file containing few-shot examples.
        - api_key_env: The environment variable name for the API key.

        """

        # Configure API key

        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set. Please export your GOOGLE_API_KEY.")
        genai.configure(api_key = api_key)
        
        # Set Configs for generation
        generation_config = {"temperature": 0.0, "max_output_tokens": 150} 

        # Disable all safety filters to interfere with math/code problems
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        # Configure the Gemini model
        self.model = genai.GenerativeModel(
            model_name=cal_model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Load few shot examples to be used in the prompt later

        print(f"Loading few shot examples from {few_shot_path}")

        with open(few_shot_path, 'r') as f:
            self.few_shot_examples=json.load(f)

        self.system_prompt_text= self._build_system_prompt_text()

        print(f"CAL Oracle: Initialized with Gemini model '{cal_model_name}'.")

        # By the end of this, Oracle is intitalized with Gemini model.


    def _build_system_prompt_text(self, ) -> str:
        """
        Builds the static few-shot prompt with refined instructions. This will be given to the model.
        """

        prompt_parts = [
            "You are a meticulous Credit Assignment LLM (CAL). Your task is to analyze a 'Correct Solution' and an 'Incorrect Solution' to a given 'Question'. You must identify the single sentence in the 'Incorrect Solution' that represents the first logical and substantial divergence from the correct answer",
            "Your output must be ONLY the exact divergent sentence text from the 'Incorrect Solution' and nothing else. Do not add explanations or introductory phrases.",
            "\nHere are some examples of the task:"
        ]


        for example in self.few_shot_examples:
            prompt_parts.extend([
            "\n---\n",
            f"Question: {example['question']}", 
            f"Correct solution: {example['correct_solution']}",
            f"Incorrect solution: {example['incorrect_solution']}",
            f"Error segment: {example['error_segment']}"
            ])

        prompt_parts.append("\n---\n")
        prompt_parts.append("Now, perform this task for the following input. Only provide the 'Error Segment'.")
        return "\n".join(prompt_parts)

    def get_error_segment(self, question: str, correct_solution:str, incorrect_solution: str) -> str:
        """
        Calls the gemini model to get the error segement.
        """

        user_prompt= f"Question: {question}\n Correct Solution: {correct_solution}\n Incorrect Solution: {incorrect_solution}\n Error Segment:"
        full_prompt= [self.system_prompt_text, user_prompt]

        try:
            response = self.model.generate_content(full_prompt)
            # Accessing the text and stripping whitespace
            return response.text.strip()
        except Exception as e:
            print(f"CAL Oracle (Gemini) Error: API call failed: {e}")
            return ""



    def get_reward(
        self,
        inputs: List[str],      # This corresponds to our 'prompts'
        responses: List[str],   # This corresponds to our 'generations'
        references: List[str],
        batch_size: int = 4,    # We can ignore this for now
    ) -> Tuple[torch.Tensor, Metric]:
        """
        Calls the CAL to get error segments and packages them as metadata.
        """
        cal_outputs = []
        # The method signature uses 'inputs' and 'responses', so we use those variable names
        for i in range(len(inputs)):
            error_segment = self.get_error_segment(
                question=inputs[i],
                correct_solution=references[i],
                incorrect_solution=responses[i]
            )
            cal_outputs.append({"error_segment": error_segment})
        
        # The contract requires us to return a Tuple of (Tensor, Metric).
        # We don't have a tensor reward, so we can return an empty tensor.
        # We will pack our important information into the Metric dictionary.
        # A Metric is just a dict.
        dummy_rewards_tensor = torch.zeros(len(inputs))
        metric_data: Metric = {"cal_outputs": cal_outputs}
        
        return dummy_rewards_tensor, metric_data
