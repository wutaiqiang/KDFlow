from typing import Callable, Optional, Dict, List, Any

import torch
from torch.utils.data import Dataset

from kdflow.models.utils import TokenizerCompareResult
from kdflow.utils.utils import zero_pad_sequences


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        student_tokenizer: Callable,
        max_length: int,
        strategy,
        tokenizer_info: Optional[TokenizerCompareResult] = None,
        teacher_tokenizer: Optional[Callable] = None,
        max_data_num: int = -1,
        input_template: Optional[str] = None,
        num_processors: int = 8,
    ) -> None:
        super().__init__()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.tokenizer_info = tokenizer_info or TokenizerCompareResult()
        self.template_identical = self.tokenizer_info.template_identical
        self.vocab_identical = self.tokenizer_info.vocab_identical

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.output_key = getattr(self.strategy.args.data, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)

        # Truncate dataset if max_data_num is specified
        if max_data_num > 0 and max_data_num < len(dataset):
            strategy.log(f"Truncating dataset from {len(dataset)} to {max_data_num}")
            dataset = dataset.select(range(max_data_num))

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
            load_from_cache_file=False,
            desc="Processing and tokenizing data",
        )
        strategy.log(f"before filter: {len(processed_dataset)}")
        self.processed_dataset = processed_dataset.filter(
            lambda x: x["stu_prompt"] is not None,
            num_proc=num_processors
        )
        strategy.log(f"after filter: {len(self.processed_dataset)}")
        self._print_sample()

    def _print_sample(self) -> None:
        """Print sample data for debugging."""
        self.strategy.log("Student input ids:")
        self.strategy.log(self.student_tokenizer.decode(self.processed_dataset[0]["stu_input_ids"]))
        if not self.template_identical or not self.vocab_identical:
            self.strategy.log("Teacher input ids:")
            self.strategy.log(self.teacher_tokenizer.decode(self.processed_dataset[0]["tea_input_ids"]))

    def _tokenize_and_build(
        self, 
        prompt_str: str, 
        resp_str: str, 
        tokenizer: Callable, 
        prefix: str
    ) -> Dict[str, Any]:
        """Tokenize prompt and response, build result dict."""
        prompt = tokenizer(prompt_str, add_special_tokens=False)
        prompt_len = len(prompt["input_ids"])
        
        if not resp_str.endswith(tokenizer.eos_token):
            resp_str += " " + tokenizer.eos_token
        resp = tokenizer(resp_str, add_special_tokens=False)
        
        return {
            f"{prefix}_prompt": prompt_str,
            f"{prefix}_response": resp_str,
            f"{prefix}_input_ids": prompt["input_ids"] + resp["input_ids"],
            f"{prefix}_attn_mask": prompt["attention_mask"] + resp["attention_mask"],
            f"{prefix}_loss_mask": [False] * prompt_len + [True] * len(resp["input_ids"]),
        }

    def process_data(self, data: Dict) -> Dict[str, Any]:
        """Process a single data sample."""
        enable_thinking = self.strategy.args.model.enable_thinking
        
        # Process student data
        stu_prompt_str, stu_resp_str = self.preprocess_data(
            data,
            self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=self.student_tokenizer.apply_chat_template,
            enable_thinking=enable_thinking
        )
        result = self._tokenize_and_build(stu_prompt_str, stu_resp_str, self.student_tokenizer, "stu")
        
        # Filter by max_length
        if len(result["stu_input_ids"]) > self.max_length:
            return {
                "stu_prompt": None, "stu_response": None,
                "stu_input_ids": [], "stu_attn_mask": [], "stu_loss_mask": [],
                "tea_prompt": None, "tea_response": None,
                "tea_input_ids": [], "tea_attn_mask": [], "tea_loss_mask": [],
            }
        
        # Process teacher data if needed
        if not self.template_identical or not self.vocab_identical:
            assert self.teacher_tokenizer is not None, "teacher_tokenizer cannot be None."
            tea_prompt_str, tea_resp_str = self.preprocess_data(
                data,
                self.input_template,
                self.input_key,
                self.output_key,
                apply_chat_template=self.teacher_tokenizer.apply_chat_template,
                enable_thinking=enable_thinking
            )
            result.update(self._tokenize_and_build(tea_prompt_str, tea_resp_str, self.teacher_tokenizer, "tea"))
        else:
            result["tea_prompt"] = result["stu_prompt"]
            result["tea_response"] = result["stu_response"]
            result["tea_input_ids"] = result["stu_input_ids"]
            result["tea_attn_mask"] = result["stu_attn_mask"]
            result["tea_loss_mask"] = result["stu_loss_mask"]
        
        return result
    
    def preprocess_data(
        self,
        data: Dict, 
        input_template: Optional[str] = None, 
        input_key: str = "input", 
        output_key: Optional[str] = None, 
        apply_chat_template: Optional[Callable] = None, 
        enable_thinking: Optional[bool] = None
    ) -> tuple:
        """Preprocess data to extract prompt and response."""
        if not apply_chat_template:
            prompt = data[input_key]
            if input_template:
                prompt = input_template.format(prompt)
            assert output_key is not None, "output_key cannot be None."
            return prompt, data[output_key]
        
        # Apply chat template
        if output_key:
            prompt_msg = data[input_key]
            resp_msg = data[output_key]

            if isinstance(prompt_msg, str) and isinstance(resp_msg, str):
                prompt_msg = [{"role": "user", "content": prompt_msg}]
                resp_msg = [{"role": "assistant", "content": resp_msg}]
            
            prompt = apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            full_text = apply_chat_template(prompt_msg + resp_msg, tokenize=False, enable_thinking=enable_thinking)
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            full_text = apply_chat_template(data[input_key], tokenize=False, enable_thinking=enable_thinking)
        
        response = full_text[len(prompt):].lstrip("<think>\n\n</think>\n\n").rstrip()
        return prompt, response

    def __len__(self) -> int:
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.processed_dataset[idx]

    def _pad_sequence(self, items: List[Dict], key: str, pad_value: int = 0) -> torch.Tensor:
        """Helper to pad sequences."""
        return zero_pad_sequences(
            [torch.LongTensor(item[key]) for item in items],
            "right", pad_value
        )

    def collate_fn(self, item_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        batch = {
            # "stu_prompts": [item["stu_prompt"] for item in item_list],
            "stu_input_ids": self._pad_sequence(item_list, "stu_input_ids", self.student_tokenizer.pad_token_id),
            "stu_attn_mask": self._pad_sequence(item_list, "stu_attn_mask"),
            "stu_loss_mask": self._pad_sequence(item_list, "stu_loss_mask", False).roll(shifts=-1, dims=1),
        }
        
        if item_list[0].get("tea_input_ids") is not None:
            tea_pad_id = self.teacher_tokenizer.pad_token_id if self.teacher_tokenizer else self.student_tokenizer.pad_token_id
            batch.update({
                "tea_input_ids": self._pad_sequence(item_list, "tea_input_ids", tea_pad_id),
                "tea_attn_mask": self._pad_sequence(item_list, "tea_attn_mask"),
                "tea_loss_mask": self._pad_sequence(item_list, "tea_loss_mask", False).roll(shifts=-1, dims=1),
            })
            
        return batch
