#coding:utf-8
from typing import List, Tuple, Dict, Any


class T5DataProcess:
    def __init__(self, custom_args, tokenizer, is_train):
        self.max_length = custom_args.max_length
        self.max_source_length = custom_args.max_source_length
        self.max_target_length = custom_args.max_target_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.input_column = custom_args.input_column ##prompt
        self.output_column = custom_args.output_column ##label
        self.training_mode = custom_args.training_mode #training_mode

    def __call__(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        input_text = example.get(self.input_column, "").strip()
        output_text = example.get(self.output_column, "").strip()
        source_text = input_text
        source_inputs = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=True  
        )
        input_ids = source_inputs["input_ids"]

        if not self.is_train:
            return {"input_ids": input_ids,
                    "labels": [-100]
                    }

        # build target text
        target_text = output_text
        
        if self.training_mode == "pretrain":
            target_text = input_text 

        target_inputs = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=False  # no bos/eos
        )
        labels = target_inputs["input_ids"] + [self.tokenizer.eos_token_id]  # add </s> to label 

        # trunct to the max_target_length
        if self.max_target_length > 0 and len(labels) > self.max_target_length:
            labels = labels[:self.max_target_length]

        return {
            "input_ids": input_ids,
            "labels": labels
        }