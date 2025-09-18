from transformers import Seq2SeqTrainer, GenerationConfig
from torch import nn
from typing import Optional, Union, Dict, Any, List, Tuple
import torch
from .log import logger
import copy
import numpy as np
from utils.predict import PredictWriter, create_predict_writer

class GRSTrainer(Seq2SeqTrainer):

    predict_output: Optional[Dict[str, Any]] = None
    predict_writer: Optional[PredictWriter] = None

    def __init__(self,
                 predict_output: Optional[Dict[str, Any]] = None,
                 **kwargs):

        model = kwargs.get('model')
        args = kwargs.get('args')
        if args.generation_config is not None and isinstance(args.generation_config, dict):
            if hasattr(model, "generation_config") and model.generation_config is not None:
                generation_config: GenerationConfig = copy.deepcopy(model.generation_config)
                generation_config.update(**args.generation_config)
                args.generation_config = generation_config
                logger.info("merge model default and user defined generation_config: %s", str(generation_config))
            else:
                args.generation_config = GenerationConfig.from_dict(args.generation_config)

        super(GRSTrainer, self).__init__(**kwargs)
        self.predict_output = None
        self.predict_writer = None

        if self.args.do_predict:
            self.predict_output = predict_output
            self.predict_writer = create_predict_writer(self.predict_output)
            if self.predict_output is None:
                raise ValueError("predict_output isn't set")
            if not self.args.predict_with_generate or self.args.prediction_loss_only:
                if not hasattr(self.model, "predict_trace_dict"):
                    raise ValueError(f"method {self.model.__class__.__name__}.predict_trace_dict() isn't defined, "
                                     f"may you should set {'predict_with_generate=True' if not self.args.predict_with_generate else 'prediction_loss_only=False'}")
            if not self.args.remove_unused_columns:
                self.accelerator.device_placement = False

    ## Generation Process:
    # 1. Define a writer based on the predict_output setting
    # 2. Write the output to the output.txt file
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        ignored_inputs = {}
        if not self.args.remove_unused_columns:
            self._set_signature_columns_if_needed()
            signature_columns = self._signature_columns
            dataset_column_names = list(inputs.keys())

            ignored_columns = list(set(dataset_column_names) - set(signature_columns))

            ignored_inputs = {ignored_column: inputs.pop(ignored_column) for ignored_column in ignored_columns}
            # inputs = send_to_device(inputs,
            #                         device=self.accelerator.device,
            #                         non_blocking=self.accelerator.non_blocking)

        results = super(GRSTrainer, self).prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        if not self.args.do_predict:
            return results

        trace_dict = None
        if hasattr(self.model, "predict_trace_dict"):
            trace_dict = self.model.predict_trace_dict()

        if trace_dict is None:
            trace_dict = {}
    

        if not trace_dict and self.args.predict_with_generate and not prediction_loss_only:
            output_columns = self.predict_output.get("columns", [])
            loss, generated_tokens, labels = results

            batch_size = self.args.per_device_eval_batch_size
            if "input_ids" in inputs:
                batch_size = inputs["input_ids"].shape[0]
            else:
                for key, value in inputs.items():
                    batch_size = value.shape[0]

            if "_generated_tokens_" in output_columns:
                if batch_size != generated_tokens.shape[0]:
                    trace_dict["_generated_tokens_"] = generated_tokens.reshape(
                        [batch_size, -1, generated_tokens.shape[-1]])
                else:
                    trace_dict["_generated_tokens_"] = generated_tokens

            # 去掉input_ids
            if "_generated_new_tokens_" in output_columns:
                generated_new_tokens = generated_tokens[:, inputs["input_ids"].size(-1):]
                if batch_size != generated_new_tokens.shape[0]:
                    trace_dict["_generated_new_tokens_"] = generated_new_tokens.reshape(
                        [batch_size, -1, generated_new_tokens.shape[-1]])
                else:
                    trace_dict["_generated_new_tokens_"] = generated_new_tokens

            if not output_columns or "_generated_text_" in output_columns:
                generated_text = self.processing_class.batch_decode(generated_tokens, skip_special_tokens=self.predict_output.get("skip_special_tokens", True))
                if batch_size != len(generated_text):
                    generated_text = np.array(generated_text).reshape([batch_size, -1]).tolist()
                trace_dict["_generated_text_"] = generated_text

            # 去掉input_ids
            if not output_columns or "_generated_new_text_" in output_columns:
                generated_new_tokens = generated_tokens[:, inputs["input_ids"].size(-1): ]
                generated_new_text = self.processing_class.batch_decode(generated_new_tokens, skip_special_tokens=self.predict_output.get("skip_special_tokens", True))
                if batch_size != len(generated_new_text):
                    generated_new_text = np.array(generated_new_text).reshape([batch_size, -1]).tolist()
                trace_dict["_generated_new_text_"] = generated_new_text

            if not output_columns:
                trace_dict.update(ignored_inputs)
                trace_dict.update(inputs)
            else:
                for output_column in output_columns:
                    if output_column in ignored_inputs:
                        trace_dict[output_column] = ignored_inputs[output_column]
                    if output_column in inputs:
                        trace_dict[output_column] = inputs[output_column]
        self.predict_writer.write(trace_dict)
        if self.compute_metrics is not None:
            return results
        # 避免gather和占用显存
        return None, None, None
