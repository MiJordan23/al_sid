#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AUTHOR  :selous(selous.zt@alibaba-inc.com)
DATE    :2025.9.2
FUNC    :一体化脚本：扩展 Qwen2.5-0.5B 的 tokenizer（加入 C0-C65536），并进行全量微调
"""
import os
import argparse
import traceback
from typing import Union
import torch.distributed as dist
from datetime import datetime, timedelta
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_dataset
from utils.common import EasyDict
from utils.util import convert_args_value_type
from utils.log import logger
from utils.trainer import GRSTrainer
from utils.data_collator import DataCollatorWrapper

## 根据参数定义Runner不同的配置
class Runner:
    config: EasyDict = None
    def __init__(self, config: EasyDict):

        self.config = config
        if self.config.envs:
            for key, value in self.config.envs.items():
                os.environ[str(key)] = str(value)
        # -------------------------------
        # 1. 解析配置参数 包括custom_args, training_args, 需要结合huggingface的配置
        # -------------------------------
        self.training_args, self.custom_args = self.init_args(self.config)
        self.predict_output = self.config.predict_output
        self.gen_kwargs = None

        self.train_dataset = None
        self.test_dataset = None
        self.preprocess_function = None

        self.trainer = None
        self.is_train = (self.training_args.do_train is True)

    def training_args_class(self):
        return Seq2SeqTrainingArguments
    
    def trainer_class(self):
        return GRSTrainer
    
    def create_compute_loss_func(self):
        return None

    def init_args(self, config: EasyDict):
        training_args_class = self.training_args_class()
        if not issubclass(training_args_class, TrainingArguments):
            raise ValueError(f"invalid training args class, it should be inherited from TrainingArguments, "
                            f"current is {training_args_class.__name__}")

        parser = HfArgumentParser(training_args_class)
        training_args = convert_args_value_type(config.get("training_args", {}), training_args_class)
        ## 需要根据配置构建一个可用的文件名
        training_args["output_dir"] = os.path.expanduser(training_args["output_dir"])

        ## Todo: 根据参数创建子文件夹
        training_args["report_to"] = "none"
        training_args, = parser.parse_dict(training_args, False) #将training_args转化成HF格式的Arguments

        job_types = [training_args.do_train,training_args.do_eval,training_args.do_predict]
        if sum(job_types) != 1 and not (job_types[0] and job_types[1]):
            # train and eval can be set at the same time
            raise ValueError(
                f"one and only one of [do_train, do_eval, do_predict] should be set as True, current is {job_types}")

        custom_args = EasyDict(config.get("custom_args", {}))
        return training_args, custom_args

    def create_preprocess(self):
        if self.config.model_type == 'qwen2_5':
            from models.qwen2_5.data import QwenDataProcess as DataProcess
        elif self.config.model_type == 't5':
            from models.t5.data import T5DataProcess as DataProcess
        preprocess_function = DataProcess(self.custom_args, self.tokenizer, self.is_train)
        return preprocess_function
        
    def create_model(self):
        ## 根据self.config创建model
        checkpoint_path = self.config.load_checkpoint_from
        device_map = self.training_args.device

        if device_map.type == "cpu":
            device_map = torch.device("cpu")
        
        if self.config.model_type == 'qwen2_5':
            from models.qwen2_5.modeling_qwen import Qwen2ForCausalLM
            model_cls = Qwen2ForCausalLM
        elif self.config.model_type == 't5':
            from models.t5.modeling_t5 import T5ForConditionalGeneration
            model_cls = T5ForConditionalGeneration
        else:
            raise ValueError(f"model_type:{self.config.model_type} is not defined yet.")

        if self.custom_args.load_func == "scratch":#读取模型的配置文件, 初始化模型结构
            config = AutoConfig.from_pretrained(checkpoint_path)
            model = model_cls(config)
        elif self.custom_args.load_func == "dense":#只 load dense
            # 1. load 参数, 将embedding层随机初始化
            model = model_cls.from_pretrained(checkpoint_path, device_map=device_map)   
            embed_layer = model.base_model.embed_tokens
            nn.init.normal_(embed_layer.weight, mean=0.0, std=0.02) ##随机初始化
            # 2. 冻结所有层
            for param in model.parameters():
                param.requires_grad = False
            # 3. 解冻嵌入层（假设嵌入层位于 model.base_model.embed_tokens）
            for param in embed_layer.parameters():
                param.requires_grad = True
        else:
            model = model_cls.from_pretrained(checkpoint_path, device_map=device_map)   
            
        #print('Old model: ', model)
        # expand model base new tokenizer
        tokenizer = self.tokenizer
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
        #print('New model: ', model)
        return model

    def create_dataset(self):
        dataset_name = self.config.dataset_name
        data_file = self.config.data_file
        if self.is_train:
            print(f"📊 加载数据集...{data_file}")
            if os.path.isfile(data_file):
                dataset = load_dataset("csv", data_files=data_file, split="train", streaming=self.config.streaming)
            else:
                dataset = load_dataset(dataset_name, data_files=data_file, split="train", streaming=self.config.streaming)
            print("🔄 正在处理数据集...")
            tokenized_train = dataset.map(self.preprocess_function, batched=False, remove_columns=["system", "user", "answer"])
            return tokenized_train, None
        else:
            print("📊 加载数据集...")
            if os.path.isfile(data_file):
                dataset = load_dataset("csv", data_files=data_file, split="all")
            else:
                ## 要根据stage不同读取不同的文件
                dataset = load_dataset(dataset_name, data_files=data_file, split="all")
            print("🔄 正在处理数据集...")
            #tokenized_test = dataset["test"].map(self.preprocess_function, batched=False, remove_columns=["instruction", "input", "output"])
            tokenized_test = dataset.map(self.preprocess_function, batched=False)
            return None, tokenized_test

    def create_data_collator(self):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True)
        if not self.is_train:
            data_collator = DataCollatorWrapper(data_collator=data_collator,
                                                extra_feature_names=["id",
                                                                     self.custom_args.instruction_column,
                                                                     self.custom_args.input_column,
                                                                     self.custom_args.output_column])
        return data_collator


    def create_tokenlizer(self, tokenizer_path) -> Union[AutoTokenizer, None]:
        # add token.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        if self.is_train:
            print('Old tokenizer length: ', len(tokenizer))
            # add special token: [SEP]
            special_tokens_dict = {'additional_special_tokens': tokenizer.all_special_tokens + ['[SEP]']}
            tokenizer.add_special_tokens(special_tokens_dict)
            # add token: C0 ~ C65535 增加一个大的词表覆盖所有的token
            tokenizer.add_tokens(['C%d' % i for i in range(0, 2 * 32768)])
            print('New tokenizer length: ', len(tokenizer))
        return tokenizer

    def create_trainer(self) -> Union[Trainer, None]:
        #创建trainer
        trainer_cls = self.trainer_class()

        kwargs = {
            "model": self.model,
            "args": self.training_args,
            "train_dataset": self.train_dataset,
            "data_collator": self.data_collator,
            "tokenizer": self.tokenizer,
            "predict_output":self.predict_output
        }

        ## 自定义损失函数？
        compute_loss_func = self.create_compute_loss_func()
        if compute_loss_func is not None:
            kwargs["compute_loss_func"] = compute_loss_func

        trainer = trainer_cls(**kwargs)
        return trainer

    def run(self,):
        self.tokenizer = self.create_tokenlizer(tokenizer_path = self.config.load_checkpoint_from)
        self.model = self.create_model()
        self.preprocess_function = self.create_preprocess()
        self.train_dataset, self.test_dataset = self.create_dataset()
        self.data_collator = self.create_data_collator()
        self.trainer = self.create_trainer()
        
        ## 根据配置文件执行训练和测试
        if self.training_args.do_train:
            self.trainer.train()
        elif self.training_args.do_predict:
            params = {"test_dataset": self.test_dataset}
            if self.gen_kwargs is not None:
                params.update(self.gen_kwargs)
            #得在每一个predict_step的过程中写入到文件中
            self.trainer.predict(**params)

    def close(self, success=True):
        # 做一些收尾工作，例如保存模型 看看要不要做？
        if self.trainer and self.trainer.predict_writer:
            return self.trainer.predict_writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    set_seed(42)

    ## python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 runner.py --config=config/t5_base_3layer.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    #支持分布式
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程数量')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--gpu', default=0, type=int, help='')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help="分布式训练中的本地排名。自动由PAI或XDL启动器输入")
    parser.add_argument('--dist_url', default='env://', help='设置分布式训练的URL')
    parser.add_argument('--distributed', action='store_true', help='是否启用分布式训练')
    args = parser.parse_args()

    config = EasyDict(args.config)
    ## output地址和配置文件名绑定
    output_dir = config.training_args.get('output_dir', './logs/')
    config.training_args['output_dir'] = os.path.join(output_dir, os.path.splitext(os.path.basename(args.config))[0])
    if 'predict_output' in config:
        config.predict_output['path'] = config.training_args['output_dir']

    #初始化分布式环境
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=1800))

    ## parameters/读取配置文件地址 ./config/xx.json
    runner = Runner(config)
    success = True
    try:
        runner.run()
        logger.info("runner run success")
    except Exception as e:
        success = False
        logger.error("runner run failed, error=%s", traceback.format_exc())
    finally:
        try:
            runner.close(success=success)
        except:
            logger.warning("runner close failed, ignore, error=%s", traceback.format_exc())
    logger.info("run end")

if __name__ == '__main__':
    main()