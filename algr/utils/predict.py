import json
import os
import time
import traceback
from typing import Union, Dict, Any, List
import torch.distributed as dist

import oss2
import torch
from .log import logger



def parse_predict_output(predict_output) -> dict:
    if isinstance(predict_output, dict):
        return predict_output
    else:
        raise Exception(f"invalid dataset format, should be dict: {predict_output}")

def create_predict_writer(predict_output):
    predict_output = parse_predict_output(predict_output)
    type = predict_output["type"]
    logger.info("create predict writer, type=%s, predict_output=%s", type, json.dumps(predict_output))
    if type == "local":
        return LocalPredictWriter(predict_output)
    else:
        raise Exception("unknown type: {}, predict_output: {}".format(type, predict_output))


class PredictWriter:
    def __init__(self, predict_output):
        self.predict_output = predict_output
        self.max_fail_count = int(predict_output.get("max_fail_count", 100))
        self.log_interval = int(predict_output.get("log_interval", 1000))
        self.write_row_count = 0
        self.last_log_count = 0
        self.write_fail_row_count = 0
        self.write_time = 0
        self.write_batch_count = 0
        self.start_time = time.time()

    def write(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        start_time = time.time()
        try:
            if isinstance(data, list):
                processed = []
                for item in data:
                    processed.append(self._process_item(item))
                self.write_batch(processed)
                self.write_row_count += len(data)
            else:
                # batch from dict of tensors
                data_new = {}
                data_len = {}
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().detach().numpy().tolist()
                    data_new[key] = value
                    data_len[key] = len(value)
                if len(set(data_len.values())) != 1:
                    raise ValueError("mismatch batch size for different values: {}".format(data_len))

                batch_size = list(data_len.values())[0]
                batch = [{k: v[i] for k, v in data_new.items()} for i in range(batch_size)]
                self.write_batch(batch)
                self.write_row_count += batch_size

        except Exception as e:
            self.write_fail_row_count += 1
            logger.error('write failed, fail_count=%s, max_fail_count=%s, error=%s',
                         str(self.write_fail_row_count), str(self.max_fail_count), traceback.format_exc())
            if self.write_fail_row_count > self.max_fail_count:
                raise Exception("fail count {} exceeds threshold {}, last error={}".format(
                    self.write_fail_row_count, self.max_fail_count, repr(e)))

        cur_time = time.time()
        self.write_time += cur_time - start_time
        self.write_batch_count += 1

        if self.write_row_count - self.last_log_count >= self.log_interval:
            self.last_log_count = self.write_row_count
            avg_batch_size = int(self.write_row_count / self.write_batch_count) if self.write_batch_count > 0 else 0
            logger.info("write_row_count=%s, write_batch_count=%s, avg_batch_size=%s, "
                        "write_time=%ss, total_time=%ss",
                        self.write_row_count, self.write_batch_count,
                        avg_batch_size,
                        int(self.write_time), int(cur_time - self.start_time))

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tensors to serializable types."""
        result = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach().numpy().tolist()
            result[k] = v
        return result

    def write_batch(self, batch: List[Dict[str, Any]]):
        """Write a batch of items. To be implemented by subclass."""
        raise NotImplementedError

    def close(self):
        logger.info("close writer, write_count=%s, write_fail_count=%s, "
                    "write_time=%ss, total_time=%ss",
                    self.write_row_count, self.write_fail_row_count,
                    int(self.write_time), int(time.time() - self.start_time))


class LocalPredictWriter(PredictWriter):
    SHARD_FILE_PREFIX = "output_rank"
    FINAL_OUTPUT_FILENAME = "output.jsonl"

    def __init__(self, predict_output):
        super().__init__(predict_output)
        self.output_dir = os.path.expanduser(predict_output["path"])
        self.mode = predict_output.get("mode", "overwrite")
        self.cleanup = predict_output.get("cleanup", True)  # 删除临时分片文件

        # 初始化分布式状态
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.is_master = (self.rank == 0)
        else:
            self.world_size = 1
            self.rank = 0
            self.is_master = True

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 分片文件路径
        self.shard_path = os.path.join(self.output_dir, f"{self.SHARD_FILE_PREFIX}{self.rank:04d}.jsonl")

        # 打开分片文件
        mode = 'w' if self.mode == "overwrite" else 'a'
        self.fp = open(self.shard_path, mode, encoding="utf-8")
        logger.info(f"Rank {self.rank} writing to shard: {self.shard_path}")

    def write_batch(self, batch: List[Dict[str, Any]]):
        try:
            for item in batch:
                self.fp.write(json.dumps(item, ensure_ascii=False) + "\n")
            self.fp.flush()
        except Exception as e:
            logger.error(f"Failed to write batch on rank {self.rank}: {str(e)}")
            raise

    def close(self):
        # 关闭当前分片
        if self.fp:
            self.fp.close()
            self.fp = None

        # 所有进程都等待写完
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # 确保所有分片已写完

        # 主进程负责合并
        if self.is_master:
            final_path = os.path.join(self.output_dir, self.FINAL_OUTPUT_FILENAME)
            if os.path.exists(final_path) and self.mode == "overwrite":
                os.remove(final_path)
            elif os.path.exists(final_path) and self.mode != "append":
                raise FileExistsError(f"File {final_path} already exists. Set 'mode=overwrite' to replace.")

            logger.info(f"Master merging {self.world_size} shard(s) into {final_path}")
            with open(final_path, "a", encoding="utf-8") as final_fp:
                for rank in range(self.world_size):
                    shard_file = os.path.join(self.output_dir, f"{self.SHARD_FILE_PREFIX}{rank:04d}.jsonl")
                    if not os.path.exists(shard_file):
                        logger.warning(f"Missing shard file: {shard_file}")
                        continue
                    with open(shard_file, "r", encoding="utf-8") as f:
                        for line in f:
                            final_fp.write(line)
                    # 清理
                    if self.cleanup:
                        os.remove(shard_file)
                        logger.debug(f"Cleaned up shard: {shard_file}")

            logger.info(f"Merge completed. Final output: {final_path}")

        # 再次 barrier，确保主进程合并完成后再退出
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        super().close()