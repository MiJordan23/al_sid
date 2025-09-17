from dataclasses import dataclass
from typing import List

from transformers import DataCollator
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorWrapper(DataCollatorMixin):

    data_collator: DataCollator = None

    # 模型输入以外的字段，比如预测时希望输出的数据集原始字段，尤其是string类型的，
    # 因为string不能生成tensor，transformers默认的DataCollator不支持
    extra_feature_names: List[str] = None

    def __call__(self, features, return_tensors=None):
        extra_batch = {}
        if self.extra_feature_names:
            extra_batch = {extra_feature_name: [] for extra_feature_name in self.extra_feature_names}
            for feature in features:
                for extra_feature_name in self.extra_feature_names:
                    additional_feature = feature.pop(extra_feature_name, None)
                    extra_batch[extra_feature_name].append(additional_feature)
            for extra_feature_name in self.extra_feature_names:
                additional_feature = extra_batch[extra_feature_name]
                if all(value is None for value in additional_feature):
                    extra_batch.pop(extra_feature_name)

        batch = self.data_collator(features=features, return_tensors=return_tensors)

        if extra_batch:
            batch.update(extra_batch)
        return batch
