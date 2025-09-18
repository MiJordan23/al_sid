from dataclasses import dataclass
from typing import List

from transformers import DataCollator
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorWrapper(DataCollatorMixin):

    data_collator: DataCollator = None

    # Fields other than model input, such as the original data set fields you want to output during prediction, especially those of string type.
    # Because strings cannot generate tensors, the default DataCollator of transformers does not support this.
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
