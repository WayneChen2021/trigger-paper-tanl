from arguments import DataTrainingArguments
from base_dataset import BaseDataset
import torch

class BatchGenerator():
    def __init__(self, data_args: DataTrainingArguments, model, device, dataset: BaseDataset, batch_size: int):
        self.data_args = data_args
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.dataset = dataset
        self.queue = []
    
    def append(self, input_ids):
        self.queue.append(input_ids)
    
    def generate(self):
        if len(self.queue) == self.batch_size:
            input_ids = torch.stack(self.queue)
            predicts = self.model.generate(
                input_ids,
                max_length=self.data_args.max_output_seq_length_eval,
                num_beams=self.data_args.num_beams if self.data_args.num_beams else 1,
                output_hidden_states=True
            )
        else:
            return None