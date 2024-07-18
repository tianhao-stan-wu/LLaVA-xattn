import torch
from transformers import LlavaForConditionalGeneration



from typing import Sequence, Dict
from llava.constants import DEFAULT_IMAGE_TOKEN

def preprocess_multimodal(sources: Sequence[str], is_multimodal=True) -> Dict:
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                # if "mmtag" in conversation_lib.default_conversation.version:
                #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

from torch.utils.data import Dataset
import json
from PIL import Image
import os
import copy
import torch

from llava.train.train import preprocess


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning (SFT) of LLaVa."""

    def __init__(self, data_path: str, image_folder, processor):
        super(SFTDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.processor = processor
        self.is_multimodal = True

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            pixel_values = self.processor.image_processor(image, return_tensors='pt').pixel_values[0]

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]),
                                            is_multimodal=self.is_multimodal)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.processor.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.processor.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict

from transformers import AutoProcessor

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)

# update model_max_length of tokenizer
processor.tokenizer.model_max_length = 2048

# from huggingface_hub import hf_hub_download

# filepath = hf_hub_download(repo_id="liuhaotian/LLaVA-Instruct-150K", filename="llava_instruct_80k.json", repo_type="dataset")

train_dataset = SFTDataset(data_path="./test_dataset/llava_instruct_10.json",
                           image_folder="../../sharedir/research/coco2017/train2017",
                           processor=processor)

from transformers import Trainer
from transformers import TrainingArguments
from llava.train.train import DataCollatorForSupervisedDataset

# based on https://github.com/haotian-liu/LLaVA/blob/9a26bd1435b4ac42c282757f2c16d34226575e96/scripts/finetune_qlora.sh
training_args = TrainingArguments(
    # bf16=True,
    output_dir="./checkpoints/finetuning_test",
    num_train_epochs=1,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=4,
    # gradient_accumulation_steps=1,
    # evaluation_strategy="no",
    save_strategy="steps",
    save_steps=50000,
    save_total_limit=1,
    learning_rate=2e-5,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    # logging_steps=1,
    # tf32=True,
    # model_max_length=2048,
    # gradient_checkpointing=True,
    dataloader_num_workers=4,
    # report_to="wandb",
)

data_collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")

trainer = Trainer(model=model,
                  data_collator=data_collator,
                  training_args=training_args)

trainer.train()
