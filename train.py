import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from datasets import Dataset, Audio, Value

import os

# Define the directory containing your text files
text_files_directory = './extracted_data/svarah/audio/'

# Define the output file paths for scp_entries and txt_entries
scp_entries_file = 'scp_entries.txt'
txt_entries_file = 'txt_entries.txt'

# Initialize lists to store entries
scp_entries = []
txt_entries = []

# Set the limit for the number of instances you want to add
limit_instances = 3000

# Iterate through text files in the directory
for text_file_name in os.listdir(text_files_directory):
    if text_file_name.endswith('.txt'):
        text_file_path = os.path.join(text_files_directory, text_file_name)
        with open(text_file_path, 'r') as text_content:
            for line in text_content:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    audio_path, audio_id, transcription = parts
                    audio_path = os.path.join(audio_path)  # You may need to adjust this based on your folder structure

                    # Create an entry for scp_entries
                    scp_entry = f"{audio_id} ./extracted_data/svarah/audio/{audio_path}\n"
                    scp_entries.append(scp_entry)

                    # Create an entry for txt_entries
                    txt_entry = f"{audio_id} {transcription}\n"
                    txt_entries.append(txt_entry)

                if len(scp_entries) >= limit_instances:
                    # If the limit is reached, break out of the loop
                    break

    if len(scp_entries) >= limit_instances:
        # If the limit is reached, break out of the loop
        break

# Write the scp_entries and txt_entries to their respective files
with open(scp_entries_file, 'w') as scp_file:
    scp_file.writelines(scp_entries)

with open(txt_entries_file, 'w') as txt_file:
    txt_file.writelines(txt_entries)

print(f'SCP entries have been written to {scp_entries_file}')
print(f'Text entries have been written to {txt_entries_file}')

scp_entries = open('./extracted_data/scp_entries.txt', 'r').readlines()
txt_entries = open("/extracted_data/txt_entries.txt", 'r').readlines()

if len(scp_entries) == len(txt_entries):
    audio_dataset = Dataset.from_dict({"audio": [audio_path.split()[1].strip() for audio_path in scp_entries],
                    "sentence": [' '.join(text_line.split()[1:]).strip() for text_line in txt_entries]})

    audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
    audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
    audio_dataset.save_to_disk('./extracted_data/svarah/audio')
    print(audio_dataset)
    print('Data preparation done')

else:
    print('Please re-check the audio_paths and text files. They seem to have a mismatch in terms of the number of entries. Both these files should be carrying the same number of lines.')

def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

audio_dataset = audio_dataset.map(prepare_dataset)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/extracted_data/whisper-small-indian-accent", 
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=audio_dataset,
    eval_dataset=audio_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

trainer.save_model("./saved_model.pt")