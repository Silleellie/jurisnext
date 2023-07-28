import os
import pickle
import random

import numpy as np
import torch

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5TokenizerFast, Adafactor

from tqdm import tqdm

RANDOM_STATE = 42


class CustomData(Dataset):

    def __init__(self, dataset: pd.DataFrame):
        # group by id and convert to dict having id as key and list of tuples where the first element of a tuple
        # contains the section and the second contains the section text

        dataset = dataset.fillna("")
        self.dataset = dataset.groupby("CELEX ID")[['Level 1', 'Level 2']].agg(list)
        self.dataset = self.dataset.T.to_dict('list')
        self.dataset = {case_id: list(zip(value[0], value[1])) for case_id, value in self.dataset.items()}

        self.all_cases = list(self.dataset.keys())

        self.tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small",
                                                         cache_dir='cache',
                                                         max_length=512)

    def __getitem__(self, idx):
        case_id = self.all_cases[idx]
        sequence = self.dataset[case_id]

        # a sequence has at least 1 data point, but it can have more depending on the length of the sequence
        # We must ensure that at least an element can be used as test set
        # in the "sliding_training_size" is included the target item
        sliding_training_size = random.randint(1, len(sequence) - 1)

        # TO DO: consider starting always from the initial paragraph, rather than varying the starting point of the seq
        start_index = random.randint(0, len(sequence) - sliding_training_size - 1)
        end_index = start_index + sliding_training_size

        train_seq = sequence[start_index:end_index]
        target_paragraph = sequence[end_index]  # the successor in the sequence is the target to predict

        return {"train_seq": train_seq, "target_paragraph": target_paragraph}

    def collate_fn_fast(self, batch):
        batch_entry = {}

        target = [entry["target_paragraph"][0] for entry in batch]

        title_sequence = []
        text_sequence = []

        for entry in batch:
            title_sequence.append(", ".join(label for label, _ in entry['train_seq']))
            text_sequence.append(entry['train_seq'][-1][1])

        # title_sequence = [", ".join(label for label, _ in entry['train_seq']) for entry in batch]
        # text_sequence = [entry['train_seq'][-1][1] for entry in batch]

        prompt = ["Predict the next element in the following sequence:\n"
                  f"{entry_sequence}\n\n"
                  "Context:\n"
                  f"{entry_context}" for entry_sequence, entry_context in zip(title_sequence, text_sequence)]

        # TO DO: consider all previous labels and text of the immediate precedent sequence to classify the next label
        output = self.tokenizer(text=prompt,
                                text_target=target,
                                padding=True,
                                truncation=True)

        input_ids = torch.LongTensor(output.input_ids)
        attention_mask = torch.FloatTensor(output.attention_mask)
        target_ids = torch.LongTensor(output.labels)

        target_ids[target_ids == 0] = -100

        batch_entry['input_ids'] = input_ids
        batch_entry['attention_mask'] = attention_mask
        batch_entry['target_ids'] = target_ids
        batch_entry['target_text'] = target

        return batch_entry

    def __len__(self):
        return len(self.all_cases)


def seed_everything(seed: int):
    """
    Function which fixes the random state of each library used by this repository with the seed
    specified when invoking `pipeline.py`

    Returns:
        The integer random state set via command line argument

    """

    # seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print(f"Random seed set as {seed}")

    return seed


if __name__ == "__main__":

    seed_everything(RANDOM_STATE)

    # PARAMETERS

    device = "cuda:0"
    max_epochs = 5
    batch_size = 4
    num_workers = 4

    with open("multi_granular_representation.pkl", "rb") as f:
        data: pd.DataFrame = pickle.load(f)

    unique_case_ids = pd.unique(data["CELEX ID"])

    # TO IMPROVE: labels could be in the test set but not in the train set (need to cover this)
    unique_labels = {x: f'{i}' for i, x in enumerate(pd.unique(data["Level 1"]))}
    data["Level 1"] = data["Level 1"].map(unique_labels)

    train_ids, test_ids = train_test_split(
        unique_case_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.1,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_set = data[data["CELEX ID"].isin(train_ids)]
    validation_set = data[data["CELEX ID"].isin(val_ids)]
    test_set = data[data["CELEX ID"].isin(test_ids)]

    sections = pd.unique(data["Level 1"]).tolist()

    labels = list(pd.unique(data['Level 1']))
    num_labels = len(labels)

    train_dataset = CustomData(train_set)
    val_dataset = CustomData(validation_set)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn_fast,
        num_workers=num_workers,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        collate_fn=val_dataset.collate_fn_fast,
        num_workers=num_workers,
        drop_last=False
    )

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir='cache').to(device)
    optimizer = Adafactor(
        list(model.parameters()),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    model.train()

    for epoch in range(0, max_epochs):

        pbar = tqdm(train_loader)

        train_loss = 0

        model.train()

        for i, batch in enumerate(pbar):

            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lm_labels = batch["target_ids"].to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=lm_labels,
            )

            loss = output.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i % 10) == 0:
                pbar.set_description(f"Epoch {epoch}, Loss -> {train_loss / (i + 1)}")

        pbar.close()

        print("VALIDATION")
        model.eval()

        pbar_val = tqdm(val_loader)

        val_loss = 0
        matches = 0

        for i, batch in enumerate(pbar_val):

            with torch.no_grad():

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                lm_labels = batch["target_ids"].to(device)
                target_text = batch["target_text"]

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=lm_labels,
                )

                loss = output.loss
                val_loss += loss.item()

                num_return_sentences = 1

                beam_outputs = model.generate(
                    batch['input_ids'].to('cuda'),
                    max_new_tokens=50,
                    num_beams=20,
                    no_repeat_ngram_size=0,
                    num_return_sequences=num_return_sentences,  # top-10 recommendation
                    early_stopping=True
                )
                generated_sents = val_dataset.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

                matches += sum(
                    [0 if truth not in generated_sents[i * num_return_sentences:(i + 1) * num_return_sentences] else 1
                     for i, truth in enumerate(target_text)])

                if (i % 10) == 0:
                    pbar_val.set_description(f"Epoch {epoch}, Val Loss -> {val_loss / (i + 1)}")

        print(matches / len(val_dataset))

        pbar_val.close()

    model.eval()
