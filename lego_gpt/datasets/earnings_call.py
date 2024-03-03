import json
import asyncio
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset


class EarningsCallDatasetLoader:

    categories = {
        "Chief Executive Officer": "CEO",
        "Chief Operating Officer": "COO",
        "Chief Financial Officer": "CFO",
        "Chief Technology Officer": "CTO",
        "Chief Commercial Officer": "CCO",
        "Chief Scientific Officer": "CSO",
        "Chief Medical Officer": "CMO",
        "Chief Marketing Officer": "CMaO",
        "Investor Relations": "IR",
        "Business Development": "BD",
        "Analyst": "Analyst",
        "Managing Director": "Analyst",
    }

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    async def aget_data(self, ticker: str) -> str:

        docs = []
        file_path = self.data_dir / f"{ticker}.json"
        if file_path.exists():
            data = json.loads(file_path.read_text())
            for quarter in data.values():
                if transcript := quarter.get('transcript'):
                    if remarks := transcript.get('Prepared Remarks'):
                        name_to_title = await EarningsCallDatasetLoader.aget_titles(transcript, fill_missing=False)
                        for name, sents in remarks.items():
                            sents = " ".join(sents)
                            if len(name) > 50: # Some text that belongs to previous person's transcript
                                docs[-1] += f" {name} {sents}"
                            else:
                                if name.lower() == "operator":
                                    title = "Operator"
                                else:
                                    title = name_to_title.get(name, "Senior person")
                                docs.append(f"<{title}> {sents}")
        return "\n".join(docs)

    async def afrom_tickers(self, tickers: List[str]):

        results = await asyncio.gather(*(self.aget_data(ticker) for ticker in tickers))
        return "\n".join(results)

    @staticmethod
    async def aget_titles(transcript: Dict, fill_missing=False):

        name_to_title = {}
        if participants := transcript.get("Call Participants"):
            for participant in participants:
                person_name, *other = participant.split("--")
                for title, abbrv in EarningsCallDatasetLoader.categories.items():
                    if title.lower() in participant.lower():
                        name_to_title[person_name.strip()] = abbrv
                        break
                else:
                    if fill_missing and len(other) == 1:
                        name_to_title[person_name.strip()] = "Analyst"
        return name_to_title


class EarningsCallDataset(Dataset, EarningsCallDatasetLoader):

    def __init__(
        self,
        data_dir: Path,
        tickers: List[str],
        tokenizer,
        max_length: int
    ):

        super(EarningsCallDataset, self).__init__(data_dir)
        self.data_dir = data_dir
        self.corpus = asyncio.run(self.afrom_tickers(tickers))
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

        encoding = tokenizer(self.corpus)["input_ids"]
        self._total_tokens = len(encoding)
        max_length += 1
        self.samples = [encoding[i:i + max_length] for i in range(0, len(encoding), max_length)]

    def __repr__(self) -> str:
        
        return f"EarningsCallDataset(samples={len(self.samples):,}, total_tokens={self._total_tokens:,})"

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        input_ids = self.samples[idx]
        input_ids, labels = input_ids[:-1], input_ids[1:]
        if len(input_ids) != self.max_length-1:
            unfilled = (self.max_length - len(input_ids))
            attention_mask = [1] * len(input_ids) + [0] * unfilled
            input_ids += unfilled * [self.pad_token_id]
            labels += unfilled * [self.pad_token_id]
        else:
            attention_mask = [1] * self.max_length

        sample = dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long)
        )
        assert all(v.shape[0] == self.max_length for v in sample.values()), "Shapes are not equal!"
        return sample