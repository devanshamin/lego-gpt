from pathlib import Path
from argparse import ArgumentParser, Namespace

from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def main(args: Namespace) -> None:

    special_tokens = dict(
        bos_token=f"[{args.bos_token}]",
        eos_token=f"[{args.eos_token}]",
        pad_token=f"[{args.pad_token}]"
    )

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False) # Convert input to a ByteLevel representation
    tokenizer.decoder = ByteLevelDecoder() # Convert tokenized input to original input
    tokenizer.post_processor = TemplateProcessing(
        single=f"[{args.bos_token}] $A", 
        special_tokens=[(f"[{args.bos_token}]", 0)]
    )

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=list(special_tokens.values()),
        show_progress=True,
    )
    tokenizer.train(files=list(Path(args.dataset_dir_path).iterdir()), trainer=trainer) # From file
    #tokenizer.train_from_iterator([dataset], trainer=trainer) # From memory

    tokenizer.model.save(str(args.output_dir))
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        **special_tokens
    )
    pretrained_tokenizer.save_pretrained(str(args.output_dir))

    enc = tokenizer.encode("<Operator> Welcome to the call")
    dec = tokenizer.decode(enc.ids)
    print(f"Token Ids: {enc.ids}")
    print(f"Encoded Tokens : {enc.tokens}")
    print(f"Decoded Tokens: {dec}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--dataset_dir_path", type=str, required=True, help="Directory path containing dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained tokenizer")
    parser.add_argument("--bos_token", type=str, default="BOS", help="Beginning of sentence token")
    parser.add_argument("--eos_token", type=str, default="EOS", help="End of sentence token")
    parser.add_argument("--pad_token", type=str, default="PAD", help="Padding token")

    args = parser.parse_args()
    main(args)
    