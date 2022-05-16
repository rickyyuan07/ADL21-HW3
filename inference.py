from argparse import ArgumentParser, Namespace
from datasets import Dataset, load_dataset
from dataset import T5_dataset
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MT5ForConditionalGeneration, MT5Tokenizer, T5TokenizerFast

from tqdm import trange, tqdm

from accelerate import Accelerator
from tw_rouge import get_rouge

def main(args):
    # prepare acclerator device
    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.to(device)

    dataset = load_dataset('json', data_files={'dev': args.test_file})
    dev_dataset = T5_dataset(tokenizer=tokenizer, dataset=dataset['dev'], mode='dev', 
                            input_length=args.in_max_length, output_length=args.out_max_length)

    val_loader = DataLoader(dataset=dev_dataset, batch_size=args.test_batch_size, shuffle=False)

    # fp-16 accelerate
    model, val_loader = accelerator.prepare(model, val_loader)

    model.load_state_dict(torch.load(args.ckpt_path))
    # validate model
    titles = []
    res = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader):
            texts = data['source_ids'].squeeze().to(device)
            masks = data['source_mask'].to(device)
            title = data['target']
            out = model.generate(
                input_ids=texts, 
                attention_mask=masks, 
                max_length=args.out_max_length, 
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature
            )
            output = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for ott, ttl in zip(output, title):
                if ott:
                    res.append(ott)
                    titles.append(ttl)

    eval_res = get_rouge(res, titles)
    print(100*eval_res['rouge-1']['f'], 100*eval_res['rouge-2']['f'], 100*eval_res['rouge-l']['f'])
    with open(args.inference_file, 'w+') as fout:
        for pred in res:
            print(pred, file=fout)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="ckpt/7_final2.ckpt")
    parser.add_argument("--test_file", type=str, default="data/public.jsonl")
    parser.add_argument("--inference_file", type=str, default="output/inference.txt")

    parser.add_argument("--test_batch_size", type=int, default=4)

    parser.add_argument("--in_max_length", type=int, default=512)
    parser.add_argument("--out_max_length", type=int, default=64)

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)