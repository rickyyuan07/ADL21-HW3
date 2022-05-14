from argparse import ArgumentParser, Namespace
from datasets import Dataset, load_dataset
from dataset import T5_dataset
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MT5ForConditionalGeneration, MT5Tokenizer, T5TokenizerFast

import json
from tqdm import trange, tqdm
from tw_rouge import get_rouge

from accelerate import Accelerator

def main(args):
    # prepare acclerator device
    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    model.to(device)

    dataset = load_dataset('json', data_files={'train': args.train_file, 'dev': args.test_file})
    train_dataset = T5_dataset(tokenizer=tokenizer, dataset=dataset['train'], mode='train', 
                            input_length=args.in_max_length, output_length=args.out_max_length)
    dev_dataset = T5_dataset(tokenizer=tokenizer, dataset=dataset['dev'], mode='dev', 
                            input_length=args.in_max_length, output_length=args.out_max_length)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dev_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # fp-16 training
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    best_f = 0
    step = 0
    # model.load_state_dict(torch.load('./ckpt/9_3.ckpt'))
    for epoch in trange(args.num_epoch):
        # train model
        model.train()
        train_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            step += 1
            text, title = data['source_ids'].to(device), data['target_ids'].to(device)
            mask, title_mask = data['source_mask'].to(device), data['target_mask'].to(device)

            loss = model(input_ids=text, labels=title, attention_mask=mask, decoder_attention_mask=title_mask).loss
            loss /= args.accu_step
            # loss.backward()
            accelerator.backward(loss)
            train_loss += loss.item()
            if step % args.accu_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            
        print(f"\nTraining loss: {train_loss / len(train_loader)}\n", flush=True)
        train_loss = 0

        torch.save(model.state_dict(), f"./ckpt/{epoch}_final.ckpt")

        # validate model
        titles = []
        res = []
        model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                text = data['source_ids'].squeeze().to(device)
                mask = data['source_mask'].to(device)
                title = data['target']
                out = model.generate(input_ids=text, attention_mask=mask, max_length=args.out_max_length, num_beams=2)
                output = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for ott, ttl in zip(output, title):
                    if ott:
                        res.append(ott)
                        titles.append(ttl)

            print("len =", len(res), len(dev_dataset))
            if len(res) >= 10:
                print(res[:10], titles[:10], sep='\n===============\n')
                eval_res = get_rouge(res, titles)
                if eval_res["rouge-l"]["f"] >= best_f:
                    print("Update best model")
                    best_f = eval_res["rouge-l"]["f"]
                    torch.save(model.state_dict(), args.ckpt_path)
                with open(f"./valid_{epoch}_final.json", "w") as fp:
                    json.dump(eval_res, fp, indent=4)
                with open("./log.txt", 'a') as fp:
                    print(f"step: {step}, eval_res", flush=True, file=fp)

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/best_final.ckpt")
    parser.add_argument("--train_file", type=str, default="./data/train.jsonl")
    parser.add_argument("--test_file", type=str, default="./data/public.jsonl")

    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    parser.add_argument("--accu_step", type=int, default=1)
    # parser.add_argument("--log_step", type=int, default=500)
    parser.add_argument("--in_max_length", type=int, default=512)
    parser.add_argument("--out_max_length", type=int, default=64)

    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--generate_method", type=str, default="mix")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)