# venv: seq2seq
# step 2
import torch
import evaluate
from datasets import Dataset
from transformers import T5Tokenizer
import os
from model.ecqg import GenCONE
from utils import set_seed, get_datasets, get_inputs, to_pt, save_best_model, get_BLEU, get_ROUGE, get_METEOR
import random
from tqdm import trange
from argparse import ArgumentParser

import datasets
datasets.logging.disable_progress_bar()
datasets.logging.set_verbosity_error()

rouge_score = evaluate.load("rouge")


@torch.no_grad()
def val_test_res_map(examples, model, model_type, tokenizer, max_source_len, max_target_len, input_key, device):
    batch_inputs = get_inputs(dataset=Dataset.from_dict(examples),
                              tokenizer=tokenizer,
                              max_source_len=max_source_len,
                              max_target_len=max_target_len,
                              input_key=input_key,
                              mode="eval")

    if model_type in ["GenCONE", "T5", "Bart"]:
        qg_sequences = model(to_pt(batch_inputs["input_ids"], device),
                             to_pt(batch_inputs["attention_mask"], device),
                             None, None, mode="eval")

    # DSQ
    if model_type == "DSQ":
        qg_sequences = model(to_pt(batch_inputs["input_ids"], device),
                             to_pt(batch_inputs["attention_mask"], device),
                             None, None, mode="eval")

    examples["decoded_q_preds"] = tokenizer.batch_decode(qg_sequences, skip_special_tokens=True)  # bs

    decoded_qs_labels = []
    for idx_batch in range(len(batch_inputs)):
        question_labels = batch_inputs["questions_labels"][idx_batch]
        decoded_q_labels = tokenizer.batch_decode(question_labels, skip_special_tokens=True)
        decoded_qs_labels.append(decoded_q_labels)
    examples["decoded_qs_labels"] = decoded_qs_labels

    return examples


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="datasets_dir", default="data/SQuAD",
                        help="directory to input datasets")
    parser.add_argument("-o", "--output", dest="save_dir", default="checkpoints",
                        help="directory to save checkpoints")
    parser.add_argument("-s", "--source", dest="max_source_len")
    parser.add_argument("-t", "--target", dest="max_target_len")
    parser.add_argument("-g", "--genlen", dest="max_gen_len")
    parser.add_argument("-e", "--epoch", dest="num_epoch")
    parser.add_argument("-l", "--lr", dest="lr")
    parser.add_argument("--seed", dest="seed", default=42)
    parser.add_argument("--train_bs", dest="train_batch_size")
    parser.add_argument("--val_bs", dest="val_batch_size")
    parser.add_argument("--gas", dest="gradient_accumulation_steps", default=1)
    parser.add_argument("--wd", dest="weight_decay")
    parser.add_argument("--gamma", dest="gamma", default=0.3)
    parser.add_argument("-m", "--mode", dest="model", help="train or test")
    parser.add_argument("--model", dest="model_type", default="GenCONE")
    parser.add_argument("--supervision", dest="supervision", help="Both or CF or QV")
    parser.add_argument("--checkpoint", dest="model_checkpoint", help="t5/bart-base/large")

    args = parser.parse_args()

    input_key = "ent_para"
    device = torch.device("cuda:0")

    model_save_dir = os.path.join(args.save_dir, args.model_type + "-" + args.supervision, args.model_checkpoint)
    result_save_dir = os.path.join(model_save_dir, "result_dataset")

    print("\n... Configuration ...")
    print(device, "  ", args.gamma, " ", args.lr, " ", args.supervision, " ", args.model_checkpoint)

    print("\n... Loading datasets ...")
    set_seed(args.seed)
    datasets = get_datasets(datasets_dir=args.datasets_dir)
    print(datasets)

    print("\n... Initializing model ...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint, model_max_length=args.max_source_len)
    model = GenCONE(args.model_checkpoint, args.gamma, args.max_gen_len, supervision=args.supervision).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.mode == "train":
        print("\n... Beginning training ...")
        best_val_rougeL = 0
        training_loss = 0
        training_step = 0

        model.eval()
        with torch.no_grad():
            val_res_dataset = datasets["val"].map(val_test_res_map,
                                                  fn_kwargs={
                                                      "model": model,
                                                      "model_type": args.model_type,
                                                      "tokenizer": tokenizer,
                                                      "max_source_len": args.max_source_len,
                                                      "max_target_len": args.max_target_len,
                                                      "input_key": input_key,
                                                      "device": device
                                                  },
                                                  batched=True,
                                                  batch_size=args.val_batch_size,
                                                  remove_columns=["ent_para"])
            ROUGE_result = get_ROUGE(references=val_res_dataset["decoded_qs_labels"],
                                     predictions=val_res_dataset["decoded_q_preds"])
            print(ROUGE_result)

        for epoch in range(args.num_epoch):
            order = list(range(len(datasets["train"])))
            random.seed(args.seed + epoch)
            random.shuffle(order)

            model.train()
            num_step_per_epoch_train = (len(datasets["train"]) - 1) // args.train_batch_size + 1
            step_trange_train = trange(num_step_per_epoch_train)
            for step in step_trange_train:
                idx_start = step * args.train_batch_size
                idx_end = min((step + 1) * args.train_batch_size, len(order))
                idx_examples = order[idx_start: idx_end]
                batch_dataset = datasets["train"].select(idx_examples)

                batch_inputs = get_inputs(dataset=batch_dataset,
                                          tokenizer=tokenizer,
                                          max_source_len=args.max_source_len,
                                          max_target_len=args.max_target_len,
                                          input_key=input_key,
                                          mode="train")

                if args.model_type in ["GenCONE", "T5", "Bart"]:
                    qg_sequences, loss = model(to_pt(batch_inputs["input_ids"], device),
                                               to_pt(batch_inputs["attention_mask"], device),
                                               to_pt(batch_inputs["question_labels"], device),
                                               to_pt(batch_inputs["tokens_tags"], device),
                                               mode="train")

                # DSQ
                if args.model_type == "DSQ":
                    qg_sequences, loss = model(to_pt(batch_inputs["input_ids"], device),
                                               to_pt(batch_inputs["attention_mask"], device),
                                               to_pt(batch_inputs["question_labels"], device),
                                               to_pt(batch_inputs["summary_labels"], device),
                                               mode="train")

                training_loss += loss.item()
                training_step += 1
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_show = ' Epoch:' + str(epoch) + " training loss:" + str(round(training_loss / training_step, 4))
                step_trange_train.set_postfix_str(loss_show)

            model.eval()
            with torch.no_grad():
                val_res_dataset = datasets["val"].map(val_test_res_map,
                                                      fn_kwargs={
                                                          "model": model,
                                                          "model_type": args.model_type,
                                                          "tokenizer": tokenizer,
                                                          "max_source_len": args.max_source_len,
                                                          "max_target_len": args.max_target_len,
                                                          "input_key": input_key,
                                                          "device": device
                                                      },
                                                      batched=True,
                                                      batch_size=args.val_batch_size,
                                                      remove_columns=["ent_para"])
                ROUGE_result = get_ROUGE(references=val_res_dataset["decoded_qs_labels"],
                                         predictions=val_res_dataset["decoded_q_preds"])
                print(ROUGE_result)

    if args.mode == "test":
        print("\n... Loading model ...")
        model_file_path = os.path.join(model_save_dir, "")
        print(model_file_path)
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        model.to(device)

        print("\n... Beginning testing ...")
        model.eval()
        with torch.no_grad():
            test_res_dataset = datasets["test"].map(val_test_res_map,
                                                    fn_kwargs={
                                                        "model": model,
                                                        "model_type": args.model_type,
                                                        "tokenizer": tokenizer,
                                                        "max_source_len": args.max_source_len,
                                                        "max_target_len": args.max_target_len,
                                                        "input_key": input_key,
                                                        "device": device
                                                    },
                                                    batched=True,
                                                    batch_size=args.val_batch_size,
                                                    remove_columns=["ent_para"])
            print("Test result save to: ", result_save_dir)
            test_res_dataset.save_to_disk(result_save_dir)
            print(test_res_dataset)

            ROUGE_result = get_ROUGE(references=test_res_dataset["decoded_qs_labels"],
                                     predictions=test_res_dataset["decoded_q_preds"])
            print(ROUGE_result)

            BLEU_result = get_BLEU(references=test_res_dataset["decoded_qs_labels"],
                                   predictions=test_res_dataset["decoded_q_preds"])
            print(BLEU_result)

            METEOR_result = get_METEOR(test_res_dataset)
            print("METEOR score: ", METEOR_result)
