# venv: seq2seq
from datasets import load_from_disk
import os
import torch
import numpy as np
import random
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import evaluate


def get_device_num(model_filename):
    desc_list = model_filename.split("-")
    device_num = -1
    for desc in desc_list:
        if desc[:4] == "cuda":
            device_num = desc[4]
            break
    return device_num


def ent_sen_map(examples):
    sep = " \\n "
    examples["ent_sen"] = [entity + sep + sentence for entity, sentence
                           in zip(examples["entity"], examples["sentence"])]
    return examples


def ent_sum_map(examples):
    sep = " \\n "
    examples["ent_sum"] = [entity + sep + sentence for entity, sentence
                           in zip(examples["entity"], examples["ec_summary"])]
    return examples


def save_best_model(model_save_dir, model, description):
    os.makedirs(model_save_dir, exist_ok=True)
    output_model_file = os.path.join(model_save_dir, description + "-best_checkpoint.pth")
    torch.save(model.state_dict(), output_model_file)


def get_METEOR(res_dataset):
    def get_meteor_map(examples):
        meteors = []
        for refs, pred in zip(examples["decoded_qs_labels"], examples["decoded_q_preds"]):
            refs = [ref.split() for ref in refs]
            pred = pred.split()
            meteor = meteor_score(references=refs, hypothesis=pred)
            meteors.append(meteor)
        examples["meteor_score"] = meteors
        return examples

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    res_dataset = res_dataset.map(get_meteor_map, batched=True)
    METEOR_result = np.mean(res_dataset["meteor_score"])

    return METEOR_result


def get_BLEU(references, predictions):
    references = [[ref.split() for ref in refs] for refs in references]
    predictions = [pred.split() for pred in predictions]

    BLEU_result = {
        "bleu1": corpus_bleu(references, predictions, weights=(1.0, 0.0, 0.0, 0.0)),
        "bleu2": corpus_bleu(references, predictions, weights=(0.5, 0.5, 0.0, 0.0)),
        "bleu3": corpus_bleu(references, predictions, weights=(0.34, 0.33, 0.33, 0.0)),
        "bleu4": corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25)),
    }

    return BLEU_result


def get_ROUGE(references, predictions):
    rouge_score = evaluate.load("rouge")
    ROUGE_result = rouge_score.compute(predictions=predictions, references=references)

    return ROUGE_result


def to_pt(inputs, device):
    return torch.tensor(inputs).long().to(device)


def get_inputs(dataset, tokenizer, max_source_len, max_target_len, input_key, mode):
    def tok_map(examples):
        # tokenize input: ent_para -> input_ids: tensor, attention_mask: tensor
        model_inputs = tokenizer(examples[input_key],
                                 padding="max_length",
                                 truncation=True,
                                 max_length=max_source_len,
                                 return_tensors="pt")

        # tokenize output: questions -> questions_labels: list of tensor
        questions_batch = examples["question"]

        if mode == "train":
            model_inputs["question_labels"] = tokenizer(questions_batch,
                                                        padding="max_length",
                                                        truncation=True,
                                                        max_length=max_target_len,
                                                        return_tensors="pt")["input_ids"]
            try:
                model_inputs["summary_labels"] = tokenizer(examples["summary"],
                                                           padding="max_length",
                                                           truncation=True,
                                                           max_length=max_target_len,
                                                           return_tensors="pt")["input_ids"]
            except:
                pass
        else:  # mode == "eval"
            questions_labels_batch = []
            for questions in questions_batch:
                questions_labels = tokenizer(questions,
                                             padding="max_length",
                                             truncation=True,
                                             max_length=max_target_len,
                                             return_tensors="pt")["input_ids"]
                questions_labels_batch.append(questions_labels)
            model_inputs["questions_labels"] = questions_labels_batch

        # # tokenize supervision: answers -> answers_labels: list of tensor
        # answers_batch = examples["answer"]
        # answers_labels_batch = []
        # for answers in answers_batch:
        #     answers_labels = tokenizer(
        #         answers, padding=True, max_length=max_target_len,
        #         truncation=True, return_tensors="pt"
        #     )["input_ids"]
        #     answers_labels_batch.append(answers_labels)
        # model_inputs["answers_labels"] = answers_labels_batch

        return model_inputs

    def ans_tag_map(examples):
        tokens_tags_list = []
        for answers, input_text, input_ids, attention_mask in zip(
                examples["answer"],
                examples[input_key],
                examples["input_ids"],
                examples["attention_mask"]
        ):
            input_words = input_text.lower().split(" ")

            if mode == "train":
                answers = [answers]

            tokens_tags = []
            for answer in answers:
                # Find answer words in input words
                ans_words = answer.lower().split(" ")
                idx_start = -1  # answer start position
                for idx_input in range(len(input_words)):
                    if ans_words[0] in input_words[idx_input]:
                        ans_candidate = input_words[idx_input: idx_input + len(ans_words)]
                        flag = True
                        for idx_ans in range(len(ans_words)):
                            if ans_words[idx_ans] not in ans_candidate[idx_ans]:
                                flag = False
                        if flag:
                            idx_start = idx_input
                            break
                if idx_start == -1:
                    exit("Answer not in input (words)")
                ans_words_tag = [0] * len(input_words)
                for idx_tag in range(idx_start, idx_start + len(ans_words)):
                    ans_words_tag[idx_tag] = 1

                # Find answer tokens in input tokens
                tokens_tag = []
                tokens_input_ids = []
                tokens_attention_mask = []
                for idx_word, word in enumerate(input_text.split(" ")):
                    tokens = tokenizer(word)
                    if idx_start <= idx_word < idx_start + len(ans_words):
                        tokens_tag += [1] * (len(tokens["input_ids"]) - 1)
                    else:
                        tokens_tag += [0] * (len(tokens["input_ids"]) - 1)
                    tokens_input_ids += tokens["input_ids"][:-1]
                    tokens_attention_mask += tokens["attention_mask"][:-1]
                tokens_tag.append(0)
                tokens_input_ids.append(1)
                tokens_attention_mask.append(1)

                # Pad and truncate
                if len(tokens_input_ids) < len(input_ids):  # pad
                    tokens_tag += [0] * (len(input_ids) - len(tokens_input_ids))
                    # tokens_input_ids += [0] * (len(input_ids) - len(tokens_input_ids))
                    # tokens_attention_mask += [0] * (len(input_ids) - len(tokens_attention_mask))
                else:  # truncate
                    tokens_tag = tokens_tag[:len(input_ids) - 1] + [0]
                    # tokens_input_ids = tokens_input_ids[:len(input_ids) - 1] + [1]
                    # tokens_attention_mask = tokens_attention_mask[:len(attention_mask)]

                # Append the tokens_tag of the current answer
                tokens_tags.append(tokens_tag)

            # Append the tokens_tags of the current example
            if mode == "train":
                tokens_tags_list.append(tokens_tags[0])
            else:
                tokens_tags_list.append(tokens_tags)

        examples["tokens_tags"] = tokens_tags_list

        return examples

    model_inputs = dataset.map(tok_map, batched=True)
    model_inputs = model_inputs.map(ans_tag_map, batched=True)
    model_inputs = model_inputs.remove_columns(dataset.column_names)

    return model_inputs


def get_datasets(datasets_dir):
    def ent_para_map(examples):
        sep = " [SEP] "
        examples["ent_para"] = [entity + sep + paragraph for entity, paragraph
                                in zip(examples["entity"], examples["paragraph"])]
        return examples

    def ans_check_map(examples):
        for answer_list, ent_para in zip(examples["answer"], examples["ent_para"]):
            # print(answer_list)
            # print(ent_para)
            for answer in answer_list:
                if answer.lower() not in ent_para.lower():
                    print(answer)
                    exit()

    datasets = load_from_disk(datasets_dir)
    # print(datasets)  # 42128: flat, 3364: group, 2338: group
    datasets = datasets.map(ent_para_map, batched=True)
    datasets = datasets.map(ans_check_map, batched=True)
    return datasets


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


if __name__ == "__main__":
    pass
