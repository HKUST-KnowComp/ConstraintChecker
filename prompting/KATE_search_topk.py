import os
import pickle
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding
)


def get_LM_embedding(args, model, tokenizer, title_list):
    preprocess_function = lambda x: tokenizer(x["title"], padding=False, max_length=128, truncation=True)
    raw_datasets = Dataset.from_dict({"title": title_list})
    title_dataset = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    title_dataloader = DataLoader(
        title_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size
    )

    encoding_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(title_dataloader, "Encoding"):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            encoding = model(**batch)
            mask = batch["attention_mask"]
            # TODO the method to get sentence embedding
            # TODO such as mean pooling, pooler_output, or max pooling
            # pooled_encoding = torch.sum(encoding["last_hidden_state"] * mask.unsqueeze(2), dim=1)
            pooled_encoding = torch.sum(encoding["last_hidden_state"] * mask.unsqueeze(2), dim=1) / \
                torch.sum(mask, dim=1, keepdim=True)
            # pooled_encoding = torch.max(encoding["last_hidden_state"], dim=1)[0]
            # pooled_encoding = encoding["last_hidden_state"][:, 0]
            # pooled_encoding = encoding["pooler_output"]
            encoding_list.append(pooled_encoding)
    encoding_list = torch.cat(encoding_list)
    return encoding_list


def cosine_similarity(train_embedding, eval_embedding, top_k):
    normalized_train = train_embedding / torch.norm(train_embedding, dim=1, keepdim=True)
    normalized_eval = eval_embedding / torch.norm(eval_embedding, dim=1, keepdim=True)
    sim = torch.matmul(normalized_eval, normalized_train.transpose(-1, -2))
    index = torch.topk(sim, k=top_k, dim=1).indices
    return index.cpu().tolist()


def jaccard_similarity(train_embedding, eval_embedding, bs=8):
    cur_list = []
    for start in tqdm(range(0, len(eval_embedding), bs), "jaccard sim"):
        end = min(start + bs, len(eval_embedding))
        min_value = torch.minimum(
            eval_embedding[start: end].unsqueeze(1),
            train_embedding.unsqueeze(0)
        )
        min_value = torch.sum(min_value, dim=-1)
        max_value = torch.maximum(
            eval_embedding[start: end].unsqueeze(1),
            train_embedding.unsqueeze(0)
        )
        max_value = torch.sum(max_value, dim=-1)
        sim = min_value / max_value
        cur_list.append(sim.argmax(dim=1))
    index = torch.cat(cur_list)
    return index.cpu().tolist()


def get_textual_format(data, template_map):
    textual_format_list = []
    for row in data:
        head, relation, tail = row["head"], row["relation"], row["tail"]
        textual_format = template_map[relation].format(head, tail)
        textual_format_list.append(textual_format)
    return textual_format_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', type=str, default=None,
                        help="where to take exemplars")
    parser.add_argument('--references_embedding_cache', type=str,
                        default='prompting_data/kate_references_embedding.cache',
                        help="where to store/load references_embedding_cache")
    parser.add_argument('--test_instances', type=str, default=None,
                        help="where the test instances need similar exemplars")
    parser.add_argument('--test_instances_embedding_cache', type=str,
                        default='prompting_data/kate_test_instances_embedding.cache',
                        help="where to store/load references_embedding_cache")
    parser.add_argument("--conversion_rule", type=str, default='',
                        choices=['', '_tq'],
                        help="read readme.md ### Data for more detail")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="where to store selected exemplars")
    parser.add_argument("--top_k", type=int, default=5,
                        help="how many most similar examples for in-context learning, regardless the label")
    parser.add_argument("--take_same_relation", type=int, default=0,
                        help="if 1 ~ True, will take most similar examples from those of the same relation")
    parser.add_argument("--model_name_or_path", type=str,
                        default="sentence-transformers/roberta-large-nli-stsb-mean-tokens",
                        help="path to load the BERT/RoBERTa model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size when running embedding model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()

def random(args):
    args.output_dir = 'prompting_data/{}_random_{}exemplars.csv'.format(
        os.path.basename(args.test_instances).rsplit('.', 1)[0][:3], args.top_k)

    references_df = pd.read_csv(args.references)
    references_df = references_df.sample(n=100000, random_state=2023, ignore_index=True)
    test_instances_df = pd.read_csv(args.test_instances)
    n = len(test_instances_df)
    exemplars_df = pd.DataFrame({'idx': list(range(n))})

    for k in range(args.top_k):
        temp = references_df.sample(n=n, random_state=k, ignore_index=True)
        exemplars_df[f'{k}'] = temp['assertion']
        exemplars_df[f'{k}_label'] = temp['label']

    for k in range(args.top_k):
        temp_tq = references_df.sample(n=n, random_state=args.top_k+k, ignore_index=True)
        exemplars_df[f'{k}_tq'] = temp_tq['assertion_tq']
        exemplars_df[f'{k}_tq_label'] = temp_tq['label']

    for k in range(args.top_k):
        temp = references_df.sample(n=n, random_state=2*args.top_k+k, ignore_index=True)
        exemplars_df[f'{args.top_k+k}'] = temp['assertion']
        exemplars_df[f'{args.top_k+k}_label'] = temp['label']

    del exemplars_df['idx']
    exemplars_df.to_csv(args.output_dir, index=False)

def kate(args):
    template_map = {
        'xWant': '{}, thus, {}',
        'oWant': '{}, thus, {}',
        'xEffect': '{}, thus as an result, {}',
        'oEffect': '{}, thus as an result, {}',
        'xReact': '{}, thus as a result on PersonX\'s emotion, {}',
        'oReact': '{}, thus as a result on PersonY\'s emotion, {}',
        'xAttr': '{}, thus it can be seen about PersonX\'s attribute that {}',
        'xIntent': '{}, thus it can be seen about PersonX\'s intention that {}',
        'xNeed': 'The event {} will not happen unless {}',
        'Causes': 'Because {}, {}',
        'xReason': '{}, because {}',
        'isBefore': 'After {}, {}',
        'isAfter': 'Before {}, {}',
        'HinderedBy': 'The event {} will not happen, if {}',
        'HasSubEvent': 'The event {} includes the event/action that {}',
    }
    args.suffix = args.conversion_rule
    args.output_dir = 'prompting_data/{}_kate_{}exemplars{}_top{}.csv'.format(
            os.path.basename(args.test_instances).rsplit('.', 1)[0][:3],
            'same_relation_' if args.take_same_relation else '',
            args.suffix, args.top_k)
    args.conversion_rule = 'assertion' + args.suffix

    references_df = pd.read_csv(args.references)
    references_df = references_df.sample(n=100000, random_state=2023, ignore_index=True)
    test_instances_df = pd.read_csv(args.test_instances)
    n = len(test_instances_df)
    exemplars_df = pd.DataFrame({'idx': list(range(n))})
    for k in range(args.top_k):
        exemplars_df[f'{k}{args.suffix}'] = ''
        exemplars_df[f'{k}{args.suffix}_label'] = 0
    print(exemplars_df)

    if not (os.path.exists(args.references_embedding_cache) and \
        os.path.exists(args.test_instances_embedding_cache)):
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModel.from_pretrained(args.model_name_or_path, from_tf=False, config=config).to(args.device)

    if not os.path.exists(args.references_embedding_cache):
        references_embedding = get_LM_embedding(args, model, tokenizer, references_df[args.conversion_rule])
        pickle.dump(references_embedding, open(args.references_embedding_cache, 'wb'))
    else:
        references_embedding = pickle.load(open(args.references_embedding_cache, 'rb'))

    if not os.path.exists(args.test_instances_embedding_cache):
        test_instances_embedding = get_LM_embedding(args, model, tokenizer, test_instances_df[args.conversion_rule])
        pickle.dump(test_instances_embedding, open(args.test_instances_embedding_cache, 'wb'))
    else:
        test_instances_embedding = pickle.load(open(args.test_instances_embedding_cache, 'rb'))

    # get similarity
    if args.take_same_relation == 1:
        for r in template_map.keys():
            ref_idx = references_df['relation'] == r
            temp_ref = references_df[ref_idx].reset_index()
            idx = test_instances_df['relation'] == r
            print(test_instances_embedding[idx].shape, references_embedding[ref_idx].shape)
            topk_exemplars = cosine_similarity(
                references_embedding[ref_idx],
                test_instances_embedding[idx],
                args.top_k
            )

            for k in range(args.top_k):
                exemplars_df[f'{k}{args.suffix}'][idx] = \
                    [temp_ref[args.conversion_rule][topk_exemplars[i][k]] for i in range(sum(idx))]
                exemplars_df[f'{k}{args.suffix}_label'][idx] = \
                    [temp_ref['label'][topk_exemplars[i][k]] for i in range(sum(idx))]

    else:
        topk_exemplars = cosine_similarity(
            references_embedding,
            test_instances_embedding,
            args.top_k
        ) # shape = no. test samples x top_k

        for k in range(args.top_k):
            exemplars_df[f'{k}{args.suffix}'] = \
                [references_df[args.conversion_rule][topk_exemplars[i][k]] for i in range(n)]
            exemplars_df[f'{k}{args.suffix}_label'] = \
                [references_df['label'][topk_exemplars[i][k]] for i in range(n)]
    
    # save exemplars
    del exemplars_df['idx']
    exemplars_df.to_csv(args.output_dir, index=False)

    # see some instances
    print(exemplars_df[f'0{args.suffix}'])
    for i in range(5):
        print(f'>>> {args.conversion_rule}:', test_instances_df[args.conversion_rule][i])
        print(f'>>> 0{args.suffix} exemplar:', exemplars_df[f'0{args.suffix}'][i])
        print(f'>>> 0{args.suffix} exemplar label:', exemplars_df[f'0{args.suffix}_label'][i])
        print()


if __name__ == "__main__":
    args = parse_args()
    # random(args)
    kate(args)

'''
CUDA_VISIBLE_DEVICES=6 python prompting/KATE_search_topk.py \
    --references prompting_data/trn_w_assertion.csv \
    --test_instances prompting_data/tst_data_to_eval.csv \
    --output_dir prompting_data \
    --top_k 5

CUDA_VISIBLE_DEVICES=4 python prompting/KATE_search_topk.py \
    --references prompting_data/trn_w_assertion.csv \
    --test_instances prompting_data/tst_data_to_eval.csv \
    --references_embedding_cache prompting_data/kate_references_tq_embedding.cache \
    --test_instances_embedding_cache prompting_data/kate_test_instances_tq_embedding.cache \
    --conversion_rule _tq \
    --output_dir prompting_data \
    --top_k 5 --batch_size 8

CUDA_VISIBLE_DEVICES=6 python prompting/KATE_search_topk.py \
    --references prompting_data/trn_w_assertion.csv \
    --references_embedding_cache prompting_data/kate_references_embedding.cache \
    --test_instances prompting_data/atomic2020_w_negsample.csv \
    --test_instances_embedding_cache prompting_data/kate_atomic_instances_embedding.cache \
    --output_dir prompting_data \
    --top_k 5 --batch_size 8 --take_same_relation 1
'''