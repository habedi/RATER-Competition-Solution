import argparse
import gc
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer

TEST_DIR = 'data/competition_data/test'
#TEST_DIR = 'data/competition_data/mini_test'

print(f"Test data directory: {TEST_DIR}")

ap = argparse.ArgumentParser()
ap.add_argument('--model_paths', nargs='+', required=True)
ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--max_len", type=int, required=True)
ap.add_argument("--batch_size", type=int, required=True)
ap.add_argument("--reuse", action='store_true', default=False,
                help="Whether to reuse tokenized data from previous runs")
args = ap.parse_args()

# Define paths for saving and loading tokenized data
TOKENIZED_TEST_DATA_PATH = f'data/processed/{Path(args.save_name).stem}_tokenized_test_data.pkl'

# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

if "longformerwithlstm" in args.save_name:
    from models.longformerwithbilstmhead import LongformerForTokenClassificationwithbiLSTM

if "debertawithlstm" in args.save_name:
    from models.deberta_lstm import DebertaForTokenClassificationwithbiLSTM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Get the number of available CPU cores
NUM_CORES = mp.cpu_count()

BATCH_SIZE = args.batch_size
MAX_SEQ_LENGTH = args.max_len
PRETRAINED_MODEL_PATHS = args.model_paths

if "debertal_chris" in args.save_name:
    print('==> using -1 in offset mapping...')

if ("v3" in args.save_name) | ("v2" in args.save_name):
    print('==> using -1 in offset mapping...')

AGG_FUNC = np.mean
print('==> using span token mean...')

MIN_TOKENS = {
    "Lead": 32,
    "Position": 5,
    "Evidence": 35,
    "Claim": 7,
    "Concluding Statement": 6,
    "Counterclaim": 6,
    "Rebuttal": 6
}

if "chris" not in args.save_name:
    ner_labels = {'O': 0,
                  'B-Lead': 1,
                  'I-Lead': 2,
                  'B-Position': 3,
                  'I-Position': 4,
                  'B-Evidence': 5,
                  'I-Evidence': 6,
                  'B-Claim': 7,
                  'I-Claim': 8,
                  'B-Concluding Statement': 9,
                  'I-Concluding Statement': 10,
                  'B-Counterclaim': 11,
                  'I-Counterclaim': 12,
                  'B-Rebuttal': 13,
                  'I-Rebuttal': 14}
else:
    print("==> Using Chris BIO")
    ner_labels = {'O': 14,
                  'B-Lead': 0,
                  'I-Lead': 1,
                  'B-Position': 2,
                  'I-Position': 3,
                  'B-Evidence': 4,
                  'I-Evidence': 5,
                  'B-Claim': 6,
                  'I-Claim': 7,
                  'B-Concluding Statement': 8,
                  'I-Concluding Statement': 9,
                  'B-Counterclaim': 10,
                  'I-Counterclaim': 11,
                  'B-Rebuttal': 12,
                  'I-Rebuttal': 13}

inverted_ner_labels = dict((v, k) for k, v in ner_labels.items())
inverted_ner_labels[-100] = 'Special Token'

test_files = os.listdir(TEST_DIR)


# accepts file path, returns tuple of (file_ID, txt split, NER labels)
def generate_text_for_file(input_filename):
    curr_id = input_filename.split('.')[0]
    with open(os.path.join(TEST_DIR, input_filename)) as f:
        curr_txt = f.read()
    return curr_id, curr_txt


with mp.Pool(NUM_CORES) as p:
    ner_test_rows = p.map(generate_text_for_file, test_files)

if ("v3" in args.save_name) or ("v2" in args.save_name):
    from transformers import DebertaTokenizerFast

    tokenizer = DebertaTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATHS[0])
else:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATHS[0])

ner_test_rows = sorted(ner_test_rows,
                       key=lambda x: len(tokenizer(x[1], max_length=MAX_SEQ_LENGTH, truncation=True)['input_ids']))


# tokenize and store word ids
def tokenize_with_word_ids(ner_raw_data):
    # ner_raw_data is shaped (num_examples, 3) where cols are (ID, words, word-level labels)
    tokenized_inputs = tokenizer([x[1] for x in ner_raw_data],
                                 max_length=MAX_SEQ_LENGTH,
                                 return_offsets_mapping=True,
                                 truncation=True, padding=False)

    tokenized_inputs['id'] = [x[0] for x in ner_raw_data]
    tokenized_inputs['offset_mapping'] = [tokenized_inputs['offset_mapping'][i] for i in range(len(ner_raw_data))]

    return tokenized_inputs


# tokenized_all = tokenize_with_word_ids(ner_test_rows)

# Load or tokenize and store word ids
if os.path.exists(TOKENIZED_TEST_DATA_PATH) and args.reuse:
    with open(TOKENIZED_TEST_DATA_PATH, 'rb') as f:
        tokenized_all = pickle.load(f)
        print("Loaded the tokenized test data from previous runs.")
else:
    tokenized_all = tokenize_with_word_ids(ner_test_rows)
    if args.reuse:
        with open(TOKENIZED_TEST_DATA_PATH, 'wb') as f:
            pickle.dump(tokenized_all, f)
            print("Saved the tokenized test data.")


# import sys
# sys.exit(-100)

class NERDataset(Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __getitem__(self, index):
        return {k: self.input_dict[k][index] for k in self.input_dict.keys() if k not in {'id', 'offset_mapping'}}

    def get_filename(self, index):
        return self.input_dict['id'][index]

    def get_offset(self, index):
        return self.input_dict['offset_mapping'][index]

    def __len__(self):
        return len(self.input_dict['input_ids'])


test_dataset = NERDataset(tokenized_all)

soft_predictions = None
hfargs = TrainingArguments(output_dir='tmp/trainer_tmp',
                           log_level='warning',
                           per_device_eval_batch_size=BATCH_SIZE)

# Check if GPU is available, if not, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for idx, curr_path in enumerate(PRETRAINED_MODEL_PATHS):

    # print(f"Model name: {args.save_name}")

    if "longformerwithlstm" in args.save_name:
        model = LongformerForTokenClassificationwithbiLSTM.from_pretrained(curr_path)
    elif "debertawithlstm" in args.save_name:
        model = DebertaForTokenClassificationwithbiLSTM.from_pretrained(curr_path)
    elif "lsg" in args.save_name or "debertal_chris" in args.save_name or "debertal" in args.save_name or "debertaxl" in args.save_name:
        model = AutoModelForTokenClassification.from_pretrained(curr_path, trust_remote_code=True)
    else:
        model = AutoModelForTokenClassification.from_pretrained(curr_path, trust_remote_code=True,
                                                                torch_dtype=torch.float16)

    trainer = Trainer(model.to(device),
                      hfargs,
                      tokenizer=tokenizer)

    curr_preds, _, _ = trainer.predict(test_dataset)
    curr_preds = curr_preds.astype(np.float16)
    curr_preds = softmax(curr_preds, -1)

    if soft_predictions is not None:
        soft_predictions = soft_predictions + curr_preds
    else:
        soft_predictions = curr_preds

    del model, trainer, curr_preds
    gc.collect()

soft_predictions = soft_predictions / len(PRETRAINED_MODEL_PATHS)

soft_claim_predictions = soft_predictions[:, :, 8]

predictions = np.argmax(soft_predictions, axis=2)
soft_predictions = np.max(soft_predictions, axis=2)


def generate_token_to_word_mapping(txt, offset):
    # GET WORD POSITIONS IN CHARS
    w = []
    blank = True
    for i in range(len(txt)):
        if not txt[i].isspace() and blank == True:
            w.append(i)
            blank = False
        elif txt[i].isspace():
            blank = True
    w.append(1e6)

    # MAPPING FROM TOKENS TO WORDS
    word_map = -1 * np.ones(len(offset), dtype='int32')
    w_i = 0
    for i in range(len(offset)):
        if offset[i][1] == 0: continue
        while offset[i][0] >= (w[w_i + 1] - ("debertal_chris" in args.save_name) - ("v3" in args.save_name) \
                               - ("v2" in args.save_name)): w_i += 1
        word_map[i] = int(w_i)

    return word_map


all_preds = []

# Clumsy gathering of predictions at word lvl - only populate with 1st subword pred
for curr_sample_id in range(len(test_dataset)):
    curr_preds = []
    sample_preds = predictions[curr_sample_id]
    sample_offset = test_dataset.get_offset(curr_sample_id)
    sample_txt = ner_test_rows[curr_sample_id][1]
    sample_word_map = generate_token_to_word_mapping(sample_txt, sample_offset)

    word_preds = [''] * (max(sample_word_map) + 1)
    word_probs = dict(zip(range((max(sample_word_map) + 1)), [0] * (max(sample_word_map) + 1)))
    claim_probs = dict(zip(range((max(sample_word_map) + 1)), [0] * (max(sample_word_map) + 1)))

    for i, curr_word_id in enumerate(sample_word_map):
        if curr_word_id != -1:
            if word_preds[curr_word_id] == '':  # only use 1st subword
                word_preds[curr_word_id] = inverted_ner_labels[sample_preds[i]]
                word_probs[curr_word_id] = soft_predictions[curr_sample_id, i]
                claim_probs[curr_word_id] = soft_claim_predictions[curr_sample_id, i]
            elif 'B-' in inverted_ner_labels[sample_preds[i]]:
                word_preds[curr_word_id] = inverted_ner_labels[sample_preds[i]]
                word_probs[curr_word_id] = soft_predictions[curr_sample_id, i]
                claim_probs[curr_word_id] = soft_claim_predictions[curr_sample_id, i]

    # Dict to hold Lead, Position, Concluding Statement
    let_one_dict = dict()  # K = Type, V = (Prob of start token, start, end)

    # If we see tokens I-X, I-Y, I-X -> change I-Y to I-X
    for j in range(1, len(word_preds) - 1):
        pred_trio = [word_preds[k] for k in [j - 1, j, j + 1]]
        splitted_trio = [x.split('-')[0] for x in pred_trio]
        if all([x == 'I' for x in splitted_trio]) and pred_trio[0] == pred_trio[2] and pred_trio[0] != pred_trio[1]:
            word_preds[j] = word_preds[j - 1]

    # B-X, ? (not B), I-X -> change ? to I-X
    for j in range(1, len(word_preds) - 1):
        if 'B-' in word_preds[j - 1] and word_preds[j + 1] == f"I-{word_preds[j - 1].split('-')[-1]}" and word_preds[
            j] != word_preds[j + 1] and 'B-' not in word_preds[j]:
            word_preds[j] = word_preds[j + 1]

    # If we see tokens I-X, O, I-X, change center token to the same for stated discourse types
    for j in range(1, len(word_preds) - 1):
        if word_preds[j - 1] in ['I-Lead', 'I-Position', 'I-Concluding Statement'] and word_preds[j - 1] == word_preds[
            j + 1] and word_preds[j] == 'O':
            word_preds[j] = word_preds[j - 1]

    j = 0  # start of candidate discourse
    while j < len(word_preds):
        cls = word_preds[j]
        cls_splitted = cls.split('-')[-1]
        end = j + 1  # try to extend discourse as far as possible

        if word_probs[j] > 0.54:
            # Must match suffix i.e., I- to I- only; no B- to I-
            while end < len(word_preds) and (
                word_preds[end].split('-')[-1] == cls_splitted if cls_splitted in ['Lead', 'Position',
                                                                                   'Concluding Statement'] else
                word_preds[
                    end] == f'I-{cls_splitted}'):
                end += 1
            # if we're here, end is not the same pred as start
            if cls != 'O' and (end - j > MIN_TOKENS[cls_splitted] or max(
                word_probs[l] for l in range(j, end)) > 0.73):  # needs to be longer than class-specified min
                if cls_splitted in ['Lead', 'Position', 'Concluding Statement']:
                    lpc_max_prob = max(word_probs[c] for c in range(j, end))
                    if cls_splitted in let_one_dict:  # Already existing, check contiguous or higher prob
                        prev_prob, prev_start, prev_end = let_one_dict[cls_splitted]
                        if cls_splitted in ['Lead',
                                            'Concluding Statement'] and j - prev_end < 49:  # If close enough, combine
                            let_one_dict[cls_splitted] = (max(prev_prob, lpc_max_prob), prev_start, end)

                            # Delete other preds that lie inside the joined LC discourse
                            for l in range(len(curr_preds) - 1, 0, -1):
                                check_span = curr_preds[l][2]
                                check_start, check_end = int(check_span[0]), int(check_span[-1])
                                if check_start > prev_start and check_end < end:
                                    del curr_preds[l]

                        elif lpc_max_prob > prev_prob:  # Overwrite if current candidate is more likely
                            let_one_dict[cls_splitted] = (lpc_max_prob, j, end)
                    else:  # Add to it
                        let_one_dict[cls_splitted] = (lpc_max_prob, j, end)
                else:
                    # Lookback and add preceding I- tokens
                    while j - 1 > 0 and word_preds[j - 1] == cls:
                        j = j - 1
                    # Try to add the matching B- tag if immediately precedes the current I- sequence
                    if j - 1 > 0 and word_preds[j - 1] == f'B-{cls_splitted}':
                        j = j - 1

                    #############################################################
                    # Run a bunch of adjustments to discourse predictions based on CV
                    adj_start, adj_end = j, end + 1

                    # Run some heuristics against previous discourse
                    if len(curr_preds) > 0:
                        prev_span = list(map(int, curr_preds[-1][2].split()))
                        prev_start, prev_end = prev_span[0], prev_span[-1]

                        # Join adjacent rebuttals
                        if cls_splitted in 'Rebuttal':
                            if curr_preds[-1][1] == cls_splitted and adj_start - prev_end < 32:
                                del curr_preds[-1]
                                combined_list = prev_span + list(range(adj_start, adj_end))
                                curr_preds.append((test_dataset.get_filename(curr_sample_id),
                                                   cls_splitted,
                                                   ' '.join(map(str, combined_list)),
                                                   AGG_FUNC([word_probs[i] for i in combined_list if
                                                             i in word_probs.keys()])))
                                j = end
                                continue

                        elif cls_splitted in 'Counterclaim':
                            if curr_preds[-1][1] == cls_splitted and adj_start - prev_end < 24:
                                del curr_preds[-1]
                                combined_list = prev_span + list(range(adj_start, adj_end))
                                curr_preds.append((test_dataset.get_filename(curr_sample_id),
                                                   cls_splitted,
                                                   ' '.join(map(str, combined_list)),
                                                   AGG_FUNC([word_probs[i] for i in combined_list if
                                                             i in word_probs.keys()])))
                                j = end
                                continue

                        elif cls_splitted in 'Evidence':
                            if curr_preds[-1][1] == cls_splitted and 8 < adj_start - prev_end < 25:
                                if max(claim_probs[l] for l in range(prev_end + 1, adj_start)) > 0.35:
                                    claim_tokens = [str(l) for l in range(prev_end + 1, adj_start) if
                                                    claim_probs[l] > 0.15]
                                    if len(claim_tokens) > 2:
                                        curr_preds.append((test_dataset.get_filename(curr_sample_id),
                                                           'Claim',
                                                           ' '.join(claim_tokens),
                                                           AGG_FUNC([word_probs[int(i)] for i in claim_tokens if
                                                                     int(i) in word_probs.keys()])))
                        # If gap with discourse of same type, extend to it
                        elif curr_preds[-1][1] == cls_splitted and adj_start - prev_end > 2:
                            adj_start -= 1

                    # Adjust discourse lengths if too long or short
                    if cls_splitted == 'Evidence':
                        if adj_end - adj_start < 45:
                            adj_start -= 9
                        else:
                            adj_end -= 1
                    elif cls_splitted == 'Claim':
                        if adj_end - adj_start > 24:
                            adj_end -= 1
                    elif cls_splitted == 'Counterclaim':
                        if adj_end - adj_start > 24:
                            adj_end -= 1
                        else:
                            adj_start -= 1
                            adj_end += 1
                    elif cls_splitted == 'Rebuttal':
                        if adj_end - adj_start > 32:
                            adj_end -= 1
                        else:
                            adj_start -= 1
                            adj_end += 1
                    adj_start = max(0, adj_start)
                    adj_end = min(len(word_preds) - 1, adj_end)
                    curr_preds.append((test_dataset.get_filename(curr_sample_id),
                                       cls_splitted,
                                       ' '.join(map(str, list(range(adj_start, adj_end)))),
                                       AGG_FUNC([word_probs[i] for i in range(adj_start, adj_end) if
                                                 i in word_probs.keys()])))

        j = end

        # Add the Lead, Position, Concluding Statement
    for k, v in let_one_dict.items():
        pred_start = v[1]
        pred_end = v[2]

        # Lookback and add preceding I- tokens
        while pred_start - 1 > 0 and word_preds[pred_start - 1] == f'I-{k}':
            pred_start = pred_start - 1
        # Try to add the matching B- tag if immediately precedes the current I- sequence
        if pred_start - 1 > 0 and word_preds[pred_start - 1] == f'B-{k}':
            pred_start = pred_start - 1

        # Extend short Leads and Concluding Statements
        if k == 'Lead':
            if pred_end - pred_start < 33:
                pred_end = min(len(word_preds), pred_end + 5)
            else:
                pred_end -= 5
        elif k == 'Concluding Statement':
            if pred_end - pred_start < 23:
                pred_start = max(0, pred_start - 1)
                pred_end = min(len(word_preds), pred_end + 10)
        elif k == 'Position':
            if pred_end - pred_start < 18:
                pred_end = min(len(word_preds), pred_end + 3)

        pred_start = max(0, pred_start)
        if pred_end - pred_start > 6:
            curr_preds.append((test_dataset.get_filename(curr_sample_id),
                               k,
                               ' '.join(map(str, list(range(pred_start, pred_end)))),
                               AGG_FUNC(
                                   [word_probs[i] for i in range(pred_start, pred_end) if i in word_probs.keys()])))

    all_preds.extend(curr_preds)

output_df = pd.DataFrame(all_preds)
output_df.columns = ['id', 'class', 'predictionstring', 'scores']
output_df.to_csv(f'{args.save_name}.csv', index=False)
