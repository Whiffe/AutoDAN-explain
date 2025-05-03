import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import openai
from tqdm import tqdm
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from collections import defaultdict, OrderedDict
from utils.string_utils import autodan_SuffixManager
import sys
import time


def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i + batch_size]
        else:
            batch_attention_mask = None
        '''
        model(input_ids=batch_input_ids, attention_mask=batch_attention_mask): CausalLMOutputWithPast(loss=None, logits=tensor([[[ 0.1042, -0.2214,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
                 [-6.5078, -2.0742,  1.4473,  ..., -4.5352, -5.9062, -2.6445],
                 [-3.8574, -4.3125, -2.1035,  ..., -4.0547, -4.6953, -0.8022],
                 ...,
                 [-4.4141, -1.9912,  2.0449,  ..., -3.5195, -4.5000, -2.5410],
                 [-3.9180,  4.2617,  5.3906,  ..., -0.7847, -3.4355,  0.1445],
                 [-4.7461, -2.1562,  2.2188,  ..., -3.7539, -4.8125, -2.6074]],
        
                [[ 0.1042, -0.2214,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
                 [-6.5078, -2.0742,  1.4473,  ..., -4.5352, -5.9062, -2.6445],
                 [-3.8574, -4.3125, -2.1035,  ..., -4.0547, -4.6953, -0.8022],
                 ...,
                 [-4.8125, -1.4053,  2.1523,  ..., -4.5000, -5.8594, -2.2188],
                 [-2.8594,  5.8477,  7.5078,  ..., -0.0772, -2.9766,  1.5596],
                 [-4.4648, -0.7119,  2.0938,  ..., -4.5039, -5.6484, -2.0918]],
        
                [[ 0.1042, -0.2214,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
                 [-6.5078, -2.0742,  1.4473,  ..., -4.5352, -5.9062, -2.6445],
                 [-3.8574, -4.3125, -2.1035,  ..., -4.0547, -4.6953, -0.8022],
                 ...,
                 [-4.5547, -2.2344,  2.1465,  ..., -3.5469, -4.5391, -2.5352],
                 [-2.8242,  5.3203,  6.7656,  ...,  0.5229, -1.8740,  2.0000],
                 [-3.2363, -0.6011,  0.4753,  ..., -4.0078, -3.3652, -2.2715]],
        
                ...,
        
                [[ 0.1042, -0.2214,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
                 [-6.5078, -2.0742,  1.4473,  ..., -4.5352, -5.9062, -2.6445],
                 [-3.8574, -4.3125, -2.1035,  ..., -4.0547, -4.6953, -0.8022],
                 ...,
                 [-3.8984, -1.6436,  1.6895,  ..., -3.2559, -4.0391, -2.3965],
                 [-3.3770,  4.2461,  6.1328,  ..., -1.4023, -3.6934,  0.8052],
                 [-3.7559,  0.0191,  0.9517,  ..., -4.3555, -4.8203, -1.5625]],
        
                [[ 0.1042, -0.2214,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
                 [-6.5078, -2.0742,  1.4473,  ..., -4.5352, -5.9062, -2.6445],
                 [-3.8574, -4.3125, -2.1035,  ..., -4.0547, -4.6953, -0.8022],
                 ...,
                 [-3.9238, -1.5938,  1.6963,  ..., -3.2598, -4.1016, -2.4512],
                 [-6.0977,  1.2100,  4.2773,  ..., -1.9600, -5.3398, -0.6475],
                 [-4.7305, -0.1689,  5.1250,  ..., -2.2441, -4.0312, -0.9390]],
        
                [[ 0.1042, -0.2214,  0.3130,  ...,  1.3281,  1.8799,  0.6436],
                 [-6.5078, -2.0742,  1.4473,  ..., -4.5352, -5.9062, -2.6445],
                 [-3.8574, -4.3125, -2.1035,  ..., -4.0547, -4.6953, -0.8022],
                 ...,
                 [-3.8320, -1.6270,  1.6377,  ..., -3.2051, -3.9473, -2.4180],
                 [-2.9492,  4.2969,  5.4570,  ..., -0.7666, -2.5605,  1.0469],
                 [-4.4023, -2.1504,  1.9395,  ..., -3.4922, -4.3281, -2.5059]]],
               device='cuda:0', dtype=torch.float16), past_key_values=None, hidden_states=None, attentions=None)

        '''
        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


### AutoDAN ###
def autodan_sample_control(control_suffixs, score_list, num_elites, batch_size, crossover=0.5,
                           num_points=5, mutation=0.01, API_key=None, reference=None, if_softmax=True, if_api=True):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_suffixs
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_suffixs = [control_suffixs[i] for i in sorted_indices]

    # Step 2: Select the elites
    elites = sorted_control_suffixs[:num_elites]

    # Step 3: Use roulette wheel selection for the remaining positions
    parents_list = roulette_wheel_selection(control_suffixs, score_list, batch_size - num_elites, if_softmax)

    # Step 4: Apply crossover and mutation to the selected parents
    offspring = apply_crossover_and_mutation(parents_list, crossover_probability=crossover,
                                                     num_points=num_points,
                                                     mutation_rate=mutation, API_key=API_key, reference=reference,
                                                     if_api=if_api)

    # Combine elites with the mutated offspring
    next_generation = elites + offspring[:batch_size-num_elites]

    assert len(next_generation) == batch_size
    return next_generation


### GA ###
def roulette_wheel_selection(data_list, score_list, num_selected, if_softmax=True):
    if if_softmax:
        selection_probs = np.exp(score_list - np.max(score_list))
        selection_probs = selection_probs / selection_probs.sum()
    else:
        total_score = sum(score_list)
        selection_probs = [score / total_score for score in score_list]

    selected_indices = np.random.choice(len(data_list), size=num_selected, p=selection_probs, replace=True)

    selected_data = [data_list[i] for i in selected_indices]
    return selected_data


def apply_crossover_and_mutation(selected_data, crossover_probability=0.5, num_points=3, mutation_rate=0.01,
                                 API_key=None,
                                 reference=None, if_api=True):
    offspring = []

    for i in range(0, len(selected_data), 2):
        parent1 = selected_data[i]
        parent2 = selected_data[i + 1] if (i + 1) < len(selected_data) else selected_data[0]

        if random.random() < crossover_probability:
            child1, child2 = crossover(parent1, parent2, num_points)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    mutated_offspring = apply_gpt_mutation(offspring, mutation_rate, API_key, reference, if_api)

    return mutated_offspring


def crossover(str1, str2, num_points):
    # Function to split text into paragraphs and then into sentences
    def split_into_paragraphs_and_sentences(text):
        paragraphs = text.split('\n\n')
        return [re.split('(?<=[,.!?])\s+', paragraph) for paragraph in paragraphs]

    paragraphs1 = split_into_paragraphs_and_sentences(str1)
    paragraphs2 = split_into_paragraphs_and_sentences(str2)

    new_paragraphs1, new_paragraphs2 = [], []

    for para1, para2 in zip(paragraphs1, paragraphs2):
        max_swaps = min(len(para1), len(para2)) - 1
        num_swaps = min(num_points, max_swaps)

        swap_indices = sorted(random.sample(range(1, max_swaps + 1), num_swaps))

        new_para1, new_para2 = [], []
        last_swap = 0
        for swap in swap_indices:
            if random.choice([True, False]):
                new_para1.extend(para1[last_swap:swap])
                new_para2.extend(para2[last_swap:swap])
            else:
                new_para1.extend(para2[last_swap:swap])
                new_para2.extend(para1[last_swap:swap])
            last_swap = swap

        if random.choice([True, False]):
            new_para1.extend(para1[last_swap:])
            new_para2.extend(para2[last_swap:])
        else:
            new_para1.extend(para2[last_swap:])
            new_para2.extend(para1[last_swap:])

        new_paragraphs1.append(' '.join(new_para1))
        new_paragraphs2.append(' '.join(new_para2))

    return '\n\n'.join(new_paragraphs1), '\n\n'.join(new_paragraphs2)

def gpt_mutate(sentence, API_key=None):
    openai.api_key = API_key
    system_msg = 'You are a helpful and creative assistant who writes well.'
    user_message = f'Please revise the following sentence with no changes to its length and only output the revised version, the sentences are: \n "{sentence}".\nPlease give me your revision directly without any explanation. Remember keep the original paragraph structure. Do not change the words "[REPLACE]", "[PROMPT]", "[KEEPER]", and "[MODEL]", if they are in the sentences.'
    revised_sentence = sentence
    received = False
    while not received:
        try:
            response = openai.ChatCompletion.create(model="gpt-4",
                                                    messages=[{"role": "system", "content": system_msg},
                                                              {"role": "user", "content": user_message}],
                                                    temperature=1, top_p=0.9)
            revised_sentence = response["choices"][0]["message"]["content"].replace('\n', '')
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:  # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError, Prompt error.")
                return None
            if error == AssertionError:
                print("Assert error:", sys.exc_info()[1])  # assert False
            else:
                print("API error:", error)
            time.sleep(1)
    if revised_sentence.startswith("'") or revised_sentence.startswith('"'):
        revised_sentence = revised_sentence[1:]
    if revised_sentence.endswith("'") or revised_sentence.endswith('"'):
        revised_sentence = revised_sentence[:-1]
    if revised_sentence.endswith("'.") or revised_sentence.endswith('".'):
        revised_sentence = revised_sentence[:-2]
    print(f'revised: {revised_sentence}')
    return revised_sentence

def apply_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, reference=None, if_api=True):
    if if_api:
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                if API_key is None:
                    offspring[i] = random.choice(reference[len(offspring):])
                else:
                    offspring[i] = gpt_mutate(offspring[i], API_key)
    else:
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = replace_with_synonyms(offspring[i])
    return offspring


def apply_init_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, if_api=True):
    for i in tqdm(range(len(offspring)), desc='initializing...'):
        if if_api:
            if random.random() < mutation_rate:
                offspring[i] = gpt_mutate(offspring[i], API_key)
        else:
            if random.random() < mutation_rate:
                offspring[i] = replace_with_synonyms(offspring[i])
    return offspring


def replace_with_synonyms(sentence, num=10):
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(sentence)
    uncommon_words = [word for word in words if word.lower() not in stop_words and word.lower() not in T]
    selected_words = random.sample(uncommon_words, min(num, len(uncommon_words)))
    for word in selected_words:
        synonyms = wordnet.synsets(word)
        if synonyms and synonyms[0].lemmas():
            synonym = synonyms[0].lemmas()[0].name()
            sentence = sentence.replace(word, synonym, 1)
    print(f'revised: {sentence}')
    return sentence


### HGA ###
def autodan_sample_control_hga(word_dict, control_suffixs, score_list, num_elites, batch_size, crossover=0.5,
                               mutation=0.01, API_key=None, reference=None, if_api=True):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_suffixs
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_suffixs = [control_suffixs[i] for i in sorted_indices]

    # Step 2: Select the elites
    elites = sorted_control_suffixs[:num_elites]
    parents_list = sorted_control_suffixs[num_elites:]

    # Step 3: Construct word list
    word_dict = construct_momentum_word_dict(word_dict, control_suffixs, score_list)
    print(f"Length of current word dictionary: {len(word_dict)}")

    # check the length of parents
    parents_list = [x for x in parents_list if len(x) > 0]
    if len(parents_list) < batch_size - num_elites:
        print("Not enough parents, using reference instead.")
        parents_list += random.choices(reference[batch_size:], k = batch_size - num_elites - len(parents_list))
        
    # Step 4: Apply word replacement with roulette wheel selection
    offspring = apply_word_replacement(word_dict, parents_list, crossover)
    offspring = apply_gpt_mutation(offspring, mutation, API_key, reference, if_api)

    # Combine elites with the mutated offspring
    next_generation = elites + offspring[:batch_size-num_elites]

    assert len(next_generation) == batch_size
    return next_generation, word_dict

def construct_momentum_word_dict(word_dict, control_suffixs, score_list, topk=-1):
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    stop_words = set(stopwords.words('english'))
    if len(control_suffixs) != len(score_list):
        raise ValueError("control_suffixs and score_list must have the same length.")

    word_scores = defaultdict(list)

    for prefix, score in zip(control_suffixs, score_list):
        words = set(
            [word for word in nltk.word_tokenize(prefix) if word.lower() not in stop_words and word.lower() not in T])
        for word in words:
            word_scores[word].append(score)

    for word, scores in word_scores.items():
        avg_score = sum(scores) / len(scores)
        if word in word_dict:
            word_dict[word] = (word_dict[word] + avg_score) / 2
        else:
            word_dict[word] = avg_score

    sorted_word_dict = OrderedDict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
    if topk == -1:
        topk_word_dict = dict(list(sorted_word_dict.items()))
    else:
        topk_word_dict = dict(list(sorted_word_dict.items())[:topk])
    return topk_word_dict


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def word_roulette_wheel_selection(word, word_scores):
    if not word_scores:
        return word
    min_score = min(word_scores.values())
    adjusted_scores = {k: v - min_score for k, v in word_scores.items()}
    total_score = sum(adjusted_scores.values())
    pick = random.uniform(0, total_score)
    current_score = 0
    for synonym, score in adjusted_scores.items():
        current_score += score
        if current_score > pick:
            if word.istitle():
                return synonym.title()
            else:
                return synonym

def replace_with_best_synonym(sentence, word_dict, crossover_probability):
    stop_words = set(stopwords.words('english'))
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    paragraphs = sentence.split('\n\n')
    modified_paragraphs = []
    min_value = min(word_dict.values())

    for paragraph in paragraphs:
        words = replace_quotes(nltk.word_tokenize(paragraph))
        count = 0
        for i, word in enumerate(words):
            if random.random() < crossover_probability:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = get_synonyms(word.lower())
                    word_scores = {syn: word_dict.get(syn, min_value) for syn in synonyms}
                    best_synonym = word_roulette_wheel_selection(word, word_scores)
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
            else:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = get_synonyms(word.lower())
                    word_scores = {syn: word_dict.get(syn, 0) for syn in synonyms}
                    best_synonym = word_roulette_wheel_selection(word, word_scores)
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
        modified_paragraphs.append(join_words_with_punctuation(words))
    return '\n\n'.join(modified_paragraphs)

def replace_quotes(words):
    new_words = []
    quote_flag = True

    for word in words:
        if word in ["``", "''"]:
            if quote_flag:
                new_words.append('“')
                quote_flag = False
            else:
                new_words.append('”')
                quote_flag = True
        else:
            new_words.append(word)
    return new_words

def apply_word_replacement(word_dict, parents_list, crossover=0.5):
    return [replace_with_best_synonym(sentence, word_dict, crossover) for sentence in parents_list]

def join_words_with_punctuation(words):
    sentence = words[0]
    previous_word = words[0]
    flag = 1
    for word in words[1:]:
        if word in [",", ".", "!", "?", ":", ";", ")", "]", "}", '”']:
            sentence += word
        else:
            if previous_word in ["[", "(", "'", '"', '“']:
                if previous_word in ["'", '"'] and flag == 1:
                    sentence += " " + word
                else:
                    sentence += word
            else:
                if word in ["'", '"'] and flag == 1:
                    flag = 1 - flag
                    sentence += " " + word
                elif word in ["'", '"'] and flag == 0:
                    flag = 1 - flag
                    sentence += word
                else:
                    if "'" in word and re.search('[a-zA-Z]', word):
                        sentence += word
                    else:
                        sentence += " " + word
        previous_word = word
    return sentence

def get_score_autodan(tokenizer, conv_template, instruction, target, model, device, test_controls=None, crit=None, dis=None):
    # Convert all test_controls to token ids and find the max length
    input_ids_list = []
    target_slices = []
    for item in test_controls:
        suffix_manager = autodan_SuffixManager(tokenizer=tokenizer,
                                               conv_template=conv_template,
                                               instruction=instruction,
                                               target=target,
                                               adv_string=item)
        input_ids = suffix_manager.get_input_ids(adv_string=item).to(device)
        input_ids_list.append(input_ids)
        target_slices.append(suffix_manager._target_slice)

    # Pad all token ids to the max length
    pad_tok = 0
    for ids in input_ids_list:
        while pad_tok in ids:
            pad_tok += 1

    # Find the maximum length of input_ids in the list
    max_input_length = max([ids.size(0) for ids in input_ids_list])

    # Pad each input_ids tensor to the maximum length
    padded_input_ids_list = []
    for ids in input_ids_list:
        pad_length = max_input_length - ids.size(0)
        padded_ids = torch.cat([ids, torch.full((pad_length,), pad_tok, device=device)], dim=0)
        padded_input_ids_list.append(padded_ids)

    # Stack the padded input_ids tensors
    input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)

    attn_mask = (input_ids_tensor != pad_tok).type(input_ids_tensor.dtype)

    # Forward pass and compute loss
    '''
    input_ids_tensor: tensor([[    1,   518, 25580,  ...,     0,     0,     0],
            [    1,   518, 25580,  ...,     0,     0,     0],
            [    1,   518, 25580,  ...,     0,     0,     0],
            ...,
            [    1,   518, 25580,  ...,     0,     0,     0],
            [    1,   518, 25580,  ...,     0,     0,     0],
            [    1,   518, 25580,  ...,     0,     0,     0]])
    '''
    logits = forward(model=model, input_ids=input_ids_tensor, attention_mask=attn_mask, batch_size=len(test_controls))
    losses = []
    for idx, target_slice in enumerate(target_slices):
        loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
        logits_slice = logits[idx, loss_slice, :].unsqueeze(0).transpose(1, 2)
        targets = input_ids_tensor[idx, target_slice].unsqueeze(0)
        if dis is not None:
            logits_cal = logits_slice.unsqueeze(0).expand_as(dis)
            loss = -crit(logits_cal, dis).mean()
            losses.append(loss)
        else:
            loss = crit(logits_slice, targets)
            losses.append(loss)

    del input_ids_list, target_slices, input_ids_tensor, attn_mask
    gc.collect()
    return torch.stack(losses)


def get_score_autodan_low_memory(tokenizer, conv_template, instruction, target, model, device, test_controls=None,
                                 crit=None):
    losses = []
    for item in test_controls:
        suffix_manager = autodan_SuffixManager(tokenizer=tokenizer,
                                               conv_template=conv_template,
                                               instruction=instruction,
                                               target=target,
                                               adv_string=item)
        input_ids = suffix_manager.get_input_ids(adv_string=item).to(device)
        input_ids_tensor = torch.stack([input_ids], dim=0)

        # Forward pass and compute loss
        logits = forward(model=model, input_ids=input_ids_tensor, attention_mask=None, batch_size=len(test_controls))

        target_slice = suffix_manager._target_slice
        loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
        logits_slice = logits[0, loss_slice, :].unsqueeze(0).transpose(1, 2)
        targets = input_ids_tensor[0, target_slice].unsqueeze(0)
        loss = crit(logits_slice, targets)
        losses.append(loss)

    del input_ids_tensor
    gc.collect()
    return torch.stack(losses)
