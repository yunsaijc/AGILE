import os

import json
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import string
from nltk.corpus import wordnet
import torch

def generate_by_text(model, tokenizer, text: str, 
                     max_length: int = 1000,
                     do_sample: bool = True,
                     temperature: float = 0.7,
                     top_p: float = 0.95,
                     top_k: int = 60,
                     repetition_penalty: float = 1.0,
                     ) -> str:
    input_txt = tokenizer.apply_chat_template([
        {"role": "user", "content": text},
    ], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_txt, return_tensors="pt").to(model.device)
    output_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                max_new_tokens=max_length, temperature=temperature, 
                                top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                                do_sample=do_sample,
                                pad_token_id=tokenizer.eos_token_id)
    output_ids = [output_ids[0][len(inputs.input_ids[0]):]]
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def generate_by_text_batch(model, tokenizer, text: list[str], 
                           max_length: int = 2000,
                           do_sample: bool = True,
                           temperature: float = 0.7,
                           top_p: float = 0.95,
                           top_k: int = 60,
                           repetition_penalty: float = 1.0,
                           ) -> list[str]:
        formated_input_texts = [[{"role": "user", "content": text_item}] for text_item in text]
        inputs = tokenizer.apply_chat_template(formated_input_texts, 
                                                    add_generation_prompt=True, tokenize=False,)
        
        model_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
        attention_mask = model_inputs["attention_mask"]
        outputs = model.generate(model_inputs.input_ids, attention_mask=attention_mask,
                                max_new_tokens=max_length, temperature=temperature, 
                                top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
                                do_sample=do_sample,
                                pad_token_id=tokenizer.eos_token_id)
        input_ids_len = model_inputs.input_ids.shape[1]
        generated_ids = outputs[:, input_ids_len:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

def get_activations_batch(model, tokenizer, texts: list[str], history: list[list[dict]], require_grad: bool = False) -> torch.Tensor:
    if history is not None:
        templated_strings = []
        assert len(history) == len(texts)
        for i in range(len(texts)):
            templated_strings.append(tokenizer.apply_chat_template(history[i] + [{"role": "user", "content": texts[i]}], tokenize=False, add_generation_prompt=True))
    else:
        templated_strings = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) for text in texts]

    inputs = tokenizer(templated_strings, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    if require_grad:
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
    else:
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
            
    # Get all hidden states
    all_hidden_states = outputs.hidden_states  # tuple of tensors, each shape (batch_size, seq_len, hidden_size)
    
    # Get the last token state of each layer
    last_token_states = []
    for layer_hidden_states in all_hidden_states:
        last_token_state = layer_hidden_states[:, -1, :]  # shape: (batch_size, hidden_size)
        last_token_states.append(last_token_state)
    
    # Stack all layer states
    activations = torch.stack(last_token_states[1:])  # shape: (num_layers, batch_size, hidden_size), remove embedding layer
    activations = activations.transpose(0, 1)  # shape: (batch_size, num_layers, hidden_size)
    return activations




def get_sent_sim(embed_model, seq_list) -> List[List[float]]:
    """
    calculate the similarity between sentences
    """
    embeddings = embed_model.encode(seq_list, show_progress_bar=False)
    similarities = embed_model.similarity(embeddings, embeddings)
    return similarities.tolist()

def build_formatted_history(history_list, response_list):
    """convert the conversation history and response to the model input format"""
    # assert len(history_list) == len(response_list), print(f"{history_list}\n{response_list}\n{len(history_list)}\n{len(response_list)}")
    history_list = history_list[-len(response_list):]
    formatted_history = []
    for history, response in zip(history_list, response_list):
        formatted_history.append({'role': 'user', 'content': history})
        formatted_history.append({'role': 'assistant', 'content': response})
    return formatted_history

def update_synonym_cache(word, syn, delta_score,
        cache_path: str = './cache/synonyms.json'):
    """
    update the score and count of the synonym cache
    delta_score: the amount of score decreased by the synonym
    """
    global synonym_cache
    delta_score = float(delta_score)
    if word not in synonym_cache.keys():
        synonym_cache[word] = {}
    if syn not in synonym_cache[word].keys():
        synonym_cache[word][syn] = [0.0, 0]

    new_count = synonym_cache[word][syn][1] + 1
    new_score = (synonym_cache[word][syn][0] * synonym_cache[word][syn][1] + delta_score) / new_count
    synonym_cache[word][syn] = (new_score, new_count)
    
    synonym_cache[word] = dict(sorted(synonym_cache[word].items(), key=lambda x: x[1][0], reverse=True))
    with open(cache_path, 'w') as f:
        json.dump(synonym_cache, f, indent=4)

def get_synonyms_batch(model, tokenizer, word, num_candidate=5,
                 cache_path: str = './cache/synonyms.json'):
    """
    get the num_candidate synonyms of the word
    if the word is in the cache, return the synonyms directly
    otherwise, use llm to generate the synonyms
    """
    global synonym_cache
    if synonym_cache is None:
        if os.path.exists(cache_path):
            if os.path.getsize(cache_path) > 0:
                with open(cache_path, 'r') as f:
                    synonym_cache = json.load(f)
            else:
                synonym_cache = {}
        else:
            synonym_cache = {}  # format: {word: {synonym1: (0.34, 4), synonym2: (0.21, 2), ...}},
                                # 0.34 represents the average score decreased by the synonym, 4 represents the number of times the synonym makes the score decrease


    if word in synonym_cache.keys() and len(synonym_cache[word]) >= num_candidate*2:
        syns = sorted(synonym_cache[word].items(), key=lambda x: x[1][0], reverse=True)
        return [syn[0] for syn in syns[:num_candidate]]  # return the top num_candidate synonyms
    
    elif word in synonym_cache.keys() and len(synonym_cache[word]) < num_candidate*2:  # if the word is in the cache, and the cache has less than num_candidate synonyms, use llm to generate the synonyms
        syns = get_llm_synonyms_batch(model, tokenizer, word, num_candidate*2 - len(synonym_cache[word]))
        syns = list(set(syns))
        syns = [syn for syn in syns if '\n' not in syn and '\t' not in syn]  # remove the items containing newline and tab
        syns = [syn for syn in syns if syn != word and len(syn) > 0]  # remove the items containing the original word itself
        if len(syns) == 0:  # if the generated synonyms are empty, return the original word
            return [word]

        for syn_item in syns:
            if syn_item not in synonym_cache[word].keys():  # if the syn is not in the cache, add it to the cache
                synonym_cache[word][syn_item] = [0.0, 0]
        
        # sorted and update the cache
        synonym_cache[word] = dict(sorted(synonym_cache[word].items(), key=lambda x: x[1][0], reverse=True))
        with open(cache_path, 'w') as f:
            json.dump(synonym_cache, f, indent=4)

        syns = list(synonym_cache[word].keys())
        if len(syns) < num_candidate:
            syns = np.random.choice(syns, size=num_candidate, replace=True)  # may not be enough num_candidate, repeat sampling
        else:
            syns = syns[:num_candidate]
        return syns
    
    else:   # if the word is not in the cache, use llm to generate the synonyms
        syns = get_llm_synonyms_batch(model, tokenizer, word, num_candidate)
        syns = list(set(syns))
        syns = [syn for syn in syns if '\n' not in syn and '\t' not in syn]  # remove the items containing newline and tab
        syns = [syn for syn in syns if syn != word and len(syn) > 0]  # remove the items containing the original word itself
        if len(syns) == 0:
            return [word]

        synonym_cache[word] = {syn: [0.0, 0] for syn in syns}
        with open(cache_path, 'w') as f:
            json.dump(synonym_cache, f, indent=4)
        return np.random.choice(syns, size=num_candidate, replace=True)  # may not be enough num_candidate, repeat sampling

def get_llm_synonyms_batch(model, tokenizer, word, num_candidate=5):
    """
    get the num_candidate synonyms of the words
    """
    synonyms = []
    prompt = f"""
        Return a synonym word/phrase of the given word: {word}.
        Wrap it with <synonym></synonym>:"""
    responses = generate_by_text_batch(model, tokenizer, [prompt]*num_candidate, max_length=100, temperature=0.95, top_k=100)
    for response in responses:
        try:
            syn = response.split('<synonym>')[1].split('</synonym>')[0]
        except:
            syn = word
        synonyms.append(syn)
    return synonyms

def softplus(x):
    return np.log(1+np.exp(x))

def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) if np.linalg.norm(x) * np.linalg.norm(y) != 0 else 0
