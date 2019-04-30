#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
pytorch model training, validating and testing function

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/29 20:10
"""
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


def model_training(model, optimizer, criterion, dataset, batch_size, max_gradient_norm, device):
    """
    Training model by batch.
    """
    # Switch the model to train mode.
    model.train()

    train_loss = 0.0
    processed = 0

    batch_index = 0
    batch_count = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)
    tqdm_batch_iterator = tqdm(dataset.gen_mini_batches(batch_size=batch_size, shuffle=True), total=batch_count)
    for batch_data in tqdm_batch_iterator:
        batch_question = torch.tensor(batch_data['question_token_ids'], dtype=torch.long).to(device)
        batch_pos_questions = torch.tensor(batch_data['pos_questions']).to(device)
        batch_pos_freq_questions = torch.tensor(batch_data['pos_freq_questions']).to(device)
        batch_keyword_questions = torch.tensor(batch_data['keyword_questions']).to(device)
        batch_question_length = torch.tensor(batch_data['question_length']).to(device)

        batch_passage_token_ids = torch.tensor(batch_data['passage_token_ids'], dtype=torch.long).to(device)
        batch_pos_passages = torch.tensor(batch_data['pos_passages']).to(device)
        batch_pos_freq_passages = torch.tensor(batch_data['pos_freq_passages']).to(device)
        batch_keyword_passages = torch.tensor(batch_data['keyword_passages']).to(device)
        batch_passage_length = torch.tensor(batch_data['passage_length']).to(device)
        batch_wiq_feature = torch.tensor(batch_data['wiq_feature']).to(device)

        batch_passage_cnts = torch.tensor(batch_data['passage_cnts']).to(device)

        batch_start_ids = batch_data['start_ids']
        batch_end_ids = batch_data['end_ids']
        batch_match_scores = batch_data['match_scores']

        batch_start_labels = []
        batch_end_labels = []
        indexes = np.argmax(batch_match_scores, axis=1)
        for idx, s, e in zip(indexes, batch_start_ids, batch_end_ids):
            batch_start_labels.append(s[idx])
            batch_end_labels.append(e[idx])

        batch_start_labels = torch.tensor(batch_start_labels).to(device)
        batch_end_labels = torch.tensor(batch_end_labels).to(device)

        # zero the gradients
        optimizer.zero_grad()
        # model forward
        start_end_probs, ans_range, _ = model.forward(question=batch_question,
                                                      context=batch_passage_token_ids,
                                                      passage_cnts=batch_passage_cnts)
        start_probs = start_end_probs[:, 0, :]
        end_probs = start_end_probs[:, 1, :]

        # compute loss
        loss = criterion(start_probs, batch_start_labels, end_probs, batch_end_labels)
        # loss back prop
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        # update params
        optimizer.step(closure=None)

        train_loss += loss.item()
        processed += len(batch_start_ids)

        description = "train loss: {:.5f}".format(train_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
        batch_index += 1

    epoch_loss = train_loss / batch_index  # the number of batch
    return epoch_loss
