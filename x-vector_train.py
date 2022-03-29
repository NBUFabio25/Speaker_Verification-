#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020
@author: krishna
"""

import torch
from torch.utils.data import DataLoader
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils import speech_collate
from utils import get_centroids
from utils import get_cossim
torch.multiprocessing.set_sharing_strategy('file_system')

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-training_filepath', type=str, default='meta/training.txt')
parser.add_argument('-testing_filepath', type=str, default='meta/testing.txt')
parser.add_argument('-enrolling_filepath', type=str, default='meta/enrolling.txt')

parser.add_argument('-input_dim', action="store_true", default=257)
#TIMIT train speaker:567 test speaker:56
parser.add_argument('-num_classes', action="store_true", default=567)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=32)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=100)
args = parser.parse_args()

### Data related
dataset_train = SpeechDataGenerator(manifest=args.training_filepath, mode='train')
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=speech_collate)

dataset_enroll = SpeechDataGenerator(manifest=args.enrolling_filepath, mode='enroll')
dataloader_enroll = DataLoader(dataset_enroll, batch_size=10, shuffle=True, collate_fn=speech_collate)

dataset_test = SpeechDataGenerator(manifest=args.testing_filepath, mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=speech_collate)

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model = X_vector(args.input_dim, args.num_classes).to(device)
# checkpoint = torch.load(r'E:\研一项目\实验1\speaker verification\Speaker_Verification-\save_model\best_check_point_99_0.653254128239128')
# model.load_state_dict(checkpoint['model'])
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()



def train(dataloader_train, epoch):
    train_loss_list = []
    full_preds = []
    full_gts = []
    model.train()
    for i_batch, sample_batched in enumerate(dataloader_train):

        features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        features, labels = features.to(device), labels.to(torch.int64).to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        #spec 送入网络。
        pred_logits, x_vec = model(features)
        #### CE loss
        loss = loss_fun(pred_logits, labels)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        # train_acc_list.append(accuracy)
        # if i_batch%10==0:
        #    print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))

        predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)

    # mean_acc = accuracy_score(full_gts, full_preds)
    # mean_loss = np.mean(np.asarray(train_loss_list))
    # print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss, mean_acc, epoch))
    #
    # model_save_path = os.path.join('save_model', 'best_check_point_' + str(epoch) + '_' + str(mean_loss))
    # state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    # torch.save(state_dict, model_save_path)


def test(dataloader_test, epoch):
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        full_preds = []
        full_gts = []
        for sample_batched in dataloader_enroll:
            features = torch.from_numpy(
                np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device), labels.to(torch.int64).to(device)
            pred_logits, enrollment_embeddings = model(features)

            #### CE loss
            # loss = loss_fun(pred_logits, labels)
            # val_loss_list.append(loss.item())
            # train_acc_list.append(accuracy)
            # predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            # for pred in predictions:
            #     full_preds.append(pred)
            # for lab in labels.detach().cpu().numpy():
            #     full_gts.append(lab)
        enrollment_centroids = get_centroids(enrollment_embeddings)
        for i_batch, sample_batched in enumerate(dataloader_test):
            features = torch.from_numpy(
                np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device), labels.to(torch.int64).to(device)
            pred_logits, verification_embeddings = model(features)
            #### CE loss
            loss = loss_fun(pred_logits, labels)
            val_loss_list.append(loss.item())
            # train_acc_list.append(accuracy)
            predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)

        mean_acc = accuracy_score(full_gts, full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('Total vlidation loss {} and Validation accuracy {} after {} epochs'.format(mean_loss, mean_acc, epoch))

        model_save_path = os.path.join('save_model', 'best_check_point_' + str(epoch) + '_' + str(mean_loss))
        state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state_dict, model_save_path)

if __name__ == '__main__':
    for epoch in range(args.num_epochs):
        train(dataloader_train, epoch)
        test(dataloader_test, epoch)
