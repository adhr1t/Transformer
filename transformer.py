# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use TransformerLayer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()

        self.vocabSize = vocab_size
        self.numPositions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal
        self.numClasses = num_classes
        self.numLayers = num_layers
        self.attnList = []

        self.Embedding = nn.Embedding(vocab_size, d_model)  
        self.PositionalEncoder = PositionalEncoding(d_model, num_positions) # can include batching if want

        ### initialize transformerlayer list
        # self.TransformerLayers = nn.ModuleList([TransformerLayer(d_model, d_internal), TransformerLayer(d_model, d_internal)])
        self.TransformerLayers = nn.ModuleList([copy.deepcopy(TransformerLayer(d_model, d_internal)) for i in range(self.numLayers)])
        # self.TransformerLayer = TransformerLayer(d_model, d_internal)

        # output linear layer
        self.o_linear = nn.Linear(d_model, num_classes)

        # raise Exception("Implement me")

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        
        # Step 1: character encoding
        x = self.Embedding(indices) # shape [20, 512] = [indices length, d_model]

        # Step 2: add positional encoding to each character encoding
        # do this by calling the PositionalEncoding class and feeding the character encodings into it
        x = self.PositionalEncoder(x)

        # Step 3: feed encodings into TransformerLayer class
        ### do the transformerlayer list here
        # x_scaledValues, x_attnWeight = self.TransformerLayer(x) # shape [seq len, d_model]; [seq len, seq len]
        for i, l in enumerate(self.TransformerLayers):
            # print(i, l, "x output", x)
            x, x_attnWeight = self.TransformerLayers[i](x)
            
            # Append the current layer's attention map to the attention map list. I'm only doing one
            # transformer layer so only one value in the attention map list
            self.attnList.append(x_attnWeight)

        x_scaledValues = x



        # Step 4: use linear layer and softmax to make the prediction
        # turn [seq_len, d_model] into [seq_len, num_classes]
        # [seq_len, d_model] * [d_model, num_classes] = [seq_len, num_classes] = [20, 3]
        x_scaledValues = self.o_linear(x_scaledValues)  # shape [seq_len, num_classes]

        # softmax
        # x_output_log_prob = nn.functional.softmax(x_output, dim=-1)  # maintains dim of [seq_len, num_classes]
        m = nn.LogSoftmax(dim=-1)  # maintains dim of [seq_len, num_classes] # used dim = 1
        x_scaledValues_log_prob = m(x_scaledValues)

        return (x_scaledValues_log_prob, self.attnList)
        # raise Exception("Implement me")


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal # d_internal is just a hyperparameter. Don't even need to use it

        
        self.q_linear = nn.Linear(self.d_model, self.d_internal)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_internal)

        self.linear_1 = nn.Linear(self.d_model, self.d_model)  # input dim is d_model
        self.linear_2 = nn.Linear(self.d_model, self.d_model)  # output dim is d_model
        # self.dropout = nn.Dropout(dropout)

        # raise Exception("Implement me")

    def attention(self, queries, keys, values, d_model): # , mask=None  
        

        q = self.q_linear(queries)  # queries dim [seq_len, d_model] * [d_model, d_internal] = [seq_len, d_internal]
        k = self.k_linear(keys) # queries dim [seq_len, d_model] * [d_model, d_internal] = [seq_len, d_internal]
        v = self.v_linear(values) # queries dim [seq_len, d_model] * [d_model, d_model] = [seq_len, d_model]

        attn_scores = torch.matmul(q, k.T) / np.sqrt(d_model) # transpose(-2, -1) # am I np.sqrt the wrong thing ? originally np.sqrt(d_model)
        # np.sqrt(d_k) where d_k is the dimensions of the queries and keys

        # [seq_len, d_internal] * [d_internal, seq_len] = [seq_len, seq_len]

        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        #     scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
    
        # if dropout is not None:
        #     scores = dropout(scores)
        
        scaled_values = torch.matmul(attn_weights, v)   # [seq_len, seq_len] * [seq_len, d_model] = [seq_len, d_model]

        return scaled_values, attn_weights    # output needs to be [seq_len, d_model] for the rest of the model to work.
                            # bc the final tensor of TransformerLayer needs to be shape [seq_len, d_model]


    def feed_forward(self, x):

        x = nn.functional.relu(self.linear_1(x))
        # x = self.dropout(x) # if I want to do dropout
        x = self.linear_2(x)

        return x


    def forward(self, input_vecs):
        """
        :param input_vecs: an input tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the log probabilities of each position in the input
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """

        x_resid_connection = input_vecs # shape [seq len, d_model]  # I would normalize x here if I wanted to

        scaledValues, attnWeights = self.attention(x_resid_connection, x_resid_connection, x_resid_connection, self.d_model)

        # self attention then residual connection
        x_resid_connection = x_resid_connection + scaledValues # shape [seq len, d_model]
        x = x_resid_connection # shape [seq len, d_model]   # I would normalize x here if I wanted to
        
        # feed forward then final residual connection
        x = x + self.feed_forward(x_resid_connection)

        # print("Within TransformerLayer.forward() this is attnWeights.shape", attnWeights.shape)

        return x, attnWeights
        # raise Exception("Implement me")


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]    # just getting num_positions (aka seq_len)
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor) # create a tensor of 0 to num_positions/seq_len
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)  # shape [input_size/seq_len, d_model]
            # [20, 512]
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)   # add the positional embedding to each character encoding


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):

    # raise Exception("Not fully implemented yet")

    # Transformer Parameters
    vocab_size = 27
    num_positions = 20
    d_model = 128   # try making these smaller like <100. Originally 512; 64
    d_internal = 256   # try making these smaller like <100. Originally 1024; 128
    num_classes = 3
    num_layers = 2

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.zero_grad()   # zero the gradient. I would do this for every batch if I was doing batching
    model.train()

    ### Potential idea for improving model training 
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)
    # # this code is very important! It initialises the parameters with a
    # # range of values that stops the signal fading or getting too big.
    # optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    num_epochs = 2  # originally 10. Profs said 5 should be enough
    for t in range(0, num_epochs):
        # print("Epoch", t)
        count = 0

        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        length = int(len(train) * .8)
        ex_idxs = [i for i in range(0, length)] # this is where I'd batch if I wanted to
        # ex_idxs = [i for i in range(0, len(train))] # this is where I'd batch if I wanted to

        random.shuffle(ex_idxs) # shuffles the indexes
        loss_fcn = nn.NLLLoss()

        for ex_idx in ex_idxs:
            # print("Start of data entry with index", ex_idx)
            # Make predictions
            output_log_prob, attnWeights = model(train[ex_idx].input_tensor)  # feed in the tensorized indices of the input
            # prediction = np.argmax(output_log_prob.detach().numpy(), axis=1)
            target = torch.from_numpy(train[ex_idx].output).long()

            # Compute the loss
            # nn.NLLLoss() throws an error if target is not long() type. Might need to change
            # the type of prediction too if it throws an error
            loss = loss_fcn(output_log_prob, target) # TODO: Run forward and compute loss

            # Calculate the backwards gradients over the learning weights
            loss.backward()

            # Tell the optimizer to perform one learning step
            # Basically adjust the model's learning weights based on the observed gradients
            # according to the optimization algo we chose.
            optimizer.step()

            loss_this_epoch += loss.item()

            # Zero the gradient for the next input
            model.zero_grad()

            ### How it originally was
            # model.zero_grad()
            # loss.backward()
            # optimizer.step()
            count += 1
            # print("End of data entry with index", ex_idx, "and loss", loss.item(), "and count", count)


            # don't need to do any accuracy checking. decode method does that

    # Set the model to evaluation mode for testing
    # This disables dropout and batchnormalization
    model.eval()
    

    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))