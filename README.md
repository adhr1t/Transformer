# Transformer: Project Overview
- A Transformer is a sequence-to-sequence/encoder-decoder model. It employs multi-headed attention to "pay attention" to different aspects of the task at hand. It is complex, renowned model that is widely used for current NLP applications like Chat-GPT
- Implemented a Transformer from scratch to complete a Natural Language Processing task for which it achieved 99% accuracy
- Task 1: Given a string of characters, predict - for each character - how many times the character at that position occured previously, maxing out at 2. Example:
  
    ![task1](https://github.com/adhr1t/Transformer/assets/72672768/caa060f4-c0e0-4514-b66b-567b9e1e7c44)
- Task 2: Given a string of characters, predict - for each character - how many times the character at that position occured before and after in the sequence, maxing out at 2. Example:

    ![task2](https://github.com/adhr1t/Transformer/assets/72672768/57d6e97a-9224-4eb1-9c39-69ea8bccb5bb)

# Transformer Architecture
![transformer](https://github.com/adhr1t/Transformer/assets/72672768/9be3628c-7674-449f-ab40-5b9150cf9481)

# Approach
1. Build model architecture: create classes for Transformer, TransformerLayer, Positional Encoding, Model Training, and Decoding
2. Create Attention mechanism:
     1) Generate query, key, and value representation of the inputs
     2) Compute the dot-product of the queries with the keys. Scale the result by the square root of d_model. These are the attention scores
     3) Optionally apply a mask to the attention scores
     4) Feed the attention scores into a softmax function to get the attention weights
     5) Compute the dot-product of the attention weights with the values to get the scaled values
    ![attn](https://github.com/adhr1t/Transformer/assets/72672768/59cdb9f4-d2af-48bf-a49a-7ce676ba1d31)
3. Complete Feed Forward layer
4. Establish Residual Connection: add Scaled Values (output of Attention mechanism) and output of Feed Forward layer to input
5. Engineer Transformer class: assemble input embedding, positional embedding, TransformerLayer(s), output linear layer, and softmax layer
6. Train Model: complete Model Training class. Determine hyperparameters for best model performance.
7. Model Results: 99% accuracy on test dataset   
