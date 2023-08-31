#%%
import os
import utils as ut

#%%
vocab_path = os.getcwd()+"/../vocab.txt"
train_path = os.getcwd()+"/../data/train"
test_path = os.getcwd()+"/../data/dev"

#%%
vocab_dict = ut.get_word2ix(vocab_path)
train_files = ut.get_files(train_path)
test_files = ut.get_files(test_path)

# %%
context_size = 5
train_data = ut.process_data(train_files, context_size, vocab_dict)
test_data = ut.process_data(test_files, context_size, vocab_dict)
# %%
print(train_data[0])
# %%
