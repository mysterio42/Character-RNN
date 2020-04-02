import random
import string

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def load_model(model, path_extension: str):
    model.load_state_dict(torch.load(path_extension))


def generate_model_name(size=5):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(model, input_seq, target_seq, learning_rate, n_epochs, save_model):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(input_seq)
        target_seq = target_seq.view(-1).long()

        '''
        100,10   100
        
        42,17      42
        '''
        # print(f'output {output.size()}  target {target_seq.size()} ')
        # input()
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}/{n_epochs} Loss:{loss.item()}')
    if save_model:
        torch.save(model.state_dict(), 'weights/model-' + generate_model_name(5) + '.pkl')
