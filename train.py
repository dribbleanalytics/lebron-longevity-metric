import torch
import time
from early_stopping import EarlyStopping


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    

def epoch_time(start_time, end_time):
    # gets time an epoch ran for
    elapsed = end_time - start_time
    elapsed_min = int(elapsed / 60)
    elapsed_sec = int(elapsed - (elapsed_min * 60))
    return elapsed_min, elapsed_sec


def train_epoch(model, iterator, optimizer, criterion, short_train):
    # trains one epoch of the model
    model.train()
    epoch_loss = 0
    for n, batch in enumerate(iterator):
        if short_train and n % 50 != 0:
            continue
        hidden = model.initHidden(batch[0].shape[0])
        batch_loss = torch.tensor(0., requires_grad=True)
        optimizer.zero_grad()
        output, hidden = model(batch[0], hidden)
        hidden = repackage_hidden(hidden)
        output = output.permute(1, 0)
        batch_loss = batch_loss + criterion(output, batch[1])
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    # evaluates model on valid and test sets
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            hidden = model.initHidden(batch[0].shape[0])
            output, hidden = model(batch[0], hidden)
            output = output.permute(1, 0)
            epoch_loss = epoch_loss + criterion(output, batch[1])
    return epoch_loss / len(iterator)


def train(model, train_iter, val_iter, test_iter, optimizer, criterion, n_epochs, short_train,
          checkpoint_name, patience, verbose=True):
    early_stopping = EarlyStopping(filename=checkpoint_name, patience=patience, verbose=verbose)
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_iter, optimizer, criterion, short_train)
        val_loss = evaluate(model, val_iter, criterion)
        end_time = time.time()

        epoch_min, epoch_sec = epoch_time(start_time, end_time)
        if verbose:
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_min}m {epoch_sec}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {val_loss:.3f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, reloading checkpoint model")
            model.load_state_dict(torch.load(checkpoint_name))
            break

    test_loss = evaluate(model, test_iter, criterion)
    model.test_loss = test_loss.item()
    print(f'Test Loss: {test_loss:.3f}')
    