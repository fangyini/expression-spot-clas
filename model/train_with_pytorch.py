import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def train_with_pytorch(model, training_loader, validation_loader, path, EPOCHS):
    loss_fn = weighted_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    def train_one_epoch(epoch_index, tb_writer=None):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels, weight = data
            inputs.to(DEVICE)
            labels.to(DEVICE)
            weight.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels, weight)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        last_loss = running_loss / 1000  # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        tb_x = epoch_index * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        running_loss = 0.

        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.
    total_havent_change = 15
    havent_change = 0

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(epoch_number, writer)
        # We don't need gradients on to do reporting
        model.eval()

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels, vweight = vdata
            vinputs.to(DEVICE)
            vlabels.to(DEVICE)
            vweight.to(DEVICE)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels, vweight)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), path+'/best')
            havent_change = 0
        else:
            havent_change += 1

        if havent_change > total_havent_change:
            return 0

        epoch_number += 1