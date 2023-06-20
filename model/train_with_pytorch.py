import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def weighted_bce_loss(input, target, weight):
    input_class_zero = input[:, :, 0]
    bce = -(target * torch.log(input_class_zero) + (1 - target) * torch.log(1 - input_class_zero))
    bce = bce * weight
    return torch.sum(bce)

def object_detection_loss(input, target, weight):
    x = torch.round(target).int().tolist()
    samples_weight_test = weight[x]
    return torch.sum(samples_weight_test * (input - target) ** 2)

    '''b = target.size()[0]
    loss = 0
    for i in range(b):
        gt_object = target[i][0]
        if gt_object == 0:
            loss += (mseLoss(target[i][0], input[i][0]) * weight[0])
        else:
            loss += (mseLoss(target[i][0], input[i][0]) * weight[1])  # todo: only use confidence score
    loss /= b
    return loss'''

def train_with_pytorch(model, training_loader, validation_loader, path, EPOCHS, class_weight):
    loss_fn = object_detection_loss
    #mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    def train_one_epoch(epoch_index, tb_writer=None):
        running_loss = 0.
        last_loss = 0.

        for i, data in tqdm(enumerate(training_loader), total=len(training_loader)):
            '''inputs, labels, weight = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            weight = weight.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels, weight)'''

            # changed to OB
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels, class_weight)

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
    total_havent_change = 5
    havent_change = 0

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch(epoch_number, writer)
        # We don't need gradients on to do reporting
        model.eval()

        running_vloss = 0.0
        for i, vdata in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            '''vinputs, vlabels, vweight = vdata
            vinputs = vinputs.to(DEVICE)
            vlabels = vlabels.to(DEVICE)
            vweight = vweight.to(DEVICE)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels, vweight)'''

            # changed to OB
            inputs, labels = vdata
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            vloss = loss_fn(outputs, labels, class_weight)

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