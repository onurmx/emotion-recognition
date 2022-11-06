import torch

def training_step(model, batch):
    x, y = batch 
    y_hat = model(x)
    training_loss = torch.nn.functional.cross_entropy(y_hat, y)
    return training_loss

def validation_per_batch(model, batch):
    x, y = batch 
    y_hat = model(x)
    validation_loss = torch.nn.functional.cross_entropy(y_hat, y)

    _, preds = torch.max(y_hat, dim=1)
    validation_accuracy = torch.tensor(torch.sum(preds == y).item() / len(preds))
    return {'val_loss': validation_loss.detach(), 'val_acc': validation_accuracy}

def validation_per_epoch(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_per_batch(model, batch) for batch in val_loader]
    return validation_per_epoch(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def logger(epoch, lrs, train_loss, val_loss, val_acc):
    print("Epoch [{}]: last_learning_rate: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, lrs, train_loss, val_loss, val_acc))

def fit(epochs, lr, model, train_loader, val_loader, factor=0.5, patience=5, weight_decay=0, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()

    optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=False)

    for epoch in range(epochs):
        model.train()
        train_losses = [] #for logging
        lrs = [] #for logging

        #training phase
        for batch in train_loader:
            loss = training_step(model, batch)
            train_losses.append(loss) #for logging
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer)) #for logging
        
        #validation phase
        validation_result = evaluate(model, val_loader)

        #update learning rate
        scheduler.step(validation_result['val_acc'])

        #logging
        logger(epoch, lrs[-1], torch.stack(train_losses).mean().item(), validation_result['val_loss'], validation_result['val_acc'])