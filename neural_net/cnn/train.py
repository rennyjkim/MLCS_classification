from cnn_network import *

train_losses = []
for epoch in range(50):
    model.train()
    for inputs, labels in train_dataloader:
        logits = model(inputs)
        loss = loss_function(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += [loss.item()]
    print('[epoch:{}] train1 loss : {:.4f}'.format(epoch, np.mean(train_losses)))