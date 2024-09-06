from train import *

# Testing phase
n = 0.
test_loss = 0.
test_acc = 0.
c1.eval()
for test_inputs, test_labels in test_dataloader:
    logits = c1(test_inputs)
    test_loss += F.cross_entropy(logits, test_labels, reduction='sum')
    test_acc += (logits.argmax(dim=1) == test_labels).float().sum()
    n += test_inputs.size(0)

test_loss /= n
test_acc /= n
print('test_acc: ', test_acc)
print('test_loss', test_loss)