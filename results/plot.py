import matplotlib.pyplot as plt

f_SGD = open('SGD.csv', 'r')
f_explicit = open('explicit.csv', 'r')

# training accuracy
train_loss_SGD      = [float(x) for x in f_SGD.readline().split(',')[1:]]
train_loss_explicit = [float(x) for x in f_explicit.readline().split(',')[1:]]

plt.xlabel('batches')
plt.ylabel('training loss')
plt.plot(train_loss_SGD, label='SGD')
plt.plot(train_loss_explicit, label='subsampled Newton')
plt.legend()
# plt.show()
plt.savefig('train_loss.png')
plt.clf()

# testing accuracy
test_acc_SGD      = [float(x) for x in f_SGD.readline().split(',')[1:]]
test_acc_explicit = [float(x) for x in f_explicit.readline().split(',')[1:]]

plt.xlabel('epochs')
plt.ylabel('testing accuracy')
plt.plot(test_acc_SGD, label='SGD')
plt.plot(test_acc_explicit, label='subsampled Newton')
plt.legend()
# plt.show()
plt.savefig('test_acc.png')
plt.clf()
