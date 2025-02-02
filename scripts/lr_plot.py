import math
import matplotlib.pyplot as plt

def lr(epoch, lr_initial):
    if epoch < 25:
        return lr_initial
    else:
        return lr_initial * math.exp(-0.1 * (epoch - 25))
def lr(epoch, lr_initial, lr_min=1e-6, max_epoch=50):
    return lr_min + 0.5 * (lr_initial - lr_min) * (1 + math.cos(math.pi * epoch / max_epoch))

# Parameters
initial_lr = 0.01
epochs = range(51)  # From epoch 0 to 50
learning_rates = [lr(epoch, initial_lr) for epoch in epochs]

# Plotting the learning rate
plt.plot(epochs, learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Decay over Epochs')
plt.grid(True)
plt.legend()
plt.show()