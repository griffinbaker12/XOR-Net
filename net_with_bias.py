import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate=0.1):
        self.inodes = 2
        self.hnodes = 2
        self.onodes = 1
        self.lr = learning_rate

        self.wih = np.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)
        )

        self.bias_hidden = np.random.normal(0.0, 1, (self.hnodes, 1))
        self.bias_output = np.random.normal(0.0, 1, (self.onodes, 1))

    def forward(self, inputs_list):
        i = np.array(inputs_list, ndmin=2).T

        hi = np.dot(self.wih, i) + self.bias_hidden
        ho = self.sigmoid(hi)

        fi = np.dot(self.who, ho) + self.bias_output
        fo = self.sigmoid(fi)

        return i, ho, fo

    def train(self, inputs_list, target):
        i, ho, fo = self.forward(inputs_list)

        targets = np.array(target, ndmin=2).T

        # Calculate errors
        output_error = targets - fo
        hidden_error = np.dot(self.who.T, output_error * fo * (1 - fo))

        self.who += self.lr * np.dot((output_error * fo * (1 - fo)), ho.T)
        self.wih += self.lr * np.dot((hidden_error * ho * (1 - ho)), i.T)

        self.bias_output += self.lr * (output_error * fo * (1 - fo))
        self.bias_hidden += self.lr * (hidden_error * ho * (1 - ho))

        return 0.5 * np.sum(output_error**2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def generate_xor_data():
    return (
        ([0, 0], 0.01),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0.01),
    )


nn = NeuralNetwork()


epochs = 100000
losses = []

for epoch in range(epochs):
    total_loss = 0
    for i, o in generate_xor_data():
        loss = nn.train(i, o)
        total_loss += loss
    avg_loss = total_loss / 4
    losses.append(avg_loss)
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")


for test_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    _, _, output = nn.forward(test_input)
    print(f"Input: {test_input}, Output: {output[0][0]:.4f}")
    print(f"iho weights: {nn.wih}, who weights: {nn.who}")
    print(f"activated weights: {nn.sigmoid(nn.wih)}, {nn.sigmoid(nn.who)}")

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.savefig("loss_plot.png")
plt.show()
