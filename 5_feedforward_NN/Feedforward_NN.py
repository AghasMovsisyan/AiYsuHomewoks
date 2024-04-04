import torch
import torchvision
import torchvision.transforms as transforms

# Define the transform to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Get input feature size
a, _ = trainset[0]
input_feat_no = a.shape[-2] * a.shape[-1]

# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(Net, self).__init__()
        input_size, h1, h2, output_size = layer_sizes
        self.fc1 = torch.nn.Linear(input_size, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, output_size)

    def forward(self, x):
        x = x.view(-1, x.shape[-2] * x.shape[-1])
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define layer sizes
layer_sizes = [input_feat_no, 256, 128, 10]

# Create network object
net = Net(layer_sizes)

# Define the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Number of epochs and steps for printing loss
epoch_num = 2
steps_for_loss = 200

# Train the neural network
for epoch in range(epoch_num):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % steps_for_loss == steps_for_loss - 1:
            print(f'epoch={epoch + 1}, step={i + 1} loss: {running_loss / steps_for_loss}')
            running_loss = 0.0

print('Finished Training')

# Test the neural network on test data
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}')
