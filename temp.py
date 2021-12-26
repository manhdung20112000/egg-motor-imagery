import torch
from torchinfo import summary
import torchmetrics
from model import EGGNet, Classifier

# pred = torch.rand((32, 1))
# label = torch.rand((32))

# loss_func = torch.nn.BCELoss()
# accuracy = torchmetrics.Accuracy()
# print(pred.shape)
# print(pred.squeeze().shape)

# print(loss_func(pred.squeeze().float(), label.float()))
# print(accuracy(pred.squeeze(), label.int()))

# model = EGGNet(n_channels=25)
# print(summary(model, (1, 1, 25, 126)))

model = Classifier(n_channels=25)
model.load_from_checkpoint('./model/pretrain_weight.ckpt')
model.eval()

print(model.forward(torch.rand((1, 1, 25, 126))))