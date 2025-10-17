import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import MoCoV2Transform, utils
from lightly.utils.scheduler import cosine_schedule


# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# # PyTorch 2: allow TF32 via matmul precision
# try:
#     torch.set_float32_matmul_precision('high')
# except Exception:
#     pass



# num_workers = 24 # 8
# batch_size = 2048 # 512
num_workers = 16 # 8
batch_size = 512 # 512
memory_bank_size = 4096
seed = 1
max_epochs = 50


# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/
path_to_train = "/datasets/cifar10/train/"
path_to_test = "/datasets/cifar10/test/"


pl.seed_everything(seed)


# disable blur because we're working with tiny images
transform = MoCoV2Transform(
    input_size=32,
    gaussian_blur=0.0,
)

# Augmentations typically used to train on cifar-10
train_classifier_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)



# # We use the moco augmentations for training moco
# dataset_train_moco = LightlyDataset(input_dir=path_to_train, transform=transform)

# # Since we also train a linear classifier on the pre-trained moco model we
# # reuse the test augmentations here (MoCo augmentations are very strong and
# # usually reduce accuracy of models which are not used for contrastive learning.
# # Our linear layer will be trained using cross entropy loss and labels provided
# # by the dataset. Therefore we chose light augmentations.)
# dataset_train_classifier = LightlyDataset(
#     input_dir=path_to_train, transform=train_classifier_transforms
# )

# dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)




dataset_train_moco = torchvision.datasets.CIFAR10(
    root="datasets/cifar10",
    train=True,
    download=True,
    transform=transform
)

dataset_test = torchvision.datasets.CIFAR10(
    root="datasets/cifar10",
    train=False,
    download=True,
    transform=test_transforms
)

dataset_train_classifier = torchvision.datasets.CIFAR10(
    root="datasets/cifar10",
    train=True,
    download=True,
    transform=train_classifier_transforms
)




dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
#     pin_memory=True,
#     persistent_workers=True,
#     prefetch_factor=6
)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    # pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=6
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    # pin_memory=True,
    # persistent_workers=True,
    # prefetch_factor=6
)


# class MocoModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         resnet = torchvision.models.resnet18()
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])
#         self.projection_head = MoCoProjectionHead(512, 512, 128)

#         self.backbone_momentum = copy.deepcopy(self.backbone)
#         self.projection_head_momentum = copy.deepcopy(self.projection_head)

#         deactivate_requires_grad(self.backbone_momentum)
#         deactivate_requires_grad(self.projection_head_momentum)

#         self.criterion = NTXentLoss(memory_bank_size=(memory_bank_size, 128))

#     def forward(self, x):
#         query = self.backbone(x).flatten(start_dim=1)
#         query = self.projection_head(query)
#         return query

#     def forward_momentum(self, x):
#         key = self.backbone_momentum(x).flatten(start_dim=1)
#         key = self.projection_head_momentum(key).detach()
#         return key

#     def training_step(self, batch, batch_idx):
#         # print("batch type", type(batch))
#         # print("batch length", len(batch))
#         # print("batch[0]", batch[0])
#         # print("batch[1]", batch[1])
#         momentum = cosine_schedule(self.current_epoch, max_epochs, 0.996, 1)
#         update_momentum(self.backbone, self.backbone_momentum, m=momentum)
#         update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
#         x_query, x_key = batch[0]
#         query = self.forward(x_query)
#         key = self.forward_momentum(x_key)
#         loss = self.criterion(query, key)
#         self.log("train_loss_moco", loss,
#             on_step=True,   
#             on_epoch=True,  
#             prog_bar=True
#         )
#         return loss

#     def configure_optimizers(self):
#         optim = torch.optim.SGD(self.parameters(), lr=0.06)
#         return optim




class MocoModel(pl.LightningModule):
    def __init__(self, lr=0.06):
        super().__init__()

        self.lr = lr

        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18", 1, num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=(memory_bank_size, 128)
        )

    def training_step(self, batch, batch_idx):
        # (x_q, x_k), _, _ = batch
        x_q, x_k = batch[0]

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log(
            "train_loss_ssl", loss,
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        pass
        # for name, params in self.named_parameters():
        #     self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]




class Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # freeze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(512, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        # x, y, _ = batch
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss)
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        pass
        # for name, params in self.named_parameters():
        #     self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        # x, y, _ = batch
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        # calculate number of correct predictions
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.validation_step_outputs.append((num, correct))
        return num, correct

    def on_validation_epoch_end(self):
        # calculate and log top1 accuracy
        if self.validation_step_outputs:
            total_num = 0
            total_correct = 0
            for num, correct in self.validation_step_outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.test_step_outputs.append((num, correct))
        return num, correct

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            total_num, total_correct = 0, 0
            for num, correct in self.test_step_outputs:
                total_num += num
                total_correct += correct
            acc = (total_correct / total_num).item()
            self.log("test_acc", acc, on_epoch=True, prog_bar=True)
            self.print(f"[TEST] accuracy = {acc:.4f}")
            self.test_step_outputs.clear()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


scaled_lr = 0.06 * (batch_size / 512)
model = MocoModel(lr=scaled_lr)
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu",
)
trainer.fit(model, dataloader_train_moco)


model.eval()
classifier = Classifier(model.backbone)
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")
trainer.fit(classifier, dataloader_train_classifier, dataloader_test)

# test the classifier
results = trainer.test(classifier, dataloaders=dataloader_test)
print("Final test results:", results)   # e.g. [{'test_acc': 0.8731}]
