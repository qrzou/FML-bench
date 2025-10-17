import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from model import EvolvingModel

from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import utils
from transform import EvolvingTransform


# num_workers = 24 # 8
# batch_size = 2048 # 512
num_workers = 16 # 8
batch_size = 512 # 512
memory_bank_size = 4096
seed = 1
max_epochs = 100
scaled_lr = 0.06 * (batch_size / 512)

path_to_train = "./datasets/cifar10/train/"
path_to_test = "./datasets/cifar10/test/"


pl.seed_everything(seed)


# disable blur because we're working with tiny images
transform = EvolvingTransform(
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
)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)




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


from pytorch_lightning.loggers import CSVLogger

csv_logger = CSVLogger(
    save_dir="results_tmp/",
    name="pretrain"
)
model = EvolvingModel(lr=scaled_lr, memory_bank_size=memory_bank_size, max_epochs=max_epochs)
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu", logger=csv_logger)
trainer.fit(model, dataloader_train_moco)


csv_logger = CSVLogger(
    save_dir="results_tmp/",
    name="linear_probing"
)
model.eval()
classifier = Classifier(model.backbone)
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu", logger=csv_logger)
trainer.fit(classifier, dataloader_train_classifier, dataloader_test)

# test the classifier
results = trainer.test(classifier, dataloaders=dataloader_test)
print("Final test results:", results)   # e.g. [{'test_acc': 0.8731}]



# ---------- Save to JSON ----------
test_acc = results[0]["test_acc"]
results = {
    "cifar10_linear_probing": {
        "means": {
            "test_acc_mean": test_acc
        },
        "stderrs": {
            "test_acc_stderr": 0.0,
        },
        "final_info_dict": {
            "test_acc": [test_acc]
        }
    }
}

from pathlib import Path
import json
output_path = Path("results_tmp/final_info.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to {output_path}")


