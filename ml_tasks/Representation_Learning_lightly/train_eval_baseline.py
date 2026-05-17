import argparse
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from model import EvolvingModel

from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import utils
from transform import EvolvingTransform


num_workers = 16
batch_size = 512
memory_bank_size = 4096
seed = 1
max_epochs = 100
scaled_lr = 0.06 * (batch_size / 512)

path_to_train = "./datasets/cifar10/train/"
path_to_test = "./datasets/cifar10/test/"


def load_split_config(path="split_config.json"):
    """Load agent-controlled split config. Returns defaults if file missing."""
    defaults = {"val_ratio": 0.057, "val_seed": 42}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


def split_test_indices(targets, split_seed=42):
    """Split test dataset indices into val/test (30/70), stratified by class.

    Returns (val_indices, test_indices): 30% val, 70% test per class.
    """
    rng = np.random.RandomState(split_seed)
    val_indices = []
    test_indices = []

    for class_idx in range(10):
        class_mask = np.where(targets == class_idx)[0]
        shuffled = rng.permutation(class_mask)
        n_val = int(0.3 * len(shuffled))
        val_indices.extend(shuffled[:n_val])
        test_indices.extend(shuffled[n_val:])

    return np.array(val_indices), np.array(test_indices)


class TransformSubset(torch.utils.data.Dataset):
    """Wraps numpy arrays (data, targets) with a transform."""
    def __init__(self, data, targets, transform):
        self.data = data          # numpy uint8 array
        self.targets = targets    # numpy int array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


class Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        deactivate_requires_grad(backbone)
        self.fc = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.validation_step_outputs.append((num, correct))
        return num, correct

    def on_validation_epoch_end(self):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], required=True,
                        help='Which evaluation split to use: val or test')
    args = parser.parse_args()

    pl.seed_everything(seed)

    # ---------- Transforms ----------
    transform = EvolvingTransform(input_size=32, gaussian_blur=0.0)

    train_classifier_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ])

    # ========== Step 1: Load raw numpy arrays ==========
    cifar_train = torchvision.datasets.CIFAR10(
        root="datasets/cifar10", train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(
        root="datasets/cifar10", train=False, download=True)

    train_data = np.array(cifar_train.data)        # (50000, 32, 32, 3) uint8
    train_targets = np.array(cifar_train.targets)   # (50000,) int
    test_data = np.array(cifar_test.data)            # (10000, 32, 32, 3) uint8
    test_targets = np.array(cifar_test.targets)      # (10000,) int

    # ========== Step 2: Carve hidden test from test set (70%, stratified, seed=42) ==========
    val_orig_indices, hidden_test_indices = split_test_indices(test_targets, split_seed=42)
    # val_orig_indices ~ 3K, hidden_test_indices ~ 7K

    # ========== Step 3: Build visible pool = ALL train (50K) + val portion of test (3K) = 53K ==========
    visible_data = np.concatenate([train_data, test_data[val_orig_indices]], axis=0)
    visible_targets = np.concatenate([train_targets, test_targets[val_orig_indices]], axis=0)
    print(f"Visible pool size: {len(visible_data)} (train={len(train_data)} + test_val={len(val_orig_indices)})")

    # Hidden test data (for test path only)
    hidden_test_data = test_data[hidden_test_indices]
    hidden_test_targets = test_targets[hidden_test_indices]
    print(f"Hidden test size: {len(hidden_test_data)}")

    checkpoint_dir = "./model_checkpoint"

    if args.split == 'val':
        # ========== Step 4 (val): Read split_config, split visible pool ==========
        cfg = load_split_config()
        print(f"split_config: {cfg}")

        n_visible = len(visible_data)
        n_val = int(n_visible * cfg["val_ratio"])
        rng = np.random.RandomState(cfg["val_seed"])
        perm = rng.permutation(n_visible)
        agent_val_idx = perm[:n_val]
        agent_train_idx = perm[n_val:]

        agent_train_data = visible_data[agent_train_idx]
        agent_train_targets = visible_targets[agent_train_idx]
        agent_val_data = visible_data[agent_val_idx]
        agent_val_targets = visible_targets[agent_val_idx]
        print(f"Agent train: {len(agent_train_data)}, Agent val: {len(agent_val_data)}")

        # Build datasets with appropriate transforms
        dataset_train_moco = TransformSubset(agent_train_data, agent_train_targets, transform)
        dataset_train_classifier = TransformSubset(agent_train_data, agent_train_targets, train_classifier_transforms)
        dataset_val = TransformSubset(agent_val_data, agent_val_targets, test_transforms)

        # DataLoaders
        dataloader_train_moco = torch.utils.data.DataLoader(
            dataset_train_moco, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=num_workers)

        dataloader_train_classifier = torch.utils.data.DataLoader(
            dataset_train_classifier, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=num_workers)

        dataloader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers)

        # ----- Stage 1: MoCo Pretraining -----
        from pytorch_lightning.loggers import CSVLogger
        csv_logger = CSVLogger(save_dir="results_tmp/", name="pretrain")
        model = EvolvingModel(lr=scaled_lr, memory_bank_size=memory_bank_size, max_epochs=max_epochs)
        trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu", logger=csv_logger)
        trainer.fit(model, dataloader_train_moco)

        # Save pretrained backbone
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "backbone.pt"))
        print(f"Saved pretrained backbone to {checkpoint_dir}/backbone.pt")

        # ----- Stage 2: Linear Probing -----
        csv_logger = CSVLogger(save_dir="results_tmp/", name="linear_probing")
        model.eval()
        classifier = Classifier(model.backbone)
        trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu", logger=csv_logger)
        trainer.fit(classifier, dataloader_train_classifier, dataloader_val)

        # Save classifier
        torch.save(classifier.state_dict(), os.path.join(checkpoint_dir, "classifier.pt"))
        print(f"Saved classifier to {checkpoint_dir}/classifier.pt")

        # Evaluate on agent_val
        dataloader_eval = dataloader_val
        results = trainer.test(classifier, dataloaders=dataloader_eval)

    else:
        # ========== Step 4 (test): Load checkpoint, evaluate on hidden test ==========
        from pytorch_lightning.loggers import CSVLogger

        dataset_eval = TransformSubset(hidden_test_data, hidden_test_targets, test_transforms)
        dataloader_eval = torch.utils.data.DataLoader(
            dataset_eval, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers)

        # Load pretrained backbone
        print(f"Loading pretrained backbone from {checkpoint_dir}/backbone.pt (skipping MoCo pretraining)")
        model = EvolvingModel(lr=scaled_lr, memory_bank_size=memory_bank_size, max_epochs=max_epochs)
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "backbone.pt"), map_location="cuda"))

        # Load trained classifier
        model.eval()
        classifier = Classifier(model.backbone)
        classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, "classifier.pt"), map_location="cuda"))
        print(f"Loaded classifier from {checkpoint_dir}/classifier.pt (skipping linear probe training)")

        csv_logger = CSVLogger(save_dir="results_tmp/", name="linear_probing")
        trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu", logger=csv_logger)
        results = trainer.test(classifier, dataloaders=dataloader_eval)

    print(f"Final {args.split} results:", results)

    eval_acc = results[0]["test_acc"]

    # ---------- Save to JSON ----------
    output = {
        "cifar10_linear_probing": {
            "means": {
                "test_acc_mean": eval_acc
            },
            "stderrs": {
                "test_acc_stderr": 0.0,
            },
            "final_info_dict": {
                "test_acc": [eval_acc]
            }
        }
    }

    os.makedirs('results_tmp', exist_ok=True)
    output_path = f'results_tmp/{args.split}_info.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
