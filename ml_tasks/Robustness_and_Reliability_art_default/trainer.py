from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.defences.preprocessor import Mixup
from art.defences.trainer import DPInstaHideTrainer
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess


def get_trainer(classifier):

    mixup = Mixup(num_classes=10, num_mix=2)

    trainer = DPInstaHideTrainer(
        classifier=classifier,
        augmentations=mixup,
        noise='laplacian',
        scale=0.3,
        clip_values=(0, 1)
    )
    return trainer

