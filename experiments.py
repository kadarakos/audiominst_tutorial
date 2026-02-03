from torch.optim import AdamW

from data import read_audio_mnist, collate_fn_digit
from model import AudioClassifier
from train import training_loop


def small_audio_cls(
    n_classes: int,
    sample_rate: int = 8000,
) -> AudioClassifier:
    return AudioClassifier(
        n_classes=n_classes,
        sample_rate=sample_rate,
        n_blocks=1,
        n_mels=100,
        hidden_size=100,
        n_heads=5,
        head_size=20,
        mlp_size=200,
    )


def baseline_gender():
    sample_rate = 8000
    model = small_audio_cls(n_classes=2, sample_rate=sample_rate)
    train_loader, test_loader = read_audio_mnist(
        n_test_speakers=10,
        resample=sample_rate,
        batch_size=16,
    )
    optimizer = AdamW(model.parameters())
    training_loop(
        model=model,
        optimizer=optimizer,
        epochs=10,
        train_loader=train_loader,
        test_loader=test_loader,
        ckpt_dir=".",
        name="gender_baseline",
    )


def baseline_digit():
    sample_rate = 8000
    model = small_audio_cls(n_classes=10, sample_rate=sample_rate)
    train_loader, test_loader = read_audio_mnist(
        n_test_speakers=10,
        resample=sample_rate,
        batch_size=16,
        collator=collate_fn_digit,
    )
    optimizer = AdamW(model.parameters())
    training_loop(
        model=model,
        optimizer=optimizer,
        epochs=10,
        train_loader=train_loader,
        test_loader=test_loader,
        ckpt_dir=".",
        name="gender_baseline",
    )


baseline_gender()
# baseline_digit()
