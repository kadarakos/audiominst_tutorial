from torch.optim import AdamW

from data import read_audio_mnist
from model import GenderClassifier
from train import training_loop


def baseline_gender():
    sample_rate = 8000
    train_loader, test_loader = read_audio_mnist(
        n_test_speakers=10,
        resample=sample_rate,
        batch_size=16,
    )
    model = GenderClassifier(
        sample_rate=sample_rate,
        n_blocks=1,
        n_mels=100,
        hidden_size=100,
        n_heads=5,
        head_size=20,
        mlp_size=200,
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
