from pathlib import Path

import pytest

from llm_followups.tuning.dataset import (
    DatasetConfig,
)
from llm_followups.tuning.train import (
    TrainConfig,
    build_lora_config,
    validate_train_config,
)


def make_config(
    **overrides,
) -> TrainConfig:
    values = {
        "model_name": "test-model",
        "output_dir": Path("test-output"),
        "dataset": DatasetConfig(),
        "device": "cpu",
    }

    values.update(overrides)

    return TrainConfig(**values)


def test_default_lora_configuration() -> None:
    cfg = make_config()

    lora = build_lora_config(cfg)

    assert lora.r == 16
    assert lora.lora_alpha == 32
    assert lora.lora_dropout == 0.05

    assert lora.target_modules == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }


def test_default_lora_config_validates() -> None:
    validate_train_config(
        make_config()
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "lora_r",
            0,
            "lora_r must be >= 1",
        ),
        (
            "lora_alpha",
            0,
            "lora_alpha must be >= 1",
        ),
        (
            "lora_dropout",
            -0.01,
            "lora_dropout must be",
        ),
        (
            "lora_dropout",
            1.0,
            "lora_dropout must be",
        ),
        (
            "lora_target_modules",
            (),
            "lora_target_modules",
        ),
        (
            "lora_target_modules",
            ("q_proj", ""),
            "lora_target_modules",
        ),
    ],
)
def test_invalid_lora_configuration(
    field: str,
    value,
    message: str,
) -> None:
    cfg = make_config(
        **{
            field: value,
        }
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        validate_train_config(cfg)


def test_lora_validation_skipped_when_disabled() -> None:
    cfg = make_config(
        use_lora=False,
        lora_r=0,
        lora_alpha=0,
        lora_dropout=2.0,
        lora_target_modules=(),
    )

    validate_train_config(cfg)
