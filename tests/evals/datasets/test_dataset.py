import pytest

from evals.datasets import DATASET_REGISTRY, Dataset, get_dataset, register_dataset


def test_public_datasets_are_registered():
    assert set(DATASET_REGISTRY) == {"babilong", "longbench-v2", "ruler"}


def test_unknown_dataset_lists_registered_names():
    with pytest.raises(SystemExit, match="Known: babilong, longbench-v2, ruler"):
        get_dataset("missing")


def test_duplicate_registration_is_rejected():
    @register_dataset("temporary")
    class First(Dataset):
        pass

    try:
        with pytest.raises(ValueError, match="already registered"):

            @register_dataset("temporary")
            class Second(Dataset):
                pass

    finally:
        DATASET_REGISTRY.pop("temporary")
