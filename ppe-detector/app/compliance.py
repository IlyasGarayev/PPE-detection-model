"""Safe/unsafe rule derived from a person's detected PPE set."""

from app import config


def is_safe(ppe_set: set) -> bool:
    return all(item in ppe_set for item in config.REQUIRED_PPE)


def missing_ppe(ppe_set: set) -> list:
    return [item for item in config.REQUIRED_PPE if item not in ppe_set]
