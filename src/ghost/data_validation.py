"""Dataset validation reports for governed dataset loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ghost.config import GhostConfig, get_config
from ghost.datasets import DatasetSpec
from ghost.metadata_store import MetadataStore

if TYPE_CHECKING:
    from ghost.data_loading import LoadedDataset


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DataValidationReport:
    """Persisted dataset validation report."""

    report_id: str
    dataset_id: str
    dataset_version: str
    status: str
    issues: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DataValidationReport:
        return cls(**payload)


class DatasetValidator:
    """Validate loaded dataset arrays and persist a validation report."""

    def __init__(
        self,
        config: GhostConfig | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.config = config or get_config()
        self.metadata_store = metadata_store or MetadataStore(
            self.config.data_cache_dir / "metadata"
        )

    def validate_loaded_dataset(
        self,
        spec: DatasetSpec,
        dataset: LoadedDataset,
        *,
        dataset_version: str = "builtin-v1",
    ) -> DataValidationReport:
        issues: list[str] = []
        train_count = int(len(dataset.train_labels))
        eval_count = int(len(dataset.eval_labels))

        if train_count == 0:
            issues.append("Training split is empty")
        if eval_count == 0:
            issues.append("Evaluation split is empty")
        if len(dataset.train_features) != train_count:
            issues.append("Training features and labels are misaligned")
        if len(dataset.eval_features) != eval_count:
            issues.append("Evaluation features and labels are misaligned")

        observed_num_classes = int(len(np.unique(dataset.train_labels))) if train_count else 0
        if observed_num_classes > spec.num_classes:
            issues.append("Observed class count exceeds dataset specification")

        report = DataValidationReport(
            report_id=self._report_id(spec.dataset_id, dataset_version),
            dataset_id=spec.dataset_id,
            dataset_version=dataset_version,
            status="passed" if not issues else "failed",
            issues=issues,
            stats={
                "train_samples": train_count,
                "eval_samples": eval_count,
                "train_feature_shape": list(dataset.train_features.shape),
                "eval_feature_shape": list(dataset.eval_features.shape),
                "observed_num_classes": observed_num_classes,
            },
        )
        self.metadata_store.save_record(
            "dataset-validation-reports",
            report.report_id,
            report.to_dict(),
        )
        return report

    def get_report(
        self,
        dataset_id: str,
        dataset_version: str,
    ) -> DataValidationReport | None:
        payload = self.metadata_store.load_record(
            "dataset-validation-reports",
            self._report_id(dataset_id, dataset_version),
        )
        if not isinstance(payload, dict):
            return None
        return DataValidationReport.from_dict(payload)

    def _report_id(self, dataset_id: str, dataset_version: str) -> str:
        safe_dataset = dataset_id.replace("/", "-")
        safe_version = dataset_version.replace("/", "-")
        return f"{safe_dataset}@{safe_version}"