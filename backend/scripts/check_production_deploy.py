from __future__ import annotations

from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_DIR = REPO_ROOT / "k8s"

REQUIRED_FILES = (
    "00-namespace.yaml",
    "10-inference-deployment.yaml",
    "11-model-pvc.yaml",
    "12-inference-service.yaml",
    "13-ingress.yaml",
    "14-keda-scaledobject.yaml",
    "15-servicemonitor.yaml",
    "16-prometheus-rule.yaml",
    "17-pdb.yaml",
)

PLACEHOLDERS = (
    "YOUR_",
    "example.com",
)

REQUIRED_DEPLOYMENT_MARKERS = (
    "nvidia.com/gpu",
    "NATIVE_MODEL_PATH",
    "NATIVE_GPU_LAYERS",
)

REQUIRED_DOCKER_MARKERS = (
    "nvidia/cuda",
    "GGML_CUDA=ON",
    "llm_followups.server.inference_app:app",
)


def check_yaml(
    path: Path,
) -> list[str]:
    errors: list[str] = []

    try:
        content = path.read_text(
            encoding="utf-8"
        )

        documents = list(
            yaml.safe_load_all(content)
        )

    except Exception as exc:
        return [
            f"{path}: invalid YAML: {exc}"
        ]

    if not any(
        document is not None
        for document in documents
    ):
        errors.append(
            f"{path}: contains no YAML documents"
        )

    return errors


def find_placeholders(
    path: Path,
) -> list[str]:
    content = path.read_text(
        encoding="utf-8"
    )

    found: list[str] = []

    for placeholder in PLACEHOLDERS:
        if placeholder in content:
            found.append(
                placeholder
            )

    return found


def check_markers(
    path: Path,
    markers: tuple[str, ...],
) -> list[str]:
    content = path.read_text(
        encoding="utf-8"
    )

    missing = [
        marker
        for marker in markers
        if marker not in content
    ]

    return missing


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    print(
        "EC Pro production deployment preflight"
    )
    print(
        "=" * 40
    )

    for filename in REQUIRED_FILES:
        path = K8S_DIR / filename

        if not path.exists():
            errors.append(
                f"Missing Kubernetes file: {path}"
            )
            continue

        yaml_errors = check_yaml(
            path
        )

        errors.extend(
            yaml_errors
        )

        placeholders = find_placeholders(
            path
        )

        if placeholders:
            warnings.append(
                (
                    f"{filename}: unresolved "
                    f"placeholder(s): "
                    f"{', '.join(placeholders)}"
                )
            )

    deployment_path = (
        K8S_DIR
        / "10-inference-deployment.yaml"
    )

    if deployment_path.exists():
        missing = check_markers(
            deployment_path,
            REQUIRED_DEPLOYMENT_MARKERS,
        )

        for marker in missing:
            errors.append(
                (
                    "10-inference-deployment.yaml: "
                    f"missing required marker "
                    f"{marker!r}"
                )
            )

    dockerfile = (
        REPO_ROOT
        / "backend"
        / "Dockerfile.gpu"
    )

    if not dockerfile.exists():
        errors.append(
            f"Missing GPU Dockerfile: {dockerfile}"
        )

    else:
        missing = check_markers(
            dockerfile,
            REQUIRED_DOCKER_MARKERS,
        )

        for marker in missing:
            errors.append(
                (
                    "Dockerfile.gpu: "
                    f"missing required marker "
                    f"{marker!r}"
                )
            )

    print()

    if errors:
        print(
            "ERRORS"
        )

        for error in errors:
            print(
                f"  [FAIL] {error}"
            )

    else:
        print(
            "Structural checks: PASS"
        )

    print()

    if warnings:
        print(
            "DEPLOYMENT PLACEHOLDERS"
        )

        for warning in warnings:
            print(
                f"  [WAIT] {warning}"
            )

        print()
        print(
            "Production deployment is not "
            "ready until these environment-"
            "specific values are resolved."
        )

    else:
        print(
            "Deployment placeholders: NONE"
        )

    print()

    if errors:
        print(
            "Preflight result: FAIL"
        )
        return 1

    if warnings:
        print(
            "Preflight result: "
            "STRUCTURALLY READY, "
            "ENVIRONMENT NOT CONFIGURED"
        )
        return 0

    print(
        "Preflight result: "
        "PRODUCTION CONFIG READY"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())