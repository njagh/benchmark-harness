"""Example: Using llm_bench as a library from another project."""

from pathlib import Path

from bench_harness import (
    StorageConfig,
    ArtifactRegistry,
    ArtifactMetadataHook,
    RunSpec,
    ArtifactKind,
    ArtifactMode,
    RuntimeKind,
    LaunchMode,
)


def main() -> None:
    # 1. Configure storage — resolve from env var or project config
    config = StorageConfig.from_env()
    config.ensure_namespaces()
    print(f"Storage root: {config.root}")

    # 2. Use the artifact registry to track models
    registry = ArtifactRegistry(config)
    print(f"Registry path: {registry.path}")

    # 3. Create a run spec programmatically
    spec = RunSpec(
        name="library-example-run",
        project="example-project",
        tags=["example", "library"],
        artifact={
            "kind": ArtifactKind.hf_checkpoint,
            "mode": ArtifactMode.external_path,
            "path": "/tmp/example-model",
            "model_id": "example/model",
        },
        runtime={
            "kind": RuntimeKind.openai_compatible,
            "launch": LaunchMode.existing,
            "host": "127.0.0.1",
            "port": 8000,
        },
        workload={
            "prompt_suite": "coding_smoke",
            "max_tokens": 256,
            "temperature": 0.0,
            "num_runs": 3,
        },
    )

    print(f"Run spec: {spec.name} | artifact: {spec.artifact.kind}")

    # 4. Register the artifact in the registry
    from bench_harness.schemas import ModelArtifact

    model_artifact = ModelArtifact(
        artifact_id="example-001",
        kind=ArtifactKind.hf_checkpoint,
        mode=ArtifactMode.managed_copy,
        source_path="/tmp/example-model",
        model_id="example/model",
    )
    registry.register(model_artifact)
    print(f"Registered artifact: {model_artifact.artifact_id}")

    # 5. Look up the artifact
    lookup = registry.lookup("example-001")
    if lookup:
        print(f"Looked up artifact: {lookup.artifact_id} at {lookup.source_path}")

    # 6. List all registered artifacts
    all_artifacts = registry.list_all()
    print(f"Total artifacts in registry: {len(all_artifacts)}")

    # 7. Filter by kind
    hf_artifacts = registry.query(kind="hf_checkpoint")
    print(f"HF checkpoint artifacts: {len(hf_artifacts)}")

    # 8. Load a run spec from YAML
    example_yaml = Path("examples/modelopt_3070ti/run-endpoint-only.yaml")
    if example_yaml.exists():
        loaded_spec = RunSpec.from_yaml(example_yaml)
        print(f"Loaded spec from YAML: {loaded_spec.name}")
        print(f"  Artifact kind: {loaded_spec.artifact.kind}")
        print(f"  Runtime kind: {loaded_spec.runtime.kind}")

    print("\nLibrary usage example complete.")


if __name__ == "__main__":
    main()
