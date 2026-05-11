"""Config validator — validates generated config files (YAML, JSON, systemd, etc.)."""

from __future__ import annotations

import shutil
import json
import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates generated configuration files such as YAML, JSON, systemd units."""

    @staticmethod
    def validate_yaml(content: str, schema: dict | None = None) -> dict[str, Any]:
        """Validate YAML content.

        Args:
            content: YAML string to validate.
            schema: Optional JSON schema dict for structural validation.

        Returns:
            Dict with keys: valid, error, parsed, schema_valid.
        """
        try:
            import yaml
        except ImportError:
            return {
                "valid": False,
                "error": "PyYAML not installed",
                "parsed": None,
                "schema_valid": None,
            }

        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as e:
            return {
                "valid": False,
                "error": str(e),
                "parsed": None,
                "schema_valid": None,
            }

        schema_valid = None
        if schema is not None:
            try:
                import jsonschema
                jsonschema.validate(instance=parsed, schema=schema)
                schema_valid = True
            except jsonschema.ValidationError as e:
                schema_valid = False
            except Exception as e:
                logger.warning("Schema validation error: %s", e)
                schema_valid = None

        return {
            "valid": parsed is not None,
            "error": None,
            "parsed": parsed,
            "schema_valid": schema_valid,
        }

    @staticmethod
    def validate_json(content: str, schema: dict | None = None) -> dict[str, Any]:
        """Validate JSON content.

        Args:
            content: JSON string to validate.
            schema: Optional JSON schema dict for structural validation.

        Returns:
            Dict with keys: valid, error, parsed, schema_valid.
        """
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": str(e),
                "parsed": None,
                "schema_valid": None,
            }

        schema_valid = None
        if schema is not None:
            try:
                import jsonschema
                jsonschema.validate(instance=parsed, schema=schema)
                schema_valid = True
            except jsonschema.ValidationError as e:
                schema_valid = False
            except Exception as e:
                logger.warning("Schema validation error: %s", e)
                schema_valid = None

        return {
            "valid": True,
            "error": None,
            "parsed": parsed,
            "schema_valid": schema_valid,
        }

    @staticmethod
    def validate_systemd_unit(content: str) -> dict[str, Any]:
        """Validate a systemd unit file format using format-based checks.

        Checks for:
        - Required sections: [Unit], [Service], [Install]
        - ExecStart directive present in [Service]
        - Invalid directives

        Args:
            content: Systemd unit file content as a string.

        Returns:
            Dict with keys: valid, error, checks, details.
        """
        issues: list[str] = []
        sections_found: set[str] = set()
        has_exec_start = False
        in_service_section = False

        for line in content.splitlines():
            stripped = line.strip()

            if not stripped or stripped.startswith("#") or stripped.startswith(";"):
                continue

            section_match = stripped.startswith("[") and stripped.endswith("]")
            if section_match:
                section_name = stripped.strip("[]")
                sections_found.add(section_name)
                in_service_section = section_name == "Service"
                continue

            if in_service_section and stripped.startswith("ExecStart="):
                has_exec_start = True

        required_sections = ["Unit", "Service", "Install"]
        for section in required_sections:
            if section not in sections_found:
                issues.append(f"Missing required section: [{section}]")

        if not has_exec_start and "Service" in sections_found:
            issues.append("Missing required directive ExecStart in [Service]")

        valid = len(issues) == 0
        return {
            "valid": valid,
            "error": "; ".join(issues) if issues else None,
            "checks": {
                "has_unit_section": "Unit" in sections_found,
                "has_service_section": "Service" in sections_found,
                "has_install_section": "Install" in sections_found,
                "has_exec_start": has_exec_start,
            },
            "details": {
                "sections_found": sorted(sections_found),
                "issues": issues,
            },
        }

    @staticmethod
    def validate_systemd_unit_via_analyze(content: str, work_dir: str) -> dict[str, Any]:
        """Use systemd-analyze to validate a unit file if available.

        Falls back to format-based validation if systemd-analyze is not
        installed or not available.

        Args:
            content: Systemd unit file content.
            work_dir: Temporary directory to write the unit file.

        Returns:
            Dict with keys: valid, error, method, details.
        """
        if not shutil.which("systemd-analyze"):
            logger.debug("systemd-analyze not available, using format validation")
            result = ConfigValidator.validate_systemd_unit(content)
            return {
                "valid": result["valid"],
                "error": result["error"],
                "method": "format_validation",
                "details": result["details"],
            }

        unit_path = os.path.join(work_dir, "benchmark_unit.service")
        try:
            with open(unit_path, "w") as f:
                f.write(content)

            proc = subprocess.run(
                ["systemd-analyze", "verify", unit_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            stderr_output = proc.stderr.strip() if proc.stderr else ""
            stdout_output = proc.stdout.strip() if proc.stdout else ""
            combined_output = stderr_output or stdout_output

            valid = proc.returncode == 0
            return {
                "valid": valid,
                "error": combined_output or None,
                "method": "systemd-analyze",
                "details": {
                    "exit_code": proc.returncode,
                    "stderr": stderr_output,
                    "stdout": stdout_output,
                },
            }

        except FileNotFoundError:
            logger.debug("systemd-analyze not found, falling back to format validation")
            result = ConfigValidator.validate_systemd_unit(content)
            return {
                "valid": result["valid"],
                "error": result["error"],
                "method": "format_validation",
                "details": result["details"],
            }
        except subprocess.TimeoutExpired:
            return {
                "valid": False,
                "error": "systemd-analyze timed out after 30s",
                "method": "systemd-analyze",
                "details": {"exit_code": -1},
            }
        except Exception as e:
            logger.warning("systemd-analyze verification failed: %s", e)
            result = ConfigValidator.validate_systemd_unit(content)
            return {
                "valid": result["valid"],
                "error": f"systemd-analyze error: {e}; format validation: {result['error']}",
                "method": "format_validation_fallback",
                "details": result["details"],
            }
