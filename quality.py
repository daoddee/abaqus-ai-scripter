import re
import json

class QualityError(Exception):
    pass


def _has_fstrings(code: str) -> bool:
    return re.search(r'(?<![A-Za-z0-9_])([rR]?[fF]|[fF][rR])(["\'])', code) is not None


def _has_print_function(code: str) -> bool:
    return re.search(r'(?m)^\s*print\s*\(', code) is not None


def _prompt_wants_last_step(prompt_l: str) -> bool:
    triggers = [
        "last step", "last analysis step", "select the last step",
        "last frame", "select the last frame", "automatically select the last",
        "final frame", "final step"
    ]
    return any(t in prompt_l for t in triggers)


def _prompt_implies_instance_elset(prompt_l: str) -> bool:
    triggers = [
        "assembly or instance", "assembly level or instance level",
        "instance level", "unknown instance", "unknown instance name",
        "may exist at the assembly", "may exist at either the assembly or instance",
        "may exist either at the assembly or instance"
    ]
    return any(t in prompt_l for t in triggers)


def _script_searches_all_instances(script: str) -> bool:
    # We require a loop over instances keys/names AND access to instance.elementSets
    # Accept common patterns
    patterns = [
        r'for\s+\w+\s+in\s+assembly\.instances\.keys\(\)\s*:',
        r'for\s+\w+\s+in\s+\w+\.instances\.keys\(\)\s*:',
        r'for\s+\w+\s+in\s+assembly\.instances\s*:',
        r'for\s+\w+\s+in\s+\w+\.instances\s*:',
    ]
    has_loop = any(re.search(p, script) for p in patterns)
    touches_elsets = re.search(r'instances\[[^\]]+\]\.elementSets', script) is not None or re.search(r'\.elementSets\[[^\]]+\]', script) is not None
    return bool(has_loop and touches_elsets)


def _script_loops_all_steps(script: str) -> bool:
    # Catch common "scan all steps" patterns
    return (
        re.search(r'for\s+\w+\s+in\s+odb\.steps\.values\(\)\s*:', script) is not None
        or re.search(r'for\s+\w+\s+in\s+odb\.steps\s*:', script) is not None
        or re.search(r'for\s+\w+\s+in\s+odb\.steps\.keys\(\)\s*:', script) is not None
    )


def _script_selects_last_step_explicitly(script: str) -> bool:
    # Accept common patterns that explicitly pick the last step/frame
    patterns = [
        r'list\(odb\.steps\.keys\(\)\)',
        r'odb\.steps\.keys\(\)\s*\[\s*-1\s*\]',
        r'stepNames\s*=\s*odb\.steps\.keys\(\)',
        r'lastStepName\s*=.*odb\.steps\.keys\(\)\s*\[\s*-1\s*\]',
        r'lastStep\s*=\s*odb\.steps\[.*\]',
    ]
    has_step_pick = any(re.search(p, script) for p in patterns)
    has_last_frame = re.search(r'\.frames\[\s*-1\s*\]', script) is not None
    return bool(has_step_pick and has_last_frame)


def validate_script(script, python_version, prompt):
    errors = []
    warnings = []

    s = script or ""
    p = (prompt or "").lower()

    # ---------- Python version checks ----------
    if python_version == "py2":
        if _has_fstrings(s):
            errors.append("Uses f-strings but Python 2 was selected.")
        if _has_print_function(s):
            errors.append("Uses print() but Python 2 was selected. Use print statements + % formatting.")
        if "pathlib" in s:
            errors.append("Uses pathlib (Python 3 only).")
        if ":=" in s:
            errors.append("Uses walrus operator (Python 3 only).")
        if re.search(r'(?m)^\s*from\s+__future__\s+import\s+print_function\s*$', s):
            errors.append("Uses from __future__ import print_function; keep pure Abaqus Python 2.7 style.")

    # ---------- ODB / Abaqus checks ----------
    odb_like = ("odb" in p) or ("openodb" in p) or ("odbaccess" in p) or ("openodb" in s.lower())
    if odb_like:
        if "openOdb" not in s:
            errors.append("ODB task but openOdb is missing.")
        if "fieldOutputs" not in s:
            errors.append("ODB task but fieldOutputs not used.")
        if "getScalarField" in s and ("Mises" in s or "mises" in s):
            errors.append("Uses getScalarField(componentLabel='Mises'); forbidden. Prefer value.mises.")
        if "frameId" in s:
            # promoted to ERROR
            errors.append("Uses frameId; must use frame.incrementNumber for increment reporting.")

    # ---------- Last step/last frame enforcement ----------
    if _prompt_wants_last_step(p):
        if _script_loops_all_steps(s):
            errors.append("Prompt requests last step/last frame; script loops over all steps. Must explicitly select last step and last frame.")
        if not _script_selects_last_step_explicitly(s):
            errors.append("Prompt requests last step/last frame; script does not explicitly select last step name and last frame index.")

    # ---------- Element set enforcement ----------
    if ("elset" in p) or ("element set" in p) or ("elementset" in p) or ("hotspot_elems" in p) or ("critical_elems" in p):
        if "getSubset(" not in s:
            errors.append("Element set requested but getSubset(region=...) is missing.")
        if re.search(r'instances\.keys\(\)\s*\[\s*0\s*\]', s):
            errors.append("Only first instance checked (instances.keys()[0]); must search all instances.")

    if _prompt_implies_instance_elset(p):
        if not _script_searches_all_instances(s):
            errors.append("Prompt implies elset may be instance-level with unknown instance; script must search ALL instances for the element set.")

    # ---------- Forbidden imports ----------
    if re.search(r'(?m)^\s*import\s+numpy\b', s) or re.search(r'(?m)^\s*from\s+numpy\b', s):
        warnings.append("numpy imported but not required for typical Abaqus scripts.")
    if re.search(r'(?m)^\s*import\s+(torch|transformers)\b', s):
        errors.append("Forbidden ML imports in Abaqus script.")

    return errors, warnings


def enforce_json(response_text: str):
    try:
        data = json.loads(response_text)
    except Exception:
        raise QualityError("Model did not return valid JSON (parse failed).")

    required = ["assumptions", "plan", "script", "how_to_run"]
    for k in required:
        if k not in data:
            raise QualityError("Missing required JSON key: %s" % k)

    if not isinstance(data["assumptions"], list):
        raise QualityError("JSON key 'assumptions' must be a list.")
    if not isinstance(data["plan"], list):
        raise QualityError("JSON key 'plan' must be a list.")
    if not isinstance(data["script"], str):
        raise QualityError("JSON key 'script' must be a string.")
    if not isinstance(data["how_to_run"], str):
        raise QualityError("JSON key 'how_to_run' must be a string.")

    return data
