import re
import json

class QualityError(Exception):
    pass


def _has_fstrings(code: str) -> bool:
    # catches: f"..." f'...' and rf"..." fr"..." etc.
    return re.search(r'(?<![A-Za-z0-9_])([rR]?[fF]|[fF][rR])(["\'])', code) is not None


def _has_print_function(code: str) -> bool:
    # catches print( ... ) which is Py3 style; in Py2 it prints tuples and is not desired here
    return re.search(r'(?m)^\s*print\s*\(', code) is not None


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
            errors.append("Uses print() but Python 2 was selected. Use print statements, e.g. print 'x' or print('x') only if using from __future__ import print_function (not allowed here).")
        if "pathlib" in s:
            errors.append("Uses pathlib (Python 3 only).")
        if ":=" in s:
            errors.append("Uses walrus operator (Python 3 only).")
        if re.search(r'(?m)^\s*from\s+__future__\s+import\s+print_function\s*$', s):
            errors.append("Uses from __future__ import print_function; keep pure Abaqus Python 2.7 style.")

    # ---------- Abaqus ODB checks ----------
    if ("odb" in p) or ("openodb" in p) or ("odbaccess" in p) or ("openodb" in s.lower()):
        if "openOdb" not in s:
            errors.append("ODB task but openOdb is missing.")
        if "fieldOutputs" not in s:
            errors.append("ODB task but fieldOutputs not used.")
        if "getScalarField" in s and ("Mises" in s or "mises" in s):
            errors.append("Uses getScalarField(componentLabel='Mises'); forbidden. Prefer value.mises.")
        if "frameId" in s:
            warnings.append("Use frame.incrementNumber instead of frameId.")

    # ---------- Element set checks ----------
    if ("elset" in p) or ("element set" in p) or ("elementset" in p):
        if "getSubset(" not in s:
            errors.append("Element set requested but getSubset(region=...) is missing.")
        if re.search(r'instances\.keys\(\)\s*\[\s*0\s*\]', s):
            errors.append("Only first instance checked (instances.keys()[0]); must search all instances.")

    # ---------- Forbidden imports ----------
    if re.search(r'(?m)^\s*import\s+numpy\b', s) or re.search(r'(?m)^\s*from\s+numpy\b', s):
        warnings.append("numpy imported but not required for typical Abaqus scripts.")
    if re.search(r'(?m)^\s*import\s+(torch|transformers)\b', s):
        errors.append("Forbidden ML imports in Abaqus script.")

    return errors, warnings


def enforce_json(response_text: str):
    """
    Expect JSON with keys:
    assumptions, plan, script, how_to_run
    """
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
