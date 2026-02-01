import re

class QualityError(Exception):
    pass


def validate_script(script, python_version, prompt):
    errors = []
    warnings = []

    s = script

    # ---------- Python version checks ----------
    if python_version == "py2":
        if re.search(r'f["\']', s):
            errors.append("Uses f-strings but Python 2 was selected.")
        if "print(" in s:
            errors.append("Uses print() but Python 2 was selected.")
        if "pathlib" in s:
            errors.append("Uses pathlib (Python 3 only).")
        if ":=" in s:
            errors.append("Uses walrus operator (Python 3 only).")

    # ---------- Abaqus ODB checks ----------
    if "odb" in prompt.lower() or "opendb" in prompt.lower() or "openodb" in s.lower():
        if "openOdb" not in s:
            errors.append("ODB task but openOdb is missing.")

        if "fieldOutputs" not in s:
            errors.append("ODB task but fieldOutputs not used.")

        if "getScalarField" in s and "Mises" in s:
            warnings.append("Avoid getScalarField(componentLabel='Mises'); prefer value.mises.")

        if "frameId" in s:
            warnings.append("Use frame.incrementNumber instead of frameId.")

    # ---------- Element set checks ----------
    if "elset" in prompt.lower() or "element set" in prompt.lower():
        if "getSubset(" not in s:
            errors.append("Element set requested but stress field is not subset using getSubset(region=...).")

        if re.search(r'instances\.keys\(\)\[0\]', s):
            errors.append("Only first instance checked; must search all instances.")

    # ---------- Forbidden imports ----------
    if re.search(r'\bimport\s+numpy\b', s):
        warnings.append("numpy imported but not required for simple Abaqus scripts.")

    if re.search(r'\bimport\s+(torch|transformers)\b', s):
        errors.append("Forbidden ML imports in Abaqus script.")

    return errors, warnings


def enforce_json(response_text):
    """
    Expect JSON with keys:
    assumptions, plan, script, how_to_run
    """
    import json
    try:
        data = json.loads(response_text)
    except Exception:
        raise QualityError("Model did not return valid JSON.")

    required = ["assumptions", "plan", "script", "how_to_run"]
    for k in required:
        if k not in data:
            raise QualityError("Missing required JSON key: %s" % k)

    return data
