from pathlib import Path


def extract_annotations(p: str | Path) -> list:
    """Reads EAF file and produces a list of (start_ms, end_ms)
    tuples where filled pauses were annotated.

    :param str | Path p: input EAF file
    :return list:list of (start_ms, end_ms) filled pauses.
    """
    from lxml import etree as ET

    doc = ET.fromstring(Path(p).read_bytes())
    ees = doc.findall(".//ALIGNABLE_ANNOTATION")
    if len(ees) == 0:
        return []
    timeline = {
        i.get("TIME_SLOT_ID"): i.get("TIME_VALUE") for i in doc.findall(".//TIME_SLOT")
    }
    return [
        (
            int(timeline[i.get("TIME_SLOT_REF1")]),
            int(timeline[i.get("TIME_SLOT_REF2")]),
        )
        for i in ees
    ]
