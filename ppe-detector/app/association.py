"""Containment math + per-person PPE assignment.

WHY containment instead of IoU: an equipment box (helmet, mask, glove...) is
tiny compared to a person box, so IoU is near-zero even for a perfect match.
Containment = intersection_area / equipment_box_area answers "what fraction
of the equipment sits inside this person?" — the right question for
small-in-large matching.
"""

from app import compliance, config


def containment(equip_box, person_box) -> float:
    ex1, ey1, ex2, ey2 = equip_box
    px1, py1, px2, py2 = person_box

    ix1, iy1 = max(ex1, px1), max(ey1, py1)
    ix2, iy2 = min(ex2, px2), min(ey2, py2)
    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    equip_area = max(0, ex2 - ex1) * max(0, ey2 - ey1)
    if equip_area <= 0:
        return 0.0

    return inter_area / equip_area


def assign_ppe(persons, equipment) -> dict:
    result = {}
    for person in persons:
        if person.track_id is None:
            continue
        result[person.track_id] = {
            "box": person.xyxy,
            "conf": person.confidence,
            "ppe": set(),
        }

    for item in equipment:
        best_track_id = None
        best_score = config.CONTAINMENT_THRESHOLD
        for person in persons:
            if person.track_id is None:
                continue
            score = containment(item.xyxy, person.xyxy)
            if score > best_score:
                best_score = score
                best_track_id = person.track_id
        if best_track_id is not None:
            ppe_item = config.PPE_CLASS_TO_ITEM.get(item.class_name, item.class_name)
            result[best_track_id]["ppe"].add(ppe_item)

    for data in result.values():
        data["missing"] = compliance.missing_ppe(data["ppe"])

    return result
