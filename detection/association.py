"""Associate plate detections with tracked vehicles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from records import PlateDet, VehicleDet


@dataclass(frozen=True)
class PlateAssociation:
    plate: PlateDet
    vehicle: Optional[VehicleDet]  # None => unassociated ("orphan") plate
    overlap: float                 # fraction of the plate area inside the vehicle box


def associate_plates(plates: Sequence[PlateDet], vehicles: Sequence[VehicleDet],
                     min_overlap: float = 0.5, one_per_vehicle: bool = False) -> List[PlateAssociation]:
    """Attach each plate to the vehicle box containing most of it.

    Uses the fraction of the PLATE inside the vehicle (not IoU) so partially visible
    plates still match. Ties prefer the smaller vehicle box (nested/overlapping vehicles).
    Plates are never dropped: if no vehicle qualifies, or a vehicle already has a
    stronger plate this frame, the plate is returned unassociated.
    """
    chosen: List[Optional[VehicleDet]] = []
    overlaps: List[float] = []
    for plate in plates:
        best, best_key = None, None
        for veh in vehicles:
            frac = plate.bbox.fraction_inside(veh.bbox)
            if frac < min_overlap:
                continue
            key = (frac, -veh.bbox.area)
            if best_key is None or key > best_key:
                best, best_key = veh, key
        chosen.append(best)
        overlaps.append(best_key[0] if best_key else 0.0)

    if one_per_vehicle:
        winners = {}
        for i, veh in enumerate(chosen):
            if veh is None:
                continue
            vid = id(veh)
            if vid not in winners or plates[i].confidence > plates[winners[vid]].confidence:
                winners[vid] = i
        for i, veh in enumerate(chosen):
            if veh is not None and winners[id(veh)] != i:
                chosen[i], overlaps[i] = None, 0.0

    return [PlateAssociation(plates[i], chosen[i], overlaps[i]) for i in range(len(plates))]
