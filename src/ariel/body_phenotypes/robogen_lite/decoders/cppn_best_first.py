import math

import mujoco
import networkx as nx
import numpy as np
import numpy.typing as npt

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_TYPES_OF_MODULES,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleRotationsTheta,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

# Matches has_self_collision's own default -- contacts/gaps shallower than this
# are treated as legitimate touching (adjacent modules meeting at a shared
# face), not real interpenetration.
PENETRATION_TOLERANCE = -0.005


def softmax(raw_scores: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    e_x = np.exp(raw_scores - np.max(raw_scores))
    return e_x / e_x.sum()


# ──────────────────────────────────────────────────────────────────────────
# Lightweight forward kinematics, used ONLY to decide bilateral-symmetry
# pairing (see MorphologyDecoderBestFirst._find_global_mirror_slot below).
#
# The abstract grid (`face_deltas`/`_get_child_coords`) treats every face as a
# fixed GLOBAL step regardless of accumulated module rotation, so it agrees
# with real 3D geometry only near the core -- once a module along a growth
# path has a nonzero `rotation` (DEG_45/90/...), the grid's notion of "which
# way is left/right" silently drifts away from the TRUE world-space left/right
# (confirmed empirically: a module attached via a chain of rotated ancestors
# can land at a nonzero real-world X even though the grid still calls it
# "on-axis"). Symmetry pairing needs the real answer, not the grid's.
#
# These tables are built ONCE (module import time) by compiling each module
# type in isolation and reading MuJoCo's own resolved site frames -- not by
# trusting any single spec attribute directly, since a site can live on a
# body nested below the module's own outer body (HingeModule's FRONT site is
# defined on `rotor`, itself offset from the outer `hinge` body that actually
# gets attached elsewhere -- reading `site.pos` directly silently drops that
# offset). Validated against a real evolved 15-module body: composing these
# tables reproduces MuJoCo's own compiled xpos to <0.0004 and xquat-derived
# rotation matrices to within accumulated float rounding from `rotate()`'s
# own `decimals=3` truncation (~0.003 by 14 levels deep).
def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                          (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                          0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _site_transforms_relative_to_body(module, body_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every site's (pos, R) relative to `body_name`'s own frame, resolved via
    real forward kinematics on this module compiled standalone -- correct
    regardless of how many sub-bodies the site is actually nested under."""
    model = module.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    body_pos = data.xpos[body_id].copy()
    body_R = data.xmat[body_id].reshape(3, 3).copy()
    out = {}
    for face, site in module.sites.items():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site.name)
        site_pos = data.site_xpos[site_id].copy()
        site_R = data.site_xmat[site_id].reshape(3, 3).copy()
        out[face.name] = (body_R.T @ (site_pos - body_pos), body_R.T @ site_R)
    return out


def _build_fk_tables():
    site_table = {
        "CORE": _site_transforms_relative_to_body(CoreModule(index=0), "core"),
        "BRICK": _site_transforms_relative_to_body(BrickModule(index=1), "brick"),
        "HINGE": _site_transforms_relative_to_body(HingeModule(index=1), "hinge"),
    }
    rotation_table = {}
    for mtype, cls in [("BRICK", BrickModule), ("HINGE", HingeModule)]:
        rotation_table[mtype] = {}
        for rname in ModuleRotationsTheta.__members__:
            inst = cls(index=1)
            inst.rotate(ModuleRotationsTheta[rname].value)
            rotation_table[mtype][rname] = _quat_to_R(np.array(inst.body.quat, dtype=float))
    rotation_table["CORE"] = {"DEG_0": np.eye(3)}
    return site_table, rotation_table


_SITE_TABLE, _ROTATION_TABLE = _build_fk_tables()

# The core is never rotated (ALLOWED_ROTATIONS[CORE] == [DEG_0]) and has no
# parent, so its local frame == world frame always -- making world-space
# "left/right" a fixed, well-defined axis for the whole body regardless of
# how any downstream module is twisted. CORE's own RIGHT/LEFT sites differ
# only in local X (confirmed via _SITE_TABLE), so that's the axis.
GLOBAL_MIRROR_AXIS = 0
GLOBAL_MIRROR_EPS = 1e-3

# Radial symmetry turns about the core's own UP axis: CORE's TOP/BOTTOM sites
# differ only in local Z (confirmed via _SITE_TABLE), same way RIGHT/LEFT pin
# down the mirror axis above. This is the axis a starfish-like body is radial
# about -- so an n-fold body keeps n copies of every limb spread around the
# core in the ground plane, rather than stacked vertically.
GLOBAL_RADIAL_AXIS = 2

# Every symmetry operation acts about the core's geometric CENTER, not about
# world (0,0,0) -- `world_pos[core]` is the core BODY's frame origin, and the
# core's box geom is deliberately offset from it (core.py: pos=[0, 0.1, 0]),
# so the two are not the same point. Derived from the opposing site pairs
# rather than hardcoded, so it tracks CORE_DIMENSIONS if those ever change.
# Only matters for rotations: the mirror plane happens to pass through the
# body origin too (CORE_CENTER[GLOBAL_MIRROR_AXIS] == 0), which is why plain
# `x *= -1` was correct before this was spelled out.
CORE_CENTER = np.array([
    (_SITE_TABLE["CORE"]["LEFT"][0][0] + _SITE_TABLE["CORE"]["RIGHT"][0][0]) / 2,
    (_SITE_TABLE["CORE"]["BACK"][0][1] + _SITE_TABLE["CORE"]["FRONT"][0][1]) / 2,
    (_SITE_TABLE["CORE"]["BOTTOM"][0][2] + _SITE_TABLE["CORE"]["TOP"][0][2]) / 2,
])


def _symmetry_ops(bilateral: bool, radial_folds: int) -> list[npt.NDArray[np.float64]]:
    """Linear parts of the symmetry group the body must be invariant under,
    identity FIRST, each applied about CORE_CENTER (see _symmetry_images).

    `radial_folds` n>1 contributes the n rotations about GLOBAL_RADIAL_AXIS;
    `bilateral` contributes the reflection in GLOBAL_MIRROR_AXIS. Both
    together give those rotations AND every rotation-composed-with-reflection
    -- i.e. the full dihedral group of order 2n, which is what "stacked"
    means here: a 4-fold radial body that is also bilaterally symmetric has
    8 copies of an off-axis limb, not 4 plus a mirror of one of them.
    Returning the composed set (rather than the two generators) is what lets
    a single pass over it reach the whole orbit of a position."""
    rotations = [np.eye(3)]
    if radial_folds > 1:
        u, v = (i for i in range(3) if i != GLOBAL_RADIAL_AXIS)
        for k in range(1, radial_folds):
            theta = 2.0 * math.pi * k / radial_folds
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rot = np.eye(3)
            rot[u, u], rot[u, v] = cos_t, -sin_t
            rot[v, u], rot[v, v] = sin_t, cos_t
            rotations.append(rot)
    if not bilateral:
        return rotations
    mirror = np.eye(3)
    mirror[GLOBAL_MIRROR_AXIS, GLOBAL_MIRROR_AXIS] = -1.0
    return [*rotations, *(mirror @ rot for rot in rotations)]


class MorphologyDecoderBestFirst:
    """Decodes a CPPN using a true greedy, best-first search strategy."""

    def __init__(
        self,
        cppn_genome: Genome,
        max_modules: int = 20,
        allow_variable_size: bool = False,
        check_geometry: bool = True,
        min_hinges: int = 2,
        max_consecutive_hinges: int = 0,
        block_core_front_face: bool = False,
        max_consecutive_same_face: int = 0,
        hinge_any_parent_face: bool = True,
        bilateral_symmetry: bool = True,
        radial_symmetry: int = 0,
    ):
        """`allow_variable_size`: if True, let the type output's argmax land on
        ModuleType.NONE (candidate is then dropped by the existing
        ALLOWED_FACES[NONE]=[] check), so decode() can stop growing before
        max_modules. If False (default, unchanged from prior behavior), NONE
        is masked out of every argmax, so decode() only stops early once the
        frontier runs out of free coordinate/face slots -- it will otherwise
        always grow to exactly max_modules.

        `check_geometry`: if True (default), each candidate is compiled and
        checked for REAL 3D penetration against every already-placed module
        (via mj_geomDistance) before being accepted -- occupied_coords alone
        only rejects two modules sharing the exact same integer grid cell, it
        has no idea that a rotated module's actual geometry can swing into a
        cell occupied by a structurally distant (non-adjacent-in-the-graph)
        module. A candidate that would overlap is dropped in favor of the
        next-best-scoring one instead of accepting it; if every candidate
        this step would overlap, growth stalls here (same outcome as running
        out of candidates entirely).

        `min_hinges`: a body with 0-1 hinges can't be meaningfully controlled
        (utils/worker.py's `model.nu < 2` guard). Rather than patch such a
        body up after the fact, decode() just discards it (returns an empty
        graph -- see the bottom of decode()) once normal growth finishes with
        fewer than this many HINGE modules. The caller already treats an
        empty graph as an invalid individual (CPPNNeatGenotype.decode raises,
        evaluate_batch scores it 0), so the genome gets replaced by the outer
        evolutionary loop next generation rather than retried here -- decode()
        is a pure function of a fixed genome, there's nothing to vary on a
        literal in-place retry. 0 disables this (any hinge count accepted).

        `max_consecutive_hinges`: caps how many HINGE modules may chain
        back-to-back (parent HINGE -> child HINGE) along any single growth
        path before a HINGE candidate there is rejected outright (falling
        through to the next-best candidate, same as a geometry rejection).
        0 disables this (no cap).

        `block_core_front_face`: if True, the core's FRONT attachment site
        (the fixed direction SPAWN_ROTATION points at the target, see
        config.py) is never offered as a growth candidate -- so a body can't
        be a single straight limb sticking out of the core's own "nose"
        toward the target. Only applies to the core; every other module's
        own FRONT face is unaffected.

        `max_consecutive_same_face`: caps how many times in a row a growth
        path may keep reusing the SAME attachment face -- i.e. how long a
        straight run can get, regardless of module type (generalizes
        max_consecutive_hinges, which only caps HINGE-type runs: a straight
        line of BRICKs, or a BRICK/HINGE mix, is capped just the same here).
        A candidate that would extend a straight run past this is rejected
        outright, same as a geometry rejection, falling through to the
        next-best candidate -- best-first is globally greedy and a chain
        that starts scoring well tends to keep winning every step over
        starting a fresh branch elsewhere, so left alone this is normally
        how a body ends up almost entirely one straight "snake" (measured:
        ~60% of bodies were a single unbranched, single-face chain with this
        off). 0 disables this (no cap).

        `hinge_any_parent_face`: ariel's own ALLOWED_FACES (config.py) lists
        HINGE as attachable via FRONT only -- checked identically regardless
        of what the PARENT is, so as shipped a hinge can't attach to a
        brick's or the core's LEFT/RIGHT/TOP/BOTTOM/BACK either, not just to
        another hinge's sides (a hinge only having one physical site is a
        real geometric constraint -- see hinge.py, only one add_site() call
        -- ALLOWED_FACES[HINGE]=[FRONT] just happens to also apply that same
        restriction to non-hinge parents, which isn't a geometric necessity:
        construct_mjspec_from_graph's attach_body() only ever reads the
        PARENT's own site, never the child's, so mechanically nothing stops
        a hinge docking onto a brick's TOP). True (default) relaxes this: a
        HINGE child may use any face its ACTUAL parent exposes when that
        parent isn't itself a HINGE. A hinge parent still only ever offers
        FRONT regardless of this flag -- that part isn't a policy choice,
        it's `ALLOWED_FACES[HINGE]` filtering the PARENT's own candidate
        faces earlier in the same loop, so hinge-onto-hinge sideways
        placement stays impossible either way. False reverts to ariel's
        stock behavior (FRONT-only for a hinge child no matter the parent).

        `bilateral_symmetry`: True (default, the long-standing behavior)
        enforces mirror symmetry across the fixed global plane through the
        core (GLOBAL_MIRROR_AXIS): a module placed off that plane is only
        ever accepted together with a same-typed module at its mirrored
        world position. False places modules with no symmetry constraint at
        all (and drops the matching fold from the CPPN's own input encoding,
        see _canonical_coords).

        `radial_symmetry`: n-fold rotational symmetry about the core's up
        axis (GLOBAL_RADIAL_AXIS), enforced the same way: an off-axis module
        is only accepted together with same-typed modules at ALL n-1 of its
        rotated world positions, so a body gets n identical limbs spread
        around the core instead of one. 0 or 1 disables it. STACKS with
        `bilateral_symmetry` -- both on means the body must be invariant
        under the full dihedral group (n rotations plus n reflected
        rotations, see _symmetry_ops), giving 2n copies of an off-axis,
        off-plane limb.

        Only n=2 and n=4 are geometrically reachable with this module set:
        the modules attach face-to-face on an axis-aligned cubic lattice
        about the core, and only quarter turns map that lattice onto itself
        (verified: the core's four side sites form one exact 4-orbit, and so
        do their descendants). A larger or odd n is accepted but has almost
        no valid off-axis placement left -- every candidate needing an
        unreachable partner slot is dropped, so bodies collapse toward the
        rotation axis and tend to be discarded by `min_hinges`.

        Both options are hard guarantees, not biases: a candidate whose
        required partner slots don't all exist is dropped in favor of the
        next-best-scoring one, and the whole set is placed (and geometry-
        checked, and rolled back) as one unit -- never half-placed. The set
        also has to fit the remaining `max_modules` budget, so a symmetric
        group is never what pushes a body over its module count."""
        self.cppn_genome = cppn_genome
        self.max_modules = max_modules
        self.allow_variable_size = allow_variable_size
        self.check_geometry = check_geometry
        self.min_hinges = min_hinges
        self.max_consecutive_hinges = max_consecutive_hinges
        self.block_core_front_face = block_core_front_face
        self.max_consecutive_same_face = max_consecutive_same_face
        self.hinge_any_parent_face = hinge_any_parent_face
        self.bilateral_symmetry = bilateral_symmetry
        self.radial_symmetry = radial_symmetry if radial_symmetry > 1 else 0
        self.symmetry_ops = _symmetry_ops(
            self.bilateral_symmetry, self.radial_symmetry)
        self.face_deltas = {
            ModuleFaces.FRONT: (1, 0, 0),
            ModuleFaces.BACK: (-1, 0, 0),
            ModuleFaces.TOP: (0, 1, 0),
            ModuleFaces.BOTTOM: (0, -1, 0),
            ModuleFaces.RIGHT: (0, 0, 1),
            ModuleFaces.LEFT: (0, 0, -1),
        }

    _OPPOSITE_FACE = {ModuleFaces.FRONT: ModuleFaces.BACK, ModuleFaces.BACK: ModuleFaces.FRONT}

    def _get_child_coords(self, parent_pos: tuple, face: ModuleFaces, x_inverted: bool = False) -> tuple:
        """`face_deltas` is a fixed GLOBAL step (FRONT always +1 on the same
        axis, everywhere), not relative to how the parent itself was reached.
        That's harmless for every face except BACK: BACK's delta is the exact
        negation of FRONT's, so a module that arrived via BACK and then tries
        to place a child via its own FRONT (the only face HINGE ever offers,
        and BRICK's dominant one too) would walk straight back onto its own
        parent's cell -- already occupied, so it silently, permanently
        dead-ends every BACK-rooted branch (measured: subtree size stuck at
        exactly 1, 80/80, before this fix). No other face pair has this
        problem: a module's own preferred continuation is FRONT, a different
        axis than LEFT/RIGHT/TOP/BOTTOM, so those never cancel out.

        `x_inverted`: True for a module whose lineage has an odd number of
        BACK attachments (see decode()) -- swaps which global step FRONT/BACK
        resolve to for THIS candidate only, so "continue via my own FRONT"
        means "keep moving away from where I came from" instead of "walk
        back into my parent." Only affects grid-position bookkeeping here --
        the actual attachment label used for ALLOWED_FACES/physical MuJoCo
        attachment (see construct_mjspec_from_graph) is untouched."""
        effective_face = self._OPPOSITE_FACE.get(face, face) if x_inverted else face
        delta = self.face_deltas[effective_face]
        return (
            parent_pos[0] + delta[0],
            parent_pos[1] + delta[1],
            parent_pos[2] + delta[2],
        )

    # Grid axes (face_deltas): FRONT/BACK is index 0, TOP/BOTTOM index 1,
    # RIGHT/LEFT index 2 -- so the grid's own counterpart of
    # GLOBAL_RADIAL_AXIS (the up axis) is index 1, and radial folding turns
    # in the (0, 2) plane around the core's cell at (0,0,0).
    _GRID_RADIAL_AXIS = 1

    def _canonical_coords(self, pos: tuple) -> tuple:
        """One abstract-grid coordinate folded into a representative that is
        the same for every position in its symmetry orbit, so symmetry-paired
        candidates are indistinguishable to the CPPN and therefore always
        score identically (see _cppn_inputs). Approximate in exactly the way
        the grid itself is (it drifts from real geometry under accumulated
        rotation) -- this only steers SCORING, never which placements are
        legal, which _find_symmetry_slots decides against real geometry."""
        if not self.radial_symmetry:
            # Unchanged original behavior: mirroring the left-right axis is
            # just abs() on it.
            lr = abs(pos[2]) if self.bilateral_symmetry else pos[2]
            return (pos[0], pos[1], lr)
        # Rotations mix two axes, so fold in polar form: keep the distance
        # from the rotation axis, and reduce the angle into one n-fold
        # sector (halved again when a reflection is also in the group -- the
        # dihedral orbit's representative).
        u, v = (i for i in range(3) if i != self._GRID_RADIAL_AXIS)
        rho = math.hypot(pos[u], pos[v])
        sector = 2.0 * math.pi / self.radial_symmetry
        phi = math.atan2(pos[v], pos[u]) % sector
        if self.bilateral_symmetry:
            phi = min(phi, sector - phi)
        out = [0.0, 0.0, 0.0]
        out[u], out[v] = rho * math.cos(phi), rho * math.sin(phi)
        out[self._GRID_RADIAL_AXIS] = pos[self._GRID_RADIAL_AXIS]
        return tuple(out)

    def _cppn_inputs(self, parent_pos: tuple, child_pos: tuple) -> list:
        """Builds the CPPN's 8 input values from a raw parent/child coordinate
        pair (RIGHT=(0,0,1)/LEFT=(0,0,-1) in face_deltas, so index 2 is the
        left-right axis):

        - symmetry folding of both positions (_canonical_coords): under
          bilateral symmetry that's abs() on index 2, so a left-side and its
          mirrored right-side candidate become indistinguishable to the
          network; they then always score identically and best-first tends
          to place both sides close together in growth order. Radial
          symmetry folds the angle about the up axis the same way, so all n
          rotated copies of a candidate score identically too. Doesn't cover
          hinge rotation (DEG_0/45/90 isn't auto-folded), only module
          type/placement.
        - appended radial distance from the core (0,0,0), for parent and
          child both, matching how the rest of this input already carries
          both positions rather than forcing the network to derive a delta
          itself. Gives gauss/hat direct access to "near core" vs "distant
          limb" without evolution needing to first discover x^2+y^2+z^2 via
          the `square` activation option. Computed from the RAW position,
          which is already symmetry-invariant on its own."""
        parent_sym = self._canonical_coords(parent_pos)
        child_sym = self._canonical_coords(child_pos)
        r_parent = math.sqrt(sum(c * c for c in parent_pos))
        r_child = math.sqrt(sum(c * c for c in child_pos))
        return [*parent_sym, *child_sym, r_parent, r_child]

    def _child_world_pos(self, parent_id, face: ModuleFaces, world_pos: dict, world_R: dict,
                          module_data: dict) -> np.ndarray:
        """Real (not abstract-grid) world position a child would land at if
        attached to `parent_id` via `face` -- independent of the CHILD's own
        `rotation` attribute, since rotating a module twists its own body
        (and therefore where ITS children land) but never moves its own
        origin, which is fixed entirely by the PARENT's site transform. Safe
        to call on a purely tentative (parent_id, face) pair -- reads only
        already-accepted nodes' tracked world_pos/world_R."""
        parent_type = module_data[parent_id]["type"].name
        site_pos, _site_R = _SITE_TABLE[parent_type][face.name]
        return world_pos[parent_id] + world_R[parent_id] @ site_pos

    def _child_world_pose(self, parent_id, face: ModuleFaces, child_type: ModuleType, child_rot: ModuleRotationsIdx,
                           world_pos: dict, world_R: dict, module_data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Full (position, orientation) a child attached via (parent_id, face)
        would have -- orientation additionally composes the child's OWN
        `rotation` attribute, since that's what determines where ITS future
        children land."""
        parent_type = module_data[parent_id]["type"].name
        site_pos, site_R = _SITE_TABLE[parent_type][face.name]
        pos = world_pos[parent_id] + world_R[parent_id] @ site_pos
        own_R = _ROTATION_TABLE[child_type.name][child_rot.name]
        R = world_R[parent_id] @ site_R @ own_R
        return pos, R

    def _face_ok(self, face: ModuleFaces, parent_type: ModuleType, child_type: ModuleType) -> bool:
        if face not in ALLOWED_FACES[parent_type]:
            return False
        return face in ALLOWED_FACES[child_type] or (
            self.hinge_any_parent_face
            and child_type == ModuleType.HINGE
            and parent_type != ModuleType.HINGE
        )

    def _open_faces(self, node_id, robot_graph: nx.DiGraph, module_data: dict) -> list[ModuleFaces]:
        used = {robot_graph.edges[node_id, c]["face"] for c in robot_graph.successors(node_id)}
        node_type = module_data[node_id]["type"]
        return [f for f in ALLOWED_FACES[node_type] if f.name not in used]

    def _symmetry_images(self, world_pos: np.ndarray) -> list[np.ndarray]:
        """Every OTHER world position the enabled symmetry group maps
        `world_pos` onto -- i.e. its orbit, minus the position itself, which
        a symmetry operation can legitimately map to (a module sitting on the
        mirror plane, or on the rotation axis, is already its own image and
        needs no partner at all; that is this decoder's old "on-axis" case,
        generalized). "Already its own image" is measured as a distance
        between the two POSITIONS, the same way a partner slot is matched --
        so for the mirror it means |x| <= eps/2, where the old hardcoded test
        read |x| <= eps. Nothing real sits in that band (placements land on
        exact lattice multiples of 0.05, and the FK tables reproduce MuJoCo
        to <0.0004), and erring toward "needs a partner" is the safe side.

        Operations act about CORE_CENTER rather than the core body origin --
        the two differ (see CORE_CENTER), which is invisible for the mirror
        but would put every rotation slightly off its own lattice.

        Duplicates are folded together within GLOBAL_MIRROR_EPS: distinct
        group operations often agree on a given position (e.g. a limb lying
        in the mirror plane of a 4-fold radial body), and each distinct
        TARGET POSITION is what costs a module, not each operation."""
        images: list[np.ndarray] = []
        offset = world_pos - CORE_CENTER
        for op in self.symmetry_ops:
            image = CORE_CENTER + op @ offset
            if float(np.linalg.norm(image - world_pos)) <= GLOBAL_MIRROR_EPS:
                continue
            if any(float(np.linalg.norm(image - seen)) <= GLOBAL_MIRROR_EPS
                   for seen in images):
                continue
            images.append(image)
        return images

    def _find_symmetry_slots(
        self, targets: list[np.ndarray], child_type: ModuleType,
        robot_graph: nx.DiGraph, module_data: dict, world_pos: dict, world_R: dict,
        occupied_coords: dict, taken_slot: tuple, taken_coord: tuple,
    ) -> list[tuple[int, ModuleFaces, tuple]] | None:
        """Where a same-typed partner module needs to attach, for each of
        `targets`, for the WHOLE body -- not just one parent/child pair -- to
        be symmetric under the enabled group (see _symmetry_images).

        Replaces the old grid-based _mirror_target, which assumed the mirror
        always attaches to "the parent's grid-mirror via the opposite
        LEFT/RIGHT face" -- true only near the core. Once accumulated
        rotation has carried a branch's local sense of left/right away from
        the true global one (confirmed: an ordinary BOTTOM-attached spine
        module can drift to a nonzero real-world X while the grid still
        calls it on-axis), the correct partner attachment can be a DIFFERENT
        parent and a DIFFERENT face entirely -- so this searches every
        already-accepted node's open faces for whichever one's real-world
        position actually lands on each target, rather than assuming which
        one it must be.

        Searching only ALREADY-ACCEPTED nodes is sufficient, not a
        limitation: symmetry is enforced from the first module on, so the
        body is invariant at the start of every step, which means the orbit
        of a depth-d slot consists of depth-d slots hanging off depth-(d-1)
        modules -- all of them placed in earlier steps.

        A module's own `rotation` twist doesn't change where ITS OWN origin
        sits (see _child_world_pos), so this search -- and the resulting
        pairing -- is entirely independent of the twist any module in the
        chain happens to have; only real-world POSITION and TYPE need to
        match, matching the project's own symmetry definition.

        `taken_slot`/`taken_coord` are the primary candidate's own
        (parent, face) and grid cell: it is being placed in this same step,
        so no partner may claim it, even though the graph doesn't show it as
        used yet.

        Returns None as soon as ANY target has no open (parent, face) slot
        within GLOBAL_MIRROR_EPS -- the caller drops the whole candidate
        rather than placing it partly-symmetric, the same hard-guarantee
        principle the old mechanism used. Each target consumes its slot, so
        two targets can never be served by one module."""
        slots = []
        for node_id, data in module_data.items():
            parent_type = data["type"]
            for face in self._open_faces(node_id, robot_graph, module_data):
                if (node_id, face) == taken_slot:
                    continue
                if not self._face_ok(face, parent_type, child_type):
                    continue
                if child_type == ModuleType.HINGE and self.max_consecutive_hinges:
                    streak = data["hinge_streak"] + 1 if parent_type == ModuleType.HINGE else 1
                    if streak > self.max_consecutive_hinges:
                        continue
                if self.max_consecutive_same_face:
                    streak = data["same_face_streak"] + 1 if data["incoming_face"] == face else 1
                    if streak > self.max_consecutive_same_face:
                        continue
                abstract_pos = self._get_child_coords(
                    data["pos"], face, data["x_inverted"])
                if abstract_pos in occupied_coords or abstract_pos == taken_coord:
                    continue
                pos = self._child_world_pos(node_id, face, world_pos, world_R, module_data)
                slots.append((node_id, face, pos, abstract_pos))

        chosen: list[tuple[int, ModuleFaces, tuple]] = []
        used_slots: set[tuple] = set()
        used_coords: set[tuple] = set()
        for target in targets:
            best = None
            best_err = GLOBAL_MIRROR_EPS
            for node_id, face, pos, abstract_pos in slots:
                if (node_id, face) in used_slots or abstract_pos in used_coords:
                    continue
                err = float(np.linalg.norm(pos - target))
                if err < best_err:
                    best = (node_id, face, abstract_pos)
                    best_err = err
            if best is None:
                return None
            chosen.append(best)
            used_slots.add((best[0], best[1]))
            used_coords.add(best[2])
        return chosen

    def _check_and_get_names(
        self, robot_graph: nx.DiGraph, prev_names: frozenset[str]
    ) -> tuple[bool, frozenset[str]]:
        """Compiles robot_graph as it currently stands (every previously
        accepted module plus the tentative newest one) and checks only
        NEW-vs-OLD geom pairs for real penetration -- a module's own internal
        geoms (e.g. a hinge's stator/rotor, which sit close together by
        design) are never compared against each other here, only against
        geoms that existed before this step.

        "New" is identified by NAME-SET DIFFERENCE against `prev_names`, not
        by geom index range: MuJoCo compiles a spec's geom array by tree
        structure, not by attach_body call order, so when a rejected
        candidate is retried against a *different* parent (same tentative
        graph-node id, different attachment point), its geoms can land
        anywhere in the compiled array -- an index-range boundary silently
        checks the wrong geoms in that case. Names are exact regardless of
        where the compiler places anything (confirmed empirically: switching
        from index-range to this fixed a real, otherwise-undetected
        interpenetration bug in this exact method).

        Returns (overlaps, names) so a successful check's compiled name set
        can be reused as next step's `prev_names` baseline instead of
        recompiling for it."""
        try:
            spec_obj = construct_mjspec_from_graph(robot_graph)
            model = spec_obj.spec.compile()
        except Exception:
            return True, prev_names   # unbuildable -- reject this candidate
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        name_of = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
        current_names = frozenset(name_of)
        new_idx = [i for i, nm in enumerate(name_of) if nm not in prev_names]
        old_idx = [i for i, nm in enumerate(name_of) if nm in prev_names]

        for i in new_idx:
            for j in old_idx:
                if model.geom_bodyid[i] == model.geom_bodyid[j]:
                    continue
                if mujoco.mj_geomDistance(model, data, i, j, 10.0, None) < PENETRATION_TOLERANCE:
                    return True, current_names
        return False, current_names

    def decode(self) -> nx.DiGraph:
        robot_graph = nx.DiGraph()
        occupied_coords = {}
        module_data = {}

        core_id, core_pos, core_type, core_rot = (
            IDX_OF_CORE,
            (0, 0, 0),
            ModuleType.CORE,
            ModuleRotationsIdx.DEG_0,
        )
        robot_graph.add_node(
            core_id, type=core_type.name, rotation=core_rot.name
        )
        occupied_coords[core_pos] = core_id
        module_data[core_id] = {
            "pos": core_pos,
            "type": core_type,
            "rot": core_rot,
            "hinge_streak": 0,
            "incoming_face": None,      # no parent -- any first hop off the core starts a fresh streak
            "same_face_streak": 0,
            "x_inverted": False,
        }

        # The frontier now contains ALL modules with potential open faces.
        frontier = [core_id]
        next_module_id = 1

        # Real (not abstract-grid) world pose per accepted node -- see
        # _find_global_mirror_slot's docstring for why this is what symmetry
        # pairing actually needs to be checked against.
        world_pos = {core_id: np.zeros(3)}
        world_R = {core_id: np.eye(3)}

        prev_names: frozenset[str] = frozenset()
        if self.check_geometry:
            core_spec = construct_mjspec_from_graph(robot_graph)
            core_model = core_spec.spec.compile()
            prev_names = frozenset(
                mujoco.mj_id2name(core_model, mujoco.mjtObj.mjOBJ_GEOM, i)
                for i in range(core_model.ngeom)
            )

        # Check for the max module count.
        while len(robot_graph) < self.max_modules:
            potential_connections = []

            # At each step, we check the ENTIRE frontier of all existing modules.
            for parent_id in frontier:
                parent_pos = module_data[parent_id]["pos"]
                parent_type = module_data[parent_id]["type"]
                parent_x_inverted = module_data[parent_id]["x_inverted"]
                for face in ModuleFaces:
                    if face not in ALLOWED_FACES[parent_type]:
                        continue
                    if self.block_core_front_face and parent_id == core_id and face == ModuleFaces.FRONT:
                        continue

                    child_pos = self._get_child_coords(parent_pos, face, parent_x_inverted)

                    if child_pos in occupied_coords:
                        continue

                    cppn_inputs = self._cppn_inputs(parent_pos, child_pos)
                    raw_outputs = self.cppn_genome.activate(cppn_inputs)

                    conn_score = raw_outputs[0]
                    type_scores = np.array(
                        raw_outputs[1 : 1 + NUM_OF_TYPES_OF_MODULES]
                    )
                    rot_scores = np.array(
                        raw_outputs[1 + NUM_OF_TYPES_OF_MODULES :]
                    )

                    type_probs = softmax(type_scores)

                    if not self.allow_variable_size:
                        type_probs[
                            ModuleType.NONE.value
                        ] = -1.0  # Ignore NONE if that's the output
                    type_probs[
                        ModuleType.CORE.value
                    ] = -1.0  # Ignore CORE if that's the output

                    child_type = ModuleType(np.argmax(type_probs))
                    child_rot = ModuleRotationsIdx(
                        np.argmax(softmax(rot_scores))
                    )

                    face_ok = face in ALLOWED_FACES[child_type] or (
                        self.hinge_any_parent_face
                        and child_type == ModuleType.HINGE
                        and parent_type != ModuleType.HINGE
                    )
                    if not (face_ok and child_rot in ALLOWED_ROTATIONS[child_type]):
                        continue

                    hinge_streak = 0
                    if child_type == ModuleType.HINGE:
                        hinge_streak = (
                            module_data[parent_id]["hinge_streak"] + 1
                            if parent_type == ModuleType.HINGE else 1
                        )
                        if self.max_consecutive_hinges and hinge_streak > self.max_consecutive_hinges:
                            continue

                    same_face_streak = (
                        module_data[parent_id]["same_face_streak"] + 1
                        if module_data[parent_id]["incoming_face"] == face else 1
                    )
                    if self.max_consecutive_same_face and same_face_streak > self.max_consecutive_same_face:
                        continue

                    child_x_inverted = parent_x_inverted != (face == ModuleFaces.BACK)

                    # Where the enabled symmetry group maps this candidate in
                    # REAL 3D space (not in the abstract grid, whose
                    # child_pos[2] != 0 only agrees with real geometry near
                    # the core -- see _find_symmetry_slots' docstring): the
                    # candidate needs a same-typed partner module SOMEWHERE
                    # in the body at EACH of those positions, found by
                    # searching real geometry rather than assumed from the
                    # grid. An image that coincides with the candidate itself
                    # (on the mirror plane / on the rotation axis) asks for
                    # nothing, so such a module places alone as before.
                    # Drop the whole candidate if any partner slot is
                    # missing, rather than finding out only after
                    # tentatively placing the primary -- same hard-guarantee
                    # principle as before, just checked against the real
                    # world instead of the grid.
                    child_world_pos = self._child_world_pos(parent_id, face, world_pos, world_R, module_data)
                    partners: list[tuple[int, ModuleFaces, tuple]] = []
                    images = self._symmetry_images(child_world_pos)
                    if images:
                        # A symmetric group is placed all-or-nothing, so it
                        # has to fit the remaining budget as a whole -- an
                        # n-fold body would otherwise overshoot max_modules
                        # by up to 2n-1 modules on its last step.
                        if len(robot_graph) + 1 + len(images) > self.max_modules:
                            continue
                        found = self._find_symmetry_slots(
                            images, child_type, robot_graph, module_data,
                            world_pos, world_R, occupied_coords,
                            taken_slot=(parent_id, face), taken_coord=child_pos)
                        if found is None:
                            continue
                        partners = found

                    potential_connections.append({
                        "score": conn_score,
                        "parent_id": parent_id,
                        "child_pos": child_pos,
                        "child_world_pos": child_world_pos,
                        "child_type": child_type,
                        "child_rot": child_rot,
                        "face": face,
                        "hinge_streak": hinge_streak,
                        "same_face_streak": same_face_streak,
                        "x_inverted": child_x_inverted,
                        # [(partner_parent_id, partner_face, partner_pos), ...]
                        "partners": partners,
                    })

            if not potential_connections:
                # Expected, frequent outcome once allow_variable_size=True lets
                # the type channel legitimately choose NONE everywhere on the
                # current frontier -- this is how bodies stop growing early,
                # not an error. No console spam; decode() just returns
                # whatever's built so far.
                break

            # Try candidates best-score-first; if check_geometry, validate
            # each against real 3D overlap before accepting -- reject only
            # the offending candidate and fall through to the next-best one,
            # rather than discarding the whole body and starting over. An
            # off-axis candidate's symmetry partners (already resolved
            # structurally above) are added and checked JOINTLY with the
            # primary here, and the whole set is rolled back together on
            # overlap -- so a candidate is only ever accepted as a complete
            # symmetric group, never half-placed.
            potential_connections.sort(key=lambda x: x["score"], reverse=True)

            accepted = None
            accepted_partner_ids: list[int] = []
            child_id = next_module_id
            for cand in potential_connections:
                robot_graph.add_node(
                    child_id, type=cand["child_type"].name, rotation=cand["child_rot"].name
                )
                robot_graph.add_edge(
                    cand["parent_id"], child_id, face=cand["face"].name
                )
                added_ids = [child_id]

                partner_ids = []
                for offset, (partner_parent_id, partner_face, _pos) in enumerate(
                    cand["partners"], start=1
                ):
                    partner_id = child_id + offset
                    # Rotation is copied verbatim, never negated or turned:
                    # a partner's own twist no longer needs to relate to the
                    # primary's at all -- global symmetry (the thing that's
                    # actually being guaranteed now) only requires matching
                    # TYPE at the symmetric POSITION, found above
                    # independently of any side's rotation attribute.
                    robot_graph.add_node(
                        partner_id, type=cand["child_type"].name, rotation=cand["child_rot"].name,
                    )
                    robot_graph.add_edge(partner_parent_id, partner_id, face=partner_face.name)
                    added_ids.append(partner_id)
                    partner_ids.append(partner_id)

                if self.check_geometry:
                    overlaps, names = self._check_and_get_names(robot_graph, prev_names)
                    if overlaps:
                        for nid in added_ids:
                            robot_graph.remove_node(nid)   # also drops tentative edges
                        continue
                    prev_names = names

                accepted = cand
                accepted_partner_ids = partner_ids
                break

            if accepted is None:
                # Every candidate this step would overlap the body built so
                # far -- stop growing here (same outcome as running out of
                # candidates entirely).
                break

            occupied_coords[accepted["child_pos"]] = child_id
            module_data[child_id] = {
                "pos": accepted["child_pos"],
                "type": accepted["child_type"],
                "rot": accepted["child_rot"],
                "hinge_streak": accepted["hinge_streak"],
                "incoming_face": accepted["face"],
                "same_face_streak": accepted["same_face_streak"],
                "x_inverted": accepted["x_inverted"],
            }
            world_pos[child_id], world_R[child_id] = self._child_world_pose(
                accepted["parent_id"], accepted["face"], accepted["child_type"], accepted["child_rot"],
                world_pos, world_R, module_data)

            # I no longer remove the parent, I just add the new child.
            # (I think this makes snakes less likely)
            frontier.append(child_id)
            next_module_id += 1

            for partner_id, (partner_parent_id, partner_face, partner_pos) in zip(
                accepted_partner_ids, accepted["partners"], strict=True
            ):
                # Streak state is recomputed fresh from the PARTNER's own
                # parent, not copied from the primary's -- under global
                # (rather than same-parent-local) symmetry the copies can
                # attach to entirely different parents with different streak
                # histories, so the primary's streak no longer describes a
                # partner's actual growth path.
                partner_parent_data = module_data[partner_parent_id]
                partner_hinge_streak = 0
                if accepted["child_type"] == ModuleType.HINGE:
                    partner_hinge_streak = (
                        partner_parent_data["hinge_streak"] + 1
                        if partner_parent_data["type"] == ModuleType.HINGE else 1
                    )
                partner_same_face_streak = (
                    partner_parent_data["same_face_streak"] + 1
                    if partner_parent_data["incoming_face"] == partner_face else 1
                )
                partner_x_inverted = partner_parent_data["x_inverted"] != (partner_face == ModuleFaces.BACK)

                occupied_coords[partner_pos] = partner_id
                module_data[partner_id] = {
                    "pos": partner_pos,
                    "type": accepted["child_type"],
                    "rot": accepted["child_rot"],
                    "hinge_streak": partner_hinge_streak,
                    "incoming_face": partner_face,
                    "same_face_streak": partner_same_face_streak,
                    "x_inverted": partner_x_inverted,
                }
                world_pos[partner_id], world_R[partner_id] = self._child_world_pose(
                    partner_parent_id, partner_face, accepted["child_type"], accepted["child_rot"],
                    world_pos, world_R, module_data)
                frontier.append(partner_id)
                next_module_id += 1

        # No top-up pass: growth above is entirely CPPN-driven and may finish
        # with too few (or zero) hinges to be controllable (utils/worker.py's
        # `model.nu < 2` guard). Rather than patch that up with placements
        # the CPPN never actually chose, the whole body is discarded here --
        # an empty graph is the existing "invalid individual" convention
        # (CPPNNeatGenotype.decode raises on 0 nodes, evaluate_batch scores
        # it 0), so this genome is naturally replaced by the outer
        # evolutionary loop next generation instead of retried in place.
        hinge_count = sum(
            1 for n in robot_graph.nodes if robot_graph.nodes[n]["type"] == ModuleType.HINGE.name
        )
        if hinge_count < self.min_hinges:
            return nx.DiGraph()

        return robot_graph
