import argparse
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class OrbitElements:
    a: float
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("Semi-major axis a must be positive.")
        if not (0 <= self.e < 1):
            raise ValueError("Eccentricity e must satisfy 0 <= e < 1 for an ellipse.")


def deg2rad(angle_deg: float) -> float:
    return angle_deg * math.pi / 180.0


def rotation_matrix_313(raan: float, inc: float, argp: float) -> List[List[float]]:
    cO = math.cos(raan)
    sO = math.sin(raan)
    ci = math.cos(inc)
    si = math.sin(inc)
    cw = math.cos(argp)
    sw = math.sin(argp)

    return [
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ]


def mat_vec_mul(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> Vector3:
    return (
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
    )


def orbit_position(orbit: OrbitElements, true_anomaly: float) -> Vector3:
    inc = deg2rad(orbit.i_deg)
    raan = deg2rad(orbit.raan_deg)
    argp = deg2rad(orbit.argp_deg)
    p = orbit.a * (1.0 - orbit.e * orbit.e)
    r = p / (1.0 + orbit.e * math.cos(true_anomaly))

    perifocal = (
        r * math.cos(true_anomaly),
        r * math.sin(true_anomaly),
        0.0,
    )
    rotation = rotation_matrix_313(raan, inc, argp)
    return mat_vec_mul(rotation, perifocal)


def squared_distance(p1: Vector3, p2: Vector3) -> float:
    return (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + (p1[2] - p2[2]) ** 2
    )


def wrap_angle(angle: float) -> float:
    return angle % (2.0 * math.pi)


def golden_section_minimize(func, center: float, half_width: float, tol: float = 1e-8) -> float:
    left = center - half_width
    right = center + half_width
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = 1.0 - invphi

    c = left + invphi2 * (right - left)
    d = left + invphi * (right - left)
    fc = func(c)
    fd = func(d)

    while right - left > tol:
        if fc < fd:
            right, d, fd = d, c, fc
            c = left + invphi2 * (right - left)
            fc = func(c)
        else:
            left, c, fc = c, d, fd
            d = left + invphi * (right - left)
            fd = func(d)

    return wrap_angle((left + right) / 2.0)


def refine_candidate(
    orbit1: OrbitElements,
    orbit2: OrbitElements,
    nu1: float,
    nu2: float,
    coarse_step: float,
    iterations: int = 20,
) -> Tuple[float, float, float]:
    half_width = coarse_step

    def objective(angle1: float, angle2: float) -> float:
        p1 = orbit_position(orbit1, wrap_angle(angle1))
        p2 = orbit_position(orbit2, wrap_angle(angle2))
        return squared_distance(p1, p2)

    best_nu1 = wrap_angle(nu1)
    best_nu2 = wrap_angle(nu2)
    best_value = objective(best_nu1, best_nu2)

    for _ in range(iterations):
        best_nu1 = golden_section_minimize(
            lambda angle: objective(angle, best_nu2), best_nu1, half_width
        )
        best_nu2 = golden_section_minimize(
            lambda angle: objective(best_nu1, angle), best_nu2, half_width
        )
        new_value = objective(best_nu1, best_nu2)
        if abs(new_value - best_value) < 1e-12:
            best_value = new_value
            break
        best_value = new_value
        half_width *= 0.6

    return best_value, best_nu1, best_nu2


def minimum_distance_between_orbits(
    orbit1: OrbitElements,
    orbit2: OrbitElements,
    coarse_samples: int = 180,
    candidate_count: int = 12,
) -> Tuple[float, float, float, Vector3, Vector3]:
    if coarse_samples < 8:
        raise ValueError("coarse_samples must be at least 8.")

    step = 2.0 * math.pi / coarse_samples
    candidates: List[Tuple[float, float, float]] = []

    for idx1 in range(coarse_samples):
        nu1 = idx1 * step
        p1 = orbit_position(orbit1, nu1)
        for idx2 in range(coarse_samples):
            nu2 = idx2 * step
            p2 = orbit_position(orbit2, nu2)
            candidates.append((squared_distance(p1, p2), nu1, nu2))

    candidates.sort(key=lambda item: item[0])

    best_value = float("inf")
    best_nu1 = 0.0
    best_nu2 = 0.0
    for _, nu1, nu2 in candidates[:candidate_count]:
        value, ref_nu1, ref_nu2 = refine_candidate(orbit1, orbit2, nu1, nu2, step)
        if value < best_value:
            best_value = value
            best_nu1 = ref_nu1
            best_nu2 = ref_nu2

    pos1 = orbit_position(orbit1, best_nu1)
    pos2 = orbit_position(orbit2, best_nu2)
    return math.sqrt(best_value), best_nu1, best_nu2, pos1, pos2


def parse_orbit(values: Sequence[float]) -> OrbitElements:
    if len(values) != 5:
        raise ValueError("Each orbit must contain exactly 5 values: a e i raan argp")
    return OrbitElements(
        a=float(values[0]),
        e=float(values[1]),
        i_deg=float(values[2]),
        raan_deg=float(values[3]),
        argp_deg=float(values[4]),
    )


def format_vector(vec: Vector3) -> str:
    return f"({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}) km"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate the minimum distance between two elliptical orbits."
    )
    parser.add_argument(
        "--orbit1",
        nargs=5,
        metavar=("a", "e", "i", "raan", "argp"),
        type=float,
        help="Orbit 1 elements in km/deg: a e i raan argp",
    )
    parser.add_argument(
        "--orbit2",
        nargs=5,
        metavar=("a", "e", "i", "raan", "argp"),
        type=float,
        help="Orbit 2 elements in km/deg: a e i raan argp",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=180,
        help="Number of coarse samples per orbit, default: 180",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.orbit1 is None or args.orbit2 is None:
        orbit1 = OrbitElements(a=7000.0, e=0.10, i_deg=30.0, raan_deg=40.0, argp_deg=20.0)
        orbit2 = OrbitElements(a=7200.0, e=0.05, i_deg=55.0, raan_deg=80.0, argp_deg=120.0)
        print("No command-line orbit provided. Running the built-in example.\n")
    else:
        orbit1 = parse_orbit(args.orbit1)
        orbit2 = parse_orbit(args.orbit2)

    min_distance, nu1, nu2, pos1, pos2 = minimum_distance_between_orbits(
        orbit1, orbit2, coarse_samples=args.samples
    )

    print("Orbit 1:", orbit1)
    print("Orbit 2:", orbit2)
    print(f"Minimum distance: {min_distance:.6f} km")
    print(f"Orbit 1 true anomaly at minimum distance: {math.degrees(nu1):.6f} deg")
    print(f"Orbit 2 true anomaly at minimum distance: {math.degrees(nu2):.6f} deg")
    print("Orbit 1 position:", format_vector(pos1))
    print("Orbit 2 position:", format_vector(pos2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
