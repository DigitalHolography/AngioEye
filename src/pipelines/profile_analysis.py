import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .core.base import ProcessPipeline, ProcessResult, registerPipeline

# fitting kit
# ============================================================


def parabola(x, A, x0, y0):
    return A * (x - x0) ** 2 + y0


def parabola_fit(V):
    segment_data = {}
    for branch_index in range(V.shape[2]):
        for circle_index in range(V.shape[3]):
            profile_complex = V[0, :, branch_index, circle_index]
            profile = np.real(profile_complex)

            if np.all(profile == 0):
                continue

            x = np.arange(len(profile))

            try:
                A_guess = -0.1
                x0_guess = np.argmax(profile)
                y0_guess = np.max(profile)

                popt, pcov = curve_fit(
                    parabola, x, profile, p0=[A_guess, x0_guess, y0_guess]
                )

                A_fit, x0_fit, y0_fit = popt

                r0_fit = np.sqrt(-y0_fit / A_fit)

                segment_data[(branch_index, circle_index)] = {
                    "r0": r0_fit,
                    "y0": y0_fit,
                    "x0": x0_fit,
                    "A": A_fit,
                }

            except Exception as e:
                print(f"Fit failed for branch={branch_index}, circle={circle_index}")

                print(e)
    return segment_data


# Distribution of A, x0, y0, r0 over ALL segments
# ============================================================


def view_distribution(segment_data, bins=15):
    A_values = []
    x0_values = []
    y0_values = []
    r0_values = []

    for data in segment_data.values():
        A_values.append(data["A"])
        x0_values.append(data["x0"])
        y0_values.append(data["y0"])
        r0_values.append(data["r0"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(A_values, bins=bins, edgecolor="black")
    axes[0, 0].set_title("Distribution of A")
    axes[0, 0].set_xlabel("A")

    axes[0, 1].hist(x0_values, bins=bins, edgecolor="black")
    axes[0, 1].set_title("Distribution of x0")
    axes[0, 1].set_xlabel("x0")

    axes[1, 0].hist(y0_values, bins=bins, edgecolor="black")
    axes[1, 0].set_title("Distribution of y0")
    axes[1, 0].set_xlabel("y0")

    axes[1, 1].hist(r0_values, bins=bins, edgecolor="black")
    axes[1, 1].set_title("Distribution of r0")
    axes[1, 1].set_xlabel("r0")

    plt.tight_layout()
    plt.show()


# Variation along circles for one branch
# ============================================================


def view_branch_variation(segment_data, branch_index):
    circles = []
    A_values = []
    x0_values = []
    y0_values = []
    r0_values = []

    for (branch, circle), data in segment_data.items():
        if branch != branch_index:
            continue

        circles.append(circle)
        A_values.append(data["A"])
        x0_values.append(data["x0"])
        y0_values.append(data["y0"])
        r0_values.append(data["r0"])

    if len(circles) == 0:
        print(f"No data for branch {branch_index}")
        return

    # sort by circle index
    circles = np.array(circles)

    sort_idx = np.argsort(circles)

    circles = circles[sort_idx]

    A_values = np.array(A_values)[sort_idx]
    x0_values = np.array(x0_values)[sort_idx]
    y0_values = np.array(y0_values)[sort_idx]
    r0_values = np.array(r0_values)[sort_idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(circles, A_values, marker="o")
    axes[0, 0].set_title(f"A variation (branch {branch_index})")
    axes[0, 0].set_xlabel("Circle index")
    axes[0, 0].set_ylabel("A")

    axes[0, 1].plot(circles, x0_values, marker="o")
    axes[0, 1].set_title(f"x0 variation (branch {branch_index})")
    axes[0, 1].set_xlabel("Circle index")
    axes[0, 1].set_ylabel("x0")

    axes[1, 0].plot(circles, y0_values, marker="o")
    axes[1, 0].set_title(f"y0 variation (branch {branch_index})")
    axes[1, 0].set_xlabel("Circle index")
    axes[1, 0].set_ylabel("y0")

    axes[1, 1].plot(circles, r0_values, marker="o")
    axes[1, 1].set_title(f"r0 variation (branch {branch_index})")
    axes[1, 1].set_xlabel("Circle index")
    axes[1, 1].set_ylabel("r0")

    plt.tight_layout()
    plt.show()


# Find and fit two branches whose y0 decreases linearly along the radius
# ============================================================


def find_linear_y0_branches(segment_data, min_circles=5, top_n=3):
    pixel_size = 20 * 1e-06  # in m
    branch_dict = {}

    for (branch, circle), data in segment_data.items():
        if branch not in branch_dict:
            branch_dict[branch] = {"circles": [], "y0": []}

        branch_dict[branch]["circles"].append(circle)
        branch_dict[branch]["y0"].append(data["y0"])

    fit_results = []

    for branch, values in branch_dict.items():
        circles = np.array(values["circles"])
        y0 = np.array(values["y0"])

        if len(circles) < min_circles:
            continue

        sort_idx = np.argsort(circles)

        circles = circles[sort_idx]
        y0 = y0[sort_idx]

        slope, intercept = np.polyfit(circles, y0, 1)

        y_fit = slope * circles + intercept

        ss_res = np.sum((y0 - y_fit) ** 2)
        ss_tot = np.sum((y0 - np.mean(y0)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        relaxation_distance = intercept / (slope / (10 * pixel_size * 128))  # in m

        fit_results.append(
            {
                "branch": branch,
                "circles": circles,
                "y0": y0,
                "y_fit": y_fit,
                "slope": slope,
                "intercept": intercept,
                "R2": r2,
                "RD": relaxation_distance,
            }
        )

    fit_results.sort(key=lambda x: x["R2"], reverse=True)

    best_results = fit_results[:top_n]

    fig, axes = plt.subplots(1, len(best_results), figsize=(6 * len(best_results), 5))

    if len(best_results) == 1:
        axes = [axes]

    for ax, res in zip(axes, best_results, strict=True):
        ax.plot(res["circles"], res["y0"], "o", label="Raw data")

        ax.plot(res["circles"], res["y_fit"], "-", label="Linear fit")

        ax.set_title(f"Branch {res['branch']}\n$R^2$ = {res['R2']:.4f}")

        ax.set_xlabel("Circle index")
        ax.set_ylabel("y0")

        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return best_results


def extract_RD_matrix(best_results):
    rows = []

    seen_branches = set()

    for result in best_results:
        branch = result["branch"]

        if branch in seen_branches:
            continue

        seen_branches.add(branch)

        RD = result["RD"]

        rows.append([branch, RD])

    matrix = np.array(rows)

    return matrix


@registerPipeline(name="ProfileAnalysis")
class ProfileAnalysis(ProcessPipeline):
    description = "Profile Analysis Pipeline"

    v_profile_path = "/AngioEye/Processing/womersleymodeling/v_pulse_fft"

    def run(self, h5file: h5py.File) -> ProcessResult:
        obj = h5file[self.v_profile_path]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"Expected a dataset at {self.v_profile_path}, but found {type(obj)}"
            )
        V = obj[:]

        segment_data = parabola_fit(V)

        # view_distribution(segment_data)

        # for i in range(20):

        #     branch_to_view = i
        #     view_branch_variation(
        #         segment_data,
        #         branch_index=branch_to_view,
        #     )

        best_results = find_linear_y0_branches(segment_data)

        RD_matrix = extract_RD_matrix(best_results)

        metrics: dict = {}
        metrics["fit_matrix"] = np.asarray(RD_matrix)

        return ProcessResult(metrics=metrics)
