#!/usr/bin/env python3
"""Generate Springer Nature figures as vector PDF and high-resolution PNG.

Main figures are built at the 174-mm width recommended for Nonlinear
Dynamics.  The standalone graphical abstract has the journal's preferred
1:2 aspect ratio.  Text is rendered through LaTeX, and all publication-figure
text is at least 8 pt at final reproduction size.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from local_analysis import transverse_cycle
from model import two_pop_equilibrium, two_pop_jacobian


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "font.sans-serif": ["Helvetica"],
        "font.monospace": ["Courier"],
        "mathtext.fontset": "cm",
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{lmodern}"
            r"\usepackage{amsmath,amssymb,bm}"
        ),
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.labelpad": 1.7,
        "xtick.major.pad": 1.7,
        "ytick.major.pad": 1.7,
        "axes.linewidth": 0.70,
        "lines.linewidth": 1.05,
        "lines.markersize": 2.8,
        "legend.handlelength": 1.30,
        "legend.handletextpad": 0.32,
        "legend.columnspacing": 0.55,
        "legend.borderaxespad": 0.25,
        "legend.borderpad": 0.28,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLUMN_WIDTH = 6.85

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GOLD = "#A66F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#555555"


def load_csv(name: str):
    return np.genfromtxt(
        DATA / name, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )


def render_png(
    fig: plt.Figure,
    pdf_source: Path,
    png_target: Path,
    raster_prefix: Path,
    dpi: int,
) -> Path | None:
    """Render PNG with Poppler when available, otherwise use Matplotlib."""
    if shutil.which("pdftocairo") is not None:
        subprocess.run(
            [
                "pdftocairo",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf_source),
                str(raster_prefix),
            ],
            check=True,
        )
        raster_png = Path(f"{raster_prefix}.png")
        os.replace(raster_png, png_target)
        return raster_png

    # The Agg backend otherwise asks for ``dvipng`` when ``text.usetex`` is
    # enabled.  The vector PDF above remains the authoritative LaTeX render;
    # this fallback uses Matplotlib mathtext only for the companion raster.
    for artist in fig.findobj(Text):
        artist.set_usetex(False)
    fig.savefig(png_target, format="png", dpi=dpi)
    return None


def sync_directory(path: Path) -> None:
    """Best-effort directory sync; directory descriptors are not portable."""
    if os.name == "nt":
        return
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sync_file(path: Path) -> None:
    """Best-effort file sync before an atomic replacement."""
    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError:
        pass


def save(fig: plt.Figure, stem: str) -> None:
    pdf = FIG / f"{stem}.pdf"
    png = FIG / f"{stem}.png"
    token = uuid.uuid4().hex
    pdf_tmp = FIG / f".{stem}.{token}.pdf.tmp"
    png_tmp = FIG / f".{stem}.{token}.png.tmp"
    raster_prefix = FIG / f".{stem}.{token}.raster"
    raster_png: Path | None = None
    try:
        fig.savefig(pdf_tmp, format="pdf")
        if not pdf_tmp.read_bytes().rstrip().endswith(b"%%EOF"):
            raise RuntimeError(f"Incomplete PDF render: {pdf_tmp}")
        raster_png = render_png(
            fig, pdf_tmp, png_tmp, raster_prefix, dpi=600
        )
        for temporary, final in ((pdf_tmp, pdf), (png_tmp, png)):
            sync_file(temporary)
            os.replace(temporary, final)
        sync_directory(FIG)
    finally:
        for temporary in (pdf_tmp, png_tmp, raster_png):
            if temporary is None:
                continue
            temporary.unlink(missing_ok=True)
        plt.close(fig)


def save_graphical_abstract(fig: plt.Figure) -> None:
    """Write a vector file and the journal-preferred 400 x 200 px raster."""
    pdf = FIG / "Graphical_Abstract.pdf"
    png = FIG / "Graphical_Abstract.png"
    token = uuid.uuid4().hex
    pdf_tmp = FIG / f".Graphical_Abstract.{token}.pdf.tmp"
    png_tmp = FIG / f".Graphical_Abstract.{token}.png.tmp"
    raster_prefix = FIG / f".Graphical_Abstract.{token}.raster"
    raster_png: Path | None = None
    try:
        fig.savefig(pdf_tmp, format="pdf")
        if not pdf_tmp.read_bytes().rstrip().endswith(b"%%EOF"):
            raise RuntimeError(f"Incomplete PDF render: {pdf_tmp}")
        raster_png = render_png(
            fig, pdf_tmp, png_tmp, raster_prefix, dpi=100
        )
        for temporary, final in ((pdf_tmp, pdf), (png_tmp, png)):
            sync_file(temporary)
            os.replace(temporary, final)
        sync_directory(FIG)
    finally:
        for temporary in (pdf_tmp, png_tmp, raster_png):
            if temporary is None:
                continue
            temporary.unlink(missing_ok=True)
        plt.close(fig)


def label(
    ax,
    text: str,
    *,
    x: float = -0.16,
    y: float = 1.02,
) -> None:
    """Place a panel letter clear of tick labels and vertical axis titles."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        fontsize=9.0,
        zorder=20,
        clip_on=False,
    )


def grid(ax) -> None:
    ax.grid(alpha=0.20, linewidth=0.50)


def compact_ticks(ax, nx: int = 4, ny: int = 4) -> None:
    """Limit linear-axis tick density for narrow panels."""
    if ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(nx))
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(ny))


def figure1_overview() -> None:
    """Compact model schematic and branch-resolved event hierarchy."""
    fig, axs = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH, 2.45), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.10)
    ax = axs[0]
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    for x, color, title in [
        (2.6, BLUE, "coherent\npopulation"),
        (7.4, GOLD, "partially coherent\npopulation"),
    ]:
        ax.add_patch(Circle((x, 3.1), 1.2, fill=False, color=color, lw=2.2))
        ax.text(x, 1.55, title, ha="center", va="top", color=color, fontsize=8.0)
    ax.add_patch(FancyArrowPatch((3.85, 3.55), (6.15, 3.55), arrowstyle="-|>",
                                 mutation_scale=13, color=GREEN, lw=2.0))
    ax.add_patch(FancyArrowPatch((6.15, 2.7), (3.85, 2.7), arrowstyle="-|>",
                                 mutation_scale=13, color=GREEN, lw=2.0))
    ax.text(5, 5.05, r"symmetric weights $K=K^{\mathsf T}$", ha="center", color=GREEN,
            fontsize=8.0)
    ax.text(5, 0.20, r"$\alpha=\pi/2-\beta$; $\mu=(1+A)/2$, $\nu=(1-A)/2$",
            ha="center", color=GRAY, fontsize=8.0)
    label(ax, "(a)")

    ax = axs[1]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        (0.88, r"saddle-node bifurcation" + "\n" + r"$A=0.178341$: locked branch", BLUE),
        (0.65, r"defective node--focus / EP" + "\n" + r"$A=0.178454$: stable spectrum", GREEN),
        (0.42, r"Hopf bifurcation" + "\n" + r"$A=0.277775$: breathing begins", ORANGE),
        (0.19, r"homoclinic bifurcation" + "\n" + r"$A=0.353697$: cycle terminates", PURPLE),
    ]
    for i, (y, text, color) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((0.04, y - 0.085), 0.92, 0.17,
                     boxstyle="round,pad=0.015", fill=False, ec=color, lw=1.5))
        ax.text(0.50, y, text, ha="center", va="center", color=color,
                fontsize=8.0)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((0.50, y - 0.075),
                         (0.50, boxes[i + 1][0] + 0.075), arrowstyle="-|>",
                         mutation_scale=10, color=GRAY, lw=0.9))
    ax.text(0.50, 0.015, r"fixed $\beta=0.10$; event positions not drawn to scale",
            ha="center", color=GRAY, fontsize=8.0)
    label(ax, "(b)")
    save(fig, "Fig1_overview")


def graphical_abstract() -> None:
    """A concise 1:2 summary for SpringerLink and the manuscript."""
    fig = plt.figure(figsize=(4.0, 2.0), layout="constrained")
    ax = fig.add_subplot(111)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    y = 0.68
    xvals = [0.10, 0.37, 0.65, 0.90]
    colors = [BLUE, GREEN, ORANGE, PURPLE]
    titles = ["saddle-node", "node--focus / EP", "Hopf", "homoclinic"]
    subtitles = ["locked branch", "relaxation changes", "breathing begins", "cycle ends"]
    for i, (x, color, title, subtitle) in enumerate(
        zip(xvals, colors, titles, subtitles)
    ):
        ax.scatter([x], [y], s=55, color=color, zorder=4)
        ax.text(x, y + 0.13, title, ha="center", va="bottom",
                fontsize=8.0, color=color, fontweight="bold")
        ax.text(x, y - 0.12, subtitle, ha="center", va="top",
                fontsize=8.0, color=GRAY)
        if i < len(xvals) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.025, y),
                    (xvals[i + 1] - 0.025, y),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color=GRAY,
                    lw=0.9,
                )
            )

    t = np.linspace(0, 1, 180)
    base_x, base_y = 0.23, 0.18
    ax.plot(base_x + 0.16 * t, base_y + 0.13 * np.exp(-4 * t),
            color=BLUE, lw=1.0)
    ax.plot(base_x + 0.16 * t, base_y + 0.13 * np.exp(-4 * t) * (1 + 2.2 * t),
            color=GREEN, lw=1.0)
    ax.plot(base_x + 0.16 * t, base_y + 0.13 * np.exp(-4 * t) * np.cos(14 * t),
            color=ORANGE, lw=1.0)
    ax.text(base_x + 0.08, 0.035, "two-rate  →  Jordan  →  damped",
            ha="center", color=GRAY, fontsize=8.0)

    tt = np.linspace(0, 2 * np.pi, 220)
    ax.plot(0.72 + 0.10 * np.cos(tt), 0.18 + 0.10 * np.sin(tt),
            color=ORANGE, lw=1.2)
    ax.scatter([0.72], [0.18], marker="x", color=GRAY, s=20)
    ax.text(0.72, 0.035, "periodic order parameter", ha="center",
            color=GRAY, fontsize=8.0)
    ax.text(
        0.5,
        0.96,
        "A stable spectral degeneracy precedes the known Hopf onset",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    save_graphical_abstract(fig)


def figure2_ep_response() -> None:
    branch = load_csv("locked_branch_declared_metric.csv")
    critical = load_csv("critical_points.csv")
    by_event = {str(row["event"]): row for row in np.atleast_1d(critical)}
    A_sn = float(by_event["SN"]["A"])
    A_ep = float(by_event["EP"]["A"])
    A_h = float(by_event["Hopf"]["A"])
    pulse = load_csv("relaxation_pulses_beta_0p10.csv")

    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.15), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.18, hspace=0.08)
    axs = grid_axes.ravel()
    ax = axs[0]
    zoom = branch["A"] < A_ep + 7.5e-4
    ax.plot(branch["A"][zoom], 1.0e3 * branch["discriminant"][zoom], color=BLUE)
    ax.axhline(0, color="black", lw=0.65)
    ax.axvline(A_sn, color=BLUE, ls="--", label="SN")
    ax.axvline(A_ep, color=GREEN, ls=":", label="EP")
    ax.set(xlabel=r"coupling disparity $A$", ylabel=r"$10^3\mathcal{D}$")
    ax.legend(frameon=True, framealpha=0.88, edgecolor="none", ncol=2,
              loc="upper right")
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(a)")

    ax = axs[1]
    ax.plot(branch["A"], branch["lambda1_real"], color=BLUE, label=r"Re $\lambda_1$")
    ax.plot(branch["A"], branch["lambda2_real"], color=SKY, label=r"Re $\lambda_2$")
    ax.plot(branch["A"], abs(branch["lambda1_imag"]), color=ORANGE, ls="--",
            label=r"$|$Im $\lambda_{1,2}|$")
    ax.plot(branch["A"], branch["lambda_transverse"], color=GRAY, ls=":",
            label=r"$\lambda_\perp$")
    ax.axhline(0, color="black", lw=0.65)
    ax.axvline(A_ep, color=GREEN, ls=":")
    ax.axvline(A_h, color=ORANGE, ls=":")
    ax.set(xlabel=r"$A$", ylabel="eigenvalue component", xlim=(A_sn, A_h + 0.012))
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(b)")

    ax = axs[2]
    mapping = {
        "stable_node": (BLUE, "node"),
        "exceptional_point": (GREEN, r"EP (Jordan)"),
        "stable_focus": (ORANGE, "focus"),
    }
    for case, (color, text) in mapping.items():
        q = pulse[pulse["case"] == case]
        ax.plot(q["time"], q["delta_r_over_epsilon"], color=color, label=text)
    ax.axhline(0, color="black", lw=0.55)
    ax.set(xlabel="time after coherence pulse", ylabel=r"$\delta r(t)/\epsilon$",
           xlim=(0, 260))
    # Direct labels sit in the data-free upper-right region and avoid hiding
    # any part of the three relaxation traces.
    ax.text(0.96, 0.92, "node", transform=ax.transAxes,
            ha="right", va="top", color=BLUE)
    ax.text(0.96, 0.82, r"Jordan EP", transform=ax.transAxes,
            ha="right", va="top", color=GREEN)
    ax.text(0.96, 0.72, "focus", transform=ax.transAxes,
            ha="right", va="top", color=ORANGE)
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    source = load_csv("locked_branch_beta_0p10.csv")
    distance, split, condition = [], [], []
    for row in np.atleast_1d(source):
        d = abs(float(row["A"]) - A_ep)
        if 1.0e-9 < d < 8.0e-3:
            J = np.array([[row["J11"], row["J12"]], [row["J21"], row["J22"]]], float)
            eig, V = np.linalg.eig(J)
            distance.append(d)
            split.append(abs(eig[0] - eig[1]))
            condition.append(np.linalg.cond(V))
    distance, split, condition = map(np.asarray, (distance, split, condition))
    order = np.argsort(distance)
    ax.loglog(distance[order], split[order], ".", ms=2.5, color=PURPLE,
              label=r"$|\Delta\lambda|$")
    guide_x = np.geomspace(max(distance.min(), 2e-8), min(distance.max(), 2e-3), 80)
    ref = np.argmin(abs(distance - 1e-5))
    ax.loglog(guide_x, split[ref] * np.sqrt(guide_x / distance[ref]), "k--",
              label=r"$\sqrt{|A-A_{EP}|}$")
    ax2 = ax.twinx()
    ax2.loglog(distance[order], condition[order], color=GOLD, lw=1.0,
               label=r"$\kappa_2(V)$")
    ax.set(xlabel=r"$|A-A_{EP}|$", ylabel="eigenvalue splitting")
    ax2.set_ylabel(r"$\kappa_2(V)$", color=GOLD)
    lines, names = ax.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, names + names2, frameon=True, framealpha=0.95,
              edgecolor="none", ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    ax.set_xticks([1.0e-9, 1.0e-6, 1.0e-3])
    grid(ax)
    label(ax, "(d)", x=-0.01, y=1.08)
    save(fig, "Fig2_EP_and_relaxation")


def figure3_hopf_floquet() -> None:
    branch = load_csv("locked_branch_declared_metric.csv")
    critical = load_csv("critical_points.csv")
    A_ep = float(critical[critical["event"] == "EP"]["A"][0])
    A_h = float(critical[critical["event"] == "Hopf"]["A"][0])
    l1 = load_csv("hopf_first_lyapunov.csv")
    floquet = load_csv("transverse_floquet_cycles.csv")
    if "inplane_floquet_exponent" not in floquet.dtype.names:
        # Backward-compatible smoke builds may reuse the earlier local-audit
        # table; the distributed independent full-spectrum table supplies the
        # missing Liouville exponent in that case.
        floquet = load_csv("full_floquet_cycles_independent.csv")
        inplane_field = "inplane_exponent"
        normal_field = "transverse_exponent"
    else:
        inplane_field = "inplane_floquet_exponent"
        normal_field = "transverse_floquet_exponent"
    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.10), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.20, hspace=0.08)
    axs = grid_axes.ravel()

    ax = axs[0]
    valid = branch["A"] >= A_ep
    mu = branch["lambda1_real"][valid]
    omega = abs(branch["lambda1_imag"][valid])
    A = branch["A"][valid]
    ax.plot(A, mu, color=BLUE, label=r"$\mu_J$")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(A_ep, color=GREEN, ls=":", label="EP")
    ax.axvline(A_h, color=ORANGE, ls=":", label="Hopf")
    ax2 = ax.twinx()
    ax2.plot(A, omega, color=GOLD, ls="--", label=r"$\Omega_J$")
    ax.set(xlabel=r"$A$", ylabel=r"$\mu_J$")
    ax2.set_ylabel(r"$\Omega_J$", color=GOLD)
    lines, names = ax.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, names + names2, frameon=True, framealpha=0.95,
              edgecolor="none", ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(a)")

    ax = axs[1]
    ax.plot(l1["beta"], l1["l1"], "o-", ms=3, color=ORANGE)
    ax.axhline(0, color="black", lw=0.6)
    ax.scatter([0.1], [l1[np.argmin(abs(l1["beta"] - .1))]["l1"]], color="black", s=14)
    ax.set(xlabel=r"$\beta$", ylabel=r"$\ell_1$")
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(b)")

    ax = axs[2]
    ax.plot(l1["beta"], l1["direct_cubic"], "o-", ms=2.5, color=BLUE,
            label="direct")
    ax.plot(l1["beta"], l1["zero_frequency_feedback"], "o-", ms=2.5,
            color=GREEN, label="zero freq.")
    ax.plot(l1["beta"], l1["second_harmonic"], "o-", ms=2.5,
            color=GOLD, label="2nd harmonic")
    ax.plot(l1["beta"], l1["l1"], color="black", lw=1.8, label="total")
    ax.axhline(0, color="black", lw=0.55)
    ax.set(xlabel=r"$\beta$", ylabel=r"contribution to $\ell_1$")
    ax.legend(frameon=True, framealpha=0.97, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.13))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    for beta, color, marker in [(0.10, BLUE, "o"), (0.06, PURPLE, "s")]:
        q = floquet[np.isclose(floquet["beta"], beta)]
        ax.plot(q["A"], q[inplane_field], marker + "-", color=color,
                ms=3, label=fr"$\parallel,\ {beta:.2f}$")
        ax.plot(q["A"], q[normal_field], marker + "--", color=color,
                alpha=.65, ms=3, label=fr"$\perp,\ {beta:.2f}$")
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xlabel=r"$A$", ylabel="nontrivial Floquet exponent")
    ax.legend(frameon=True, framealpha=0.97, edgecolor="none", ncol=2,
              fontsize=8.0, loc="lower center", bbox_to_anchor=(.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(d)", x=-0.01, y=1.08)
    save(fig, "Fig3_Hopf_and_Floquet")


def figure4_homoclinic(cycles006: dict[float, dict]) -> None:
    periods = load_csv("homoclinic_periods_beta_0p06.csv")
    windows = load_csv("homoclinic_fit_windows_beta_0p06.csv")
    Aall, Tall = periods["A"], periods["period"]
    fit = periods[Aall >= 0.335]

    def log_model(A, C, c, Ac):
        return C - c * np.log(Ac - A)

    def snic_model(A, C, k, Ac):
        return C + k / np.sqrt(Ac - A)

    plog, _ = curve_fit(
        log_model, fit["A"], fit["period"], p0=(-40, 31, .39616),
        bounds=([-5000, 0, .39600001], [5000, 5000, .45]), maxfev=200000
    )
    psnic, _ = curve_fit(
        snic_model, fit["A"], fit["period"], p0=(20, 7, .397),
        bounds=([-5000, 0, .39600001], [5000, 5000, .45]), maxfev=200000
    )
    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.30), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.20, hspace=0.08)
    axs = grid_axes.ravel()
    ax = axs[0]
    ax.plot(Aall, Tall, "o", ms=3.5, color=BLUE, label="data")
    xx = np.linspace(fit["A"].min(), min(.39605, plog[2] - 1e-6), 400)
    ax.plot(xx, log_model(xx, *plog), color=ORANGE, label="log")
    ax.plot(xx, snic_model(xx, *psnic), color=GRAY, ls="--", label="SNIC")
    ax.axvspan(Aall.min(), fit["A"].min(), color="0.93", label="excluded")
    ax.set(xlabel=r"$A$", ylabel="period $T$")
    ax.legend(frameon=True, framealpha=0.97, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(a)")

    ax = axs[1]
    ax.plot(fit["A"], fit["period"] - log_model(fit["A"], *plog), "o-",
            ms=3, color=ORANGE, label="log")
    ax.plot(fit["A"], fit["period"] - snic_model(fit["A"], *psnic), "s--",
            ms=3, color=GRAY, label="SNIC")
    ax.axhline(0, color="black", lw=0.55)
    row10 = windows[windows["n_last_points"] == 10][0]
    ax.text(0.07, 0.93,
            "$\\Delta{\\rm AICc}$\n"
            fr"$={row10['delta_AICc_log_over_snic']:.1f}$",
            transform=ax.transAxes, ha="left", va="top", linespacing=0.92)
    ax.set(xlabel=r"$A$", ylabel="fit residual")
    ax.legend(frameon=True, framealpha=0.88, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    ax.set_xticks([0.35, 0.39])
    ax.set_xticklabels([r"$0.35$", r"$0.39$"])
    grid(ax)
    label(ax, "(b)")

    ax = axs[2]
    for A, color, linestyle in zip(
        (0.35, 0.385, 0.395), (GREEN, SKY, ORANGE), ("-", "--", "-.")
    ):
        cycle = cycles006[A]
        ax.plot(np.sin(cycle["orbit_psi"]), cycle["orbit_r"],
                color=color, ls=linestyle, label="_nolegend_")
    Ac = float(row10["Ac_log"])
    saddle = two_pop_equilibrium(Ac, 0.06, (0.982, -0.197))
    J = two_pop_jacobian(saddle, Ac, 0.06)
    eig, V = np.linalg.eig(J)
    ax.scatter([np.sin(saddle[1])], [saddle[0]], marker="X",
               s=35, color="black", label="saddle")
    for idx, color, name, display_name in [
        (np.argmax(eig.real), ORANGE, "unstable", "unst."),
        (np.argmin(eig.real), BLUE, "stable", "stable"),
    ]:
        v = np.real(V[:, idx])
        v /= np.linalg.norm(v)
        vx = np.cos(saddle[1]) * v[1]
        scale = 0.028
        ax.plot([np.sin(saddle[1]) - scale * vx, np.sin(saddle[1]) + scale * vx],
                [saddle[0] - scale * v[0], saddle[0] + scale * v[0]],
                color=color, lw=1.8, ls="-" if name == "unstable" else "--",
                label=display_name)
    ax.set(xlabel=r"$\sin\psi$", ylabel=r"$r$")
    for y, text, color in [
        (0.70, r"$A=.350$", GREEN),
        (0.62, r"$A=.385$", SKY),
        (0.54, r"$A=.395$", ORANGE),
    ]:
        ax.text(0.50, y, text, transform=ax.transAxes,
                ha="center", va="center", color=color, fontsize=8.0)
    ax.legend(frameon=True, framealpha=0.97, edgecolor="none", ncol=1,
              loc="center", bbox_to_anchor=(0.50, 0.32))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    n = windows["n_last_points"]
    ax.errorbar(n, windows["inv_c_log"], yerr=windows["inv_c_standard_error"],
                fmt="o-", ms=3, color=GOLD, capsize=2, label=r"$c^{-1}$")
    ax.plot(n, windows["lambda_u_at_Ac"], "s--", ms=3, color=GREEN,
            label=r"$\lambda_u$")
    ax.set(xlabel="points retained", ylabel="rate")
    ax2 = ax.twinx()
    ax2.plot(n, windows["delta_AICc_log_over_snic"], "^-",
             ms=3, color=PURPLE, label=r"$\Delta$AICc")
    ax2.set_ylabel(r"$\Delta\mathrm{AICc}$", color=PURPLE)
    lines, names = ax.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, names + names2, frameon=True, framealpha=0.97,
              edgecolor="none", ncol=1, loc="upper left")
    compact_ticks(ax, nx=4, ny=4)
    compact_ticks(ax2, nx=4, ny=4)
    grid(ax)
    label(ax, "(d)", x=-0.20, y=1.04)
    save(fig, "Fig4_homoclinic_evidence")


def figure4_same_cut_homoclinic() -> None:
    """Direct manifold shooting and period scaling on the main beta=0.10 cut."""
    periods = load_csv("homoclinic_periods_beta_0p10.csv")
    windows = load_csv("homoclinic_fit_windows_beta_0p10.csv")
    orbit = load_csv("homoclinic_orbit_beta_0p10.csv")
    summary = json.loads((DATA / "same_cut_homoclinic_summary.json").read_text())
    Ac = float(summary["A_homoclinic_manifold"])
    A, T = periods["A"], periods["period"]
    fit = periods[-6:]

    Xlog_all = np.column_stack([np.ones(len(periods)), -np.log(Ac - A)])
    Clog_all, clog_all = np.linalg.lstsq(Xlog_all, T, rcond=None)[0]
    Xsnic_all = np.column_stack([np.ones(len(periods)), 1.0 / np.sqrt(Ac - A)])
    Csnic_all, ksnic_all = np.linalg.lstsq(Xsnic_all, T, rcond=None)[0]
    log_model_all = lambda x: Clog_all - clog_all * np.log(Ac - x)
    snic_model_all = lambda x: Csnic_all + ksnic_all / np.sqrt(Ac - x)

    Xlog = np.column_stack([np.ones(len(fit)), -np.log(Ac - fit["A"])])
    Clog, clog = np.linalg.lstsq(Xlog, fit["period"], rcond=None)[0]
    Xsnic = np.column_stack([np.ones(len(fit)), 1.0 / np.sqrt(Ac - fit["A"])])
    Csnic, ksnic = np.linalg.lstsq(Xsnic, fit["period"], rcond=None)[0]
    log_model = lambda x: Clog - clog * np.log(Ac - x)
    snic_model = lambda x: Csnic + ksnic / np.sqrt(Ac - x)

    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.30), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.20, hspace=0.08)
    axs = grid_axes.ravel()

    ax = axs[0]
    ax.plot(A, T, "o", ms=2.8, color=BLUE, label="cycles")
    xx = np.linspace(A.min(), A.max(), 500)
    ax.plot(xx, log_model_all(xx), color=ORANGE, label="log")
    ax.plot(xx, snic_model_all(xx), color=GRAY, ls="--", label="SNIC")
    ax.set(xlabel=r"$A$", ylabel="period $T$", xlim=(.30, .3542))
    ax.legend(frameon=True, framealpha=.97, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(a)")

    ax = axs[1]
    residual_log = fit["period"] - log_model(fit["A"])
    residual_snic = fit["period"] - snic_model(fit["A"])
    endpoint_gap = 1.0e6 * (Ac - fit["A"])
    ax.plot(endpoint_gap, residual_log, "o-", ms=3, color=ORANGE, label="log")
    ax.plot(endpoint_gap, residual_snic, "s--", ms=3, color=GRAY, label="SNIC")
    ax.axhline(0, color="black", lw=.55)
    ax.set(xlabel=r"$10^6(A_{\rm hom}-A)$", ylabel="fit residual")
    ax.invert_xaxis()
    ax.legend(frameon=True, framealpha=.95, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(b)")

    ax = axs[2]
    for segment, color, style, name in [
        ("unstable_forward_to_section", ORANGE, "-", "unstable"),
        ("section_forward_to_stable", BLUE, "--", "stable"),
    ]:
        q = orbit[orbit["segment"] == segment]
        ax.plot(np.sin(q["psi"]), q["r"], color=color, ls=style, lw=1.15,
                label=name)
    saddle = np.asarray(summary["saddle"])
    ax.scatter([np.sin(saddle[1])], [saddle[0]], marker="X", s=34,
               color="black", label="saddle", zorder=5)
    ax.set(xlabel=r"$\sin\psi$", ylabel=r"$r$")
    ax.legend(frameon=True, framealpha=.97, edgecolor="none", fontsize=8.0,
              loc="lower center", ncol=1)
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    n = windows["n_last_points"]
    ax.plot(n, windows["inv_c_log_fixed_Ac"], "o-", ms=3, color=GREEN,
            label=r"$c^{-1}$")
    ax.axhline(float(summary["lambda_u"]), color=ORANGE, ls="--",
               label=r"$\lambda_u$")
    ax.set(xlabel="points retained", ylabel="rate")
    ax2 = ax.twinx()
    ax2.plot(n, windows["delta_AICc_log_over_snic_fixed_Ac"], "^-",
             ms=3, color=PURPLE, label=r"$\Delta$AICc")
    ax2.set_ylabel(r"$\Delta\mathrm{AICc}$", color=PURPLE)
    lines, names = ax.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, names + names2, frameon=True, framealpha=.95,
              edgecolor="none", fontsize=8.0, loc="lower left")
    compact_ticks(ax, nx=4, ny=4)
    compact_ticks(ax2, nx=4, ny=4)
    grid(ax)
    label(ax, "(d)", x=-0.20, y=1.03)
    save(fig, "Fig4_homoclinic_evidence")


def figure5_parameter_synthesis() -> None:
    loci = load_csv("codimension_one_loci.csv")
    branch = load_csv("locked_branch_declared_metric.csv")
    dispersion = load_csv("dispersive_event_loci.csv")
    critical = load_csv("critical_points.csv")
    bt = critical[critical["event"] == "BT"][0]
    ep = critical[critical["event"] == "EP"][0]
    A_ep = float(ep["A"])
    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.15), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.22, hspace=0.08)
    axs = grid_axes.ravel()

    ax = axs[0]
    ax.plot(loci["beta"], loci["A_SN"], color=BLUE, label="SN")
    ax.plot(loci["beta"], loci["A_EP"], color=GREEN, ls="--",
            label="EP")
    ax.plot(loci["beta"], loci["A_H"], color=ORANGE, label="Hopf")
    ax.scatter([bt["beta"]], [bt["A"]], marker="*", s=55,
               color="black", zorder=5, label="BT")
    ax.set(xlabel=r"$\beta$", ylabel="critical $A$")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(a)")

    ax = axs[1]
    ax.semilogy(loci["beta"], loci["gap_EP_minus_SN"], color=GREEN,
                label=r"$A_{EP}-A_{SN}$")
    ax.semilogy(loci["beta"], loci["gap_H_minus_EP"], color=ORANGE,
                label=r"$A_H-A_{EP}$")
    ax.axvline(bt["beta"], color="black", ls=":", lw=0.8)
    ax.set(xlabel=r"$\beta$", ylabel="parameter separation")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=1,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(b)")

    ax = axs[2]
    near = abs(branch["A"] - A_ep) < 8.0e-4
    ax.plot(branch["A"][near], branch["g_declared_metric"][near] ** 2,
            color=PURPLE, label=r"$g_{\rm a}^2$")
    ax.plot(branch["A"][near], branch["gcrit_declared_metric"][near] ** 2,
            color=BLUE, ls="--", label=r"$g_{\rm s}^2$")
    ax.axvline(A_ep, color=GREEN, ls=":")
    ax.set(xlabel=r"$A$", ylabel="squared restricted rate")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    ax.plot(dispersion["Delta"], dispersion["A_SN"], color=BLUE,
            label="SN")
    ax.plot(dispersion["Delta"], dispersion["A_EP"], color=GREEN, ls="--",
            label="EP")
    ax.plot(dispersion["Delta"], dispersion["A_Hopf"], color=ORANGE,
            label="Hopf")
    ax.set(xlabel=r"frequency half-width $\Delta$", ylabel="critical $A$")
    ax.legend(frameon=True, framealpha=0.97, edgecolor="none", ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(d)", x=-0.20, y=1.03)
    save(fig, "Fig5_parameter_synthesis")


def figure8_dispersion_and_pulse() -> None:
    """Supplementary details for the positive-dispersion robustness test."""
    dispersion = load_csv("dispersive_event_loci.csv")
    pulses = load_csv("dispersive_pulse_responses.csv")
    microscopic = load_csv("microscopic_pulse_validation.csv")
    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.25), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.20, hspace=0.08)
    axs = grid_axes.ravel()

    ax = axs[0]
    ax.plot(dispersion["Delta"], dispersion["A_SN"], color=BLUE, label="SN")
    ax.plot(dispersion["Delta"], dispersion["A_EP"], color=GREEN, ls="--",
            label="EP")
    ax.plot(dispersion["Delta"], dispersion["A_Hopf"], color=ORANGE,
            label="Hopf")
    ax.set(xlabel=r"$\Delta$", ylabel="critical $A$")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=4, ny=4)
    grid(ax)
    label(ax, "(a)")

    ax = axs[1]
    ax.semilogy(dispersion["Delta"], dispersion["SN_to_EP"], color=GREEN,
                label=r"$A_{\rm EP}-A_{\rm SN}$")
    ax.semilogy(dispersion["Delta"], dispersion["EP_to_Hopf"], color=ORANGE,
                label=r"$A_{\rm H}-A_{\rm EP}$")
    ax.set(xlabel=r"$\Delta$", ylabel="parameter separation")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=1,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=4, ny=4)
    grid(ax)
    label(ax, "(b)", x=-0.20, y=1.03)

    ax = axs[2]
    for case, color, name in [
        ("node", BLUE, "node"),
        ("EP", GREEN, "EP"),
        ("focus", ORANGE, "focus"),
    ]:
        q = pulses[pulses["case"] == case]
        ax.plot(q["time"], q["nonlinear_drho2_over_epsilon"], color=color,
                label=name)
        ax.plot(q["time"], q["linear_drho2_over_epsilon"], color=color,
                ls=":", lw=0.8)
    ax.set(xlabel="time", ylabel=r"$\delta\rho_2/\epsilon$")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=4, ny=4)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    Ns = np.unique(microscopic["N"]).astype(int)
    means = []
    sems = []
    phase_means = []
    for N in Ns:
        q = microscopic[microscopic["N"] == N]
        amp = np.abs(q["amplitude_error"])
        phase = np.abs(q["phase_shift"])
        means.append(float(np.mean(amp)))
        sems.append(float(np.std(amp, ddof=1) / np.sqrt(len(amp))))
        phase_means.append(float(np.mean(phase)))
    ax.loglog(Ns, means, "o-", color=BLUE, label="amplitude error")
    ax.fill_between(
        Ns,
        np.maximum(np.asarray(means) - np.asarray(sems), 1e-12),
        np.asarray(means) + np.asarray(sems),
        color=BLUE,
        alpha=0.18,
    )
    ax.loglog(Ns, phase_means, "s--", color=PURPLE, label="phase shift")
    ax.set(xlabel="oscillators $N$", ylabel="finite-$N$ reset error")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=1,
              loc="lower left")
    grid(ax)
    label(ax, "(d)", x=-0.20, y=1.03)
    save(fig, "Fig8_dispersion_and_pulse")


def figure6_modal_reduction() -> None:
    ring4 = load_csv("ring_M4_modal_branch.csv")
    ring5 = load_csv("ring_M5_modal_branch.csv")
    counter = load_csv("ring_modal_counterexample.csv")
    eps = json.loads((DATA / "ring_modal_EP_summary.json").read_text())
    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.30), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.20, hspace=0.08)
    axs = grid_axes.ravel()

    ax = axs[0]
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    items = [
        (0.13, 0.78, 0.74, 0.16, "full $J$\n$n=2M-1$", BLUE),
        (0.13, 0.54, 0.74, 0.16, "Schur plane\n$Q^TQ=I$", GREEN),
        (0.13, 0.30, 0.74, 0.16, "modal block\n$B=Q^TJQ$", ORANGE),
    ]
    for x, y, w, h, text, color in items:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fill=False, ec=color, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=color, fontsize=8.0)
    for y0, y1 in [(0.78, 0.70), (0.54, 0.46)]:
        ax.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1), arrowstyle="-|>",
                                     mutation_scale=9, color=GRAY))
    ax.text(0.5, 0.21, r"$\delta=(a-d)/2,\quad s=(b+c)/2$",
            ha="center", fontsize=8.0)
    ax.text(0.5, 0.13, r"$g=(b-c)/2$", ha="center", fontsize=8.0)
    ax.text(0.5, 0.045,
            r"complex iff $\mathcal{R}_B>1/\sqrt{2}$",
            ha="center", color=GRAY, fontsize=8.0)
    label(ax, "(a)")

    ax = axs[1]
    for data, color, M, linestyle, marker in [
        (ring4, BLUE, 4, "-", "o"), (ring5, ORANGE, 5, "--", "s")
    ]:
        ax.plot(data["eta"], data["R_B"], color=color, ls=linestyle,
                marker=marker, markevery=12, label=fr"$M={M}$")
        ax.axvline(eps[f"M{M}"]["eta_EP"], color=color, ls=":", lw=0.9)
    ax.axhline(1 / np.sqrt(2), color="black", ls="--", lw=0.8,
               label=r"$1/\sqrt{2}$")
    ax.set(xlabel=r"directionality $\eta$", ylabel=r"$\mathcal{R}_B$")
    ax.legend(frameon=True, framealpha=0.97, edgecolor="none", ncol=1,
              loc="lower right")
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(b)")

    ax = axs[2]
    for data, color, M, marker in [
        (ring4, BLUE, 4, "o"), (ring5, ORANGE, 5, "s")
    ]:
        ax.plot(data["eta"], abs(data["lambda1_imag"]), color=color,
                marker=marker, markevery=12,
                label=fr"$|\mathrm{{Im}}\,\lambda|,{M}$")
        ax.plot(data["eta"], data["spectral_gap"], color=color, ls="--",
                marker=marker, markevery=12,
                label=fr"gap, ${M}$")
    ax.set(xlabel=r"$\eta$", ylabel="rate")
    ax.legend(frameon=True, framealpha=0.88, edgecolor="none", ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    for M, color, linestyle, marker in [
        (4, BLUE, "-", "o"), (5, ORANGE, "--", "s")
    ]:
        q = counter[counter["M"] == M]
        ax.plot(q["eta"], abs(q["g"]), color=color, ls=linestyle,
                marker=marker, label=fr"$M={M}$")
    ax.set(xlabel=r"$\eta$", ylabel=r"projected $|g|$")
    deviations = []
    for M in (4, 5):
        q = counter[(counter["M"] == M) & (counter["eta"] > 0)]
        deviations.extend(abs(q["cond_eigenvectors"] - 1.0))
    ax.text(0.05, 0.93, "normal block",
            transform=ax.transAxes, color=GRAY)
    ax.text(0.05, 0.84, r"$B=aI+g\Omega$",
            transform=ax.transAxes, color=GRAY)
    ax.text(0.05, 0.75, r"$|\kappa_2-1|$",
            transform=ax.transAxes, color=GRAY, fontsize=8.0)
    ax.text(0.05, 0.67, r"$<5\times10^{-8}$",
            transform=ax.transAxes, color=GRAY, fontsize=8.0)
    ax.text(0.96, 0.17, r"$M=4$", transform=ax.transAxes,
            ha="right", color=BLUE)
    ax.text(0.96, 0.07, r"$M=5$", transform=ax.transAxes,
            ha="right", color=ORANGE)
    compact_ticks(ax, nx=3, ny=4)
    grid(ax)
    label(ax, "(d)", x=-0.20, y=1.03)
    save(fig, "Fig6_modal_projection_M4_M5")


def figure7_finite_n() -> None:
    raw04 = np.loadtxt(
        DATA / "finite_n_raw_true_quotient_dt004.csv", delimiter=",", skiprows=1
    )
    raw02 = np.loadtxt(
        DATA / "finite_n_raw_true_quotient_dt002.csv", delimiter=",", skiprows=1
    )
    summary = load_csv("finite_n_summary.csv")
    fit = json.loads((DATA / "finite_n_fit_summary.json").read_text())
    N = summary["N"]
    oa_mean = float(np.mean(np.load(DATA / "figure_cycle_cache.npz")["r_033"]))
    fig, grid_axes = plt.subplots(
        2, 2, figsize=(COLUMN_WIDTH, 4.10), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.035, h_pad=0.035, wspace=0.20, hspace=0.08)
    axs = grid_axes.ravel()

    ax = axs[0]
    ax.errorbar(N, summary["mean_r2"], yerr=summary["sem_r2"], fmt="o-",
                ms=3.5, capsize=2, color=BLUE, label=r"$\Delta t=.04$")
    means02 = [np.mean(raw02[raw02[:, 0] == n, 3]) for n in N]
    ax.plot(N, means02, "x", color=ORANGE, label=r"$\Delta t=.02$")
    ax.axhline(oa_mean, color="black", ls="--", label="OA mean")
    ax.set(xscale="log", xlabel="oscillators per population $N$",
           ylabel=r"$\langle |Z_2|\rangle_t$")
    ax.legend(frameon=True, framealpha=0.97, edgecolor="none", ncol=1,
              loc="upper right")
    grid(ax)
    label(ax, "(a)")

    ax = axs[1]
    distance = summary["mean_rms_cycle_distance"]
    error = summary["sem_rms_cycle_distance"]
    ax.errorbar(N, distance, yerr=error, fmt="o", ms=3.5, capsize=2,
                color=GREEN, label="chordal quotient")
    xx = np.geomspace(N.min(), N.max(), 100)
    ax.plot(xx, np.exp(fit["intercept_dt_0p04"]) * xx ** fit["slope_dt_0p04"],
            color="black", ls="--", label=fr"fit $N^{{{fit['slope_dt_0p04']:.3f}}}$")
    lo, hi = fit["bootstrap_95_percent_CI"]
    ax.text(0.05, 0.10, r"CI $[-.54,-.46]$",
            transform=ax.transAxes, color=GRAY, fontsize=8.0)
    ax.set(xscale="log", yscale="log", xlabel="$N$",
           ylabel="RMS OA-cycle distance")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=1,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    grid(ax)
    label(ax, "(b)", x=-0.01, y=1.08)

    ax = axs[2]
    for i, n in enumerate(N):
        q = raw04[raw04[:, 0] == n, 7]
        jitter = np.linspace(-0.06, 0.06, len(q))
        ax.scatter(np.full_like(q, i) + jitter, q, s=12, color=BLUE, alpha=.75)
        ax.plot([i - .16, i + .16], [np.mean(q), np.mean(q)], color="black", lw=1.2)
    ax.set(xticks=np.arange(len(N)), xticklabels=["64", "128", "256", "512", "1k", "2k"],
           xlabel="$N$ (seeds)", ylabel="RMS cycle distance")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=35)
    grid(ax)
    label(ax, "(c)")

    ax = axs[3]
    mean04 = [np.mean(raw04[raw04[:, 0] == n, 7]) for n in N]
    mean02 = [np.mean(raw02[raw02[:, 0] == n, 7]) for n in N]
    ax.loglog(N, mean04, "o-", ms=3, color=BLUE,
              label=fr"$\Delta t=.04$ ({fit['slope_dt_0p04']:.3f})")
    ax.loglog(N, mean02, "x--", color=ORANGE,
              label=fr"$\Delta t=.02$ ({fit['slope_dt_0p02']:.3f})")
    ax.set(xlabel="$N$", ylabel="RMS cycle distance")
    ax.legend(frameon=True, framealpha=0.95, edgecolor="none", ncol=1,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    grid(ax)
    label(ax, "(d)", x=-0.01, y=1.08)
    ax.text(0.03, 0.08, "finite $T$\nPoisson kernel",
            transform=ax.transAxes, color=GRAY, fontsize=8.0)
    save(fig, "Fig7_finite_N_validation")


def main() -> None:
    cache_path = DATA / "figure_cycle_cache.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        cycle033 = {
            "orbit_time": cache["t_033"], "orbit_r": cache["r_033"],
            "orbit_psi": cache["psi_033"],
        }
        cycles006 = {
            A: {
                "orbit_time": cache[f"t_{key}"],
                "orbit_r": cache[f"r_{key}"],
                "orbit_psi": cache[f"psi_{key}"],
            }
            for A, key in [(0.35, "0350"), (0.385, "0385"), (0.395, "0395")]
        }
    else:
        cycle033 = transverse_cycle(0.33, 0.10)
        cycles006 = {A: transverse_cycle(A, 0.06) for A in (0.35, 0.385, 0.395)}
        np.savez_compressed(
            cache_path,
            t_033=cycle033["orbit_time"], r_033=cycle033["orbit_r"],
            psi_033=cycle033["orbit_psi"],
            t_0350=cycles006[0.35]["orbit_time"],
            r_0350=cycles006[0.35]["orbit_r"], psi_0350=cycles006[0.35]["orbit_psi"],
            t_0385=cycles006[0.385]["orbit_time"],
            r_0385=cycles006[0.385]["orbit_r"], psi_0385=cycles006[0.385]["orbit_psi"],
            t_0395=cycles006[0.395]["orbit_time"],
            r_0395=cycles006[0.395]["orbit_r"], psi_0395=cycles006[0.395]["orbit_psi"],
        )
    graphical_abstract()
    figure1_overview()
    figure2_ep_response()
    figure3_hopf_floquet()
    figure4_same_cut_homoclinic()
    figure5_parameter_synthesis()
    figure6_modal_reduction()
    figure7_finite_n()
    figure8_dispersion_and_pulse()
    print(f"Generated figures in {FIG}")


if __name__ == "__main__":
    main()
