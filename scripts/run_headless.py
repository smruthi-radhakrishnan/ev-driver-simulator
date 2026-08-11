#!/usr/bin/env python3
"""Run a scenario headlessly and write charts + CSVs to disk (§7.5).

    python scripts/run_headless.py
    python scripts/run_headless.py --config my_scenario.json --n-agents 5000 --outdir out

Charts are written as self-contained interactive HTML. PNG copies are also written
when a compatible plotly/kaleido pair is installed, but they are not required.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# An incompatible plotly/kaleido pair warns loudly at import time about static image
# export. We fall back to HTML for exactly that case, so the warning is just noise.
warnings.filterwarnings("ignore", message=r"(?s).*version of Kaleido.*")

# Allow running as a plain script from a clone with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ScenarioConfig, Simulator  # noqa: E402
from model.aggregate import (  # noqa: E402
    compute_archetype_breakdown,
    compute_charging_demand,
    compute_demand_per_agent,
    compute_occupancy_bars,
    compute_soc_band,
    headline_metrics,
)
from model.plotting import (  # noqa: E402
    build_agent_trace_chart,
    build_demand_per_agent_box,
    build_timing_preview,
    build_soc_distribution_preview,
    build_breakdown_chart,
    build_combined_chart,
    build_demand_chart,
    build_plugin_soc_histogram,
)
from model.population import population_summary  # noqa: E402
from model.validate import validate_energy_closure, validate_io_average  # noqa: E402

DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "outputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="scenario JSON (as exported from Tab 4)")
    parser.add_argument("--n-agents", type=int, help="override the scenario's population size")
    parser.add_argument("--seed", type=int, help="override the scenario's random seed")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"where to write charts and CSVs (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--agent-trace",
        type=int,
        default=0,
        help="agent_id to render an individual trace chart for (default: 0)",
    )
    return parser.parse_args(argv)


def load_scenario(args: argparse.Namespace) -> ScenarioConfig:
    if args.config:
        scenario = ScenarioConfig.from_dict(json.loads(args.config.read_text()))
    else:
        scenario = ScenarioConfig()
    if args.n_agents is not None:
        scenario.n_agents = args.n_agents
    if args.seed is not None:
        scenario.seed = args.seed
    return scenario


def write_figure(fig, outdir: Path, stem: str) -> None:
    fig.write_html(outdir / f"{stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(outdir / f"{stem}.png", width=1200, height=650, scale=2)
    except Exception:
        # PNG export needs a compatible kaleido; the HTML is the real deliverable.
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scenario = load_scenario(args)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running {scenario.n_agents} agents, seed {scenario.seed} ...")
    result = Simulator.run(scenario)

    charts = {
        "combined_occupancy_and_soc": build_combined_chart(result),
        "archetype_breakdown": build_breakdown_chart(result),
        "archetype_plugin_rate": build_breakdown_chart(result, normalise=True),
        "charging_demand": build_demand_chart(result),
        "charging_demand_per_agent": build_demand_per_agent_box(result),
        "combined_occupancy_and_soc_whole_fleet": build_combined_chart(
            result, connected_only=False
        ),
        "plugin_soc_distribution": build_plugin_soc_histogram(result),
        f"agent_{args.agent_trace}_trace": build_agent_trace_chart(result, args.agent_trace),
    }
    for cfg in result.archetypes:
        stem = cfg.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        charts[f"assumptions_{stem}_soc"] = build_soc_distribution_preview(cfg)
        timing = build_timing_preview(cfg)
        if timing is not None:  # always-connected archetypes have no timing chart
            charts[f"assumptions_{stem}_timing"] = timing

    for stem, fig in charts.items():
        write_figure(fig, outdir, stem)

    tables = {
        "occupancy": compute_occupancy_bars(result),
        "soc_band": compute_soc_band(result),
        "charging_demand": compute_charging_demand(result),
        "archetype_breakdown": compute_archetype_breakdown(result),
        "event_log": result.events,
        "population_summary": population_summary(result.agents, result.archetypes),
        "validation_io_average": validate_io_average(result),
        "validation_energy_closure": validate_energy_closure(result),
        "demand_per_agent": compute_demand_per_agent(result),
    }
    for stem, frame in tables.items():
        frame.to_csv(outdir / f"{stem}.csv", index=False)

    (outdir / "scenario.json").write_text(json.dumps(scenario.to_dict(), indent=2))

    metrics = headline_metrics(result)
    print("\nHeadline metrics")
    print(f"  agents                     {metrics['agents']:,.0f}")
    print(f"  peak % plugged in          {metrics['peak_pct_plugged_in']:.1f}%")
    print(
        f"  peak charging demand       {metrics['peak_demand_kw']:,.0f} kW "
        f"at {metrics['peak_demand_time']}"
    )
    print(f"  total energy delivered     {metrics['total_energy_kwh']:,.0f} kWh")
    print(f"  agents with an event       {metrics['pct_with_event']:.1f}%")
    print(f"  median plug-in SoC         {metrics['median_plugin_soc']:.1f}%")
    print(f"  median charge duration     {metrics['median_charge_duration_hrs']:.2f} hrs")
    print(f"  median plug-out SoC        {metrics['median_plugout_soc']:.1f}%")
    print(f"  left below target          {metrics['pct_left_below_target']:.2f}%")

    validation = validate_io_average(result)
    if not validation.empty:
        print("\nIO Average vs report (informational)")
        print(validation.to_string(index=False))

    closure = validate_energy_closure(result)
    if not closure.empty:
        print("\nEnergy closure: driving depletion vs charge requirement")
        print(closure.to_string(index=False))

    print(f"\nWrote {len(charts)} charts and {len(tables)} CSVs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
