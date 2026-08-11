"""Archetype configuration for the EV charging agent-based model.

Structural parameters (§3.1) and distributional parameters (§3.2) live on the same
dataclass. Every field is editable from the app -- there is no locked/unlocked
distinction between archetypes.

Clock hours are stored as floats on a 24-hour clock (18.5 == 18:30). They are
converted to offsets into the noon-anchored simulation window by
:func:`model.timegrid.to_window_offset`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

# Behaviour keys -- select which event generator in model.events is used.
UNMANAGED = "unmanaged"
MANAGED = "managed"
INFREQUENT_CHARGING = "infrequent_charging"
ALWAYS_PLUGGED_IN = "always_plugged_in"

BEHAVIOURS = (UNMANAGED, MANAGED, INFREQUENT_CHARGING, ALWAYS_PLUGGED_IN)


@dataclass
class ArchetypeConfig:
    """One driver archetype: structural facts plus the distributions it samples from."""

    # --- structural (§3.1) ---
    name: str
    population_pct: float
    battery_kwh: float
    charger_kw: float
    window_start_hr: float  # clock hour the plug-in window opens
    window_end_hr: float  # clock hour the driver needs the car back (deadline)
    # Seeds target_soc_choices when no distribution is given, and nothing else. Read
    # mean_target_soc -- or an agent's own sampled target -- everywhere else.
    default_target_soc: float  # percent
    managed: bool
    behaviour: str = UNMANAGED

    # --- timing distributions (§3.2, §4.5) ---
    plugin_time_mean_hr: float = 18.5
    plugin_time_std_hr: float = 1.0

    # Ready-by is a value the driver enters, so it is modelled as a weighted choice
    # among a few times rather than a continuous density -- nobody sets 07:13. Each
    # entry is [clock_hour, weight]; weights are normalised, so they can be counts,
    # percentages or fractions.
    readyby_choices: list[list[float]] = field(default_factory=list)

    # Actual plug-out is the driver's own deadline plus this offset, plus noise.
    # Ready-by is what they asked the charger for; departure is what they do. Keeping
    # the offset at zero centres departure on the deadline, but it need not.
    plugout_offset_hr: float = 0.0
    plugout_time_std_hr: float = 30.0 / 60.0

    # Target state of charge as a weighted choice, for the same reason as ready-by: it is a
    # value the driver picks in an app, and the CNZ report shows the population genuinely split
    # across 80/90/100 rather than agreeing on one figure. Each entry is [percent, weight].
    # Empty means "one choice at default_target_soc".
    
    target_soc_choices: list[list[float]] = field(default_factory=list)

    # --- SoC-at-plug-in distribution (§3.2, §4.6) ---
    plugin_soc_mean: float = 60.0
    plugin_soc_std: float = 10.0

    # --- driving, used to deplete SoC after plug-out ---
    # Annual mileage is split into how *often* the driver drives and how far each trip
    # is, rather than being flattened into a daily average.
    miles_per_year: float = 9435.0
    efficiency_mi_per_kwh: float = 3.5
    driving_days_per_week: float = 7.0

    # How often the driver plugs in. Separate from driving frequency: a driver can drive
    # daily and still plug in every other day. 
    plugin_days_per_week: float = 7.0

    # --- managed archetypes only (§3.2, §4.4) ---
    cheap_window_start_hr: Optional[float] = None
    cheap_window_end_hr: Optional[float] = None
    charge_placement_std_hr: Optional[float] = None

    # --- Infrequent Charging only (§3.2, §4.7) ---
    interplug_gap_min_days: Optional[float] = None
    interplug_gap_mode_days: Optional[float] = None
    interplug_gap_max_days: Optional[float] = None

    # --- Always Plugged-In only (§3.2, §4.8) ---
    trip_duration_hrs: Optional[float] = None

    def __post_init__(self) -> None:
        if self.behaviour not in BEHAVIOURS:
            raise ValueError(f"unknown behaviour {self.behaviour!r}; expected one of {BEHAVIOURS}")
        if self.behaviour == MANAGED and (
            self.cheap_window_start_hr is None or self.cheap_window_end_hr is None
        ):
            raise ValueError(f"{self.name}: managed archetypes need a cheap window")
        if self.behaviour == INFREQUENT_CHARGING and self.interplug_gap_mode_days is None:
            raise ValueError(f"{self.name}: infrequent charging needs interplug gap parameters")
        if self.behaviour == ALWAYS_PLUGGED_IN and self.trip_duration_hrs is None:
            raise ValueError(f"{self.name}: always-plugged-in needs a trip duration")
        if not self.target_soc_choices:
            self.target_soc_choices = [[self.default_target_soc, 1.0]]
        for entry in self.target_soc_choices:
            if len(entry) != 2:
                raise ValueError(f"{self.name}: target_soc_choices entries must be [pct, weight]")
        if sum(w for _, w in self.target_soc_choices) <= 0:
            raise ValueError(f"{self.name}: target_soc_choices weights must be positive")
        if not self.readyby_choices:
            self.readyby_choices = [[self.window_end_hr, 1.0]]
        for entry in self.readyby_choices:
            if len(entry) != 2:
                raise ValueError(f"{self.name}: readyby_choices entries must be [hour, weight]")
        if sum(w for _, w in self.readyby_choices) <= 0:
            raise ValueError(f"{self.name}: readyby_choices weights must sum to something positive")

    # --- target state of charge ------------------------------------------------

    @property
    def target_soc_distribution(self) -> list[tuple[float, float]]:
        """(percent, normalised weight) pairs, heaviest first."""
        total = sum(w for _, w in self.target_soc_choices)
        pairs = [(float(t), float(w) / total) for t, w in self.target_soc_choices if w > 0]
        return sorted(pairs, key=lambda p: -p[1])

    @property
    def mean_target_soc(self) -> float:
        """Weighted mean target, which is what the energy figures should be read against."""
        return sum(t * w for t, w in self.target_soc_distribution)

    @property
    def modal_target_soc(self) -> float:
        return self.target_soc_distribution[0][0]

    # --- ready-by --------------------------------------------------------------

    @property
    def readyby_distribution(self) -> list[tuple[float, float]]:
        """(clock_hour, normalised weight) pairs, heaviest first."""
        total = sum(w for _, w in self.readyby_choices)
        pairs = [(float(h), float(w) / total) for h, w in self.readyby_choices if w > 0]
        return sorted(pairs, key=lambda p: -p[1])

    @property
    def mean_readyby_hr(self) -> float:
        """Weighted mean deadline, as an offset-safe clock hour.

        Averaged in window-offset space so that a mix of 23:00 and 01:00 deadlines
        does not average to midday.
        """
        from model.timegrid import to_clock_hour, window_offsets

        weighted = sum(
            window_offsets(self.window_start_hr, h)[1] * w for h, w in self.readyby_distribution
        )
        return to_clock_hour(weighted)

    @property
    def mean_required_duration_hrs(self) -> float:
        """Charge length implied by the *mean* plug-in SoC, for diagnostics only."""
        deficit = max(self.mean_target_soc - self.plugin_soc_mean, 0.0)
        energy = deficit / 100.0 * self.battery_kwh
        return energy / self.charger_kw if self.charger_kw > 0 else 0.0

    @property
    def is_continuous(self) -> bool:
        """True when the archetype is nominally plugged in for the whole window (§4.8)."""
        return self.behaviour == ALWAYS_PLUGGED_IN

    @property
    def charging_kind(self) -> str:
        """Which of the three kinds of charging this is: unmanaged, scheduled or managed.

        Derived, because ``behaviour`` cannot answer it: scheduled and managed share the
        MANAGED dispatch key and the same placement code. What separates them is the
        placement spread -- a timer fires when its window opens, so zero spread *is* the
        scheduled case. Deriving it from the mechanism means the label follows a user who
        edits the spread rather than contradicting them, which a stored flag would not.
        """
        if self.behaviour != MANAGED:
            return "unmanaged"
        return "scheduled" if not self.charge_placement_std_hr else "managed"

    # --- driving depletion, all derived rather than stored ---------------------

    @property
    def expected_plugins_per_day(self) -> float:
        """How often this archetype plugs in.

        Derived from the cadence parameters for Infrequent Charging so it cannot
        contradict the sliders; every other archetype plugs in once per window.
        """
        if self.behaviour == INFREQUENT_CHARGING:
            mean_gap = (
                self.interplug_gap_min_days
                + self.interplug_gap_mode_days
                + self.interplug_gap_max_days
            ) / 3.0
            return 1.0 / mean_gap if mean_gap > 0 else 1.0
        return max(self.plugin_days_per_week / 7.0, 0.0)

    @property
    def daily_miles(self) -> float:
        """Annual mileage spread across all days, driving or not."""
        return self.miles_per_year / 365.0

    @property
    def driving_probability_per_day(self) -> float:
        """Chance this driver takes a trip on any given day."""
        return min(max(self.driving_days_per_week / 7.0, 0.0), 1.0)

    @property
    def trip_miles(self) -> float:
        """Miles driven on a day when the driver does drive.

        Divided by the *driving* days rather than all days, so cutting the frequency
        lengthens each trip instead of shrinking it.
        """
        probability = self.driving_probability_per_day
        if probability <= 0:
            return 0.0
        return self.miles_per_year / (365.0 * probability)

    @property
    def trip_energy_kwh(self) -> float:
        if self.efficiency_mi_per_kwh <= 0:
            return 0.0
        return self.trip_miles / self.efficiency_mi_per_kwh

    @property
    def trip_soc_drop_pct(self) -> float:
        """Percentage points consumed by one trip, on a day the driver drives."""
        if self.battery_kwh <= 0:
            return 0.0
        return self.trip_energy_kwh / self.battery_kwh * 100.0

    @property
    def expected_daily_soc_drop_pct(self) -> float:
        """Average daily depletion across driving and non-driving days alike."""
        return self.trip_soc_drop_pct * self.driving_probability_per_day

    @property
    def expected_plugin_gap_days(self) -> float:
        """Days between plug-in events."""
        rate = self.expected_plugins_per_day
        return 1.0 / rate if rate > 0 else 1.0

    @property
    def driving_soc_drop_between_plugins_pct(self) -> float:
        """Depletion accumulated between one plug-in and the next.

        Should land close to ``mean_target_soc - plugin_soc_mean``: the energy taken out
        driving is the energy that has to go back in. Unlike the old formulation this
        is a daily *rate* multiplied by the plug-in cadence, so it validates the two
        against each other rather than only checking a total. See
        :func:`model.validate.validate_energy_closure`.
        """
        return self.expected_daily_soc_drop_pct * self.expected_plugin_gap_days

    # --- serialisation (Tab 4) ---
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchetypeConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown archetype fields: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ScenarioConfig:
    """A full run: the archetype mix plus population size and seed."""

    archetypes: list[ArchetypeConfig] = field(default_factory=lambda: default_archetypes())
    n_agents: int = 2000
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "seed": self.seed,
            "archetypes": [a.to_dict() for a in self.archetypes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioConfig":
        return cls(
            archetypes=[ArchetypeConfig.from_dict(a) for a in data["archetypes"]],
            n_agents=int(data.get("n_agents", 2000)),
            seed=int(data.get("seed", 42)),
        )

    def total_population_pct(self) -> float:
        return sum(a.population_pct for a in self.archetypes)


# --- defaults -----------------------------------------------------------------
# Structural values come from the task brief, distributions from the Centre for Net Zero /
# Octopus report. Which figure came from where, what was recalibrated after reading the
# report, and where its numbers fail to reconcile: README, "Calibration against the
# published report". Keep the two in step when changing anything below.

_IO_CHEAP_START = 23.5  # 23:30 (report Table 1)
_IO_CHEAP_END = 5.5  # 05:30

# Report Table 2's plug-in cadence, in days per week. Exported rather than applied:
# `plugin_days_per_week` defaults to daily, and adopting this figure also requires
# re-deriving mileage. See README, "Plug-in frequency is the open question".
CNZ_PLUGIN_DAYS_PER_WEEK = 5.18

# Ready-by settings, from report Figure 2 (see README's calibration table).
_CNZ_READYBY_CHOICES = [[6.0, 0.15], [6.5, 0.15], [7.0, 0.35], [7.5, 0.10], [8.0, 0.25]]

# Target state of charge, from report Figure 2. The top three weights are measured; 70/60/85
# are estimates spread over the heatmap's unlabelled lower rows. Mean 84.3%, mode 80%.
# Deliberately *not* rescaled to make the energy arithmetic balance -- see README, "Target
# state of charge is a distribution, and it moved the plug-in SoC with it".
CNZ_TARGET_SOC_CHOICES = [
    [80.0, 0.25],
    [90.0, 0.24],
    [100.0, 0.23],
    [70.0, 0.14],
    [60.0, 0.08],
    [85.0, 0.06],
]

# Timing, from report Figures 4 and 5 (see README's calibration table).
_CNZ_WINDOW_START = 17.0
_CNZ_PLUGIN_MEAN = 18.0
_CNZ_PLUGOUT_OFFSET = 1.0
_CNZ_PLUGOUT_STD = 45.0 / 60.0


def default_archetypes() -> list[ArchetypeConfig]:
    """The six archetypes of §3.1 with their default distributions (§3.2)."""
    return [
        ArchetypeConfig(
            name="Average (UK)",
            population_pct=40.0,
            battery_kwh=60.0,
            charger_kw=7.0,
            window_start_hr=_CNZ_WINDOW_START,
            window_end_hr=7.0,
            default_target_soc=80.0,
            managed=False,
            behaviour=UNMANAGED,
            plugin_time_mean_hr=_CNZ_PLUGIN_MEAN,
            plugin_time_std_hr=1.0,
            plugin_soc_mean=72.0,
            plugin_soc_std=10.0,
            target_soc_choices=list(CNZ_TARGET_SOC_CHOICES),
            readyby_choices=list(_CNZ_READYBY_CHOICES),
            plugout_offset_hr=_CNZ_PLUGOUT_OFFSET,
            plugout_time_std_hr=_CNZ_PLUGOUT_STD,
            miles_per_year=9435.0,
        ),
        ArchetypeConfig(
            name="Intelligent Octopus Average",
            population_pct=30.0,
            battery_kwh=72.5,
            charger_kw=7.0,
            window_start_hr=_CNZ_WINDOW_START,
            window_end_hr=7.0,
            default_target_soc=80.0,
            managed=True,
            behaviour=MANAGED,
            plugin_time_mean_hr=_CNZ_PLUGIN_MEAN,
            plugin_time_std_hr=1.0,
            plugin_soc_mean=52.0,  # report Figure 7
            plugin_soc_std=25.0,
            target_soc_choices=list(CNZ_TARGET_SOC_CHOICES),
            readyby_choices=list(_CNZ_READYBY_CHOICES),
            plugout_offset_hr=_CNZ_PLUGOUT_OFFSET,
            plugout_time_std_hr=_CNZ_PLUGOUT_STD,
            miles_per_year=28105.0,
            cheap_window_start_hr=_IO_CHEAP_START,
            cheap_window_end_hr=_IO_CHEAP_END,
            charge_placement_std_hr=1.0,
        ),
        ArchetypeConfig(
            name="Infrequent Charging",
            population_pct=10.0,
            battery_kwh=60.0,
            charger_kw=7.0,
            window_start_hr=_CNZ_WINDOW_START,
            window_end_hr=7.0,
            default_target_soc=80.0,
            managed=False,
            behaviour=INFREQUENT_CHARGING,
            plugin_time_mean_hr=_CNZ_PLUGIN_MEAN,
            plugin_time_std_hr=1.0,
            plugin_soc_mean=19.0,
            plugin_soc_std=10.0,
            target_soc_choices=list(CNZ_TARGET_SOC_CHOICES),
            readyby_choices=list(_CNZ_READYBY_CHOICES),
            plugout_offset_hr=_CNZ_PLUGOUT_OFFSET,
            plugout_time_std_hr=_CNZ_PLUGOUT_STD,
            miles_per_year=9435.0,
            interplug_gap_min_days=3.0,
            interplug_gap_mode_days=5.0,
            interplug_gap_max_days=8.0,
        ),
        ArchetypeConfig(
            name="Infrequent Driving",
            population_pct=10.0,
            battery_kwh=60.0,
            charger_kw=7.0,
            window_start_hr=_CNZ_WINDOW_START,
            window_end_hr=7.0,
            default_target_soc=80.0,
            managed=False,
            behaviour=UNMANAGED,
            plugin_time_mean_hr=_CNZ_PLUGIN_MEAN,
            plugin_time_std_hr=1.0,
            plugin_soc_mean=77.0,
            plugin_soc_std=10.0,
            target_soc_choices=list(CNZ_TARGET_SOC_CHOICES),
            readyby_choices=list(_CNZ_READYBY_CHOICES),
            plugout_offset_hr=_CNZ_PLUGOUT_OFFSET,
            plugout_time_std_hr=_CNZ_PLUGOUT_STD,
            miles_per_year=5700.0,
            driving_days_per_week=3.0,  # the archetype's defining characteristic
        ),
        ArchetypeConfig(
            name="Scheduled Charging",
            population_pct=9.0,
            battery_kwh=60.0,
            charger_kw=7.0,
            # The driver plugs in when they get home, like everyone else -- what makes this
            # archetype distinctive is the timer, not an unusually late arrival. The window
            # has to open before the timer does, or the timer can never be what starts the
            # charge: placement floors at the plug-in, so a car arriving at 22:30 would
            # begin charging on arrival and the 21:00 setting would be inert.
            window_start_hr=_CNZ_WINDOW_START,
            window_end_hr=9.0,
            default_target_soc=80.0,
            managed=True,
            behaviour=MANAGED,
            plugin_time_mean_hr=_CNZ_PLUGIN_MEAN,
            plugin_time_std_hr=1.0,
            plugin_soc_mean=72.0,
            plugin_soc_std=10.0,
            target_soc_choices=list(CNZ_TARGET_SOC_CHOICES),
            miles_per_year=9435.0,
            # A timer, not a tariff window: it opens at 21:00 and the charge starts the
            # moment it does, every night. The zero placement spread is what separates
            # this archetype from managed charging in the code -- both share the placement
            # mechanism, and the spread is the only thing that made them differ.
            cheap_window_start_hr=21.0,
            cheap_window_end_hr=_IO_CHEAP_END,
            charge_placement_std_hr=0.0,
        ),
        ArchetypeConfig(
            name="Always Plugged-In",
            population_pct=1.0,
            battery_kwh=60.0,
            charger_kw=7.0,
            window_start_hr=0.0,
            window_end_hr=23.983333,  # 23:59
            default_target_soc=80.0,
            managed=False,
            behaviour=ALWAYS_PLUGGED_IN,
            plugin_time_mean_hr=0.0,  # unused: connection is continuous (§4.8)
            plugin_time_std_hr=0.0,
            plugin_soc_mean=72.0,
            plugin_soc_std=10.0,
            target_soc_choices=list(CNZ_TARGET_SOC_CHOICES),
            miles_per_year=9435.0,
            trip_duration_hrs=1.0,
        ),
    ]


DEFAULT_ARCHETYPES = default_archetypes()
