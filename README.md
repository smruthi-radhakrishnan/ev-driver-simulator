# ev-driver-simulator

An agent-based model of EV home charging behaviour across six driver archetypes,
simulated over a single 24-hour window, plus a Streamlit app for configuring and
viewing it.

Each agent is simulated individually (plug-in and plug-out times, charge schedule,
State of Charge over time). The population is then aggregated into the headline view:
share of the fleet plugged in, and the State of Charge distribution, over 24 hours.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/run_headless.py          # charts + CSVs into outputs/
.venv/bin/python -m streamlit run app/streamlit_app.py
.venv/bin/python -m pytest                        # 56 tests, ~17s
```

The package is importable on its own and never imports Streamlit:

```python
from model import ScenarioConfig, Simulator
from model.plotting import build_combined_chart

result = Simulator.run(ScenarioConfig(n_agents=5000, seed=42))
build_combined_chart(result).show()
```

## Layout

```
model/            simulation package (no UI dependency)
  config.py        ArchetypeConfig / ScenarioConfig and the six defaults
  timegrid.py      24hr noon-anchored window at 15-minute resolution
  population.py    agent construction, archetype allocation, renewal phases
  events.py        per-agent event generation (unmanaged / scheduled / managed, trips)
  simulator.py     discretises events onto the grid; emits the event log
  aggregate.py     occupancy, SoC percentiles, charging demand, archetype breakdown
  distributions.py analytic densities behind the assumptions previews
  validate.py      IO Average vs the published report (informational)
  plotting.py      Plotly figures, shared by the script and the app
app/               Streamlit wrapper: introduction (summary), approach/assumptions/
                   rationale (static write-up), population (mix and resulting
                   distributions), assumptions (every editable number), results, export
scripts/           run_headless.py
tests/             56 tests -- model behaviour, not presentation detail
```

## How an agent is simulated

The model is agent-based rather than a curve fitted to aggregate data: every driver is
constructed and simulated on its own, and the fleet-level charts are nothing more than
column means over those individual traces. Reading the mechanics in order:

**1. Build the population.** `build_population` allocates `n_agents` across the
archetypes by largest remainder, so a 1% archetype gets exactly 1% rather than whatever
a multinomial draw happens to give. Each agent is a small record: an id, its archetype,
and — for Infrequent Charging only — a renewal gap and a random phase within it.

**2. Generate one event per agent.** `generate_event` dispatches on the archetype's
behaviour and draws, in order:

| Quantity | Drawn from |
|---|---|
| Plug-in time | Truncated normal distribution, held inside the plug-in window |
| Plug-in state of charge | Normal distribution, truncated to 0–100% (rejection, not clipping) |
| Ready-by time | Weighted choice among the archetype's configured times |
| Departure time | Ready-by + offset + normal noise |
| Whether it drives after departure | Bernoulli distribution on `driving_days_per_week / 7` |

Charge duration is *not* drawn. It follows from the state of charge gap, the battery and
the charge rate. Where the charge sits then depends on the behaviour: unmanaged agents
start at plug-in, managed agents place the block inside the cheap window, and the block
is truncated if the driver departs before it finishes.

The result is an `AgentEvent` — connection intervals, charge start and end, energy
delivered, and the state of charge at each transition.

**3. Discretise onto the grid.** `_discretise` converts those continuous-time events into
`(n_agents, 96)` arrays, fully vectorised: per-agent scalars are stacked into columns
and broadcast against the 96 timesteps in one pass, so a 20,000-agent run is a handful
of NumPy operations rather than 20,000 Python loops.

**4. Aggregate.** Column means and percentiles over those arrays, plus a one-row-per-agent
event log for the explorer and the CSV exports.

**Randomness.** Two independent streams: one seeded `seed` for population construction,
one seeded `seed + 1` for behaviour. Changing the population size therefore does not
reshuffle the behavioural draws of unrelated agents. The same seed and scenario reproduce
a run exactly.

**What an agent does not have.** No memory across windows, no interaction with other
agents, no response to price beyond its archetype's fixed cheap window, and no feedback
from the grid. Agents are independent draws from their archetype's distributions, which
is what makes the closed-form previews on the Population tab exact.

## The model in brief

**One 24-hour window, anchored at noon**, at 15-minute resolution. Noon-anchoring
matters: every archetype's overnight cycle (earliest plug-in 17:00, latest deadline
09:00) then sits inside one window with no wrap-around. There is no multi-day
calendar, no warm-up period, and no weekday/weekend variation.

**Charge duration is never a stored parameter.** It is derived per event:

```
energy_required_kwh = (target_soc - sampled_plugin_soc)/100 * battery_kwh
charge_duration_hrs = energy_required_kwh / charger_kw
```

This is physically correct (charging stops at the target) and avoids a stated duration
figure disagreeing with the battery/charger/SoC numbers it should follow from.

**Three kinds of charging**, distinguished by what decides when the charge happens:

- **Unmanaged charging** runs in one contiguous block from the moment of plug-in.
- **Scheduled charging** runs to a timer the driver sets once, opening at 21:00. It starts
  at the same time every night, independently of the tariff and of when the car is next
  needed. On the defaults 99.8% of these agents begin charging at exactly 21:00; the
  remainder arrive home after the timer has already opened and charge on arrival.
- **Managed charging** is placed by the platform inside a cheap-rate window, defaulting to
  23:30–05:30 — distributed across this window in one block of charging per agent, because 
  Intelligent Octopus charging is scheduled against wholesale prices.

**Ready-by, plug-out and the connection window are three separate things.**

*Ready-by* is a value the driver enters, so it is modelled as a **weighted choice among
times** rather than a continuous density — nobody sets 07:13. Each archetype carries a
`readyby_choices` table of `[time, weight]`, defaulting to a single row at its own
deadline. Managed charging optimises against each agent's own draw from this.

*Plug-out* is when the car physically leaves: the agent's ready-by time, plus a
configurable offset (**+1 hour** on the defaults), plus noise (45-minute spread,
deliberately tighter than the 60-minute plug-in spread, since morning disconnects cluster
more than evening plug-ins). A driver departing before the charge finishes leaves below target;
that share is a headline metric, around 0.02% on the defaults.

*The connection window* is derived from the other two, not configured:
`P(plugged in at t) = P(plug-in ≤ t) × P(plug-out > t)`. That closed form has been
checked against a 40,000-agent simulation and agrees to within 0.65 percentage points,
which is the simulation's own sampling noise; the one term that breaks independence (a
guard forcing plug-out after plug-in) binds in 0.0000% of events.

A spread of ready-by times interacts with the cheap window in two distinct ways, and the
app reports each because the difference is easy to get wrong:

- A deadline **before the window opens** means the driver cannot use it at all and
  charges from plug-in, behaving like an unmanaged driver.
- A deadline **inside the window** merely curtails it. Charging still happens within the
  window, just earlier. It does *not* push charging out.

Where the deadline sets the finish time, the charge ends up hard against it, so
departure noise leaves a material share of that cohort short. The config reports which
drivers are constrained (`readyby_weight_deadline_binds`, exactly computable); how many
end up below target is left to the simulation, which measures it directly.

**After plug-out, state of charge falls.** Annual mileage is split into how *often* the
driver drives and how *far* each trip is, rather than being flattened into a daily
average:

```
trip_miles      = miles_per_year / (365 × driving_probability)
trip_energy_kwh = trip_miles / efficiency_mi_per_kwh
trip_soc_drop   = trip_energy_kwh / battery_kwh × 100      # on a day they drive
```

A low-mileage driver is someone who takes a normal trip less often, not someone who 
takes a token trip every day. Infrequent Driving at 3 days a week takes
a **36.4 mile** trip on 43% of days — *longer* than the average driver's 25.9 miles —
rather than 15.6 miles daily. Same annual total, same mean depletion, but the realised
distribution is bimodal (17.4pp or nothing) instead of a uniform trickle.

Each window shows one trip, or none. How long the drive takes is a fixed assumption
(`DRIVE_DURATION_HRS`, 1h) rather than an input, since it only sets how steeply the line
falls, not how far.

### The six archetypes

| # | Archetype | Share | Battery | Plug-in window | Plug-in SoC | Managed |
|---|---|---|---|---|---|---|
| 1 | Average (UK) | 40% | 60 kWh | 17:00–07:00 | 72% ± 10 | No |
| 2 | Intelligent Octopus Average | 30% | 72.5 kWh | 17:00–07:00 | 52% ± 25 | Yes |
| 3 | Infrequent Charging | 10% | 60 kWh | 17:00–07:00 | 19% ± 10 | No |
| 4 | Infrequent Driving | 10% | 60 kWh | 17:00–07:00 | 77% ± 10 | No |
| 5 | Scheduled Charging | 9% | 60 kWh | 17:00–09:00 | 72% ± 10 | Yes |
| 6 | Always Plugged-In | 1% | 60 kWh | continuous | 72% ± 10 | No |

All six draw a target state of charge from the report's preference mix (mean 84.3%, mode
80%) and charge on a 7 kW home chargepoint by default. Every value above, and every
distribution behind it, is editable in the app.

Two archetypes have their own cadence rather than one plug-in per window.
**Infrequent Charging** draws its gap between events from a triangular distribution
(3/5/8 days) and gets a random phase within that gap at build time, so roughly 19% of
those agents have an event due on the simulated window.
**Always Plugged-In** is connected for the whole window bar one trip of configurable
length, starting at random between 13:00 and 18:00 — an afternoon trip, so the car is back
with the rest of the window ahead of it to charge in.

### The defaults

The Intelligent Octopus Average archetype is grounded in *"Learning from Intelligent
Octopus"* (Centre for Net Zero / Octopus Energy, May 2022), which analysed telemetry
from 2,500+ customers on Octopus's automated smart-charging tariff: 72.5 kWh average
battery, 52% median plug-in SoC, 2.5 hour median charge duration, an 80%-by-07:00
target, and the fixed 23:30–05:30 cheap window at 7.5p/kWh. Its Figure 4 (evening
plug-ins, morning plug-outs) sets the timing distribution shape used as a starting
default for all six archetypes, re-centred on each one's own window.

The other five archetypes are a hybrid; they take their structural values — population 
share, mileage and mean plug-in SOC from the project's task brief, and their 
distribution shapes from the CNZ report, re-centred on each one's own plug-in window.


### Aggregation

The two headline series are different kinds of quantity, and the code keeps them
distinct:

- **% plugged in** is already a population aggregate at each instant, so it carries no
  uncertainty band. Plotted as black bars on the left axis.
- **State of Charge** distribution across agents, showing the mean, 5th and 95th
  percentiles. Plotted on the right axis and defaults to currently-connected agents, 
  which is the home-charging view. It can be switched to the whole fleet, which 
  includes agents that have unplugged and driven away.

Supporting views: archetype breakdown, aggregate charging power, an agent-level
explorer, and the validation panel.

### Validation

`model/validate.py` compares the simulated IO Average archetype against the report's
figures — median charge duration within ±20% relative, median plug-in SoC within ±5
percentage points. It is a sanity check displayed as a table; nothing else in the model
or app depends on its outcome. On the defaults it lands at 2h51 against 2h30, and 52.5%
against 52%.

The simulated interquartile range is narrower than the report's (2h01–3h48 against
0h40–4h40), because plug-in SoC is drawn from a single normal distribution while real
behaviour is more dispersed.

## Out of scope in this version

- No half-hour fragmentation of charge sessions into price-optimised slots — one
  contiguous block per session.
- Plug-in SoC is sampled directly rather than accumulated from a journey model; the
  driving model runs forward from plug-out only.
- Whether an agent drives on a given day is stochastic, but trip length is not — no
  mileage variance between agents of the same type, or between that agent's trips.
- No weekday/weekend behavioural variation.
- No bump-charge or manual-override modelling.
