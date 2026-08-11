"""Tab 1 -- approach, assumptions and the reasoning behind them.

Static prose, deliberately. This is the tab someone reads when they need to defend a result
to a third party, so it states what was assumed, what was built, what was left out and why,
without depending on the current scenario. The live per-archetype numbers belong on
Introduction (as configured) and Assumptions (editable).
"""

from __future__ import annotations

import streamlit as st

_STRUCTURAL = """
- One 24-hour window at 15-minute resolution, noon-anchored so every archetype's overnight
  cycle fits without wrap-around. No multi-day calendar, warm-up, or state between days.
- Plug-in state of charge is therefore an input, drawn per agent rather than accumulated
  from history; the driving model runs forward from plug-out only.
- Managed charging is allocated to a fixed cheap window (23:30–05:30) in one contiguous
  block, rather than optimised against day-ahead prices. The window is not anchored to the
  driver's ready-by time, since Intelligent Octopus schedules against wholesale prices.
- Constant charge rate, one home chargepoint. No taper, losses, public charging or V2G.
- Agents are independent: no interaction, network constraint, or feedback from price or
  grid. This is also what makes the app's distribution previews exact.
- One trip per window or none. Trip incidence is stochastic; trip length is fixed per
  archetype.
"""

_BEHAVIOURAL = """
- Unmanaged archetypes use IO customers' plug-in timing and consumption, re-centred on
  their own windows. IO customers are engaged early adopters on a proposition designed to
  increase plug-in frequency, so this likely makes the unmanaged cohorts look more
  consistent than they are — the assumption I would revisit first.
- All archetypes plug in daily. The report's Table 2 implies about 5.2 days a week, which
  takes peak occupancy from 91.8% to 69.0% and peak demand down 26%. Left at daily because
  the brief's mileage figures were themselves derived assuming daily plug-in, so changing
  one without the other breaks the energy balance. Exposed as a dial with that caveat.
- No weekday/weekend variation, so the brief's "long trips at weekends" archetype has no
  day of week to sit on. Infrequent Charging reproduces its day-averaged signature — a
  minority with large sessions, the rest absent — but those agents are desynchronised by
  design, so it cannot show a real Saturday peak. The archetype is expressible in
  configuration alone, since plug-in frequency gates event incidence:
  `driving_days_per_week = 2`, `plugin_days_per_week = 2`, mileage set for the trip length
  wanted.
- No bump charging or mid-day top-ups: the model assumes one charge block per agent per
  window. No change in preferences or tariff over time, which does follow from the
  single-window decision.
- Ready-by is a weighted choice among times rather than a continuous density, since a
  driver enters it.
"""

_SPENT = """
- **The data model.** One dataclass per archetype holding structural and distributional
  parameters, with everything derivable as a property rather than a field: charge duration,
  trip miles, plug-in frequency, driving probability, the connection window.
- **Scenario and assumption configurability, default assumptions per archetype.** Deciding 
  what to expose and what to state as fixed — e.g. battery capacity and charge rate are shown 
  but not editable. The brief gave point values, so the shapes and spreads around them were mine.
  Battery capacity and charge rate are common across archetypes so output differences come from 
  behaviour rather than hardware. Gaussian distributions are defined for behavioural attributes,
  with weighted distributions for ready-by and target state of charge with the brief's point 
  values as means.
- **The population-level views.** Choosing which aggregates answer the brief's second
  sketch: the combined plug-in and state-of-charge chart, archetype breakdown, aggregate
  charging demand, and per-agent demand distributions. 
- **Reconciling the report against the archetypes, which disagree.** Five parameters were
  recalibrated after reading the report. One result falls out unfitted: about 16% of IO
  charges exceed the 6-hour cheap window, against the report's separate claim that 85% of
  plug events could be shifted.
"""

_DEFERRED = """
- **A full week.** It would buy day-of-week variation and its synchronisation — a real
  Saturday peak, which desynchronised infrequent chargers cannot produce — state of charge
  carried from the previous day rather than sampled, and plug-in decisions following from
  how depleted a driver is rather than a set probability. Against that, the state-of-charge
  gain is partly illusory: the brief's 28,105 mi/yr is the report's ~22 kWh per plug event
  times an assumed daily plug-in, so deriving state of charge from mileage derives it from a
  figure back-derived from state of charge, and a genuine check needs an independent mileage
  source. It also turns event generation from a pure per-agent function into a sequential
  sweep with state, which would require more complicated logic.
- **Day-ahead price optimisation.** The behaviour dispatch gives the insertion point, so
  deferring costs one function.
- **Independent validation.** The comparison against the report is partly circular —
  simulated median plug-in state of charge lands at 51% because I set the input to 52%.
  Real validation needs observed aggregate demand for unmanaged cohorts, which I did not
  have. Outputs indicate direction and shape rather than calibrated levels.
"""

_DESIGN = """
The intended user changes an assumption, sees the effect on fleet load, then has to defend
the result to someone else. Five things follow.

**Individual simulation, and traceability back to it.** Each agent is simulated on its own
and population views are aggregations over those traces, so any fleet result decomposes into
the agents producing it. The agent explorer plots one agent's connection, charging and state
of charge for that purpose.

**Scenarios as data.** A scenario serialises to JSON, and the same file drives the app and
`run_headless.py`, so a scenario explored interactively can be rerun as a batch job or
reproduced exactly later.

**Assumptions visible, adjustable and guarded.** Every number is editable in one place, with
a preview of each archetype's sampling distribution, so a change's effect is
visible before running. Because everything is editable, the model can be driven into a
physically inconsistent state, so the closure table reports disagreement between mileage,
plug-in state of charge and plug-in frequency per archetype.

**Comparability between runs.** Behaviour and population construction draw from separate
random streams, so changing population size does not reshuffle unrelated agents' behavioural
draws. A difference between two scenarios is a result of changes in assumptions rather than 
resampling noise.
"""


def render() -> None:
    st.subheader("What assumptions I made")

    st.markdown("**Structural**")
    st.markdown(_STRUCTURAL)

    st.markdown("**Behavioural**")
    st.markdown(_BEHAVIOURAL)

    st.divider()
    st.subheader("How I designed the system bearing in mind its end use")
    st.markdown(_DESIGN)

    st.divider()
    st.subheader("Where I spent time, and what I deferred")

    st.markdown("**Spent**")
    st.markdown(_SPENT)

    st.markdown("**Deferred**")
    st.markdown(_DEFERRED)

    
