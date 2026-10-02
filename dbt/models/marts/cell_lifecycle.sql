-- One row per cell: what the first 100 cycles looked like, and how long it lasted.
with early as (
    select
        cell_id,
        max(case when cycle = 2 then discharge_capacity_ah end)       as q_cycle2_ah,
        arg_max(discharge_capacity_ah, cycle)                          as q_early_last_ah,
        max(discharge_capacity_ah) - min(discharge_capacity_ah)        as q_early_range_ah,
        avg(case when cycle <= 6 then charge_time_min end)             as charge_time_first5_min,
        arg_min(internal_resistance_ohm, cycle)                        as ir_early_first_ohm,
        arg_max(internal_resistance_ohm, cycle)                        as ir_early_last_ohm,
        max(temp_max_c)                                                as temp_max_early_c,
        regr_slope(discharge_capacity_ah, cycle)                       as fade_slope_ah_per_cycle
    from {{ ref('stg_cycles') }}
    where cycle between 2 and {{ var('early_cycles') }}
    group by cell_id
),

milestones as (
    select
        cell_id,
        min(case when capacity_retention < 0.95 then cycle end)       as cycles_to_95pct,
        min(case when capacity_retention < 0.90 then cycle end)       as cycles_to_90pct,
        max(cycle)                                                     as last_cycle,
        arg_max(discharge_capacity_ah, cycle)                          as q_final_ah
    from {{ ref('stg_cycles') }}
    group by cell_id
)

select
    c.cell_id,
    c.batch,
    c.charge_policy,
    c.cycle_life,
    c.has_early_window,
    e.q_cycle2_ah,
    e.q_early_last_ah,
    e.q_early_range_ah,
    e.fade_slope_ah_per_cycle,
    e.charge_time_first5_min,
    e.ir_early_first_ohm,
    e.ir_early_last_ohm - e.ir_early_first_ohm as ir_early_change_ohm,
    e.temp_max_early_c,
    m.cycles_to_95pct,
    m.cycles_to_90pct,
    m.last_cycle,
    m.q_final_ah,
    c.source
from {{ ref('stg_cells') }} c
left join early e using (cell_id)
left join milestones m using (cell_id)
