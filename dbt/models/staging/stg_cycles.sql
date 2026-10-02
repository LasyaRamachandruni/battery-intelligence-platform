-- One row per valid cell x cycle, with capacity retention relative to cycle 2.
-- Cycle 1 is a formation-like cycle in this dataset, so the paper uses cycle 2 as the reference.
with cycles as (
    select * from {{ source('lake', 'cycle_summary') }}
),

reference as (
    select cell_id, arg_min(qd, cycle) as q_ref
    from cycles
    where cycle >= 2
    group by cell_id
)

select
    c.cell_id,
    c.batch,
    c.cycle,
    c.qd               as discharge_capacity_ah,
    c.qc               as charge_capacity_ah,
    c.ir               as internal_resistance_ohm,
    c.tavg             as temp_avg_c,
    c.tmin             as temp_min_c,
    c.tmax             as temp_max_c,
    c.chargetime       as charge_time_min,
    c.qd / r.q_ref     as capacity_retention,
    c.qd / nullif(c.qc, 0) as coulombic_efficiency
from cycles c
join reference r using (cell_id)
