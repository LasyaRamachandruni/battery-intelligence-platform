-- A cell's recorded cycle life should be close to where its capacity actually
-- crossed the end-of-life threshold. Large disagreement means a labelling or
-- ingestion problem. Tolerance: 10% or 50 cycles.
with crossing as (
    select cell_id, min(cycle) as eol_cycle
    from {{ ref('stg_cycles') }}
    where discharge_capacity_ah < {{ var('end_of_life_ah') }}
    group by cell_id
)
select l.cell_id, l.cycle_life, c.eol_cycle
from {{ ref('cell_lifecycle') }} l
join crossing c using (cell_id)
where abs(l.cycle_life - c.eol_cycle) > greatest(50, 0.1 * l.cycle_life)
