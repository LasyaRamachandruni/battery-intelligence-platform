-- Capacity retention every 10 cycles per cell: the trajectories behind the fade plots.
select cell_id, batch, cycle, discharge_capacity_ah, capacity_retention
from {{ ref('stg_cycles') }}
where cycle % 10 = 0 or cycle = 2
