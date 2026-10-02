select
    cell_id,
    batch,
    charge_policy,
    cycle_life,
    cycles_recorded,
    valid_cycles,
    has_early_window,
    source
from {{ source('lake', 'cells') }}
