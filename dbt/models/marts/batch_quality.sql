-- Per production/test batch: spread of early-life metrics and of cycle life.
select
    batch,
    count(*)                                         as cells,
    round(avg(q_cycle2_ah), 5)                       as q_cycle2_mean_ah,
    round(stddev_samp(q_cycle2_ah), 5)               as q_cycle2_sd_ah,
    round(avg(ir_early_first_ohm) * 1000, 4)         as ir_mean_mohm,
    round(stddev_samp(ir_early_first_ohm) * 1000, 4) as ir_sd_mohm,
    round(avg(charge_time_first5_min), 3)            as charge_time_mean_min,
    quantile_cont(cycle_life, 0.1)                   as cycle_life_p10,
    quantile_cont(cycle_life, 0.5)                   as cycle_life_p50,
    quantile_cont(cycle_life, 0.9)                   as cycle_life_p90
from {{ ref('cell_lifecycle') }}
group by batch
