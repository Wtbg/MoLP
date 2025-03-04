all_topk=6
for topk in $(seq 1 $all_topk)
do
    echo "cal avg for topk $topk ↓"
    python evaluate/cal_avg_for_all.py --result_dir /sda/kongming/3d-cake/script/MoLP/evaluate/v4_repeat_3part_0.1_3expert
done