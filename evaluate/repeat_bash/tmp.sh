top_k=6
for topk in $(seq 1 $top_k)
do
    echo "evaluating topk $topk ↓"
    bash evaluate/repeat_bash/multi_evaluate_parent_3part_topk.sh $topk
done