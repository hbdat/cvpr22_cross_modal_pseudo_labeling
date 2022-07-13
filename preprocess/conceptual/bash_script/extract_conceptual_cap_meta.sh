n_partition=40
echo $n_partition
for (( idx_partition=0; idx_partition<$n_partition; idx_partition++ ))
do
    python ./preprocess/extract_conceptual_cap_meta.py ${idx_partition} ${n_partition}&
done
wait