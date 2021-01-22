data_dir=$1
out_dir=$2
question=$3
part=$4
if [[ ${question} == "1" ]]; then
python q1.py $data_dir $out_dir $part
fi
if [[ ${question} == "2" ]]; then
python q2.py $data_dir $out_dir $part
fi
if [[ ${question} == "3" ]]; then
python q3.py $data_dir $out_dir $part
fi
if [[ ${question} == "4" ]]; then
python q4.py $data_dir $out_dir $part
fi