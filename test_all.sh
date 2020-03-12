export WANDB_MODE=""
export UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.2-preped/"
while IFS=" " read -a words; do
  FT_LANG="${words[0]}";
  for i in $(seq 0 2); do
    for method in meta multi;
    do
      ckpt_name=${method}_${i}_${FT_LANG}_$2
      if [[ ! -f ckpts/${ckpt_name}/best.th ]];
      then
        ln -s ckpts/${method}/best.th ckpts/${ckpt_name}/best.th
        echo "no train; skipping"
      else
        bash test.sh ${ckpt_name} ${FT_LANG} || exit 1;
      fi
    done
  done
done < $1
