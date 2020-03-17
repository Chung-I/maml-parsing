export WANDB_MODE=""
export UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.2-preped/"
while IFS=" " read -a words; do
  FT_LANG="${words[0]}";
  for i in 0 1; do
    NUM_EPOCHS=40;
    if [[ ! -f ${UD_ROOT}${FT_LANG}-ud-train.conllu ]];
    then
      # NUM_EPOCHS=1;
      echo "no train; skipping"
      # bash test.sh meta  $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
      # bash test.sh multi $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
    else
      for method in meta-fomaml multi;
      do
        if [[ ! -f "ckpts/${method}_${i}_${FT_LANG}_$2/best.th" ]];
        then
          rm -r ckpts/${method}_${i}_${FT_LANG}_$2;
          bash fine-tune.sh $method $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
        fi
      done
    fi
  done
done < $1
