export WANDB_MODE=""
export UD_ROOT="/home/nlpmaster/ssd-1t/corpus/ud/ud-v2.2-preped/"
while IFS=" " read -a words; do
  FT_LANG="${words[0]}";
  for i in 1 0 2 3 4 5 6 7 8 9 10; do
    NUM_EPOCHS=40;
    if ! ls ${UD_ROOT}${FT_LANG}*-ud-train.conllu 1> /dev/null 2>&1;
    then
      # NUM_EPOCHS=1;
      echo "no train; skipping"
      # bash test.sh meta  $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
      # bash test.sh multi $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
    else
      for method in meta-fixed-big multi-fixed-big;
      do
        if [[ ! ( -s "ckpts/${method}_${i}_${FT_LANG}_$2/result.txt" ) && -f "ckpts/${method}/model_state_epoch_${i}.th" ]];
        then
          rm -r ckpts/${method}_${i}_${FT_LANG}_$2;
          bash fine-tune.sh $method $i $FT_LANG ${NUM_EPOCHS} $2 || exit 1;
        else
          echo "no ckpts/${method}/model_state_epoch_${i}.th, or already runned; continue"
        fi
      done
    fi
  done
done < $1
