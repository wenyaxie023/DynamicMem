for id in 006 007 008 009 010; do
  nohup python test_membench.py --size large --sample-id ${id}_user_${id} > memos_user_${id}_predictions.txt 2>&1 &
done