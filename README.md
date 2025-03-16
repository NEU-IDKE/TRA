# TRA
## START
```
cd TRA
pip install -r requirement.txt
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train  --do_test --data_path data/GST --model RotatE -alpha 0.01 -n 256 -b 256 -d 1000 -g 24.0 -a 1.0 -adv -lr 0.0001 --max_steps 30000 -save models/GST--test_batch_size 1 --cuda -de
```
## topological relation calculated process
the code and results are in dictionary `TRA/init_topological_relation`.