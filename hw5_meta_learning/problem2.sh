python train_policy.py 'pm' --exp_name history_10 --history 10 --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name history_10_rnn --history 10 --discount 0.90 -lr 5e-4 -n 60 --recurrent
python train_policy.py 'pm' --exp_name history_60 --history 60 --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name history_100 --history 100 --discount 0.90 -lr 5e-4 -n 60
python train_policy.py 'pm' --exp_name history_100_rnn --history 100 --discount 0.90 -lr 5e-4 -n 60 --recurrent