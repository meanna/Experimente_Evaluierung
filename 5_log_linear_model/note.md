# optimal hyperparameters

learning_rate=0.05, mu=0.01, num_epoch=7,8,9, and higher

## to train
model.train(args.train_dir, args.paramfile, learning_rate=0.05, mu=0.01, num_epoch=10, dev_dir=args.dev_dir)

## to run test on the best model
python3 test.py paramfile_lr0.05_mu0.01_epoch7_dev_acc0.98 ../data/test

- early stopping does not work with L2. It stops too early.