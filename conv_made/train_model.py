# -*- coding: utf-8 -*-

import numpy as np

from model import ConvMADE
from utils import load_binarized_mnist
from utils import Timer

from smartlearner import Trainer
from smartlearner import tasks
from smartlearner import views
from smartlearner import stopping_criteria

from smartlearner.optimizers import SGD, AdaGrad
from smartlearner.direction_modifiers import ConstantLearningRate
from smartlearner.batch_schedulers import FullBatchScheduler, MiniBatchScheduler
from smartlearner.losses.distribution_losses import BinaryCrossEntropy as NLL


def train_model():
    with Timer("Loading dataset"):
        trainset, validset, testset = load_binarized_mnist()
        # The target for distribution estimator is the input
        trainset._targets_shared = trainset.inputs
        validset._targets_shared = validset.inputs
        testset._targets_shared = testset.inputs
        trainset.symb_targets = trainset.symb_inputs
        validset.symb_targets = validset.symb_inputs
        testset.symb_targets = testset.symb_inputs

    with Timer("Creating model"):
        #hidden_size = 50
        image_shape = (28, 28)
        filter_shape = (2, 2)
        nb_filters = 1
        model = ConvMADE(image_shape, filter_shape, nb_filters)
        model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        loss = NLL(model, trainset)
        #optimizer = SGD(loss=loss)
        #optimizer.append_direction_modifier(ConstantLearningRate(0.001))
        optimizer = AdaGrad(loss=loss, lr=0.1)

    with Timer("Building trainer"):
        # Train for 10 epochs
        # Train using mini batches of 100 examples
        batch_scheduler = MiniBatchScheduler(trainset, 100)

        trainer = Trainer(optimizer, batch_scheduler)
        trainer.append_task(stopping_criteria.MaxEpochStopping(100))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Create a callback that will shuffle the mask every 100 epochs.
        filter_size = int(np.prod(model.filter_shape))
        mask = np.arange(filter_size).reshape(model.filter_shape)
        rng = np.random.RandomState(42)

        def shuffle_mask(task, status):
            o = rng.randint(filter_size)
            model.mask.set_value((mask <= o).astype(model.mask.dtype))

        trainer.append_task(tasks.Callback(shuffle_mask, each_k_update=100))

        # Log training error
        loss_monitor = views.MonitorVariable(loss.loss)
        avg_loss = tasks.AveragePerEpoch(loss_monitor)
        trainer.append_task(avg_loss)
        trainer.append_task(tasks.Print("Trainset - Avg. loss: {0:.1f}", avg_loss[0]))

        # Print NLL mean/stderror.
        nll = views.LossView(loss=NLL(model, validset), batch_scheduler=FullBatchScheduler(validset))
        trainer.append_task(tasks.Print("Validset - NLL          : {0:.1f} ± {1:.1f}",
                                        nll.mean, nll.stderror))

        trainer.append_task(stopping_criteria.EarlyStopping(nll, lookahead=10))

    with Timer("Building Theano graph"):
        trainer.build_theano_graph()

    with Timer("Training"):
        trainer.train()


if __name__ == "__main__":
    train_model()
