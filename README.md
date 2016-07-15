
Meta-Learning with Memory Augmented Neural Networks
===================================================================================

A chainer implementation of *Meta-Learning with Memory Augmented Neural Networks*  
(This paper is also known as *One-shot Learning with Memory Augmented Neural Networks* )

- Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap, *Meta-Learning with Memory-Augmented Neural Networks*, [[link](http://jmlr.org/proceedings/papers/v48/santoro16.html)]
- Some code is taken from [tristandeleu's implementation](https://github.com/tristandeleu/ntm-one-shot) with Lasagne.



How to run 
--------------

1. Download the [Omniglot dataset](https://github.com/brendenlake/omniglot) and place it in the `data/` folder.
2. Run the scripts in `data/omniglot` to prepare dataset.
3. Run `scripts/train_omniglot.py` (Use gpu option if needed)



Summary of the paper
-------------------------

The authors attack the problem of one-shot learning by the approach of meta-learning.
They propose Memory Augmented Neural Network, which is a variant of Neural Turing Machine,
and train it to learn "how to memorize unseen characters."
After the training, the model can learn unseen characters in a few shot.











