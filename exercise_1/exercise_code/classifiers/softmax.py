"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_examples = X.shape[0]
    num_classes = W.shape[1]
    
    for ex in range(num_examples):
        score = X[ex].dot(W)
        score_stable = score - np.max(score)
        score_sum = np.sum(np.exp(score_stable))
        
        softmax = lambda y_i: np.exp(score_stable[y_i])/score_sum
        loss += -np.log(softmax(y[ex]))
        
        for c in range(num_classes):
            softmax_class = softmax(c)
            dW[:,c] = (softmax_class -(c == y[ex])) * X[ex]
    
    loss /= num_examples
    loss += 0.5 * reg * np.sum(W*W)
    
    dW /= num_examples
    dW += reg*W
    
    return loss, dW
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_examples = X.shape[0]
    
    scores = X.dot(W)
    scores_stable = scores - np.max(scores, axis = 1, keepdims = True)
    scores_sum = np.sum(np.exp(scores_stable), axis = 1, keepdims = True)
    
    softmax = np.exp(scores_stable)/scores_sum
    correct_softmax = softmax[range(num_examples),y]
    
    loss = np.sum(-np.log(correct_softmax))
    loss /= num_examples
    loss += 0.5 * reg * np.sum(W*W)
    
    softmax[range(num_examples),y] -= 1
    dW = X.T.dot(softmax) / num_examples
    dW += reg*W
    
    return loss, dW
    
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7, 1e-8]
    regularization_strengths = [1e3, 25e3, 5e3 , 1e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    num_iters = 1500
    for learning_rate in learning_rates:
        for reg in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate, reg, num_iters)
            
            pred_train = softmax.predict(X_train)
            pred_val = softmax.predict(X_val)
            
            train_accuracy = np.mean(pred_train == y_train)
            validation_accuracy = np.mean(pred_val == y_val)
            
            all_classifiers.append(softmax)
            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_softmax = softmax
            
            results[(learning_rate, reg)] = train_accuracy, validation_accuracy
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
