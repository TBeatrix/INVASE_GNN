from INVASE_MLP.data_genration_For_MLP import generate_dataset

from invase_pytorch import *
"""# **Main**
(1) Data generation

(2) Train INVASE or INVASE-

(3) Evaluate INVASE on ground truth feature importance and prediction
"""

if __name__ == '__main__':
    #Paraméterel megadása a modellnek
    class args:
      data_type: str
      train_no: int
      test_no: int
      dim : int
      model_type : str
      actor_h_dim : int
      critic_h_dim :int
      n_layer: int
      batch_size : int
      iteration : int
      activation : str
      learning_rate : float
      lamda : float

    # Inputs for the main function
    args.data_type = 'syn1'
    args.train_no = 10000
    args.test_no = 10000
    args.dim = 11
    args.model_type = 'invase'
    args.actor_h_dim = 100
    args.critic_h_dim = 200
    args.n_layer = 3
    args.batch_size = 1000
    args.iteration = 10000
    args.activation = 'relu'
    args.learning_rate = 0.0001
    args.lamda = 0.1

    # Generate dataset
    x_train, y_train, g_train = generate_dataset (n = args.train_no,
                                                  dim = args.dim,
                                                  data_type = args.data_type,
                                                  seed = 0)

    x_test, y_test, g_test = generate_dataset (n = args.test_no,
                                                dim = args.dim,
                                                data_type = args.data_type,
                                                seed = 0)

    model_parameters = {'lamda': args.lamda,
                        'actor_h_dim': args.actor_h_dim,
                        'critic_h_dim': args.critic_h_dim,
                        'n_layer': args.n_layer,
                        'batch_size': args.batch_size,
                        'iteration': args.iteration,
                        'activation': args.activation,
                        'learning_rate': args.learning_rate}

    # Train the model

    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    model = Invase(x_train_tensor, y_train_tensor, args.model_type, model_parameters)

    model.train(x_train_tensor, y_train_tensor)

    ## Evaluation
    # Compute importance score
    x_test_tensor = torch.from_numpy(x_test).float()

    g_hat = model.importance_score(x_test_tensor)
    importance_score = 1.*(g_hat > 0.5)

    # Evaluate the performance of feature importance
    mean_tpr, std_tpr, mean_fdr, std_fdr = \
    feature_performance_metric(g_test, importance_score)

    # Print the performance of feature importance
    print('TPR mean: ' + str(np.round(mean_tpr,1)) + '\%, ' + \
          'TPR std: ' + str(np.round(std_tpr,1)) + '\%, ')
    print('FDR mean: ' + str(np.round(mean_fdr,1)) + '\%, ' + \
          'FDR std: ' + str(np.round(std_fdr,1)) + '\%, ')

    # Predict labels
    y_hat = model.predict(x_test_tensor)

    # Evaluate the performance of feature importance
    auc, apr, acc = prediction_performance_metric(y_test, y_hat)

    # Print the performance of feature importance
    print('AUC: ' + str(np.round(auc, 3)) + \
          ', APR: ' + str(np.round(apr, 3)) + \
          ', ACC: ' + str(np.round(acc, 3)))

    performance = {'mean_tpr': mean_tpr, 'std_tpr': std_tpr,
                    'mean_fdr': mean_fdr, 'std_fdr': std_fdr,
                    'auc': auc, 'apr': apr, 'acc': acc}
    print(performance)