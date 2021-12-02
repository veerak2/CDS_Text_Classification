def evaluate(model,test_data):

    accuracy, loss = model.evaluate(test_data)

    return accuracy,loss
