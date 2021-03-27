
# TimeseriesGenerator
# dataX = []
    # dataY = []
    # for i in range(len(sequences)):
    #     if len(sequences[i]) >= 3:
    #         generator = TimeseriesGenerator(sequences[i], sequences[i], length=2, batch_size=1)
    #         for j in range(len(generator)):
    #             x, y = generator[i]
    #             dataX.append(x)
    #             dataY.append(y)

#best_model = None
#min_loss = 1000
#predictions = []
#predictionY = []
#for i in range(len(dataX)):
    #if len(dataX[i]) > 4:
        #lstm = Lstm(nb_features, 1, dataX, dataY)
        #lstm.train()

        #loss, prediction = lstm.evaluate_model()
        # if loss < min_loss:
        #     best_model = lstm
        #     min_loss = loss
        # if prediction is not None:
        #     predictions.append(prediction[0])
        #     predictionY.append(dataY[i][len(dataY[i])-1])

# best_model.model.save('../models/lstm.h5')
# print("----------Best model----------")
# best_loss, best_output = best_model.evaluate_model()

# predictions = np.array(predictions)
# predictionY = np.array(predictionY)
# clf = linear_model.LinearRegression(fit_intercept=False)
# clf.fit(predictions, predictionY)
# print("----------Linear Regression----------")
# print("prediction: ", clf.predict(predictions))
# print("actual: ", predictionY)
# print(clf.score(predictions, predictionY))