import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


class LinearRegression():

    def __init__(self, isBias=False, isReg=False, regLambda=1.0):
        self.isBias = isBias
        self.isReg = isReg
        self.regLambda = regLambda
        self.weights = []

    def fit(self, predictors, targets):
        rows, _ = predictors.shape
        if self.isBias:
            aug = np.c_[predictors.values, np.ones([rows, 1])]
        else:
            aug = np.c_[predictors.values]

        X = np.asmatrix(aug)
        y = np.asmatrix(targets.values)

        if self.isReg:
            tmp = (X.T * X) + rows / 2 * self.regLambda * np.eye(X.shape[1])
            weights = np.linalg.inv(tmp) * X.T * y
        else:
            weights = np.linalg.pinv(X) * y

        self.weights = weights

    def RMSE(self, predictors, targets):
        rows, _ = predictors.shape
        if self.isBias:
            X = np.asmatrix(np.c_[predictors.values, np.ones([rows, 1])])
            y_predict = X * self.weights
        else:
            X = np.asmatrix(predictors.values)
            y_predict = X * self.weights

        y = np.asmatrix(targets.values)
        sqare_error = np.square(y_predict - y)

        return np.sqrt(sum(sqare_error) / rows)

    def predict(self, X):
        if self.isBias:
            X = np.asmatrix(np.c_[X.values, np.ones([X.shape[0], 1])])
        else:
            X = np.asmatrix(X.values)
        return X * self.weights


class BayesianLinearRegression():

    def __init__(self, alpha=1.0):
        self.isBias = True
        self.alpha = alpha
        self.weights = []

    def fit(self, predictors, targets):
        rows, cols = predictors.shape
        X = np.asmatrix(np.c_[predictors.values, np.ones([rows, 1])])
        y = np.asmatrix(targets.values)

        gamma_0_inv = np.linalg.inv(1 / self.alpha * np.eye(cols + 1))
        gamma_m = np.linalg.pinv(X.T * X + gamma_0_inv)

        mu_0 = np.asmatrix(0.0 * np.ones([cols + 1, 1]))
        mu_m = gamma_m * (X.T * y + gamma_0_inv * mu_0)

        self.weights = mu_m

    def RMSE(self, predictors, targets):
        rows, _ = predictors.shape
        if self.isBias:
            X = np.asmatrix(np.c_[predictors.values, np.ones([rows, 1])])
        y_predict = X * self.weights

        y = np.asmatrix(targets.values)
        sqare_error = np.square(y_predict - y)
        ret = np.sqrt(sum(sqare_error) / rows)
        return ret

    def predict(self, X):
        if self.isBias:
            X = np.asmatrix(np.c_[X.values, np.ones([X.shape[0], 1])])
        else:
            X = np.asmatrix(X.values)
        return X * self.weights


class LogisticRegression():

    def __init__(self):
        self.weights = []
        self.maxIter = 2000
        self.stepSize = 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, predictors, targets):
        rows, cols = predictors.shape
        X = np.asmatrix(predictors.values)
        y = np.asmatrix(targets.values)
        weights = np.asmatrix(np.zeros([cols, 1]))
        prev_weights = weights
        residue = []

        for i in range(self.maxIter):
            grad = (1 / rows) * X.T * self.sigmoid(X * weights - y)
            weights = weights - self.stepSize * grad
            residue.append(np.linalg.norm(prev_weights - weights))
            prev_weights = weights

        # plt.plot(range(len(residue)), residue)
        self.weights = weights

    def predict(self, predictors):
        # rows, cols = predictors.shape
        X = np.asmatrix(predictors.values)

        return self.sigmoid(X * self.weights)


def evaluate(predcited, truth):
    """
    Parameters:

    Returns:
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(truth)):
        if truth[i] and predcited[i]:
            tp += 1
        elif truth[i] and (not predcited[i]):
            fn += 1
        elif (not truth[i]) and predcited[i]:
            fp += 1
        elif (not truth[i]) and (not predcited[i]):
            tn += 1

    confusion_matrix = np.matrix([[tn, fn], [fp, tp]])

    precision = tp / (tp + fp) if tp + fp != 0 else np.NaN
    accuracy = (tp+tn) / (tp+fn+tn+fp)
    return confusion_matrix, precision, accuracy


def get_entries(raw):
    """
    Parameters:

    Returns:
    """
    entries = ['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities',
               'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
               'Walc', 'health', 'absences']

    want = raw[entries]
    one_hot = pd.get_dummies(want)
    return one_hot


def train_test_split(raw, task='regression', random_state=42):
    """
    Parameters:

    Returns:
    """
    entries = ['school', 'sex', 'age', 'famsize', 'studytime', 'failures', 'activities',
               'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
               'Walc', 'health', 'absences', 'G3']

    want = raw[entries]

    if task == 'classification':
        want[['G3']] = want[['G3']].apply(
            lambda x: [1 if y >= 10 else 0 for y in x])

    one_hot = pd.get_dummies(want)

    train = one_hot.sample(frac=.8, random_state=random_state)
    y_train = train[['G3']]
    X_train = train.drop(columns=['G3'])
    X_train = (X_train - X_train.mean()) / X_train.std()

    test = one_hot.drop(train.index)
    y_test = test[['G3']]
    X_test = test.drop(columns=['G3'])
    X_test = (X_test - X_test.mean()) / X_test.std()

    return (X_train, X_test, y_train, y_test)


def quantize(raw, thres=0.5):
    """
    Parameters:

    Returns:
    """
    arr = np.asarray(raw).reshape(-1)
    for idx, val in enumerate(arr):
        arr[idx] = 1 if val >= thres else 0
    return arr


def plot_weights(weights=[]):
    """
    Parameters:

    Returns:
    """
    print(len(weights))
    for w in weights:
        plt.plot(range(len(w)), np.abs(w))
    plt.show()


def plot_G3(truth, predicted, n=300):
    """
    Parameters:

    Returns:
    """
    n = min(n, len(truth))
    x = range(n)
    plt.plot(x, truth.iloc[0:n], label='Ground Truth')
    for label, value in predicted:
        plt.plot(x, value[0:n], label=label)
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.legend(loc='lower right', prop={'size': 6})
    plt.grid()
    plt.show()


def plot_confusion_matrix(confusion, title):
    """
    Parameters:

    Returns:
    """
    cm_df = pd.DataFrame(confusion,
                         index=['predict = 0', 'predict = 1'],
                         columns=['true = 0', 'true = 1'])
    ax = plt.axes()
    sn.heatmap(cm_df, annot=True, fmt="d", ax=ax)
    ax.set_title(title)
    plt.show()


def Problem1():
    """
    Parameters:

    Returns:
    """
    raw = pd.read_csv('train.csv')
    X_train, X_test, y_train, y_test = train_test_split(raw, task='regression')

    predicted_G3 = []
    w = []

    lr = LinearRegression(isBias=False)
    lr.fit(X_train, y_train)
    lr_train_score = float(lr.RMSE(X_train, y_train))
    lr_test_score = float(lr.RMSE(X_test, y_test))
    predicted_G3.append(
        ('({:.3f})Linear Regression'.format(lr_test_score), lr.predict(X_test)))
    # print("Training RMSE:{}".format(lr_train_score))
    # print("Testing RMSE :{}".format(lr_test_score))
    w.append(lr.weights)

    lr_reg = LinearRegression(isBias=False, isReg=True)
    lr_reg.fit(X_train, y_train)
    lr_reg_train_score = float(lr_reg.RMSE(X_train, y_train))
    lr_reg_test_score = float(lr_reg.RMSE(X_test, y_test))
    predicted_G3.append(
        ('({:.3f})Linear Regression (/reg)'.format(lr_reg_test_score), lr_reg.predict(X_test)))
    # print("Training RMSE with reg:{}".format(lr_reg_train_score))
    # print("Testing RMSE with reg:{}".format(lr_reg_test_score))
    w.append(lr_reg.weights)

    """
    lr_b = LinearRegression(isBias=True)
    lr_b.fit(X_train, y_train)
    print("Training RMSE with bias:{}".format(float(lr_b.RMSE(X_train, y_train))))
    print("Testing RMSE with bias:{}".format(float(lr_b.RMSE(X_test, y_test))))
    predicted_G3.append( ('Linear Regression (/b)', lr_b.predict(X_test)) )
    # w.append(lr_b.weights)
    """

    lr_b_reg = LinearRegression(isBias=True, isReg=True)
    lr_b_reg.fit(X_train, y_train)
    lr_b_reg_train_score = float(lr_b_reg.RMSE(X_train, y_train))
    lr_b_reg_test_score = float(lr_b_reg.RMSE(X_test, y_test))
    predicted_G3.append(
        ('({:.3f}) Linear Regression (r/b)'.format(lr_b_reg_test_score), lr_b_reg.predict(X_test)))
    # print("Training RMSE with bias and reg:{}".format(lr_b_reg_train_score))
    # print("Testing RMSE with bias and reg:{}".format(lr_b_reg_test_score))
    w.append(lr_b_reg.weights)

    blr = BayesianLinearRegression()
    blr.fit(X_train, y_train)
    blr_train_score = float(blr.RMSE(X_train, y_train))
    blr_test_score = float(blr.RMSE(X_test, y_test))
    predicted_G3.append(('({:.3f}) Bayesian Linear Regression'.format(
        blr_test_score), blr.predict(X_test)))
    # print("Training RMSE with Bayesian LR:{}".format(blr_train_score))
    # print("Testing RMSE with Bayesian LR:{}".format(blr_test_score))
    # w.append(blr.weights)

    plot_G3(y_test, predicted_G3)
    # plot_weights(w)


def Problem2():
    raw = pd.read_csv('train.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        raw, task='classification')
    lr = LinearRegression(isBias=True, isReg=True)
    lr.fit(X_train, y_train)

    logiR = LogisticRegression()
    logiR.fit(X_train, y_train)

    threshold = [0.1, 0.5, 0.9]
    precision = np.zeros([2, 3])

    for idx, thres in enumerate(threshold):
        lr_predict = quantize(lr.predict(X_test), thres=thres)
        lr_cm, lr_precision, _ = evaluate(lr_predict, y_test.values)
        # plot_confusion_matrix(lr_cm, "Linear Regression with threshold {}".format(thres))
        precision[0][idx] = lr_precision

        logiR_predict = quantize(logiR.predict(X_test), thres=thres)
        logiR_cm, logiR_precison, _ = evaluate(
            logiR_predict, y_test.values)
        # plot_confusion_matrix(logiR_cm, "Logistic Regression with threshold {}".format(thres))
        precision[1][idx] = logiR_precison

    # print(precision)


def Problem3():
    to_be_predicted = get_entries(pd.read_csv('test_no_G3.csv'))
    raw = pd.read_csv('train.csv')

    X_train, X_test, y_train, y_test = train_test_split(raw, task='regression')
    lr = LinearRegression(isBias=True, isReg=True, regLambda=0.05)
    lr.fit(X_train, y_train)
    lr_train_score = float(lr.RMSE(X_train, y_train))
    lr_test_score = float(lr.RMSE(X_test, y_test))
    print("Training RMSE with bias and reg:{}".format(lr_train_score))
    print("Testing RMSE with bias and reg:{}".format(lr_test_score))
    predict = lr.predict(to_be_predicted)
    f = open('1.txt', 'w')
    for idx, value in enumerate(np.array(predict)):
        # print("{}\t{}".format(1001+idx, value[0]))
        f.write("{}\t{}\n".format(1001+idx, value[0]))

    # X_train, X_test, y_train, y_test = train_test_split(
    #     raw, task='classification')
    # logiR = LogisticRegression()
    # logiR.fit(X_train, y_train)
    # logiR_predict = quantize(logiR.predict(to_be_predicted), thres=0.45)

    # f = open('2.txt', 'w')
    # for idx, value in enumerate(logiR_predict):
    #     # print("{}\t{}".format(1001+idx, value))
    #     f.write("{}\t{}\n".format(1001+idx, value))

    """Test hyper-parameter
    # threshold = np.arange(0.2, 0.7, .01)
    # threshold = [0.1,0.5,0.9]
    # for idx, thres in enumerate(threshold):
    #     logiR_predict = quantize(logiR.predict(X_test), thres=thres)
    #     logiR_cm, logiR_precison, logiR_accuracy = evaluate(logiR_predict, y_test.values)
    #     # plot_confusion_matrix(logiR_cm, "Logistic Regression with threshold {}".format(thres))
    #     print("thres={:.3f}, prec={:.3f}, acc={:.3f}".format(thres, logiR_precison, logiR_accuracy))
    """

if __name__ == "__main__":
    # Problem1()
    # Problem2()
    Problem3()
    pass
