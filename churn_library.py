'''
A module to find customers who car likely to chrun

Author: HanByul Yang
Date: Aug 7, 2022
'''

# import libraries
import os
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    eda_img_path = r'./images/eda/'

    plt.figure(figsize=(20, 10))
    churn_hist = df['Churn'].hist()
    churn_hist.get_figure().savefig(os.path.join(eda_img_path, 'Churn.png'))

    plt.figure(figsize=(20, 10))
    cust_age_hist = df['Customer_Age'].hist()
    cust_age_hist.get_figure().savefig(os.path.join(eda_img_path, 'Customer_Age.png'))

    plt.figure(figsize=(20, 10))
    martail_stat = df.Marital_Status.value_counts('normalize').plot(kind='bar')
    martail_stat.get_figure().savefig(os.path.join(eda_img_path, 'Marital_Status.png'))

    plt.figure(figsize=(20, 10))
    total_trans_plot = sns.histplot(
        df['Total_Trans_Ct'], stat='density', kde=True)
    fig = total_trans_plot.get_figure()
    fig.savefig(os.path.join(eda_img_path, 'Total_Trans_Ct.png'))

    plt.figure(figsize=(20, 10))
    dark2_r_plot = sns.heatmap(
        df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    dark2_r_plot.get_figure().savefig(os.path.join(eda_img_path, 'Dark2_r.png'))


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                      used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        cat_lst = []
        cat_groups = df.groupby(category).mean()[response]

        for val in df[category]:
            cat_lst.append(cat_groups.loc[val])

        response_name = category + '_' + response
        df[response_name] = cat_lst

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
                        used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    results_img_path = './images/results/'

    plt.figure()
    # plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(os.path.join(results_img_path, 'random_forest.png'))

    plt.figure()
    # plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(os.path.join(results_img_path, 'logistic_regression.png'))


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(os.path.join(output_pth, 'feature_importances.png'))


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    results_img_path = './images/results/'
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        results_img_path)

    # roc curve plot
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join(results_img_path, 'roc_curves.png'))


if __name__ == "__main__":
    DATA_PATH = r"./data/bank_data.csv"
    df_bank = import_data(DATA_PATH)

    CHURN_COL = 'Churn'

    df_bank[CHURN_COL] = df_bank['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(df_bank)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # quant_columns = [
    #     'Customer_Age',
    #     'Dependent_count',
    #     'Months_on_book',
    #     'Total_Relationship_Count',
    #     'Months_Inactive_12_mon',
    #     'Contacts_Count_12_mon',
    #     'Credit_Limit',
    #     'Total_Revolving_Bal',
    #     'Avg_Open_To_Buy',
    #     'Total_Amt_Chng_Q4_Q1',
    #     'Total_Trans_Amt',
    #     'Total_Trans_Ct',
    #     'Total_Ct_Chng_Q4_Q1',
    #     'Avg_Utilization_Ratio'
    # ]

    df_encoded = encoder_helper(df_bank, cat_columns, CHURN_COL)

    # cat_encoded_colnums = [x + '_' + CHURN_COL for x in cat_columns]

    # keep_cols = quant_columns + cat_encoded_colnums

    # for c in keep_cols:
    #     print(c)

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_encoded, CHURN_COL)

    train_models(X_train, X_test, y_train, y_test)

    # rfc_model = joblib.load('./models/rfc_model.pkl')
    # lr_model = joblib.load('./models/logistic_model.pkl')

    # y_train_preds_rf = rfc_model.predict(X_train)
    # y_test_preds_rf = rfc_model.predict(X_test)

    # y_train_preds_lr = lr_model.predict(X_train)
    # y_test_preds_lr = lr_model.predict(X_test)

    # classification_report_image(y_train,
    #                             y_test,
    #                             y_train_preds_lr,
    #                             y_train_preds_rf,
    #                             y_test_preds_lr,
    #                             y_test_preds_rf)

    # feature_importance_plot(rfc_model, X_train, r'./images/results/')
