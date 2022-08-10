'''
Testing & Logging churn_library

Author: HanByul Yang
Date: Aug 10, 2022
'''
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        cls.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err

    try:
        eda_img_path = r'./images/eda/'

        assert os.path.isfile(os.path.join(eda_img_path, 'Churn.png'))
        assert os.path.isfile(os.path.join(eda_img_path, 'Customer_Age.png'))
        assert os.path.isfile(os.path.join(eda_img_path, 'Marital_Status.png'))
        assert os.path.isfile(os.path.join(eda_img_path, 'Total_Trans_Ct.png'))
        assert os.path.isfile(os.path.join(eda_img_path, 'Dark2_r.png'))
    except AssertionError as err:
        logging.error("Testing perform_eda: some figures are not found")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        df_bank = cls.import_data("./data/bank_data.csv")

        CHURN_COL = 'Churn'

        df_bank[CHURN_COL] = df_bank['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df_encoded = cls.encoder_helper(df_bank, cat_columns, CHURN_COL)
        logging.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: The file wasn't found")
        raise err

    try:
        cat_encoded_colnums = [x + '_' + CHURN_COL for x in cat_columns]
        for encoded_col in cat_encoded_colnums:
            assert encoded_col in df_encoded
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: some encoded column aren't found")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        df_bank = cls.import_data("./data/bank_data.csv")

        CHURN_COL = 'Churn'

        df_bank[CHURN_COL] = df_bank['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df_encoded = cls.encoder_helper(df_bank, cat_columns, CHURN_COL)

        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df_encoded, CHURN_COL)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_feature_engineering: The file wasn't found")
        raise err

    try:
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: some features are not created")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        df_bank = cls.import_data("./data/bank_data.csv")

        CHURN_COL = 'Churn'

        df_bank[CHURN_COL] = df_bank['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df_encoded = cls.encoder_helper(df_bank, cat_columns, CHURN_COL)

        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df_encoded, CHURN_COL)
        cls.train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The file wasn't found")
        raise err

    try:
        img_path = './images/results/'

        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile(os.path.join(img_path, 'random_forest.png'))
        assert os.path.isfile(
            os.path.join(
                img_path,
                'logistic_regression.png'))
        assert os.path.isfile(
            os.path.join(
                img_path,
                'feature_importances.png'))
        assert os.path.isfile(os.path.join(img_path, 'roc_curves.png'))
    except AssertionError as err:
        logging.error(
            "Testing train_models: train models doesn't have expected result files")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
