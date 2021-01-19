import os
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from torch.utils import data
from collections import namedtuple

def load_dataset(path, s_label):
    data = pd.read_csv(path)
    # Preprocessing taken from https://www.kaggle.com/islomjon/income-prediction-with-ensembles-of-decision-trees
    
    # replace missing values with majority class
    data['workclass'] = data['workclass'].replace('?','Private')
    data['occupation'] = data['occupation'].replace('?','Prof-specialty')
    data['native-country'] = data['native-country'].replace('?','United-States')

    # education category
    data.education = data.education.replace(['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th'],'left')
    data.education = data.education.replace('HS-grad','school')
    data.education = data.education.replace(['Assoc-voc','Assoc-acdm','Prof-school','Some-college'],'higher')
    data.education = data.education.replace('Bachelors','undergrad')
    data.education = data.education.replace('Masters','grad')
    data.education = data.education.replace('Doctorate','doc')

    # marital status
    data['marital-status'] = data['marital-status'].replace(['Married-civ-spouse','Married-AF-spouse'],'married')
    data['marital-status'] = data['marital-status'].replace(['Never-married','Divorced','Separated','Widowed', 'Married-spouse-absent'], 'not-married')

    # income
    data.income = data.income.replace('<=50K', 0)
    data.income = data.income.replace('>50K', 1)

    # sex
    data.gender = data.gender.replace('Male', 0)
    data.gender = data.gender.replace('Female', 1)

    # encode categorical values
    data1 = data.copy()
    data1 = pd.get_dummies(data1)
    data1 = data1.drop(['income', s_label], axis=1)

    X = StandardScaler().fit(data1).transform(data1)
    y = data['income'].values
    s = data[s_label].values

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size=0.3, random_state=1)

    X_train = torch.Tensor(X_train).float()
    X_test = torch.Tensor(X_test).float()
    y_train = torch.Tensor(y_train).int()
    y_test = torch.Tensor(y_test).int()
    s_train = torch.Tensor(s_train).int()
    s_test = torch.Tensor(s_test).int()

    return X_train, X_test, y_train, y_test, s_train, s_test


class ADULT(data.Dataset):


    def __init__(self, root="data/adult", split="train", sensible_attribute="gender"):
        assert split in ["train", "test"]
        path = os.path.join(root, "adult.csv")
        X_train, X_test, y_train, y_test, s_train, s_test = load_dataset(path, sensible_attribute)
        if split == 'train':
            self.X = X_train
            self.y = y_train
            self.s = s_train
        else:
            self.X = X_test
            self.y = y_test
            self.s = s_test
        print("loaded {} instances for split {}. y positives={}, {} positives={}".format(
            len(self.y), split, sum(self.y), sensible_attribute, sum(self.s)))


    def __len__(self):
        """__len__"""
        return len(self.X)
    

    def __getitem__(self, index):
        return dict(data=self.X[index], labels=self.y[index], sensible_attribute=self.s[index])


    def task_names(self):
        return None


    
if __name__ == "__main__":
    dataset = ADULT(split="train")
    trainloader = data.DataLoader(dataset, batch_size=256, num_workers=0)

    for i, data in enumerate(trainloader):
        break