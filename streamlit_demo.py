import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def main():
    st.title("Dataset Classifier with Streamlit Example")

    st.write("""
    ### Explore different classifiers
    Which one is the best?\n
    This one
            """)

    st.sidebar.write("#\n#\n#\n#")  #add spacing in sidebar before dropdowns and sliders
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

    X,y = get_dataset(dataset_name)

    st.write("#")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape of dataset", X.shape)
    with col2:
        st.write("number of classes", len(np.unique(y)))

    params = add_parameter_ui(classifier_name)
    clf = get_classifier(classifier_name, params)

    xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=1234) 
    clf.fit(xtrain, ytrain)

    yhat = clf.predict(xtest)
    accuracy = accuracy_score(ytest, yhat)

    st.write("#")

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Classifier', str(clf).split('(')[0])
    with col2:
        st.metric('Accuracy', f'{accuracy:0.1%}')

    #PLOT using PCA
    pca = PCA(n_components=2)
    x_projected = pca.fit_transform(X)

    x1 = x_projected[:, 0]    
    x2 = x_projected[:, 1]    

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
    plt.xlabel("Principal Component 1")
    plt.xlabel("Principal Component 2")
    plt.colorbar()
    plt.tight_layout()

    st.pyplot(fig)



def get_dataset(dataset_name: str) -> tuple:
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider('c', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)

        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name =="SVM":
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                                     n_jobs=-1, random_state=1234)
    return clf


if __name__ == '__main__':
    main()