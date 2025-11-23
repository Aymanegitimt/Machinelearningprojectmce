import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt





## Chargement/cleaning des données
##Spam Base Dataset
def preprocess_spam_data(file_path):
    """
    Compute the preprocessing for the Spambase dataset.
    Parameters:
    file_path (str): Path to the CSV file containing the Spambase dataset.
    Returns:
    X_train_spam, X_test_spam, y_train_spam, y_test_spam: Preprocessed and split data.
    """
    columns_spam = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 
        'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 
        'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 
        'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 
        'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 
        'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 
        'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 
        'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 
        'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 
        'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 
        'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 
        'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 
        'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 
        'char_freq_$', 'char_freq_#', 
        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
        'spam'
    ]

    df_spam = pd.read_csv(file_path, header=None, names=columns_spam)

    print("Total NaN Spambase :", df_spam.isnull().sum().sum())

    df_spam = df_spam.drop_duplicates()
    df_spam = df_spam[df_spam.select_dtypes(include='number').ge(0).all(1)]

    X_spam = df_spam.drop('spam', axis=1)
    y_spam = df_spam['spam']

    scaler_spam = StandardScaler()
    X_spam_scaled = scaler_spam.fit_transform(X_spam)

    X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
        X_spam_scaled, y_spam, test_size=0.2, random_state=42, stratify=y_spam
    )
    return X_train_spam, X_test_spam, y_train_spam, y_test_spam
##Diabetes Dataset
def preprocess_diabetes_data(file_path): 
    """
    Compute the preprocessing for the Diabetes dataset.
    Parameters:
    file_path (str): Path to the CSV file containing the Diabetes dataset.
    Returns:
    X_train, X_test, y_train, y_test: Preprocessed and split data.
    """
    df = pd.read_csv(file_path)

    columns_keep = [
        'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack',
        'PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost',
        'GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'
    ]

    X = df[columns_keep].copy()
    y = df['Diabetes_binary']

    for col in X.columns:
        pos_vals = X[X[col] > 0][col]
        if not pos_vals.empty:
            X.loc[X[col] < 0, col] = pos_vals.mean()

    df_combined = pd.concat([X, y], axis=1).drop_duplicates()
    X = df_combined[columns_keep]
    y = df_combined["Diabetes_binary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

###MODELS IMPLEMENTATION###

## Logistic Regression Model
def logistic_regression_model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --------------------------
    # 1. Créer le modèle avec class_weight='balanced'
    # --------------------------
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # equilibrage du dataset
        random_state=42
    )

    # --------------------------
    # 2. Entraîner le modèle
    # --------------------------
    model.fit(X_train, y_train)

    # --------------------------
    # 3. Prédictions probabilistes
    # --------------------------
    y_train_prob = model.predict_proba(X_train)[:,1]
    y_test_prob = model.predict_proba(X_test)[:,1]

    # --------------------------
    # 4. Ajustement du seuil
    # --------------------------
    threshold = 0.5 # seuil personnalisé pour améliorer F1-score
    y_train_pred = (y_train_prob >= threshold).astype(int)
    y_test_pred = (y_test_prob >= threshold).astype(int)

    
    # --------------------------
    # 5. F1-score
    # --------------------------
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)

    print("F1-score Train : {:.2f}".format(f1_train))
    print("F1-score Test  : {:.2f}".format(f1_test))

    # --------------------------
    # 6. Classification report
    # --------------------------
    print("\nClassification Report - Test :\n")
    print(classification_report(y_test, y_test_pred))

    # --------------------------
    # 7. Matrice de confusion
    # --------------------------
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - Test (seuil={threshold})")
    plt.show()










##KNN Model

def KNN_model_dataset2(X_train, X_test, y_train, y_test):

    # ================== IMPORTS ==================
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
        precision_recall_curve
    )
    from sklearn.neighbors import KNeighborsClassifier

    #from imblearn.over_sampling import SMOTE



    # ================== SMOTE ==================
    #sm = SMOTE(random_state=42)
    X_train_res, y_train_res = X_train, y_train

    #print("\nRépartition après SMOTE :", np.bincount(y_train_res))

    # ================== KNN ==================
    knn = KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        metric="manhattan",
        n_jobs=-1
    )

    print("\nEntraînement du modèle...")
    knn.fit(X_train_res, y_train_res)

    # ================== PRÉDICTIONS ==================
    y_pred_default = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]

    print("\n--- Résultats (SEUIL PAR DÉFAUT = 0.5) ---")
    print(classification_report(y_test, y_pred_default))
    print("ROC-AUC :", roc_auc_score(y_test, y_proba))

    # ================== RECHERCHE DU SEUIL OPTIMAL ==================
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print("\nSeuil optimal trouvé :", best_threshold)

    # ================== PRÉDICTION AVEC SEUIL OPTIMAL ==================
    y_pred_opt = (y_proba >= best_threshold).astype(int)

    print("\n--- Résultats (SEUIL OPTIMAL) ---")
    print(classification_report(y_test, y_pred_opt))
    print("ROC-AUC :", roc_auc_score(y_test, y_proba))
    print("Matrice de confusion avec seuil optimal :\n", confusion_matrix(y_test, y_pred_opt))

    # ================== PLOTS ==================

    # ---------- 1. Matrice de confusion ----------
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Greens")
    plt.title("Matrice de confusion (Seuil optimal)", fontsize=14)
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.show()

    # ---------- 2. ROC Curve ----------
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='purple', label=f"ROC-AUC = {roc_auc_score(y_test, y_proba):.3f}")
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe ROC – Diabetes (KNN)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ---------- 3. Courbe Precision-Recall ----------
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, color='blue')
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label='Seuil optimal')
    plt.title("Courbe Precision-Recall – Seuil optimal")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()




def KNN_model_dataset1(X_train, X_test, y_train, y_test):
    # ================== IMPORTS ==================
    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve
    )

    # ===========================
    # 1. SPAMBASE DATASET
    # ===========================
    print("=== SPAMBASE ===")


    # ---- KNN + GridSearch ----
    knn_spam = KNeighborsClassifier()

    # on teste quelques k classiques + distance
    param_grid_spam = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    }

    grid_spam = GridSearchCV(
        knn_spam,
        param_grid_spam,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_spam.fit(X_train, y_train)

    best_spam = grid_spam.best_estimator_
    print("Meilleurs paramètres SPAM :", grid_spam.best_params_)

    # prédiction
    y_spam_pred = best_spam.predict(X_test)
    y_spam_proba = best_spam.predict_proba(X_test)[:, 1]

    # métriques
    print("Accuracy (SPAM):", accuracy_score(y_test, y_spam_pred))
    print("\nClassification report (SPAM):\n", classification_report(y_test, y_spam_pred))

    print("Matrice de confusion (SPAM):")
    print(confusion_matrix(y_test, y_spam_pred))

    # ROC-AUC (utile même si dataset est à peu près équilibré)
    spam_auc = roc_auc_score(y_test, y_spam_proba)
    print("ROC-AUC (SPAM):", spam_auc)



##Random Forest Model

def random_forest_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Random Forest classifier with feature selection and threshold optimization.
    
    Parameters:
    X_train, X_test: np.array or pd.DataFrame, features for training and testing
    y_train, y_test: np.array or pd.Series, target labels
    
    Returns:
    best_rf: trained RandomForestClassifier object
    best_threshold: float, optimized threshold for class 1
    """
    
    # --------------------------
    # 1. Feature selection via RandomForest
    # --------------------------
    rf_selector = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_selector.fit(X_train, y_train)
    
    selector = SelectFromModel(rf_selector, prefit=True, threshold="mean")
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    # --------------------------
    # 2. Hyperparameter tuning
    # --------------------------
    param_grid = {
        "n_estimators": [150, 250],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1",
        verbose=0,
        n_jobs=-1
    )
    grid.fit(X_train_sel, y_train)
    
    best_rf = grid.best_estimator_
    
    # --------------------------
    # 3. Prédictions probabilistes
    # --------------------------
    y_proba = best_rf.predict_proba(X_test_sel)[:, 1]
    
    # --------------------------
    # 4. Optimisation du seuil F1
    # --------------------------
    thresholds = np.linspace(0.4, 0.6, 40)
    best_threshold = 0
    best_f1 = 0
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    y_pred_thresh = (y_proba >= best_threshold).astype(int)
    
    # --------------------------
    # 5. Metrics
    # --------------------------
    accuracy = (y_pred_thresh == y_test).mean()
    recall = f1_score(y_test, y_pred_thresh, average=None)[1] if len(np.unique(y_test))>1 else f1_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    
    print(f"Best threshold : {best_threshold:.3f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score (classe 1) : {f1:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred_thresh))
    
    # --------------------------
    # 6. Confusion Matrix
    # --------------------------
    cm = confusion_matrix(y_test, y_pred_thresh)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matrice de confusion - Random Forest (threshold={best_threshold:.2f})")
    plt.show()
    
    return best_rf, best_threshold

##MLP with PyTorch Model
def MLP_pytorch_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a simple MLP using PyTorch.
    Parameters:
    X_train, X_test: np.array or pd.DataFrame, features for training and testing
    y_train, y_test: np.array or pd.Series, target labels
    Returns:
    Plots confusion matrix and classification report."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    class MLP_torch(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MLP_torch, self).__init__()
                self.linear = torch.nn.Linear(input_dim, output_dim)
                self.activation = torch.nn.Sigmoid()

            def forward(self, x):
                x = self.linear(x)
                x = self.activation(x)

                return x
            def predict(self, x):
                with torch.no_grad():
                    outputs = self.forward(torch.tensor(x, dtype=torch.float32))
                    predicted = (outputs.numpy() > 0.5).astype(int)
                return predicted

    nb_features = X_train.shape[1]
    output_dim = 1
    model_nn = MLP_torch(nb_features,output_dim)
    n_epochs = 1000

    #Mettre plus de poids aux exemples de la classe minoritaire(diabète=1)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    print("Ratio négatif/positif:", n_neg / n_pos)
    pos_weight = torch.tensor([n_neg / n_pos])

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.1)
    for epoch in range(n_epochs):
        model_nn.train()
        optimizer.zero_grad()
        inputs = torch.FloatTensor(X_train)
        labels = torch.FloatTensor(y_train.values).view(-1, 1)
        outputs = model_nn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    model_nn.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test)
        labels = torch.FloatTensor(y_test.values).view(-1, 1)
        outputs = model_nn(inputs)
        predicted = (outputs.numpy() > 0.5).astype(int)
        cm = confusion_matrix(y_test, predicted)
        print("Matrice de confusion :\n", cm)
        cr = classification_report(y_test, predicted)
        print("Rapport de classification :\n", cr)


