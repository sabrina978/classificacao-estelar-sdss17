# %%
from pycaret.classification import *
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
dados = pd.read_csv('star_classification_limpo.csv')
dados.head()

#%%
# Matriz Correlação
plt.figure(figsize=(10, 8))
sns.heatmap(pd.get_dummies(dados, columns=['class']).corr(), # Transformar classe em dummies (One-hot) e calcular correlação
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlação')
plt.show()

#%%
dados = dados.drop(['alpha', 'delta'], axis=1)
dados.columns

#%%
le = LabelEncoder()     # trocar variavel de categorica para numerica
dados['class'] = le.fit_transform(dados['class'])
print(dict(enumerate(le.classes_))) # Ver número atribuído à classe

train, test = train_test_split(dados, test_size=0.2, # stratify: mesma distribuição original
                               stratify=dados['class'], random_state=10)

X_train, y_train = SMOTE(random_state=10).fit_resample(train.drop(columns=['class']), 
                                                       train['class']) #treino
dados_train = pd.concat([X_train, y_train], axis=1) # x = features, y = class: target

X_test, y_test = test.drop(columns=['class']), test['class'] # teste

plt.figure(figsize=(8, 5))
sns.countplot(x='class', data=dados_train, 
              order = dados_train['class'].value_counts().index)
plt.title('Frequência por Classe', fontsize=14, pad=15)
plt.xlabel('Classes', fontsize=12)
plt.ylabel('Quantidade de Registros', fontsize=12)
plt.show()

#%%
# Base random florest
crf = ClassificationExperiment(target='class', fold=5, session_id=123, 
                               verbose=False).fit(dados_train)
rf_base = crf.create_model('rf', max_depth=3, min_samples_split=2, verbose=False)

# Otimizar modelo
grid_rf = {'n_estimators':[50, 100, 200, 300],
           'max_depth':[3, 5, 7], 'min_samples_split':[2]} # Valores que serão avaliados
tuned_rf = crf.tune_model(rf_base.pipeline, custom_grid=grid_rf, 
                          optimize='F1_weighted', verbose=False) # Otimizar F1

# Consolidar no treino inteiro
final_rf = crf.finalize_model(tuned_rf.pipeline)
# Avaliar no teste
pred_rf = final_rf.pipeline.predict(X_test)

#%%
from IPython.display import display, Markdown

#Definir função para usar futuramente no svm
def tabela_metricas(y_true, y_pred, nomes_classes=None, modelo=""):
    # Tabela métricas
    print("Tabela de Métricas - ",modelo)
    relatorio = classification_report(y_true, y_pred, 
                                      target_names=nomes_classes, output_dict=True)
    df_metricas = pd.DataFrame(relatorio).T.round(4)
    # Kappa como uma linha extra na tabela
    df_metricas.loc['kappa'] = [cohen_kappa_score(y_true, y_pred), None, None, None]
    # Formato de tabela Markdown
    display(Markdown(df_metricas.fillna('-').to_markdown()))

tabela_metricas(y_test, pred_rf, nomes_classes=le.classes_, 
                modelo='Floresta Aleatória')

#%%
# Base Modelo SVM Não Linear

# Amostra estratificada de 15.000 linhas apenas para achar os hiperparâmetros rápido
amostra_tune = dados_train.groupby('class', group_keys=False).apply(lambda x: x.sample(5000, random_state=42))
csvm = ClassificationExperiment(target='class', fold=3, normalize=True, session_id=123).fit(amostra_tune)
svm = csvm.create_model('rbfsvm')

# Otimizar o modelo não linear
grid_svm = {'C':[1, 10, 50], 'gamma': ['scale', 0.01, 0.1]}
# Apenas 4 iterações com 3 folds (12 treinos rápidos em vez de 50)
tuned_svm = csvm.tune_model(svm.pipeline, custom_grid=grid_svm, n_iter=4, optimize='F1_weighted', verbose=False)

# Treina o modelo final na base treino completa usando os melhores parâmetros encontrados
csvm_full = ClassificationExperiment(target='class', fold=3, normalize=True, session_id=123).fit(dados_train)
final_svm = csvm_full.create_model('rbfsvm', **tuned_svm.best_params)

# Avaliar no teste
pred_svm = final_svm.pipeline.predict(X_test)

tabela_metricas(y_test, pred_svm, nomes_classes=le.classes_)

#%%
from sklearn.inspection import permutation_importance

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Importância Random Florest
pd.Series(final_rf.pipeline.named_steps['rf'].feature_importances_,
    index=X_train.columns).sort_values().plot(kind='barh', color='steelblue',
    title='Importância das Variáveis (Random Forest)', ax=axes[0])
axes[0].set_xlabel('Importância Relativa')
axes[0].grid(axis='x', linestyle='--', alpha=0.6)

# Importância por Permutação (SVM)
# Calcula a importância embaralhando os dados de teste (n_repeats=5 para ser rápido)
perm = permutation_importance(final_svm.pipeline, X_test, y_test,n_repeats=5,
random_state=42, scoring='f1_weighted')

pd.Series(perm.importances_mean,
index=X_test.columns).sort_values().plot(kind='barh', color='darkorange',
title='Importância por Permutação (SVM)', ax=axes[1])
axes[1].set_xlabel('Queda Média no F1-Score')
axes[1].grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

#%%
# Precision-Recall
y_test_bin = LabelBinarizer().fit_transform(y_test)
y_prob_rf = final_rf.pipeline.predict_proba(X_test)
y_prob_svm = final_svm.pipeline.predict_proba(X_test)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Gráfico Random Forest
for i, classe in enumerate(le.classes_):
    PrecisionRecallDisplay.from_predictions(y_test_bin[:, i], y_prob_rf[:, i], 
        name=classe, ax=axes[0])
axes[0].set_title('Curvas Precision-Recall por Classe (Random Forest)')
axes[0].grid(linestyle='--', alpha=0.5)

# Gráfico SVM
for i, classe in enumerate(le.classes_):
    PrecisionRecallDisplay.from_predictions(y_test_bin[:, i], y_prob_svm[:, i], 
        name=classe, ax=axes[1])        
axes[1].set_title('Curvas Precision-Recall por Classe (SVM)')
axes[1].grid(linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Matrizes de Confusão Lado a Lado
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Random Forest 
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_rf, display_labels=le.classes_, normalize='true',
    values_format='.1%', cmap='Blues', ax=axes[0], colorbar=False
)
axes[0].set_title('Random Forest (% por Classe Real)')
# SVM
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_svm, display_labels=le.classes_, normalize='true',
    values_format='.1%', cmap='Oranges', ax=axes[1], colorbar=False
)
axes[1].set_title('SVM (% por Classe Real)')

plt.tight_layout()
plt.show()

# %%
