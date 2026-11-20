# Classificação Estelar com Modelagem Preditiva

Projeto acadêmico desenvolvido no Departamento de Estatística da **Universidade Federal de Pernambuco (UFPE)** — Centro de Ciências Exatas e da Natureza (CCEN).

O objetivo deste trabalho é realizar uma Análise Exploratória de Dados (AED) e aplicar algoritmos de aprendizado de máquina supervisionado (**Random Forest** e **Support Vector Machine - SVM**) para automatizar a classificação de objetos astronômicos (**Estrelas, Galáxias e Quasares**) utilizando dados fotométricos e de *redshift* do Sloan Digital Sky Survey (SDSS-IV DR17).

---

## Autoria
* **Autora:** Thays Sabrina Brito de Souza
* **Data:** Novembro de 2026

---

## Estrutura do Repositório

```text
├── graficos/                       # Visualizações criadas a partir da AED e avaliação dos modelos
├── models.py                       # Arquivo Python com modelagem dos dados
├── relatorio.qmd                   # Código do relatório em Quarto (R + Python)
├── relatorio.pdf                   # Relatório final em PDF
├── star_classification.csv         # Base de dados original (SDSS-IV DR17 - 100.000 registros)
├── star_classification_limpo.csv   # Base de dados tratada (sem valores inválidos e outliers extremos)
└── README.md                       # Documentação do projeto
```

---

## Base de Dados

Os dados são da **Data Release 17 (DR17)** do *Sloan Digital Sky Survey* (disponibilizada no Kaggle (https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)), contendo 100.000 observações e as seguintes variáveis principais utilizadas na análise:

* **Posição Celeste:** `alpha` (ascensão reta) e `delta` (declinação).
* **Filtros Fotométricos:** `u` (ultravioleta), `g` (verde), `r` (vermelho), `i` (infravermelho próximo) e `z` (infravermelho).
* **Espectroscopia:** `redshift` (desvio para o vermelho).
* **Variável Alvo (`class`):** `GALAXY` (Galáxia), `STAR` (Estrela) ou `QSO` (Quasar).

---

## Metodologia

1. **Limpeza e AED (R):** Remoção de identificadores, filtragem de valores anômalos (`-9999`) e exclusão de outliers extremos via critério interquartil.
2. **Pré-processamento (Python):** Divisão estratificada em treino/teste (80/20) e balanceamento das classes no conjunto de treino utilizando **SMOTE**.
3. **Modelagem e Otimização (Python):** Treinamento e ajuste de hiperparâmetros com **PyCaret** e **Scikit-Learn**.

---

## Pré-requisitos

* **R** (com os pacotes `tidyverse`, `e1071`, `patchwork`, `ggcorrplot`, `GGally`, `reticulate`)
* **Python 3.13.5** (com as bibliotecas `pandas`, `scikit-learn`, `pycaret`, `imbalanced-learn`, `matplotlib`, `seaborn`)
* **Quarto** (para o relatório)

---
