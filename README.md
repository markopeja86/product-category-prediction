# Product Category Prediction

Ovaj projekat razvija model mašinskog učenja koji automatski predlaže kategoriju proizvoda na osnovu njegovog naziva (*Product Title*).  
Cilj je da se ubrza i automatizuje unos novih proizvoda u online trgovinu, kako bi se smanjila ručna klasifikacija i rizik od greške.

Model uzima tekst naslova proizvoda i predviđa vrednost kolone **Category Label** (npr. *Mobile Phones, Laptops, Washing Machines, Fridge Freezers*…).

---

## 📁 Struktura projekta

```text
product-category-prediction/
├─ data/
│  └─ products.csv
├─ notebooks/
│  └─ 01_exploration_and_model.ipynb
├─ src/
│  ├─ train_model.py
│  └─ predict_category.py
├─ models/
│  └─ product_category_model.pkl      # generiše se nakon treniranja
├─ venv/                              # lokalno virtuelno okruženje (nije u repou)
├─ requirements.txt
└─ README.md
