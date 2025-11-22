# Product Category Prediction

### Automatska klasifikacija proizvoda na osnovu naslova

Ovaj projekat implementira model mašinskog učenja koji automatski
predlaže kategoriju proizvoda na osnovu njegovog naziva (*Product
Title*).\
Rešenje pomaže online trgovinama da ubrzaju proces unosa novih artikala,
smanji ručni rad i poveća tačnost kategorizacije.

Model je razvijen korišćenjem realnog skupa podataka sa preko 30.000
proizvoda iz različitih kategorija.

------------------------------------------------------------------------

## 📁 Struktura projekta

    product-category-prediction/
    │
    ├── data/
    │   └── products.csv
    │
    ├── models/
    │   └── product_category_model.pkl
    │
    ├── notebooks/
    │   └── 01_exploration_and_model.ipynb
    │
    ├── src/
    │   ├── train_model.py
    │   └── predict_category.py
    │
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 📄 Opis projekta

Projekat koristi **TF-IDF vektorizaciju** nad tekstom i **Logistic
Regression** za predviđanje kategorije proizvoda.\
Treniranje i evaluacija se izvode u Jupyter notebook-u i Python skripti,
dok se predikcija vrši kroz interaktivnu CLI aplikaciju.

### Šta model radi?

➡️ Ulaz: naziv proizvoda (npr. "iphone 7 32gb gold")\
➡️ Izlaz: preporučena kategorija (npr. "Mobile Phones")

------------------------------------------------------------------------

## 🧪 Dataset

Dataset se nalazi u folderu `data/products.csv` i sadrži sledeće ključne
kolone:

  Kolona             Opis
  ------------------ ------------------------------------------
  `Product Title`    naziv proizvoda -- ulaz u model
  `Category Label`   ciljna kategorija -- izlaz modela
  ostale kolone      dodatni podaci (nisu korišćeni u modelu)

Napomena: dataset sadrži kolone sa razmacima → pri učitavanju se uvek
radi:

``` python
df.columns = df.columns.str.strip()
```

------------------------------------------------------------------------

## ⚙️ Instalacija i pokretanje projekta

### 1. Kloniranje repozitorijuma

``` bash
git clone https://github.com/markopeja86/product-category-prediction.git
cd product-category-prediction
```

### 2. Kreiranje i aktivacija virtualnog okruženja

``` bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Instalacija zavisnosti

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🧠 Treniranje modela

Za treniranje modela koristi se skripta:

    src/train_model.py

Pokretanje:

``` bash
python src/train_model.py \
    --data-path data/products.csv \
    --model-path models/product_category_model.pkl
```

Skripta će:

-   učitati CSV,
-   očistiti nazive kolona,
-   podeliti podatke na trening i test skup,
-   trenirati TF-IDF + Logistic Regression pipeline,
-   ispisati accuracy i classification report,
-   sačuvati model u `models/product_category_model.pkl`.

### 🎯 Rezultati modela

**Accuracy:** 0.91

------------------------------------------------------------------------

# 🔮 Interaktivna predikcija kategorije

Za ručno testiranje koristi se skripta:

    src/predict_category.py

Pokretanje:

``` bash
python src/predict_category.py --model-path models/product_category_model.pkl
```

Primer interakcije:

    === Predikcija kategorije proizvoda ===
    Unesi naziv proizvoda (ili 'exit' za izlaz):
    > iphone 7 32gb gold
    → Predikcija kategorije: Mobile Phones

Za izlaz:

    > exit

------------------------------------------------------------------------

## 🧪 Ručni test primeri

  -----------------------------------------------------------------------
  Ulaz (Product Title)                           Očekivana kategorija
  ---------------------------------------------- ------------------------
  iphone 7 32gb gold                             Mobile Phones

  olympus e m10 mark iii geh use silber          Digital Cameras

  kenwood k20mss15 solo                          Microwaves

  bosch wap28390gb 8kg 1400 spin                 Washing Machines

  bosch serie 4 kgv39vl31g                       Fridge Freezers

  smeg sbs8004po                                 Fridge Freezers
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 📓 Notebook -- analiza i razvoj modela

Notebook `notebooks/01_exploration_and_model.ipynb` sadrži:

-   učitavanje i pregled podataka,
-   čišćenje kolona i pripremu skupa,
-   inicijalni EDA,
-   testiranje više algoritama,
-   izbor finalnog modela,
-   evaluaciju na test skupu.

------------------------------------------------------------------------

## 🚀 Dalji razvoj i unapređenja

-   testirati LinearSVC za još bolje rezultate,\
-   dodati feature engineering (dužina naslova, broj reči, prisustvo
    cifara),\
-   izvlačenje brenda iz naslova (Apple, Samsung, Bosch...),\
-   balansiranje klasa kod ređih kategorija,\
-   deploy modela putem FastAPI / Streamlit aplikacije.

------------------------------------------------------------------------

## ✔️ Status projekta

-   ✓ Notebook sa analizom\
-   ✓ Trenutni model u `.pkl` formatu\
-   ✓ Train skripta\
-   ✓ Interaktivna predikcija\
-   ✓ Public GitHub repo\
-   ✓ Dokumentovan README

------------------------------------------------------------------------

# 🎉 Hvala!
