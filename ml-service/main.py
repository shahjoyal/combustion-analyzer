# ml-service/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os, joblib, json, datetime, re
from sklearn.model_selection import train_test_split
from typing import List, Optional
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AFT XGBoost service")

# allow CORS from your node server during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# directories & files
BASE_DIR = "ml-service"
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
PREPROCESS_PATH = os.path.join(BASE_DIR, "preprocessing.json")
SUMMARY_PATH = os.path.join(BASE_DIR, "xgb_summary.json")
METADATA_PATH = os.path.join(BASE_DIR, "training_metadata.json")

# helper: formulaAFT (kept as in your snippet)
# def formulaAFT(row):
#     # expects at least first 9 oxide values
#     arr = np.asarray(row, dtype=float).flatten()
#     if arr.size < 9:
#         raise ValueError("formulaAFT expects at least 9 values (SiO2..TiO2)")
#     SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2 = arr[:9]
#     sumSiAl = SiO2 + Al2O3
#     if sumSiAl < 55:
#         return (1245 + 1.1*SiO2 + 0.95*Al2O3 - 2.5*Fe2O3 - 2.98*CaO - 4.5*MgO
#                 - 7.89*(Na2O+K2O) - 1.7*SO3 - 0.63*TiO2)
#     elif sumSiAl < 75:
#         return (1323 + 1.45*SiO2 + 0.683*Al2O3 - 2.39*Fe2O3 - 3.1*CaO - 4.5*MgO
#                 - 7.49*(Na2O+K2O) - 2.1*SO3 - 0.63*TiO2)
#     else:
#         return (1395 + 1.2*SiO2 + 0.9*Al2O3 - 2.5*Fe2O3 - 3.1*CaO - 4.5*MgO
#                 - 7.2*(Na2O+K2O) - 1.7*SO3 - 0.63*TiO2)

# def formulaAFT(row):
#     # expects at least first 9 oxide values
#     arr = np.asarray(row, dtype=float).flatten()
#     if arr.size < 9:
#         raise ValueError("formulaAFT expects at least 9 values (SiO2..TiO2)")
#     SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2 = arr[:9]
#     sumSiAl = SiO2 + Al2O3

#     if sumSiAl < 55:
#         return (1377.0
#                 + 0.0243   * SiO2
#                 + 0.0198   * Al2O3
#                 - 3.582    * Fe2O3
#                 - 0.0298   * CaO
#                 - 0.0450   * MgO
#                 - 12.3579  * (Na2O + K2O)
#                 - 3.383    * SO3
#                 - 1.2537   * TiO2)
#     elif sumSiAl < 75:
#         return (1447.2
#                 + 0.0145   * SiO2
#                 + 1.35917  * Al2O3
#                 - 4.7561   * Fe2O3
#                 - 3.1000   * CaO
#                 - 8.9550   * MgO
#                 - 7.4900   * (Na2O + K2O)
#                 - 4.1790   * SO3
#                 - 0.0063   * TiO2)
#     else:
#         return (1517.4
#                 + 0.0127   * SiO2
#                 + 1.8000   * Al2O3
#                 - 4.9750   * Fe2O3
#                 - 6.1690   * CaO
#                 - 8.9550   * MgO
#                 - 0.0720   * (Na2O + K2O)
#                 - 3.3830   * SO3
#                 - 1.2537   * TiO2)



# def formulaAFT(row):
#     """
#     row: sequence-like of oxide values in order:
#     [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2]
#     If K2O is missing and the caller passed a combined NaK at index 5,
#     this function will still work (K2O treated as 0).
#     """
#     arr = np.asarray(row, dtype=float).flatten()
#     if arr.size < 8:
#         raise ValueError("formulaAFT expects at least 8 values (SiO2..TiO2)")

#     # Unpack with defaults so missing K2O is treated as 0
#     SiO2 = arr[0]
#     Al2O3 = arr[1]
#     Fe2O3 = arr[2]
#     CaO = arr[3]
#     MgO = arr[4]
#     Na2O = arr[5] if arr.size > 5 else 0.0
#     K2O  = arr[6] if arr.size > 6 else 0.0
#     SO3  = arr[7] if arr.size > 7 else 0.0
#     TiO2 = arr[8] if arr.size > 8 else 0.0

#     NaK = Na2O + K2O
#     sumSiAl = SiO2 + Al2O3

#     if sumSiAl < 45:
#         # HIGH
#         return (949.3601
#                 + 0.5646   * SiO2
#                 + 2.1826   * Al2O3
#                 + 1.6339   * Fe2O3
#                 + 6.3412   * CaO
#                 + 8.6798   * MgO
#                 + 5.0377   * NaK
#                 + 0.7961   * SO3
#                 - 2.1892   * TiO2)
#     elif sumSiAl < 55:
#         # MODERATE TO HIGH
#         return (1124.2069
#                 - 0.0797   * SiO2
#                 + 4.2971   * Al2O3
#                 - 0.1462   * Fe2O3
#                 + 4.7533   * CaO
#                 + 3.0594   * MgO
#                 + 3.0923   * NaK
#                 - 2.2814   * SO3
#                 - 4.1400   * TiO2)
#     elif sumSiAl < 65:
#         # MODERATE
#         return (1177.4801
#                 + 0.1559   * SiO2
#                 + 7.8153   * Al2O3
#                 - 1.7562   * Fe2O3
#                 + 1.9570   * CaO
#                 - 4.3791   * MgO
#                 - 2.8309   * NaK
#                 - 4.1712   * SO3
#                 + 1.7976   * TiO2)
#     elif sumSiAl < 75:
#         # LOW TO MODERATE
#         return (1140.8756
#                 + 3.0803   * SiO2
#                 + 6.9504   * Al2O3
#                 - 1.9173   * Fe2O3
#                 - 1.7021   * CaO
#                 - 9.6014   * MgO
#                 - 5.8391   * NaK
#                 - 4.5557   * SO3
#                 + 4.9372   * TiO2)
#     else:
#         # LOW (sumSiAl >= 75)
#         return (1619.5948
#                 - 1.1828   * SiO2
#                 + 0.5515   * Al2O3
#                 - 6.6603   * Fe2O3
#                 - 12.7041  * CaO
#                 - 20.8388  * MgO
#                 + 4.6097   * NaK
#                 - 5.2634   * SO3
#                 + 10.5468  * TiO2)



# def formulaAFT(row):
#     """
#     row: [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2]
#     returns numeric AFT. If K2O missing, treated as 0.
#     NOTE: formulas for h1 and mh1 were not provided; the function currently
#     falls back to the main-bin formula (h2 / mh2) for those two cases.
#     """
#     arr = np.asarray(row, dtype=float).flatten()
#     if arr.size < 8:
#         raise ValueError("formulaAFT expects at least 8 values (SiO2..TiO2)")

#     SiO2 = arr[0]
#     Al2O3 = arr[1]
#     Fe2O3 = arr[2]
#     CaO = arr[3]
#     MgO = arr[4]
#     Na2O = arr[5] if arr.size > 5 else 0.0
#     K2O  = arr[6] if arr.size > 6 else 0.0
#     SO3  = arr[7] if arr.size > 7 else 0.0
#     TiO2 = arr[8] if arr.size > 8 else 0.0

#     NaK = Na2O + K2O
#     sumSiAl = SiO2 + Al2O3
#     FeCa = Fe2O3 + CaO
#     FECA_THRESHOLD = 12.6

#     # High (h1/h2)
#     if sumSiAl < 45:
#         if FeCa <= FECA_THRESHOLD:
#             # h1 not supplied → fallback to high (h2) formula
#             return (949.3601
#                     + 0.5646   * SiO2
#                     + 2.1826   * Al2O3
#                     + 1.6339   * Fe2O3
#                     + 6.3412   * CaO
#                     + 8.6798   * MgO
#                     + 5.0377   * NaK
#                     + 0.7961   * SO3
#                     - 2.1892   * TiO2)
#         else:
#             # h2 (provided)
#             return (949.3601
#                     + 0.5646   * SiO2
#                     + 2.1826   * Al2O3
#                     + 1.6339   * Fe2O3
#                     + 6.3412   * CaO
#                     + 8.6798   * MgO
#                     + 5.0377   * NaK
#                     + 0.7961   * SO3
#                     - 2.1892   * TiO2)

#     # Moderate to high (mh1/mh2)
#     elif sumSiAl < 55:
#         if FeCa <= FECA_THRESHOLD:
#             # mh1 not supplied -> fallback to mh2
#             return (1124.2069
#                     - 0.0797   * SiO2
#                     + 4.2971   * Al2O3
#                     - 0.1462   * Fe2O3
#                     + 4.7533   * CaO
#                     + 3.0594   * MgO
#                     + 3.0923   * NaK
#                     - 2.2814   * SO3
#                     - 4.1400   * TiO2)
#         else:
#             # mh2 (provided)
#             return (1124.2069
#                     - 0.0797   * SiO2
#                     + 4.2971   * Al2O3
#                     - 0.1462   * Fe2O3
#                     + 4.7533   * CaO
#                     + 3.0594   * MgO
#                     + 3.0923   * NaK
#                     - 2.2814   * SO3
#                     - 4.1400   * TiO2)

#     # Moderate (m1/m2)
#     elif sumSiAl < 65:
#         if FeCa <= FECA_THRESHOLD:
#             # m1 (provided)
#             return (1309.86
#                     - 0.8026   * SiO2
#                     + 1.2965   * Al2O3
#                     + 2.7164   * Fe2O3
#                     - 2.9016   * CaO
#                     + 0.4322   * MgO
#                     + 2.4479   * NaK
#                     - 2.5621   * SO3
#                     + 0.0      * TiO2)
#         else:
#             # m2 (provided)
#             return (1181.3094
#                     - 0.4266   * SiO2
#                     + 8.1218   * Al2O3
#                     - 1.2093   * Fe2O3
#                     + 2.9888   * CaO
#                     - 4.6965   * MgO
#                     - 4.7050   * NaK
#                     - 3.9495   * SO3
#                     + 1.1836   * TiO2)

#     # Low to moderate (ml1/ml2)
#     elif sumSiAl < 75:
#         if FeCa <= FECA_THRESHOLD:
#             # ml1 (provided)
#             return (1402.7002
#                     - 0.8606   * SiO2
#                     + 3.2272   * Al2O3
#                     - 1.0213   * Fe2O3
#                     - 0.6618   * CaO
#                     - 1.4658   * MgO
#                     - 4.1053   * NaK
#                     - 3.1920   * SO3
#                     - 0.1844   * TiO2)
#         else:
#             # ml2 (provided)
#             return (1021.4717
#                     + 3.8985   * SiO2
#                     + 7.1950   * Al2O3
#                     + 2.1680   * Fe2O3
#                     + 2.4863   * CaO
#                     - 7.8374   * MgO
#                     - 8.0358   * NaK
#                     - 4.1760   * SO3
#                     + 5.7583   * TiO2)

#     # Low (l1/l2)
#     else:
#         if FeCa <= FECA_THRESHOLD:
#             # l1 (provided)
#             return (1515.9840
#                     - 0.0582   * SiO2
#                     + 1.5964   * Al2O3
#                     - 3.8771   * Fe2O3
#                     - 10.1779  * CaO
#                     - 13.1584  * MgO
#                     + 2.9112   * NaK
#                     - 7.4455   * SO3
#                     + 4.9705   * TiO2)
#         else:
#             # l2 (provided)
#             return (1430.9261
#                     - 2.5702   * SiO2
#                     + 2.6075   * Al2O3
#                     + 4.6287   * Fe2O3
#                     - 2.6753   * CaO
#                     - 1.1602   * MgO
#                     + 1.2016   * NaK
#                     + 1.6603   * SO3
#                     + 2.9714   * TiO2)


# def formulaAFT(row):
#     arr = np.asarray(row, dtype=float).flatten()
#     if arr.size < 8:
#         raise ValueError("formulaAFT expects at least 8 values (SiO2..TiO2)")

#     SiO2 = arr[0]
#     Al2O3 = arr[1]
#     Fe2O3 = arr[2]
#     CaO = arr[3]
#     MgO = arr[4]
#     Na2O = arr[5] if arr.size > 5 else 0.0
#     K2O  = arr[6] if arr.size > 6 else 0.0
#     SO3  = arr[7] if arr.size > 7 else 0.0
#     TiO2 = arr[8] if arr.size > 8 else 0.0

#     NaK = Na2O + K2O
#     sumSiAl = SiO2 + Al2O3
#     FeCa = Fe2O3 + CaO
#     FECA_THRESHOLD = 12.6

#     if sumSiAl < 45:
#         if FeCa <= FECA_THRESHOLD:
#             return (949.3601
#                     + 0.5646   * SiO2
#                     + 2.1826   * Al2O3
#                     + 1.6339   * Fe2O3
#                     + 6.3412   * CaO
#                     + 8.6798   * MgO
#                     + 5.0377   * NaK
#                     + 0.7961   * SO3
#                     - 2.1892   * TiO2)
#         else:
#             return (949.3601
#                     + 0.5646   * SiO2
#                     + 2.1826   * Al2O3
#                     + 1.6339   * Fe2O3
#                     + 6.3412   * CaO
#                     + 8.6798   * MgO
#                     + 5.0377   * NaK
#                     + 0.7961   * SO3
#                     - 2.1892   * TiO2)
#     elif sumSiAl < 55:
#         if FeCa <= FECA_THRESHOLD:
#             return (1139.9324
#                     - 0.2734   * SiO2
#                     + 4.5495   * Al2O3
#                     - 0.5943   * Fe2O3
#                     + 4.4863   * CaO
#                     + 2.9609   * MgO
#                     + 3.6016   * NaK
#                     - 2.4116   * SO3
#                     - 3.7743   * TiO2)
#         else:
#             return (1139.9324
#                     - 0.2734   * SiO2
#                     + 4.5495   * Al2O3
#                     - 0.5943   * Fe2O3
#                     + 4.4863   * CaO
#                     + 2.9609   * MgO
#                     + 3.6016   * NaK
#                     - 2.4116   * SO3
#                     - 3.7743   * TiO2)
#     elif sumSiAl < 65:
#         if FeCa <= FECA_THRESHOLD:
#             return (1137.4961
#                     + 1.1956   * SiO2
#                     + 3.5062   * Al2O3
#                     + 3.3419   * Fe2O3
#                     - 12.5158  * CaO
#                     - 2.2775   * MgO
#                     + 0.0163   * NaK
#                     + 6.7002   * SO3
#                     + 1.9344   * TiO2)
#         else:
#             return (1149.2441
#                     - 0.5063   * SiO2
#                     + 8.8672   * Al2O3
#                     - 1.1065   * Fe2O3
#                     + 3.3827   * CaO
#                     - 1.9235   * MgO
#                     - 0.3508   * NaK
#                     - 4.5738   * SO3
#                     + 1.6308   * TiO2)
#     elif sumSiAl < 75:
#         if FeCa <= FECA_THRESHOLD:
#             return (1194.5549
#                     + 2.7079   * SiO2
#                     + 6.4134   * Al2O3
#                     - 3.0781   * Fe2O3
#                     - 6.7834   * CaO
#                     - 6.4518   * MgO
#                     - 5.4799   * NaK
#                     - 2.4833   * SO3
#                     + 2.2483   * TiO2)
#         else:
#             return (1025.9764
#                     + 3.2181   * SiO2
#                     + 6.3438   * Al2O3
#                     + 4.2982   * Fe2O3
#                     + 5.4756   * CaO
#                     - 5.5719   * MgO
#                     - 6.8764   * NaK
#                     - 5.3138   * SO3
#                     + 4.3039   * TiO2)
#     else:
#         if FeCa <= FECA_THRESHOLD:
#             return (1638.5785
#                     - 1.3546   * SiO2
#                     + 0.3045   * Al2O3
#                     - 6.5022   * Fe2O3
#                     - 13.7161  * CaO
#                     - 20.4375  * MgO
#                     + 4.0117   * NaK
#                     - 4.9265   * SO3
#                     + 10.4268  * TiO2)
#         else:
#             return (1303.6482
#                     + 1.2428   * SiO2
#                     + 0.6550   * Al2O3
#                     + 2.2788   * Fe2O3
#                     + 0.9765   * CaO
#                     - 1.6222   * MgO
#                     + 0.2739   * NaK
#                     - 2.3258   * SO3
#                     + 0.2792   * TiO2)

import numpy as np

def formulaAFT(row):
    arr = np.asarray(row, dtype=float).flatten()
    if arr.size < 8:
        raise ValueError("formulaAFT expects at least 8 values (SiO2..TiO2)")

    SiO2 = arr[0]
    Al2O3 = arr[1]
    Fe2O3 = arr[2]
    CaO = arr[3]
    MgO = arr[4]
    Na2O = arr[5] if arr.size > 5 else 0.0
    K2O  = arr[6] if arr.size > 6 else 0.0
    SO3  = arr[7] if arr.size > 7 else 0.0
    TiO2 = arr[8] if arr.size > 8 else 0.0

    NaK = Na2O + K2O
    sumSiAl = SiO2 + Al2O3
    FeCa = Fe2O3 + CaO
    FECA_THRESHOLD = 12.6

    if sumSiAl < 45:
        if FeCa <= FECA_THRESHOLD:
            return (949.3601
                    + 0.5646 * SiO2
                    + 2.1826 * Al2O3
                    + 1.6339 * Fe2O3
                    + 6.3412 * CaO
                    + 8.6798 * MgO
                    + 5.0377 * NaK
                    + 0.7961 * SO3
                    - 2.1892 * TiO2)
        else:
            return (949.3601
                    + 0.5646 * SiO2
                    + 2.1826 * Al2O3
                    + 1.6339 * Fe2O3
                    + 6.3412 * CaO
                    + 8.6798 * MgO
                    + 5.0377 * NaK
                    + 0.7961 * SO3
                    - 2.1892 * TiO2)
    elif sumSiAl < 55:
        if FeCa <= FECA_THRESHOLD:
            return (1124.2069
                    - 0.0797 * SiO2
                    + 4.2971 * Al2O3
                    - 0.1462 * Fe2O3
                    + 4.7533 * CaO
                    + 3.0594 * MgO
                    + 3.0923 * NaK
                    - 2.2814 * SO3
                    - 4.1400 * TiO2)
        else:
            return (1124.2069
                    - 0.0797 * SiO2
                    + 4.2971 * Al2O3
                    - 0.1462 * Fe2O3
                    + 4.7533 * CaO
                    + 3.0594 * MgO
                    + 3.0923 * NaK
                    - 2.2814 * SO3
                    - 4.1400 * TiO2)
    elif sumSiAl < 65:
        if FeCa <= FECA_THRESHOLD:
            return (1137.4961
                    + 1.1956 * SiO2
                    + 3.5062 * Al2O3
                    + 3.3419 * Fe2O3
                    - 12.5158 * CaO
                    - 2.2775 * MgO
                    + 0.0163 * NaK
                    + 6.7002 * SO3
                    + 1.9344 * TiO2)
        else:
            return (1177.4801
                    + 0.1559 * SiO2
                    + 7.8153 * Al2O3
                    - 1.7562 * Fe2O3
                    + 1.9570 * CaO
                    - 4.3791 * MgO
                    - 2.8309 * NaK
                    - 4.1712 * SO3
                    + 1.7976 * TiO2)
    elif sumSiAl < 75:
        if FeCa <= FECA_THRESHOLD:
            return (-1452.0645
                    + 45.6395 * SiO2
                    + 43.3054 * Al2O3
                    - 31.2287 * Fe2O3
                    - 67.6606 * CaO
                    - 37.1564 * MgO
                    - 28.9117 * NaK
                    + 36.5424 * SO3
                    + 1.4591 * TiO2)
        else:
            return (1103.1611
                    + 3.1111 * SiO2
                    + 6.8292 * Al2O3
                    + 0.2934 * Fe2O3
                    + 0.8048 * CaO
                    - 8.8943 * MgO
                    - 6.3487 * NaK
                    - 4.9725 * SO3
                    + 4.9459 * TiO2)
    else:
        if FeCa <= FECA_THRESHOLD:
            return (1468.9852
                    + 0.4432 * SiO2
                    + 2.0587 * Al2O3
                    - 3.8776 * Fe2O3
                    - 6.5077 * CaO
                    - 8.5353 * MgO
                    + 3.2333 * NaK
                    - 10.3822 * SO3
                    + 3.3885 * TiO2)
        else:
            return (1109.9520
                    + 1.4242 * SiO2
                    + 7.7724 * Al2O3
                    + 3.3740 * Fe2O3
                    - 8.8021 * CaO
                    - 15.7613 * MgO
                    + 6.8226 * NaK
                    + 9.8592 * SO3
                    + 22.3574 * TiO2)


def compute_stats(preds, actuals):
    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    abs_err = np.abs(preds - actuals)
    N = len(preds)
    if N == 0:
        return {"rows": 0}
    return {
        "rows": int(N),
        "avgAbs": float(np.round(abs_err.mean(), 2)),
        "maxError": float(np.round(abs_err.max(), 6)),
        "minError": float(np.round(abs_err.min(), 6)),
        "gt20": int((abs_err > 20).sum()),
        "gt40": int((abs_err > 40).sum()),
        "gt60": int((abs_err > 60).sum())
    }

# Preprocessing: winsorize 1%/99% and save cutoffs
def winsorize_and_save(df_X, save_path=PREPROCESS_PATH):
    cutoffs = {"lower": [], "upper": [], "columns": list(df_X.columns)}
    X = df_X.copy()
    for col in X.columns:
        col_arr = X[col].dropna().values
        if len(col_arr) == 0:
            lo = hi = None
        else:
            lo = float(np.quantile(col_arr, 0.01))
            hi = float(np.quantile(col_arr, 0.99))
        cutoffs["lower"].append(lo)
        cutoffs["upper"].append(hi)
        if lo is not None:
            X[col] = X[col].clip(lower=lo)
        if hi is not None:
            X[col] = X[col].clip(upper=hi)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(cutoffs, f, indent=2)
    return X, cutoffs

# load CSV robustly and support optional P2O5, S
def load_csv(path):
    df = pd.read_csv(path)
    # canonical expected columns (required except AFT)
    required = ["SiO2","Al2O3","Fe2O3","CaO","MgO","Na2O","K2O","SO3","TiO2","AFT"]
    optional = ["P2O5", "S"]
    # normalize column names (strip spaces, convert common unicode subscript numerals)
    colmap = {}
    for c in df.columns:
        key = c.strip().replace(" ", "").replace("₃","3").replace("₂","2")
        matched = False
        for e in required + optional:
            if key.lower() == e.lower():
                colmap[c] = e
                matched = True
                break
        # leave unmatched columns as-is (ignored later)
    df = df.rename(columns=colmap)
    # identify which optional present
    optional_present = [c for c in optional if c in df.columns]
    # verify required present
    if not all(k in df.columns for k in required):
        raise HTTPException(status_code=400, detail=f"CSV missing required columns. Found: {list(df.columns)}")
    # final columns for ML: required without AFT, plus optional present, then AFT
    final_cols = [c for c in required if c != "AFT"] + optional_present + ["AFT"]
    df = df[final_cols].dropna()
    return df, final_cols[:-1]  # return df and list of feature column names (AFT excluded)

class TrainFromPathPayload(BaseModel):
    path: str

@app.post("/train_from_path")
def train_from_path(payload: TrainFromPathPayload):
    path = payload.path
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="CSV path not found: " + path)
    df, feature_cols = load_csv(path)
    result = _train_and_save(df, feature_cols, training_path=path)
    return result

@app.post("/train")
async def train(file: UploadFile = File(...)):
    contents = await file.read()
    temp_dir = BASE_DIR
    os.makedirs(temp_dir, exist_ok=True)
    temp = os.path.join(temp_dir, "tmp_upload.csv")
    with open(temp,"wb") as f:
        f.write(contents)
    df, feature_cols = load_csv(temp)
    result = _train_and_save(df, feature_cols, training_path=temp)
    return result

def _train_and_save(df, feature_cols: List[str], training_path: Optional[str] = None):
    """
    df: DataFrame which contains feature_cols and AFT (AFT is last column in df)
    feature_cols: list of str - columns to use as features (in order)
    training_path: optional string to record where df came from
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # features as provided (may include optional P2O5, S)
    X_raw = df[feature_cols].astype(float)
    Y = df["AFT"].astype(float).values

    # winsorize & save cutoffs (based on current feature columns)
    X_winsor, cutoffs = winsorize_and_save(X_raw, save_path=PREPROCESS_PATH)

    # compute base formula from first 9 oxides only (formulaAFT handles that)
    # NOTE: ensure the first 9 oxide columns are in X_raw columns; if optional columns exist they will be after idx 9
    # We will pass the first 9 numeric values from the DataFrame rows to formulaAFT
    def compute_base(row_vals):
        # take first 9 numbers (SiO2..TiO2). If your feature_cols order is different, ensure they remain that way.
        return formulaAFT(row_vals[:9])

    base_series = X_raw.apply(lambda r: compute_base(r.values), axis=1)

    # build hybrid features: winsorized features + base
    X_hybrid = X_winsor.copy()
    X_hybrid["baseAFT"] = base_series.values

    # train/test split
    Xp, Xt, Yp, Yt = train_test_split(X_winsor.values, Y, test_size=0.2, random_state=42)
    Xhp, Xht, Yhp, Yht = train_test_split(X_hybrid.values, Y, test_size=0.2, random_state=42)

    out = {}
    metadata_models = {}

    def train_xgb(X_train, y_train, X_test, y_test, name):
        model = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            tree_method='auto'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_round = np.round(preds).astype(int)
        stats = compute_stats(preds_round, y_test)
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        # try to read n_features_in_ (sklearn-compatible)
        n_features_in = int(getattr(model, "n_features_in_", X_train.shape[1]))
        return {"stats": stats, "modelPath": model_path, "n_features": n_features_in}

    out["pure_xgb"] = train_xgb(Xp, Yp, Xt, Yt, "pure_xgb")
    out["hybrid_xgb"] = train_xgb(Xhp, Yhp, Xht, Yht, "hybrid_xgb")

    # Save a summary + metadata (includes which file was used)
    summary = {
        "rows": int(len(df)),
        "train_rows": int(len(df) - len(Yt)),
        "test_rows": int(len(Yt)),
        "saved_models": {
            "pure": out["pure_xgb"]["modelPath"],
            "hybrid": out["hybrid_xgb"]["modelPath"]
        }
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump({"results": out, "summary": summary, "preprocessing": cutoffs}, f, indent=2)

    # training metadata keeps training_path, timestamp, features and models input sizes
    metadata = {
        "training_path": training_path or None,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "features": list(feature_cols),
        "preprocessing": cutoffs,
        "models": {
            "pure_xgb": {
                "path": out["pure_xgb"]["modelPath"],
                "n_features": out["pure_xgb"].get("n_features")
            },
            "hybrid_xgb": {
                "path": out["hybrid_xgb"]["modelPath"],
                "n_features": out["hybrid_xgb"].get("n_features")
            }
        },
        "summary": summary
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"success": True, "results": out, "summary": summary, "metadata": metadata}

# prediction endpoint: adapt to variable feature count
class PredictPayload(BaseModel):
    values: List[float]
    model: str = "hybrid_xgb"  # "pure_xgb" or "hybrid_xgb"

@app.post("/predict")
def predict(payload: PredictPayload):
    values = payload.values
    model_name = payload.model or "hybrid_xgb"

    # load preprocessing cutoffs / metadata to know expected feature length
    cutoffs = None
    if os.path.exists(PREPROCESS_PATH):
        with open(PREPROCESS_PATH, "r") as f:
            cutoffs = json.load(f)

    expected_n = None
    # If metadata exists, prefer reading expected n_features from the saved model info
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                md = json.load(f)
            model_info = md.get("models", {}).get(model_name, {})
            expected_n = int(model_info.get("n_features")) if model_info.get("n_features") is not None else None
        except Exception:
            expected_n = None

    # fallback: if preprocessing cutoffs exist, use len(cutoffs["lower"])
    if expected_n is None and cutoffs is not None:
        expected_n = len(cutoffs.get("lower", []))
        # hybrid model has +1 for base feature
        if model_name == "hybrid_xgb":
            expected_n = expected_n + 1

    # last fallback
    if expected_n is None:
        expected_n = 9 if model_name == "pure_xgb" else 10

    if not (isinstance(values, list) and len(values) >= expected_n - (1 if model_name == "hybrid_xgb" else 0)):
        # we accept fewer if user supplies at least first 9 (for base computation) but we still require exact later
        raise HTTPException(status_code=400, detail=f"values must be array of at least {expected_n - (1 if model_name == 'hybrid_xgb' else 0)} features (found {len(values)})")

    # prepare numeric array for features (we will truncate/pad as necessary)
    arr = np.array(values, dtype=float)

    # apply winsorize cutoffs if available (cutoffs length corresponds to pre-base features)
    if cutoffs:
        lowers = cutoffs.get("lower", [])
        uppers = cutoffs.get("upper", [])
        for i in range(min(len(lowers), len(arr))):
            lo = lowers[i]
            hi = uppers[i]
            if lo is not None and arr[i] < lo:
                arr[i] = lo
            if hi is not None and arr[i] > hi:
                arr[i] = hi

    # Build feature vector expected by model
    if model_name == "hybrid_xgb":
        # compute base from first 9 oxides
        if len(arr) < 9:
            raise HTTPException(status_code=400, detail="Need at least first 9 oxide values (SiO2..TiO2) to compute baseAFT")
        base = formulaAFT(arr[:9])
        # match expected_n: if model expects more features (e.g., optional P2O5,S were present at training), we must pad/truncate accordingly
        # expected pre-base feature count:
        pre_base_n = expected_n - 1
        feat_pre = arr[:pre_base_n] if len(arr) >= pre_base_n else np.pad(arr, (0, pre_base_n - len(arr)), 'constant', constant_values=0)
        feat = np.append(feat_pre, base).reshape(1, -1)
    else:
        pre_base_n = expected_n
        feat_pre = arr[:pre_base_n] if len(arr) >= pre_base_n else np.pad(arr, (0, pre_base_n - len(arr)), 'constant', constant_values=0)
        feat = feat_pre.reshape(1, -1)

    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="model not found: " + model_path)
    model = joblib.load(model_path)
    pred = model.predict(feat)[0]
    pred = int(round(max(1000, min(1600, pred))))
    return {"prediction": pred, "model": model_name}

@app.get("/summary")
def summary():
    if not os.path.exists(SUMMARY_PATH):
        raise HTTPException(status_code=404, detail="No summary available")
    with open(SUMMARY_PATH,"r") as f:
        return json.load(f)

@app.get("/training_info")
def training_info():
    """
    Returns metadata about the last training run: training_path, timestamp, features used, preprocessing cutoffs, and saved models.
    """
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="No training metadata available")
    with open(METADATA_PATH,"r") as f:
        return json.load(f)