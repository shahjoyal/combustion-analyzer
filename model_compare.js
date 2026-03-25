// model_compare.js

const fs = require('fs');
const csv = require('csv-parser');
const { RandomForestRegression } = require('ml-random-forest');
const brain = require('brain.js');
const path = require('path');
//const fs = require('fs');


// ---------------- UTILITIES ----------------

function asFloat(v) {
  const n = parseFloat(String(v).replace(/,/g,''));
  return Number.isFinite(n) ? n : null;
}

function mulberry32(a) {
  return function() {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

function seededShuffle(arr, seed=42) {
  const rnd = mulberry32(seed);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}


// ---------------- METRICS ----------------

function computeStats(preds, actuals, metaRows) {

  let totalAbs = 0;
  let maxError = -Infinity;
  let minError = Infinity;

  let c20=0,c40=0,c60=0;

  for (let i=0;i<preds.length;i++) {

    const err = Math.abs(preds[i] - actuals[i]);

    totalAbs += err;

    if(err > maxError) maxError = err;
    if(err < minError) minError = err;

    if(err > 20) c20++;
    if(err > 40) c40++;
    if(err > 60) c60++;
  }

  return {
    rows: preds.length,
    avgAbs: Number((totalAbs/preds.length).toFixed(2)),
    maxError: Number(maxError.toFixed(6)),
    minError: Number(minError.toFixed(6)),
    gt20: c20,
    gt40: c40,
    gt60: c60
  };
}


// ---------------- BASE FORMULA ----------------

// NOTE: formulaAFT destructures only the first 9 values:
// it will IGNORE any extra features appended to the input array (e.g. P2O5, S).
//new

// function formulaAFT(v) {
//   const [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2] = v;
//   const sum = SiO2 + Al2O3;

//   // High slagging (sum < 55)
//   if (sum < 55) {
//     return 1338.75
//       + 0.3645  * SiO2
//       + 0.2970  * Al2O3
//       - 3.33    * Fe2O3
//       - 0.447   * CaO
//       - 0.675   * MgO
//       - 0.9315  * (Na2O + K2O)
//       - 1.7     * SO3
//       - 1.1655  * TiO2;
//   }

//   // Medium slagging (55 <= sum < 75)
//   else if (sum < 75) {
//     return 1407.0
//       + 1.45     * SiO2
//       + 1.26355  * Al2O3
//       - 4.4215   * Fe2O3
//       - 5.735    * CaO
//       - 8.325    * MgO
//       - 7.49     * (Na2O + K2O)
//       - 3.885    * SO3
//       - 0.0945   * TiO2;
//   }

//   // Low slagging (sum >= 75)
//   else {
//     return 1334.75
//       + 2.3495  * SiO2
//       + 3.33    * Al2O3
//       - 2.5     * Fe2O3
//       - 5.735   * CaO
//       - 8.325   * MgO
//       - 1.08    * (Na2O + K2O)
//       - 3.145   * SO3
//       - 1.1655  * TiO2;
//   }
// }

// function formulaAFT(v) {
//   const [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2] = v;
//   const sum = SiO2 + Al2O3;

//   // High slagging (sum < 55)
//   if (sum < 55) {
//     return 1377.0
//       + 0.0243   * SiO2
//       + 0.0198   * Al2O3
//       - 3.582    * Fe2O3
//       - 0.0298   * CaO
//       - 0.0450   * MgO
//       - 12.3579  * (Na2O + K2O)
//       - 3.383    * SO3
//       - 1.2537   * TiO2;
//   }

//   // Medium slagging (55 <= sum < 75)
//   else if (sum < 75) {
//     return 1447.2
//       + 0.0145   * SiO2
//       + 1.35917  * Al2O3
//       - 4.7561   * Fe2O3
//       - 3.1000   * CaO
//       - 8.9550   * MgO
//       - 7.4900   * (Na2O + K2O)
//       - 4.1790   * SO3
//       - 0.0063   * TiO2;
//   }

//   // Low slagging (sum >= 75)
//   else {
//     return 1517.4
//       + 0.0127   * SiO2
//       + 1.8000   * Al2O3
//       - 4.9750   * Fe2O3
//       - 6.1690   * CaO
//       - 8.9550   * MgO
//       - 0.0720   * (Na2O + K2O)
//       - 3.3830   * SO3
//       - 1.2537   * TiO2;
//   }
// }

// v: array-like with oxides in this order:
// [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2]
// If the caller provides only a single "NaK" combined value at index 5,
// the code will treat K2O as 0 (so NaK = v[5] effectively).
// function formulaAFT(v) {
//   const [
//     SiO2 = 0,
//     Al2O3 = 0,
//     Fe2O3 = 0,
//     CaO = 0,
//     MgO = 0,
//     Na2O = 0,
//     K2O = 0,
//     SO3 = 0,
//     TiO2 = 0
//   ] = v;

//   const NaK = (Na2O || 0) + (K2O || 0);
//   const sumSiAl = SiO2 + Al2O3;

//   // Ranges mapping (non-overlapping):
//   // sum < 45          -> high
//   // 45 <= sum < 55    -> moderate to high
//   // 55 <= sum < 65    -> moderate
//   // 65 <= sum < 75    -> low to moderate
//   // sum >= 75         -> low

//   if (sumSiAl < 45) {
//     // HIGH
//     return 949.3601
//       + 0.5646   * SiO2
//       + 2.1826   * Al2O3
//       + 1.6339   * Fe2O3
//       + 6.3412   * CaO
//       + 8.6798   * MgO
//       + 5.0377   * NaK
//       + 0.7961   * SO3
//       - 2.1892   * TiO2;
//   } else if (sumSiAl < 55) {
//     // MODERATE TO HIGH
//     return 1124.2069
//       - 0.0797   * SiO2
//       + 4.2971   * Al2O3
//       - 0.1462   * Fe2O3
//       + 4.7533   * CaO
//       + 3.0594   * MgO
//       + 3.0923   * NaK
//       - 2.2814   * SO3
//       - 4.1400   * TiO2;
//   } else if (sumSiAl < 65) {
//     // MODERATE
//     return 1177.4801
//       + 0.1559   * SiO2
//       + 7.8153   * Al2O3
//       - 1.7562   * Fe2O3
//       + 1.9570   * CaO
//       - 4.3791   * MgO
//       - 2.8309   * NaK
//       - 4.1712   * SO3
//       + 1.7976   * TiO2;
//   } else if (sumSiAl < 75) {
//     // LOW TO MODERATE
//     return 1140.8756
//       + 3.0803   * SiO2
//       + 6.9504   * Al2O3
//       - 1.9173   * Fe2O3
//       - 1.7021   * CaO
//       - 9.6014   * MgO
//       - 5.8391   * NaK
//       - 4.5557   * SO3
//       + 4.9372   * TiO2;
//   } else {
//     // LOW (sumSiAl >= 75)
//     return 1619.5948
//       - 1.1828   * SiO2
//       + 0.5515   * Al2O3
//       - 6.6603   * Fe2O3
//       - 12.7041  * CaO
//       - 20.8388  * MgO
//       + 4.6097   * NaK
//       - 5.2634   * SO3
//       + 10.5468  * TiO2;
//   }
// }

// v: [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2]
// returns numeric AFT
// function formulaAFT(v) {
//   const [
//     SiO2 = 0,
//     Al2O3 = 0,
//     Fe2O3 = 0,
//     CaO = 0,
//     MgO = 0,
//     Na2O = 0,
//     K2O = 0,
//     SO3 = 0,
//     TiO2 = 0
//   ] = v;

//   const NaK = (Na2O || 0) + (K2O || 0);
//   const sumSiAl = SiO2 + Al2O3;
//   const FeCa = (Fe2O3 || 0) + (CaO || 0);
//   const FECA_THRESHOLD = 12.6;

//   // Range bins:
//   // sum < 45          -> high  (h1/h2)
//   // 45 <= sum < 55    -> moderate to high (mh1/mh2)
//   // 55 <= sum < 65    -> moderate (m1/m2)
//   // 65 <= sum < 75    -> low to moderate (ml1/ml2)
//   // sum >= 75         -> low (l1/l2)

//   // ---- HIGH (h1/h2) ----
//   if (sumSiAl < 45) {
//     if (FeCa <= FECA_THRESHOLD) {
//       // h1: formula not provided -> FALLBACK to the available HIGH formula (h2)
//       // If you later supply a specific h1 formula, replace this block.
//       return 949.3601
//         + 0.5646   * SiO2
//         + 2.1826   * Al2O3
//         + 1.6339   * Fe2O3
//         + 6.3412   * CaO
//         + 8.6798   * MgO
//         + 5.0377   * NaK
//         + 0.7961   * SO3
//         - 2.1892   * TiO2;
//     } else {
//       // h2 (you provided)
//       return 949.3601
//         + 0.5646   * SiO2
//         + 2.1826   * Al2O3
//         + 1.6339   * Fe2O3
//         + 6.3412   * CaO
//         + 8.6798   * MgO
//         + 5.0377   * NaK
//         + 0.7961   * SO3
//         - 2.1892   * TiO2;
//     }
//   }

//   // ---- MODERATE TO HIGH (mh1/mh2) ----
//   else if (sumSiAl < 55) {
//     if (FeCa <= FECA_THRESHOLD) {
//       // mh1: formula not provided -> FALLBACK to available moderate-to-high formula (mh2)
//       // Replace when you have a distinct mh1 formula.
//       return 1124.2069
//         - 0.0797   * SiO2
//         + 4.2971   * Al2O3
//         - 0.1462   * Fe2O3
//         + 4.7533   * CaO
//         + 3.0594   * MgO
//         + 3.0923   * NaK
//         - 2.2814   * SO3
//         - 4.1400   * TiO2;
//     } else {
//       // mh2 (you provided)
//       return 1124.2069
//         - 0.0797   * SiO2
//         + 4.2971   * Al2O3
//         - 0.1462   * Fe2O3
//         + 4.7533   * CaO
//         + 3.0594   * MgO
//         + 3.0923   * NaK
//         - 2.2814   * SO3
//         - 4.1400   * TiO2;
//     }
//   }

//   // ---- MODERATE (m1/m2) ----
//   else if (sumSiAl < 65) {
//     if (FeCa <= FECA_THRESHOLD) {
//       // m1 (you provided)
//       return 1309.86
//         - 0.8026   * SiO2
//         + 1.2965   * Al2O3
//         + 2.7164   * Fe2O3
//         - 2.9016   * CaO
//         + 0.4322   * MgO
//         + 2.4479   * NaK
//         - 2.5621   * SO3
//         + 0.0      * TiO2;
//     } else {
//       // m2 (you provided)
//       return 1181.3094
//         - 0.4266   * SiO2
//         + 8.1218   * Al2O3
//         - 1.2093   * Fe2O3
//         + 2.9888   * CaO
//         - 4.6965   * MgO
//         - 4.7050   * NaK
//         - 3.9495   * SO3
//         + 1.1836   * TiO2;
//     }
//   }

//   // ---- LOW TO MODERATE (ml1/ml2) ----
//   else if (sumSiAl < 75) {
//     if (FeCa <= FECA_THRESHOLD) {
//       // ml1 (you provided)
//       return 1402.7002
//         - 0.8606   * SiO2
//         + 3.2272   * Al2O3
//         - 1.0213   * Fe2O3
//         - 0.6618   * CaO
//         - 1.4658   * MgO
//         - 4.1053   * NaK
//         - 3.1920   * SO3
//         - 0.1844   * TiO2;
//     } else {
//       // ml2 (you provided)
//       return 1021.4717
//         + 3.8985   * SiO2
//         + 7.1950   * Al2O3
//         + 2.1680   * Fe2O3
//         + 2.4863   * CaO
//         - 7.8374   * MgO
//         - 8.0358   * NaK
//         - 4.1760   * SO3
//         + 5.7583   * TiO2;
//     }
//   }

//   // ---- LOW (l1/l2) ----
//   else {
//     if (FeCa <= FECA_THRESHOLD) {
//       // l1 (you provided)
//       return 1515.9840
//         - 0.0582   * SiO2
//         + 1.5964   * Al2O3
//         - 3.8771   * Fe2O3
//         - 10.1779  * CaO
//         - 13.1584  * MgO
//         + 2.9112   * NaK
//         - 7.4455   * SO3
//         + 4.9705   * TiO2;
//     } else {
//       // l2 (you provided)
//       return 1430.9261
//         - 2.5702   * SiO2
//         + 2.6075   * Al2O3
//         + 4.6287   * Fe2O3
//         - 2.6753   * CaO
//         - 1.1602   * MgO
//         + 1.2016   * NaK
//         + 1.6603   * SO3
//         + 2.9714   * TiO2;
//     }
//   }
// }


// function formulaAFT(v) {
//   const [
//     SiO2 = 0,
//     Al2O3 = 0,
//     Fe2O3 = 0,
//     CaO = 0,
//     MgO = 0,
//     Na2O = 0,
//     K2O = 0,
//     SO3 = 0,
//     TiO2 = 0
//   ] = v;

//   const NaK = (Na2O || 0) + (K2O || 0);
//   const sumSiAl = SiO2 + Al2O3;
//   const FeCa = (Fe2O3 || 0) + (CaO || 0);
//   const FECA_THRESHOLD = 12.6;

//   if (sumSiAl < 45) {
//     if (FeCa <= FECA_THRESHOLD) {
//       return 949.3601
//         + 0.5646   * SiO2
//         + 2.1826   * Al2O3
//         + 1.6339   * Fe2O3
//         + 6.3412   * CaO
//         + 8.6798   * MgO
//         + 5.0377   * NaK
//         + 0.7961   * SO3
//         - 2.1892   * TiO2;
//     } else {
//       return 949.3601
//         + 0.5646   * SiO2
//         + 2.1826   * Al2O3
//         + 1.6339   * Fe2O3
//         + 6.3412   * CaO
//         + 8.6798   * MgO
//         + 5.0377   * NaK
//         + 0.7961   * SO3
//         - 2.1892   * TiO2;
//     }
//   } else if (sumSiAl < 55) {
//     if (FeCa <= FECA_THRESHOLD) {
//       return 1139.9324
//         - 0.2734   * SiO2
//         + 4.5495   * Al2O3
//         - 0.5943   * Fe2O3
//         + 4.4863   * CaO
//         + 2.9609   * MgO
//         + 3.6016   * NaK
//         - 2.4116   * SO3
//         - 3.7743   * TiO2;
//     } else {
//       return 1139.9324
//         - 0.2734   * SiO2
//         + 4.5495   * Al2O3
//         - 0.5943   * Fe2O3
//         + 4.4863   * CaO
//         + 2.9609   * MgO
//         + 3.6016   * NaK
//         - 2.4116   * SO3
//         - 3.7743   * TiO2;
//     }
//   } else if (sumSiAl < 65) {
//     if (FeCa <= FECA_THRESHOLD) {
//       return 1137.4961
//         + 1.1956   * SiO2
//         + 3.5062   * Al2O3
//         + 3.3419   * Fe2O3
//         - 12.5158  * CaO
//         - 2.2775   * MgO
//         + 0.0163   * NaK
//         + 6.7002   * SO3
//         + 1.9344   * TiO2;
//     } else {
//       return 1149.2441
//         - 0.5063   * SiO2
//         + 8.8672   * Al2O3
//         - 1.1065   * Fe2O3
//         + 3.3827   * CaO
//         - 1.9235   * MgO
//         - 0.3508   * NaK
//         - 4.5738   * SO3
//         + 1.6308   * TiO2;
//     }
//   } else if (sumSiAl < 75) {
//     if (FeCa <= FECA_THRESHOLD) {
//       return 1194.5549
//         + 2.7079   * SiO2
//         + 6.4134   * Al2O3
//         - 3.0781   * Fe2O3
//         - 6.7834   * CaO
//         - 6.4518   * MgO
//         - 5.4799   * NaK
//         - 2.4833   * SO3
//         + 2.2483   * TiO2;
//     } else {
//       return 1025.9764
//         + 3.2181   * SiO2
//         + 6.3438   * Al2O3
//         + 4.2982   * Fe2O3
//         + 5.4756   * CaO
//         - 5.5719   * MgO
//         - 6.8764   * NaK
//         - 5.3138   * SO3
//         + 4.3039   * TiO2;
//     }
//   } else {
//     if (FeCa <= FECA_THRESHOLD) {
//       return 1638.5785
//         - 1.3546   * SiO2
//         + 0.3045   * Al2O3
//         - 6.5022   * Fe2O3
//         - 13.7161  * CaO
//         - 20.4375  * MgO
//         + 4.0117   * NaK
//         - 4.9265   * SO3
//         + 10.4268  * TiO2;
//     } else {
//       return 1303.6482
//         + 1.2428   * SiO2
//         + 0.6550   * Al2O3
//         + 2.2788   * Fe2O3
//         + 0.9765   * CaO
//         - 1.6222   * MgO
//         + 0.2739   * NaK
//         - 2.3258   * SO3
//         + 0.2792   * TiO2;
//     }
//   }
// }

function formulaAFT(v) {
  const [
    SiO2 = 0,
    Al2O3 = 0,
    Fe2O3 = 0,
    CaO = 0,
    MgO = 0,
    Na2O = 0,
    K2O = 0,
    SO3 = 0,
    TiO2 = 0
  ] = v;

  const NaK = (Na2O || 0) + (K2O || 0);
  const sumSiAl = SiO2 + Al2O3;
  const FeCa = (Fe2O3 || 0) + (CaO || 0);
  const FECA_THRESHOLD = 12.6;

  if (sumSiAl < 45) {
    if (FeCa <= FECA_THRESHOLD) {
      return 949.3601
        + 0.5646 * SiO2
        + 2.1826 * Al2O3
        + 1.6339 * Fe2O3
        + 6.3412 * CaO
        + 8.6798 * MgO
        + 5.0377 * NaK
        + 0.7961 * SO3
        - 2.1892 * TiO2;
    } else {
      return 949.3601
        + 0.5646 * SiO2
        + 2.1826 * Al2O3
        + 1.6339 * Fe2O3
        + 6.3412 * CaO
        + 8.6798 * MgO
        + 5.0377 * NaK
        + 0.7961 * SO3
        - 2.1892 * TiO2;
    }
  } else if (sumSiAl < 55) {
    if (FeCa <= FECA_THRESHOLD) {
      return 1124.2069
        - 0.0797 * SiO2
        + 4.2971 * Al2O3
        - 0.1462 * Fe2O3
        + 4.7533 * CaO
        + 3.0594 * MgO
        + 3.0923 * NaK
        - 2.2814 * SO3
        - 4.1400 * TiO2;
    } else {
      return 1124.2069
        - 0.0797 * SiO2
        + 4.2971 * Al2O3
        - 0.1462 * Fe2O3
        + 4.7533 * CaO
        + 3.0594 * MgO
        + 3.0923 * NaK
        - 2.2814 * SO3
        - 4.1400 * TiO2;
    }
  } else if (sumSiAl < 65) {
    if (FeCa <= FECA_THRESHOLD) {
      return 1137.4961
        + 1.1956 * SiO2
        + 3.5062 * Al2O3
        + 3.3419 * Fe2O3
        - 12.5158 * CaO
        - 2.2775 * MgO
        + 0.0163 * NaK
        + 6.7002 * SO3
        + 1.9344 * TiO2;
    } else {
      return 1177.4801
        + 0.1559 * SiO2
        + 7.8153 * Al2O3
        - 1.7562 * Fe2O3
        + 1.9570 * CaO
        - 4.3791 * MgO
        - 2.8309 * NaK
        - 4.1712 * SO3
        + 1.7976 * TiO2;
    }
  } else if (sumSiAl < 75) {
    if (FeCa <= FECA_THRESHOLD) {
      return -1452.0645
        + 45.6395 * SiO2
        + 43.3054 * Al2O3
        - 31.2287 * Fe2O3
        - 67.6606 * CaO
        - 37.1564 * MgO
        - 28.9117 * NaK
        + 36.5424 * SO3
        + 1.4591 * TiO2;
    } else {
      return 1103.1611
        + 3.1111 * SiO2
        + 6.8292 * Al2O3
        + 0.2934 * Fe2O3
        + 0.8048 * CaO
        - 8.8943 * MgO
        - 6.3487 * NaK
        - 4.9725 * SO3
        + 4.9459 * TiO2;
    }
  } else {
    if (FeCa <= FECA_THRESHOLD) {
      return 1468.9852
        + 0.4432 * SiO2
        + 2.0587 * Al2O3
        - 3.8776 * Fe2O3
        - 6.5077 * CaO
        - 8.5353 * MgO
        + 3.2333 * NaK
        - 10.3822 * SO3
        + 3.3885 * TiO2;
    } else {
      return 1109.9520
        + 1.4242 * SiO2
        + 7.7724 * Al2O3
        + 3.3740 * Fe2O3
        - 8.8021 * CaO
        - 15.7613 * MgO
        + 6.8226 * NaK
        + 9.8592 * SO3
        + 22.3574 * TiO2;
    }
  }
}

//sial sums
//above 75 -> low
//65 to 75 -> low to moderate
//55 to 65->moderate
//45 to 55->moderate to high
//below 45-> high

// ---------------- MODELS ----------------


function trainPureRF(X,Y,trainIdx,testIdx){

  const rf = new RandomForestRegression({
    nEstimators:150,
    maxDepth:12,
    minNumSamples:3,
    seed:42
  });

  rf.train(trainIdx.map(i=>X[i]),trainIdx.map(i=>Y[i]));

  const preds = rf.predict(testIdx.map(i=>X[i]));
  const actual = testIdx.map(i=>Y[i]);

  return computeStats(preds,actual);
}



// hybrid RF (formula as extra feature)

function trainHybridRF(X,Y,trainIdx,testIdx){

  const X_hybrid = X.map(v=>[...v,formulaAFT(v)]);

  const rf = new RandomForestRegression({
    nEstimators:150,
    maxDepth:12,
    minNumSamples:3,
    seed:42
  });

  rf.train(trainIdx.map(i=>X_hybrid[i]),trainIdx.map(i=>Y[i]));

  const preds = rf.predict(testIdx.map(i=>X_hybrid[i]));
  const actual = testIdx.map(i=>Y[i]);

  return computeStats(preds,actual);
}



// residual hybrid (BEST approach usually)

function trainResidualHybridRF(X,Y,trainIdx,testIdx){

  const baseTrain = trainIdx.map(i=>formulaAFT(X[i]));
  const baseTest = testIdx.map(i=>formulaAFT(X[i]));

  const residualY = trainIdx.map((i,j)=>Y[i]-baseTrain[j]);

  const rf = new RandomForestRegression({
    nEstimators:150,
    maxDepth:12,
    minNumSamples:3,
    seed:42
  });

  rf.train(trainIdx.map(i=>X[i]),residualY);

  const residualPred = rf.predict(testIdx.map(i=>X[i]));

  const preds = residualPred.map((r,i)=>r + baseTest[i]);
  const actual = testIdx.map(i=>Y[i]);

  return computeStats(preds,actual);
}



// ANN

function trainANN(X,Y,trainIdx,testIdx){

  const net = new brain.NeuralNetwork({
    hiddenLayers:[32,16]
  });

  const training = trainIdx.map(i=>({
    input:X[i],
    output:[Y[i]/2000]
  }));

  net.train(training,{iterations:5000,log:false});

  const preds = testIdx.map(i=>net.run(X[i])[0]*2000);
  const actual = testIdx.map(i=>Y[i]);

  return computeStats(preds,actual);
}



// base formula

function runBaseFormula(X,Y,testIdx){

  const preds = testIdx.map(i=>formulaAFT(X[i]));
  const actual = testIdx.map(i=>Y[i]);

  return computeStats(preds,actual);
}



// ---------------- MAIN ----------------


async function runAllModels(datasetPath, options = {}) {
  const { onProgress } = options;

  const rows = [];
  await new Promise((resolve, reject) => {
    fs.createReadStream(datasetPath)
      .pipe(csv())
      .on("data", (row) => rows.push(row))
      .on("end", resolve)
      .on("error", reject);
  });

  if (rows.length === 0) {
    throw new Error("Dataset empty");
  }

 
  const featureNames = [
    "SiO2","Al2O3","Fe2O3","CaO","MgO",
    "Na2O","K2O","SO3","TiO2",
    "P2O5","S"
  ];
  const target = "AFT";

  const X = [];
  const Y = [];

  for (const r of rows) {
  
    const rowLower = {};
    for (const k in r) rowLower[k.trim().toLowerCase()] = r[k];

    const v = featureNames.map(f => asFloat(rowLower[f.toLowerCase()]));
    const y = asFloat(rowLower[target.toLowerCase()]);

    if (v.some(a => a === null) || y === null) continue;

    X.push(v);
    Y.push(y);
  }

  console.log("runAllModels: valid rows parsed =", X.length);

  if (X.length < 10) {
    throw new Error("Not enough valid rows");
  }

  const idx = seededShuffle([...Array(X.length).keys()], 42);
  const split = Math.floor(idx.length * 0.8);
  const trainIdx = idx.slice(0, split);
  const testIdx = idx.slice(split);

  // collect results with both internal and UI-friendly keys
  const results = {};

  // Pure RF
  try {
    onProgress?.({ event: "rf_start" });
    console.log("runAllModels: starting Pure RF");
    const pureRfStats = trainPureRF(X, Y, trainIdx, testIdx);
    results.rf = pureRfStats;           // old key
    results.pureRF = pureRfStats;       // UI key
    onProgress?.({ event: "rf_done" });
    console.log("runAllModels: pure RF done");
  } catch (e) {
    console.error("RF error", e);
  }

  // Hybrid RF (formula as extra feature)
  try {
    onProgress?.({ event: "hybrid_rf_start" });
    console.log("runAllModels: starting Hybrid RF");
    const hybridStats = trainHybridRF(X, Y, trainIdx, testIdx);
    results.hybridRF = hybridStats;     // both names already match UI
    onProgress?.({ event: "hybrid_rf_done" });
    console.log("runAllModels: hybrid RF done");
  } catch (e) {
    console.error("Hybrid RF error", e);
  }

  // Residual hybrid RF
  try {
    onProgress?.({ event: "residual_rf_start" });
    console.log("runAllModels: starting Residual Hybrid RF");
    const residualStats = trainResidualHybridRF(X, Y, trainIdx, testIdx);
    results.residualRF = residualStats;
    onProgress?.({ event: "residual_rf_done" });
    console.log("runAllModels: residual RF done");
  } catch (e) {
    console.error("Residual RF error", e);
  }

  // ANN
  try {
    onProgress?.({ event: "ann_start" });
    console.log("runAllModels: starting ANN");
    const annStats = trainANN(X, Y, trainIdx, testIdx);
    results.ann = annStats;
    onProgress?.({ event: "ann_done" });
    console.log("runAllModels: ANN done");
  } catch (e) {
    console.error("ANN error", e);
  }

  // Base formula
  try {
    onProgress?.({ event: "formula_start" });
    console.log("runAllModels: starting Base Formula");
    const formulaStats = runBaseFormula(X, Y, testIdx);
    results.formula = formulaStats;         // old key
    results.baseFormula = formulaStats;     // UI key
    onProgress?.({ event: "formula_done" });
    console.log("runAllModels: Base Formula done");
  } catch (e) {
    console.error("Formula error", e);
  }

  return results;
}

module.exports={runAllModels};