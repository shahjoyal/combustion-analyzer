// hybrid.js
// node hybrid.js [optional_csv_path]
// Prefers last_training_upload.csv if present; otherwise accepts CLI path.

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { RandomForestRegression } = require('ml-random-forest');

const CSV_FILE = process.argv[2] || path.join(__dirname, 'last_training_upload.csv');

if (!fs.existsSync(CSV_FILE)) {
  console.error('CSV not found (hybrid.js expects last_training_upload.csv or CLI path):', CSV_FILE);
  process.exit(1);
}

function asFloat(v) {
  if (v === null || v === undefined) return null;
  const n = parseFloat(String(v).replace(/,/g, ''));
  return Number.isFinite(n) ? n : null;
}
function quantile(sortedArr, q) {
  if (!sortedArr.length) return 0;
  if (q <= 0) return sortedArr[0];
  if (q >= 1) return sortedArr[sortedArr.length-1];
  const pos = (sortedArr.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sortedArr[base+1] !== undefined) {
    return sortedArr[base] + rest * (sortedArr[base+1] - sortedArr[base]);
  } else {
    return sortedArr[base];
  }
}
function meanAbs(a,b){ let s=0; for(let i=0;i<a.length;i++) s+=Math.abs(a[i]-b[i]); return s/a.length; }
function rmseCalc(a,b){ let s=0; for(let i=0;i<a.length;i++) s+=Math.pow(a[i]-b[i],2); return Math.sqrt(s/a.length); }
function percentWithin(a,b,tol){ let c=0; for(let i=0;i<a.length;i++) if (Math.abs(a[i]-b[i])<=tol) c++; return (c/a.length)*100; }
function shuffleArray(arr){ for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; }

// function formulaAFT(v) {
//   const [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2] = v;
//   const sum = SiO2 + Al2O3;
//   if (sum < 55)
//     return 1245 + 1.1*SiO2 + 0.95*Al2O3 - 2.5*Fe2O3 - 2.98*CaO - 4.5*MgO - 7.89*(Na2O+K2O) - 1.7*SO3 - 0.63*TiO2;
//   else if (sum < 75)
//     return 1323 + 1.45*SiO2 + 0.683*Al2O3 - 2.39*Fe2O3 - 3.1*CaO - 4.5*MgO - 7.49*(Na2O+K2O) - 2.1*SO3 - 0.63*TiO2;
//   else
//     return 1395 + 1.2*SiO2 + 0.9*Al2O3 - 2.5*Fe2O3 - 3.1*CaO - 4.5*MgO - 7.2*(Na2O+K2O) - 1.7*SO3 - 0.63*TiO2;
// }


function formulaAFT(v) {
  const [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2] = v;
  const sum = SiO2 + Al2O3;

  // High slagging (sum < 55)
  if (sum < 55) {
    return 1338.75
      + 0.3645  * SiO2
      + 0.2970  * Al2O3
      - 3.33    * Fe2O3
      - 0.447   * CaO
      - 0.675   * MgO
      - 0.9315  * (Na2O + K2O)
      - 1.7     * SO3
      - 1.1655  * TiO2;
  }

  // Medium slagging (55 <= sum < 75)
  else if (sum < 75) {
    return 1407.0
      + 1.45     * SiO2
      + 1.26355  * Al2O3
      - 4.4215   * Fe2O3
      - 5.735    * CaO
      - 8.325    * MgO
      - 7.49     * (Na2O + K2O)
      - 3.885    * SO3
      - 0.0945   * TiO2;
  }

  // Low slagging (sum >= 75)
  else {
    return 1334.75
      + 2.3495  * SiO2
      + 3.33    * Al2O3
      - 2.5     * Fe2O3
      - 5.735   * CaO
      - 8.325   * MgO
      - 1.08    * (Na2O + K2O)
      - 3.145   * SO3
      - 1.1655  * TiO2;
  }
}


const rows = [];
fs.createReadStream(CSV_FILE).pipe(csv()).on('data', r => rows.push(r)).on('end', main).on('error', e=>{console.error(e);process.exit(1);});

function main(){
  const data=[];
  rows.forEach((row, i) => {
    const values = [
      asFloat(row.SiO2), asFloat(row.Al2O3), asFloat(row.Fe2O3),
      asFloat(row.CaO), asFloat(row.MgO), asFloat(row.Na2O),
      asFloat(row.K2O), asFloat(row.SO3), asFloat(row.TiO2)
    ];
    const aft = asFloat(row.AFT);
    if (values.some(v => v === null) || aft === null) return;
    data.push({ index: i+1, values, aft });
  });
  if (!data.length) { console.error('No valid rows'); process.exit(1); }
  console.log('Rows:', data.length);

  // Build X (9 features) and Y
  const Xraw = data.map(d => d.values.slice(0,9));
  const Y = data.map(d => d.aft);

  // Winsorize each feature at 1%/99% and save cutoffs
  const Xt = Xraw.map(r => r.slice());
  const cutoffs = { lower: [], upper: [] };
  for (let c = 0; c < Xt[0].length; c++) {
    const col = Xt.map(r => r[c]).slice().sort((a,b)=>a-b);
    const lo = quantile(col, 0.01), hi = quantile(col, 0.99);
    cutoffs.lower.push(lo); cutoffs.upper.push(hi);
    for (let r = 0; r < Xt.length; r++) {
      if (Xt[r][c] < lo) Xt[r][c] = lo;
      if (Xt[r][c] > hi) Xt[r][c] = hi;
    }
  }
  fs.writeFileSync('preprocessing.json', JSON.stringify(cutoffs, null, 2));
  console.log('Saved preprocessing.json');

  // Shuffle & split (80/20) — save test indices for reproducibility
  const ids = shuffleArray(Array.from({length: Xt.length}, (_,i) => i));
  const split = Math.floor(0.8 * ids.length);
  const trainIdx = ids.slice(0, split), testIdx = ids.slice(split);
  fs.writeFileSync('test_indices.json', JSON.stringify({ testIdx, trainIdx }, null, 2));

  // Build hybrid features (append formulaAFT as 10th feature)
  const X_hybrid = Xraw.map(row => {
    const base = formulaAFT(row);
    return [...row, base];
  });

  const X_train = trainIdx.map(i => X_hybrid[i]), Y_train = trainIdx.map(i => Y[i]);
  const X_test  = testIdx.map(i => X_hybrid[i]), Y_test  = testIdx.map(i => Y[i]);

  console.log('Train size:', X_train.length, 'Test size:', X_test.length);

  // grid
  const grid = [
    { nEstimators: 100, maxDepth: 10, minNumSamples: 4, replacement: false },
    { nEstimators: 150, maxDepth: 12, minNumSamples: 3, replacement: false },
    { nEstimators: 200, maxDepth: 14, minNumSamples: 2, replacement: false }
  ];

  let best = { mae: Infinity, params: null, modelJSON: null, preds: null };

  for (const cfg of grid) {
    console.log('Training', cfg);
    const rf = new RandomForestRegression({
      nEstimators: cfg.nEstimators,
      maxFeatures: Math.max(1, Math.floor(Math.sqrt(X_train[0].length))),
      maxDepth: cfg.maxDepth,
      minNumSamples: cfg.minNumSamples,
      replacement: !!cfg.replacement,
      seed: 42
    });
    rf.train(X_train, Y_train);
    const preds = rf.predict(X_test);
    const mae = meanAbs(Y_test, preds);
    const rmse = rmseCalc(Y_test, preds);
    const within20 = percentWithin(Y_test, preds, 20);
    console.log(` -> MAE ${mae.toFixed(2)}, RMSE ${rmse.toFixed(2)}, %within±20 ${within20.toFixed(1)}%`);
    if (mae < best.mae) {
      best.mae = mae; best.params = cfg; best.modelJSON = rf.toJSON(); best.preds = preds;
    }
  }

  if (!best.modelJSON) { console.error('No model trained'); process.exit(1); }

  // Save model JSON to aft_model.json (app.js loads this file name)
  fs.writeFileSync('aft_model.json', JSON.stringify(best.modelJSON, null, 2));
  fs.writeFileSync('model_meta.json', JSON.stringify({
    params: best.params,
    date: new Date().toISOString(),
    train_size: trainIdx.length,
    test_size: testIdx.length
  }, null, 2));
  console.log('Saved aft_model.json and model_meta.json. Best MAE:', best.mae);

  // per-row analysis (rounded predictions)
  const predsRounded = best.preds.map(p => Math.round(p));
  const results = [];
  let maxAbs=-Infinity, minAbs=Infinity, totalAbs=0, c20=0,c40=0,c60=0;
  for (let i=0;i<predsRounded.length;i++){
    const idx = testIdx[i];
    const actual = Y_test[i];
    const pred = predsRounded[i];
    const absE = Math.abs(actual - pred);
    totalAbs += absE;
    if (absE > maxAbs) maxAbs = absE;
    if (absE < minAbs) minAbs = absE;
    if (absE > 20) c20++; if (absE>40) c40++; if (absE>60) c60++;
    results.push({ row: data[idx].index, actual, pred, absE, features: data[idx].values });
  }

  const summary = {
    model: 'hybrid_rf',
    params: best.params,
    MAE_on_test: Number(best.mae.toFixed(4)),
    avg_abs_error: Number((totalAbs / predsRounded.length).toFixed(4)),
    max_abs_error: maxAbs, min_abs_error: minAbs,
    count_abs_gt_20: c20, count_abs_gt_40: c40, count_abs_gt_60: c60,
    test_rows: predsRounded.length
  };

  fs.writeFileSync('results_hybrid_rf_prod.csv', [
    'row,actualAFT,predictedAFT,absError,SiO2,Al2O3,Fe2O3,CaO,MgO,Na2O,K2O,SO3,TiO2',
    ...results.map(r => `${r.row},${r.actual},${r.pred},${r.absE},${r.features.join(',')}`)
  ].join('\n'));
  fs.writeFileSync('summary_hybrid_rf_prod.json', JSON.stringify(summary, null, 2));
  fs.writeFileSync('best_model_hybrid_rf.json', JSON.stringify(best.modelJSON, null, 2));
  console.log('Saved results_hybrid_rf_prod.csv, summary_hybrid_rf_prod.json, best_model_hybrid_rf.json');
}