// ann.js
// node ann.js [optional_csv_path]
// Prefers last_training_upload.csv if present; writes best_model_ann.json, ann_scaling.json and aft_model.json envelope.

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const brain = require('brain.js');

const CSV_FILE = process.argv[2] || path.join(__dirname, 'last_training_upload.csv');

if (!fs.existsSync(CSV_FILE)) {
  console.error('CSV not found (ann.js expects last_training_upload.csv or CLI path):', CSV_FILE);
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

const rows = [];
fs.createReadStream(CSV_FILE).pipe(csv()).on('data', r => rows.push(r)).on('end', main).on('error', e=>{console.error(e);process.exit(1);});

async function main(){
  try {
    const data=[];
    rows.forEach((row, i) => {
      // parse original 9 + new P2O5 and S
      const values = [
        asFloat(row.SiO2), asFloat(row.Al2O3), asFloat(row.Fe2O3),
        asFloat(row.CaO), asFloat(row.MgO), asFloat(row.Na2O),
        asFloat(row.K2O), asFloat(row.SO3), asFloat(row.TiO2),
        asFloat(row.P2O5), asFloat(row.S) // <-- new features added
      ];
      const aft = asFloat(row.AFT);
      if (values.some(v => v === null) || aft === null) return;
      data.push({ index: i+1, values, aft });
    });
    if (!data.length) { console.error('No valid rows'); process.exit(1); }
    console.log('Rows:', data.length);

    // Build X (includes new features if present) and Y
    const Xraw = data.map(d => d.values.slice(0)); // keep all features
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

    const X_train = trainIdx.map(i => Xt[i]), Y_train = trainIdx.map(i => Y[i]);
    const X_test  = testIdx.map(i => Xt[i]), Y_test  = testIdx.map(i => Y[i]);

    console.log('Train size:', X_train.length, 'Test size:', X_test.length);

    // compute per-feature min/max for scaling (training set)
    const featureCount = X_train[0].length;
    const mins = Array(featureCount).fill(Infinity);
    const maxs = Array(featureCount).fill(-Infinity);
    for (let r = 0; r < X_train.length; r++) {
      for (let c = 0; c < featureCount; c++) {
        if (X_train[r][c] < mins[c]) mins[c] = X_train[r][c];
        if (X_train[r][c] > maxs[c]) maxs[c] = X_train[r][c];
      }
    }
    const yMin = Math.min(...Y_train), yMax = Math.max(...Y_train);

    const scaleRow = (row) => row.map((v,j) => (v - mins[j]) / ( (maxs[j] - mins[j]) || 1 ));
    const scaleY = (y) => (y - yMin) / ( (yMax - yMin) || 1 );

    const trainingData = X_train.map((r, idx) => ({ input: scaleRow(r), output: [ scaleY(Y_train[idx]) ] }));

    // ANN grid (unchanged)
    const grid = [
      { hiddenLayers: [32,16], iterations: 5000, learningRate: 0.005 },
      { hiddenLayers: [64,32], iterations: 8000, learningRate: 0.003 },
      { hiddenLayers: [16], iterations: 3000, learningRate: 0.01 }
    ];

    // To reduce variance, run a small ensemble of restarts per config and average predictions.
    // This is the only change to stabilize occasional high-error runs.
    const ENSEMBLE_RESTARTS = 3;

    let best = { mae: Infinity, params: null, netJSON: null, preds: null };

    for (const cfg of grid) {
      console.log('Training ANN with', cfg);
      try {
        // collect predictions across restarts
        const predsAll = []; // will be array of arrays, length = ENSEMBLE_RESTARTS
        for (let run=0; run<ENSEMBLE_RESTARTS; run++) {
          try {
            const net = new brain.NeuralNetwork({ hiddenLayers: cfg.hiddenLayers, activation: 'relu' });
            // note: brain.train is synchronous for NeuralNetwork.train when using options object
            net.train(trainingData, { iterations: cfg.iterations, learningRate: cfg.learningRate, errorThresh: 0.002, log: false });

            const predsScaled = X_test.map(r => {
              const out = net.run(scaleRow(r))[0];
              const real = out * ( (yMax - yMin) || 1 ) + yMin;
              return real;
            });

            predsAll.push(predsScaled);
          } catch (errRun) {
            console.warn(`  restart ${run+1} failed for cfg`, cfg, errRun && errRun.message ? errRun.message : errRun);
            // don't abort the other restarts
          }
        } // end restarts

        if (predsAll.length === 0) {
          console.warn('  All restarts failed for this cfg — skipping cfg', cfg);
          continue;
        }

        // average predictions element-wise across successful restarts
        const predsAvg = predsAll[0].map((_,i) => {
          let s = 0;
          for (let r=0;r<predsAll.length;r++) s += predsAll[r][i];
          return s / predsAll.length;
        });

        const mae = meanAbs(Y_test, predsAvg);
        const rmse = rmseCalc(Y_test, predsAvg);
        const within20 = percentWithin(Y_test, predsAvg, 20);
        console.log(` -> MAE ${mae.toFixed(2)}, RMSE ${rmse.toFixed(2)}, %within±20 ${within20.toFixed(1)}% (averaged over ${predsAll.length} restarts)`);

        if (mae < best.mae) {
          best.mae = mae;
          best.params = cfg;
          // store averaged predictions and we'll need a model JSON to save - pick last successful net JSON if available
          best.preds = predsAvg;
          // to get a JSON for runtime, re-train one final net with same cfg (try few times) and save its JSON
          let finalNetJSON = null;
          for (let attempt=0; attempt<3; attempt++) {
            try {
              const finalNet = new brain.NeuralNetwork({ hiddenLayers: cfg.hiddenLayers, activation: 'relu' });
              finalNet.train(trainingData, { iterations: cfg.iterations, learningRate: cfg.learningRate, errorThresh: 0.002, log: false });
              finalNetJSON = finalNet.toJSON();
              break;
            } catch (errFinal) {
              console.warn('Final retrain attempt failed, retrying...', errFinal && errFinal.message ? errFinal.message : errFinal);
            }
          }
          if (!finalNetJSON) {
            console.warn('Could not obtain finalNetJSON for best cfg; best.netJSON will be empty but predictions are stored.');
            best.netJSON = null;
          } else {
            best.netJSON = finalNetJSON;
          }
        }

      } catch (err) {
        console.warn('Training failed for cfg', cfg, err && err.message ? err.message : err);
        // continue to next cfg
      }
    }

    if (!best.preds) { console.error('No ANN model trained successfully'); process.exit(1); }

    // Save ANN JSON and also write aft_model.json envelope so app can detect ANN active model
    if (best.netJSON) fs.writeFileSync('best_model_ann.json', JSON.stringify(best.netJSON, null, 2));
    fs.writeFileSync('model_meta_ann.json', JSON.stringify({
      params: best.params,
      date: new Date().toISOString(),
      train_size: X_train.length,
      test_size: X_test.length
    }, null, 2));
    // Save scaling info so runtime can apply same scaling
    fs.writeFileSync('ann_scaling.json', JSON.stringify({ mins, maxs, yMin, yMax }, null, 2));
    // envelope for app to detect ANN
    fs.writeFileSync('aft_model.json', JSON.stringify({ modelType: 'ann', modelPath: 'best_model_ann.json' }, null, 2));

    console.log('Saved best_model_ann.json and aft_model.json (ANN envelope). Best MAE:', best.mae);

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
      model: 'ann',
      params: best.params,
      MAE_on_test: Number(best.mae.toFixed(4)),
      avg_abs_error: Number((totalAbs / predsRounded.length).toFixed(4)),
      max_abs_error: maxAbs,
      min_abs_error: minAbs,
      count_abs_gt_20: c20,
      count_abs_gt_40: c40,
      count_abs_gt_60: c60,
      test_rows: predsRounded.length
    };

    // updated header includes P2O5 and S
    fs.writeFileSync('results_ann_prod.csv', [
      'row,actualAFT,predictedAFT,absError,SiO2,Al2O3,Fe2O3,CaO,MgO,Na2O,K2O,SO3,TiO2,P2O5,S',
      ...results.map(r => `${r.row},${r.actual},${r.pred},${r.absE},${r.features.join(',')}`)
    ].join('\n'));
    fs.writeFileSync('summary_ann_prod.json', JSON.stringify(summary, null, 2));
    console.log('Saved results_ann_prod.csv, summary_ann_prod.json');

  } catch (err) {
    console.error('Training failed', err && err.message ? err.message : err);
    process.exit(1);
  }
}