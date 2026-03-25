// app.js - unified server for both sites (Express + MongoDB)
// Install dependencies:
// npm i express body-parser mongoose multer xlsx exceljs dotenv cors express-session bcryptjs

const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const xlsx = require('xlsx');
const multer = require('multer');
const mongoose = require('mongoose');
const ExcelJS = require('exceljs');
const cors = require('cors');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const fs = require("fs");
const { RandomForestRegression } = require("ml-random-forest");
const { runAllModels } = require('./model_compare');
const { spawn } = require('child_process');

require('dotenv').config();



const app = express();

// --- active model loader (replaces direct aft_model.json load) ---
const brain = require('brain.js'); // ANN support

// in-memory active model container
// shape: { type: 'pure_rf'|'hybrid_rf'|'ann'|'base_formula', model: <loaded model>, raw: <raw json/meta>, annScaling: <optional> }
let activeModel = { type: 'pure_rf', model: null, raw: null, annScaling: null };

function loadActiveModel() {
  try {
    // prefer explicit active_model.json (written when user activates)
    let active = null;
    if (fs.existsSync('active_model.json')) {
      try { active = JSON.parse(fs.readFileSync('active_model.json', 'utf8')); }
      catch(e) { console.warn('active_model.json exists but could not be parsed', e && e.message); }
    }

    if (!active || !active.type) {
      // fallback: if aft_model.json exists we'll try to infer it's an RF model
      if (fs.existsSync('aft_model.json')) {
        try {
          const aftJ = JSON.parse(fs.readFileSync('aft_model.json', 'utf8'));
          // if aft_model.json looks like an ANN envelope, detect it
          if (aftJ && aftJ.modelType === 'ann' && aftJ.modelPath) {
            active = { type: 'ann', modelPath: aftJ.modelPath };
          } else {
            active = { type: 'pure_rf' };
          }
        } catch (e) {
          console.warn('Failed parsing aft_model.json fallback:', e && e.message);
          active = { type: 'base_formula' };
        }
      } else {
        active = { type: 'base_formula' };
      }
    }

    activeModel.type = active.type;

    if (activeModel.type === 'pure_rf' || activeModel.type === 'hybrid_rf') {
      // load RandomForest JSON from aft_model.json
      if (!fs.existsSync('aft_model.json')) {
        console.warn('aft_model.json not found for RF model load.');
        activeModel.model = null;
        activeModel.raw = null;
        return;
      }
      const aftJSON = JSON.parse(fs.readFileSync('aft_model.json', 'utf8'));
      activeModel.raw = aftJSON;
      activeModel.model = RandomForestRegression.load(aftJSON);
      console.log('Loaded RF model into memory (type):', activeModel.type);
      return;
    }

    if (activeModel.type === 'ann') {
      // Expect aft_model.json to be an envelope: { modelType: 'ann', modelPath: 'best_model_ann.json' }
      let annPath = null;
      if (fs.existsSync('aft_model.json')) {
        try {
          const aft = JSON.parse(fs.readFileSync('aft_model.json', 'utf8'));
          if (aft && aft.modelPath) annPath = path.join(__dirname, aft.modelPath);
        } catch(e) { /* ignore */ }
      }
      // fallback default filename
      if (!annPath) annPath = path.join(__dirname, 'best_model_ann.json');

      if (!fs.existsSync(annPath)) {
        console.warn('ANN model JSON not found at', annPath);
        activeModel.model = null;
        activeModel.raw = null;
        return;
      }

      const netJson = JSON.parse(fs.readFileSync(annPath, 'utf8'));
      const net = new brain.NeuralNetwork();
      // restore network
      if (typeof net.fromJSON === 'function') net.fromJSON(netJson);
      else if (typeof net.fromJSON === 'function') net.fromJSON(netJson); // defensive
      activeModel.model = net;
      activeModel.raw = netJson;
      // try to load optional ann_scaling.json for min/max used at training time
      try {
        const scalingPath = path.join(__dirname, 'ann_scaling.json');
        if (fs.existsSync(scalingPath)) {
          activeModel.annScaling = JSON.parse(fs.readFileSync(scalingPath, 'utf8'));
          console.log('Loaded ANN scaling info (ann_scaling.json)');
        } else {
          activeModel.annScaling = null;
        }
      } catch (e) {
        activeModel.annScaling = null;
      }

      console.log('Loaded ANN model into memory from', annPath);
      return;
    }

    if (activeModel.type === 'base_formula') {
      activeModel.model = null;
      activeModel.raw = null;
      activeModel.annScaling = null;
      console.log('Active model set to base_formula (no ML model loaded).');
      return;
    }

    console.warn('Unrecognized active model type:', activeModel.type);
  } catch (err) {
    console.warn('loadActiveModel error (files may be missing):', err && err.message ? err.message : err);
  }
}

// call loader at startup
loadActiveModel();

const API_BASE = "https://combustionanalyzer.onrender.com";
const MODEL_INPUT_MODE = process.env.MODEL_INPUT_MODE || 'pure'; 


function convertExcelToCSV(buffer) {
  // buffer: Buffer from multer in-memory storage
  const workbook = xlsx.read(buffer, { type: 'buffer' });
  if (!workbook.SheetNames || workbook.SheetNames.length === 0) {
    throw new Error('No sheets found in Excel workbook');
  }

  // choose the first sheet by default
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  const csv = xlsx.utils.sheet_to_csv(sheet);

  // Normalize line endings and remove stray empty lines at start/end
  const normalized = csv.replace(/\r\n/g, '\n').split('\n').map(r => r.trimEnd()).join('\n').trim() + '\n';
  return normalized;
}


// load preprocessing cutoffs saved during training (optional)
let preprocessing = null;
try {
  preprocessing = JSON.parse(fs.readFileSync('preprocessing.json'));
  console.log('Loaded preprocessing.json');
  // sanity-check shape
  if (!Array.isArray(preprocessing.lower) || !Array.isArray(preprocessing.upper)) {
    console.warn('preprocessing.json missing expected lower/upper arrays — ignoring');
    preprocessing = null;
  }
} catch (err) {
  console.warn('preprocessing.json not found or invalid — will skip clipping:', String(err));
  preprocessing = null;
}

// ---------- CONFIG ----------
const MONGODB_URI = process.env.MONGODB_URI || 'YOUR_MONGODB_URI_HERE';
const PORT = process.env.PORT || 5000; // default 5000 to match your fetch
console.log('MONGODB_URI config status:', (MONGODB_URI && !MONGODB_URI.includes('YOUR_MONGODB_URI_HERE')) ? 'using env' : 'MONGODB_URI not set - edit .env or app.js');

// ---------- MIDDLEWARE ----------
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.static('js'));

// If your frontend is served from a different origin and you use credentials (sessions),
// configure CORS accordingly. For simple same-origin setups, default cors() is okay.
app.use(cors()); // dev-friendly; tighten origin & credentials in production

// If behind a reverse proxy (nginx, load balancer), enable trust proxy so req.ip and x-forwarded-for behave correctly.
// If unsure and you are deploying behind a proxy, set true.
app.set('trust proxy', true);

// ---------- SESSION ----------
app.use(session({
  secret: process.env.SESSION_SECRET || 'please_change_this_to_a_strong_secret',
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    // maxAge = 7 days (adjust as needed)
    maxAge: 7 * 24 * 3600 * 1000
    // In production use secure: true if using HTTPS
  }
}));

// ---------- MONGOOSE CONNECT ----------
mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 30000
})
.then(() => console.log('MongoDB connected successfully'))
.catch(err => {
  console.error('MongoDB connection error:', err);
});

// ---------- SCHEMA & MODELS ----------
// flexible generic schema for existing collections
const flexibleSchema = new mongoose.Schema({}, { strict: false });
// const SlaggingData = mongoose.models.SlaggingData || mongoose.model('SlaggingData', flexibleSchema);

// Force Coal model to use 'coals' collection (as in your DB)
let Coal;
try {
  Coal = mongoose.model('Coal');
} catch (e) {
  Coal = mongoose.model('Coal', flexibleSchema, 'coals'); // explicit collection name
}
console.log('Coal collection name:', Coal.collection && Coal.collection.name);

// ---------- USER MODEL (for trials / subscription) ----------
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true, index: true },
  passwordHash: { type: String, required: true },
  trialsLeft: { type: Number, default: 5 },            // number of remaining trials
  lockedUntil: { type: Date, default: null },          // when lock expires
  lastIP: { type: String, default: null },             // last known IP
  ipHistory: [{ ip: String, when: Date }],             // optional history
  createdAt: { type: Date, default: Date.now }
}, { timestamps: true });

const User = mongoose.models.User || mongoose.model('User', userSchema);

// ---------- HELPERS ----------
function excelSerialToDate(serial) {
  const excelEpoch = new Date(Date.UTC(1900, 0, 1));
  const daysOffset = serial - 1;
  const date = new Date(excelEpoch.getTime() + daysOffset * 24 * 60 * 60 * 1000);
  return date.toISOString().split('T')[0];
}

/**
 * normalizeCoalDoc: converts a DB document (various field-naming variants)
 * into a canonical object expected by frontends:
 */
function normalizeCoalDoc(raw) {
  if (!raw) return null;
  const o = (raw.toObject ? raw.toObject() : Object.assign({}, raw));
  const id = String(o._id || o.id || '');

  const coalName = o.coal || o.name || o['Coal source name'] || o['Coal Source Name'] || '';
  const transportId = o['Transport ID'] || o.transportId || o.transport_id || null;

  const canonicalKeys = ['SiO2','Al2O3','Fe2O3','CaO','MgO','Na2O','K2O','TiO2','SO3','P2O5','Mn3O4','Sulphur (S)','GCV'];

  const aliasMap = {
    'SiO2': 'SiO2', 'SiO₂': 'SiO2',
    'Al2O3': 'Al2O3', 'Al₂O₃': 'Al2O3',
    'Fe2O3': 'Fe2O3', 'Fe₂O₃': 'Fe2O3',
    'CaO': 'CaO',
    'MgO': 'MgO',
    'Na2O': 'Na2O',
    'K2O': 'K2O',
    'TiO2': 'TiO2', 'TiO₂': 'TiO2',
    'SO3': 'SO3', 'SO₃': 'SO3',
    'P2O5': 'P2O5', 'P₂O₅': 'P2O5',
    'Mn3O4': 'Mn3O4', 'Mn₃O₄': 'Mn3O4',
    'Sulphur (S)': 'Sulphur (S)',
    'SulphurS': 'Sulphur (S)', 'Sulphur': 'Sulphur (S)', 'S': 'Sulphur (S)',
    'GCV': 'GCV', 'Gcv': 'GCV', 'gcv': 'GCV'
  };

  const properties = {};
  canonicalKeys.forEach(k => properties[k] = null);

  function collectFrom(obj) {
    if (!obj) return;
    Object.keys(obj).forEach(k => {
      const trimmed = String(k).trim();
      let mapped = aliasMap[trimmed] || null;
      if (!mapped) {
        const normalizedKey = trimmed.replace(/₂/g,'2').replace(/₃/g,'3').replace(/₄/g,'4');
        mapped = aliasMap[normalizedKey] || null;
      }
      if (mapped) {
        const val = obj[k];
        properties[mapped] = (val === '' || val === null || val === undefined) ? null : (isNaN(Number(val)) ? val : Number(val));
      }
    });
  }

  collectFrom(o);
  if (o.properties && typeof o.properties === 'object') collectFrom(o.properties);

  if ((properties['GCV'] === null || properties['GCV'] === undefined) && (o.gcv || o.GCV || o.Gcv)) {
    properties['GCV'] = o.gcv || o.GCV || o.Gcv;
  }

  Object.keys(properties).forEach(k => {
    const v = properties[k];
    if (v !== null && v !== undefined && !isNaN(Number(v))) properties[k] = Number(v);
  });

  const gcvVal = properties['GCV'];

  return {
    _id: o._id,
    id,
    coal: coalName,
    coalType: coalName,
    transportId,
    gcv: gcvVal,
    properties
  };
}

// IP helper (works with X-Forwarded-For when trust proxy=true)
function getClientIp(req) {
  const xff = req.headers['x-forwarded-for'];
  if (xff) return xff.split(',')[0].trim();
  return req.ip || (req.connection && req.connection.remoteAddress) || null;
}

// Auth middleware
async function requireAuth(req, res, next) {
  try {
    const uid = req.session && req.session.userId;
    if (!uid) return res.status(401).json({ error: 'Not authenticated' });
    const user = await User.findById(uid);
    if (!user) {
      req.session.destroy?.(()=>{});
      return res.status(401).json({ error: 'User not found' });
    }
    // check lock
    if (user.lockedUntil && user.lockedUntil > new Date()) {
      return res.status(403).json({ error: 'Account locked until ' + user.lockedUntil.toISOString() });
    }
    req.currentUser = user;
    next();
  } catch (err) {
    console.error('auth error', err);
    res.status(500).json({ error: 'Authentication error' });
  }
}

// // ---------- ROUTES ----------
// // root route - keep existing login page
// app.get('/', (req, res) => {
//   res.sendFile(path.join(__dirname, 'public', 'login.html'));
// });

// // download template
// // download template (coal-oriented)
// // download template (coal-oriented) — optional data export with ?includeData=true
// app.get("/download-template", async (req, res) => {
//   try {
//     const includeData = String(req.query.includeData || '').toLowerCase() === 'true';

//     const workbook = new ExcelJS.Workbook();
//     const worksheet = workbook.addWorksheet("Coal Upload Template");

//     const instructionText = `Instruction for filling the sheet:
// 1. Row 1 contains instructions (delete this row when uploading).
// 2. Row 2 must be headers. Required header: "Coal" (name).
// 3. Other helpful headers: SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, TiO2, SO3, P2O5, Mn3O4, Sulphur, GCV, Cost, Transport ID, Shipment date.
// 4. Leave empty any missing values. Save as .xlsx and upload using the 'file' field.`;

//     worksheet.mergeCells('A1:Q1');
//     const instructionCell = worksheet.getCell('A1');
//     instructionCell.value = instructionText;
//     instructionCell.alignment = { vertical: 'top', horizontal: 'left', wrapText: true };
//     instructionCell.font = { bold: true };
//     instructionCell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFEFEFEF' } };
//     worksheet.getRow(1).height = 90;

//     // Header row (row index 2)
//     const headers = [
//       "Coal", "SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O",
//       "TiO2", "SO3", "P2O5", "Mn3O4", "Sulphur", "GCV", "Cost", "Transport ID", "Shipment date"
//     ];
//     worksheet.addRow(headers);
//     const headerRow = worksheet.getRow(2);
//     headerRow.font = { bold: true };
//     headerRow.alignment = { horizontal: 'center', vertical: 'middle' };
//     headerRow.eachCell((cell) => {
//       cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FFB0E0E6' } };
//       cell.border = {
//         top: { style: 'thin' }, left: { style: 'thin' },
//         bottom: { style: 'thin' }, right: { style: 'thin' }
//       };
//     });

//     // Set reasonable column widths
//     headers.forEach((_, index) => worksheet.getColumn(index + 1).width = 18);

//     if (includeData) {
//       // Fetch docs from coals collection
//       const docs = await Coal.find({}, { __v: 0 }).lean().exec();

//       // Helper to choose existing field variants
//       function pick(o, ...keys) {
//         for (const k of keys) {
//           if (o && o[k] !== undefined && o[k] !== null) return o[k];
//         }
//         return '';
//       }

//       // Append each DB row into the sheet in the same header order
//       for (const d of docs) {
//         const rowValues = [
//           // Coal name
//           pick(d, 'coal', 'Coal', 'name', 'Coal source name'),
//           // oxides & properties — try common canonical and alternate keys
//           pick(d, 'SiO2', 'SiO₂', 'SiO 2'),
//           pick(d, 'Al2O3', 'Al₂O₃'),
//           pick(d, 'Fe2O3', 'Fe₂O₃'),
//           pick(d, 'CaO'),
//           pick(d, 'MgO'),
//           pick(d, 'Na2O'),
//           pick(d, 'K2O'),
//           pick(d, 'TiO2', 'TiO₂'),
//           pick(d, 'SO3', 'SO₃'),
//           pick(d, 'P2O5', 'P₂O₅'),
//           pick(d, 'Mn3O4', 'Mn₃O₄', 'MN3O4'),
//           // Sulphur field may be SulphurS / Sulphur / S / Sulphur (S)
//           pick(d, 'SulphurS', 'Sulphur (S)', 'Sulphur', 'S'),
//           // GCV/gcv
//           pick(d, 'GCV', 'gcv', 'Gcv'),
//           // cost
//           pick(d, 'cost', 'Cost'),
//           // transport id and shipment date
//           pick(d, 'Transport ID', 'transportId', 'transport_id'),
//           // for shipment date prefer ISO string if Date object or string
//           (() => {
//             const sd = pick(d, 'shipmentDate', 'Shipment date', 'shipment_date');
//             if (!sd) return '';
//             if (sd instanceof Date) return sd.toISOString().split('T')[0];
//             // if mongo stores as object like {"$date": "..."} or as string, coerce to string
//             return String(sd);
//           })()
//         ];
//         worksheet.addRow(rowValues);
//       }
//     }

//     res.setHeader("Content-Disposition", `attachment; filename=${includeData ? 'Coal_Data_Export.xlsx' : 'Coal_Upload_Template.xlsx'}`);
//     res.setHeader("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
//     await workbook.xlsx.write(res);
//     res.end();
//   } catch (err) {
//     console.error('/download-template error:', err);
//     res.status(500).send('Template generation failed');
//   }
// });



// // multer memory storage for uploads
// const storage = multer.memoryStorage();
// const upload = multer({ storage });

// // upload excel -> SlaggingData collection
// // upload excel -> insert into 'coals' collection
// // upload excel -> insert into 'coals' collection (robust header detection & normalization)
// app.post("/upload-excel", upload.single("file"), async (req, res) => {
//   if (!req.file) return res.status(400).json({ error: "No file uploaded" });
//   try {
//     const workbook = xlsx.read(req.file.buffer, { type: "buffer" });
//     if (!workbook.SheetNames || workbook.SheetNames.length === 0) {
//       return res.status(400).json({ error: "No sheets found in workbook" });
//     }

//     const sheetName = workbook.SheetNames[0];
//     const sheet = workbook.Sheets[sheetName];

//     // Read as rows array so we can detect header row robustly (handles instruction/merged header rows)
//     const rows = xlsx.utils.sheet_to_json(sheet, { header: 1, defval: null });

//     // find header row index by looking for typical header tokens
//     const headerRowIndex = rows.findIndex(r => Array.isArray(r) && r.some(cell => {
//       if (!cell) return false;
//       const s = String(cell).toLowerCase();
//       return /coal|sio2|sio₂|al2o3|gcv|sulphur|si o2|al₂o₃|fe2o3/.test(s);
//     }));

//     if (headerRowIndex === -1) {
//       return res.status(400).json({ error: "Could not find header row in sheet. Ensure headers like 'Coal' or 'SiO2' exist." });
//     }

//     const rawHeaders = rows[headerRowIndex].map(h => (h === null || h === undefined) ? '' : String(h).trim());
//     const dataRows = rows.slice(headerRowIndex + 1);

//     // header map (variants -> canonical keys)
//     const headerMap = {
//       'coal': 'coal', 'coal source name': 'coal', 'coal source': 'coal', 'name': 'coal',

//       'sio2': 'SiO2', 'sio₂': 'SiO2', 'si o2': 'SiO2',
//       'al2o3': 'Al2O3', 'al₂o₃': 'Al2O3',
//       'fe2o3': 'Fe2O3', 'fe₂o₃': 'Fe2O3',
//       'cao': 'CaO', 'mgo': 'MgO',
//       'na2o': 'Na2O', 'k2o': 'K2O',
//       'tio2': 'TiO2', 'tio₂': 'TiO2',
//       'so3': 'SO3', 'so₃': 'SO3',
//       'p2o5': 'P2O5', 'p₂o₅': 'P2O5',
//       'mn3o4': 'Mn3O4', 'mn₃o₄': 'Mn3O4',

//       'sulphur (s)': 'SulphurS', 'sulphur': 'SulphurS', 'sulphurs': 'SulphurS', 's': 'SulphurS',
//       'gcv': 'gcv', 'gcv.': 'gcv', 'g c v': 'gcv',
//       'cost': 'cost', 'price': 'cost',
//       'transport id': 'Transport ID', 'data uploaded by tps': 'uploadedBy', 'shipment date': 'shipmentDate', 'type of transport': 'transportType'
//     };

//     // helper to canonicalize header string
//     function canonicalHeader(h) {
//       if (h === null || h === undefined) return '';
//       const s = String(h).trim();
//       const simple = s.replace(/[\s_\-\.]/g, '').replace(/₂/g,'2').replace(/₃/g,'3').replace(/₄/g,'4').toLowerCase();
//       // try exact headerMap keys first
//       const direct = Object.keys(headerMap).find(k => k.toLowerCase() === s.toLowerCase());
//       if (direct) return headerMap[direct];
//       const found = Object.keys(headerMap).find(k => k.replace(/[\s_\-\.]/g,'').replace(/₂/g,'2').replace(/₃/g,'3').replace(/₄/g,'4').toLowerCase() === simple);
//       return found ? headerMap[found] : s; // fallback to original header text if not found
//     }

//     // build canonical headers array
//     const canonicalHeaders = rawHeaders.map(h => canonicalHeader(h));

//     // map rows to objects
//     const parsed = dataRows.map((row, rowIndex) => {
//       // skip completely empty rows
//       if (!Array.isArray(row) || row.every(c => c === null || (typeof c === 'string' && c.trim() === ''))) return null;

//       const out = {};
//       for (let i = 0; i < canonicalHeaders.length; i++) {
//         const key = canonicalHeaders[i];
//         // skip empty header slots
//         if (!key) continue;
//         let val = row[i] === undefined ? null : row[i];

//         // convert excel date serials if header indicates date (optional)
//         if (key.toLowerCase().includes('date') && typeof val === 'number' && val > 0 && val < 2958465) {
//           val = excelSerialToDate(val);
//         } else if (val === '') {
//           val = null;
//         }

//         // convert numeric-like strings to Number
//         if (val !== null && typeof val !== 'number') {
//           const maybeNum = Number(String(val).replace(/,/g, '').trim());
//           if (!Number.isNaN(maybeNum)) val = Math.round(maybeNum * 100) / 100;
//         }

//         out[key] = val;
//       }

//       // require coal name
//       if (!out.coal || String(out.coal).trim() === '') return null;
//       return out;
//     }).filter(Boolean);

//     if (!parsed.length) {
//       return res.status(400).json({ error: "No valid data rows found after header (check your Excel file)" });
//     }

//     // Insert into coals collection; unordered so one bad row won't stop others
//     const inserted = await Coal.insertMany(parsed, { ordered: false });
//     res.json({ message: "Data uploaded successfully", rowsParsed: parsed.length, rowsInserted: inserted.length, sample: inserted.slice(0,5) });

//   } catch (error) {
//     console.error("Error processing file (upload-excel):", error);
//     res.status(500).json({ error: "Failed to process file", details: String(error) });
//   }
// });



// // fetch raw SlaggingData
// // fetch all coals
// app.get("/fetch-data", async (req, res) => {
//   try {
//     const data = await Coal.find({}, { __v: 0 }).lean();
//     res.json(data);
//   } catch (error) {
//     console.error("Error fetching data:", error);
//     res.status(500).json({ error: "Failed to fetch data" });
//   }
// });

// // delete route -> remove docs from 'coals'
// app.delete("/delete-data", async (req, res) => {
//   try {
//     const { ids } = req.body;
//     if (!Array.isArray(ids) || ids.length === 0) return res.status(400).json({ error: "No IDs provided" });

//     const result = await Coal.deleteMany({ _id: { $in: ids } });
//     if (result.deletedCount === 0) return res.status(404).json({ error: "No data found" });
//     res.json({ message: `${result.deletedCount} data deleted successfully` });
//   } catch (error) {
//     console.error("Error deleting data:", error);
//     res.status(500).json({ error: "Failed to delete data" });
//   }
// });

// --- replace the existing calculateAFT with this async version ---
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8001';

// Unified calculateAFT that uses the in-memory activeModel
// NOTE: this is now async because XGBoost (xgb) predictions are proxied to Python over HTTP.
// Unified calculateAFT that uses the in-memory activeModel
async function calculateAFT(values) {
  try {

    const features = buildFeaturesForModel(values);
    const prepped = applyPreprocessing(features);

    if (!activeModel || !activeModel.type) {
      return Math.round(formulaAFT(values));
    }

    // =========================
    // RF MODELS
    // =========================
    if (activeModel.type === 'pure_rf' || activeModel.type === 'hybrid_rf') {

      if (!activeModel.model || typeof activeModel.model.predict !== 'function') {
        console.warn('RF model not loaded — fallback formula.');
        return Math.round(formulaAFT(values));
      }

      try {
        const pred = activeModel.model.predict([prepped])[0];
        return Math.round(Math.max(1000, Math.min(1600, pred)));
      } catch (err) {

        console.warn("RF feature mismatch, retrying with padding");

        const expected = activeModel.model.n_features || prepped.length;

        let aligned = prepped.slice(0, expected);

        while (aligned.length < expected) aligned.push(0);

        const pred = activeModel.model.predict([aligned])[0];
        return Math.round(Math.max(1000, Math.min(1600, pred)));
      }
    }

    // =========================
    // ANN
    // =========================
    if (activeModel.type === 'ann') {

      if (!activeModel.model || typeof activeModel.model.run !== 'function') {
        return Math.round(formulaAFT(values));
      }

      const out = activeModel.model.run(prepped);
      const raw = Array.isArray(out) ? out[0] : out;

      return Math.round(Math.max(1000, Math.min(1600, raw)));
    }

    // =========================
    // XGBOOST
    // =========================
    if (activeModel.type === 'pure_xgb' || activeModel.type === 'hybrid_xgb') {

      try {

        const response = await fetch(`${ML_SERVICE_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            values: values,
            model: activeModel.type
          })
        });

        if (!response.ok) {
          const txt = await response.text();
          console.warn("Python predict failed:", txt);
          return Math.round(formulaAFT(values));
        }

        const json = await response.json();
        return json.prediction;

      } catch (err) {

        console.warn("XGB prediction error:", err.message);
        return Math.round(formulaAFT(values));
      }
    }

    return Math.round(formulaAFT(values));

  } catch (err) {

    console.warn("calculateAFT error:", err.message);
    return Math.round(formulaAFT(values));
  }
}

// Replace your /calculate-aft endpoint to use calculateAFT (this keeps your existing input validation)
// Replace the existing /calculate-aft route with this block
app.post("/calculate-aft", async (req, res) => {
  try {
    const payload = req.body && (req.body.values !== undefined ? req.body.values : req.body);
    if (!payload) {
      return res.status(400).json({ error: "No input found. Send either an array in `values` or an object with oxide keys." });
    }

    // canonical training feature order (11 features)
    const featureOrder = ['SiO2','Al2O3','Fe2O3','CaO','MgO','Na2O','K2O','SO3','TiO2','P2O5','S'];

    // helper to normalize keys (remove non-alphanum, lower-case)
    const norm = s => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');

    // build normalized map if payload is object
    const buildArrayFromObject = (obj) => {
      const normMap = {};
      Object.keys(obj).forEach(k => {
        normMap[norm(k)] = obj[k];
      });

      const arr = [];
      const missing = [];
      featureOrder.forEach(f => {
        const n = norm(f);                     // e.g. 'siO2' -> 'sio2'
        // also accept 'sulphur', 'sulphurs', 'sulphur(s)' etc because norm() strips punctuation
        let val = normMap[n];
        // special tolerance: if key 's' not present but 'sulphur' or 'sulphurs' present, norm handles it.
        if (val === undefined) {
          // fallback: look for a few common aliases
          if (f === 'S') val = normMap['sulphur'] ?? normMap['sulphurs'] ?? normMap['sulphurs'] ?? normMap['s'];
          if (f === 'P2O5') val = val ?? normMap['p205'];
        }
        if (val === undefined) missing.push(f);
        arr.push(val);
      });

      return { arr, missing };
    };

    // build array when payload is an array
    let featureArray = null;
    if (Array.isArray(payload)) {
      if (payload.length !== featureOrder.length) {
        return res.status(400).json({
          error: `Array input must have ${featureOrder.length} values in this order: ${featureOrder.join(', ')}`
        });
      }
      featureArray = payload.map(v => Number(v));
    } else if (typeof payload === 'object') {
      const { arr, missing } = buildArrayFromObject(payload);
      if (missing.length) {
        return res.status(400).json({
          error: `Missing features: ${missing.join(', ')}. Acceptable keys include exact names or aliases like 'Sulphur(s)' or 'S' and 'P2O5' or 'p205'.`,
        });
      }
      featureArray = arr.map(v => Number(v));
    } else {
      return res.status(400).json({ error: "Unsupported payload format. Send JSON array or object." });
    }

    // validate numeric
    const notNumber = featureArray.map((v,i)=> isFinite(v) ? null : featureOrder[i]).filter(Boolean);
    if (notNumber.length) {
      return res.status(400).json({ error: `These features are not numeric: ${notNumber.join(', ')}` });
    }

    // prepare training object (keys = training feature names)
    const trainingObj = {};
    featureOrder.forEach((k, i) => trainingObj[k] = featureArray[i]);

    // prepare coal storage object: user-facing coal properties should use 'Sulphur(s)'
    const storageObj = { ...trainingObj };
    // copy S -> Sulphur(s) and remove the training key S if you only want the storage representation to have Sulphur(s)
    storageObj['Sulphur(s)'] = storageObj['S'];
    // (optional) delete storageObj['S']; // uncomment if you don't want 'S' in stored coal properties

    // call the calculation (your calculateAFT should accept the 11-element array)
    const predicted = await calculateAFT(featureArray);

    // method label
    const method = (activeModel && activeModel.type) ? activeModel.type : 'base_formula';

    return res.status(200).json({
      prediction: predicted,
      method,
      // include both representations so you can see what was fed to the model and what you'd store as coal props
      trainingFeatures: trainingObj,
      coalPropertiesToStore: storageObj
    });
  } catch (err) {
    console.error('/calculate-aft error:', err);
    return res.status(500).json({ error: 'Internal error calculating AFT', details: String(err) });
  }
});



function formulaAFT(values) {
  const [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2] = values;
  const sumSiAl = SiO2 + Al2O3;

  if (sumSiAl < 55) {
    return 1245 +
      1.1 * SiO2 +
      0.95 * Al2O3 -
      2.5 * Fe2O3 -
      2.98 * CaO -
      4.5 * MgO -
      7.89 * (Na2O + K2O) -
      1.7 * SO3 -
      0.63 * TiO2;
  } else if (sumSiAl < 75) {
    return 1323 +
      1.45 * SiO2 +
      0.683 * Al2O3 -
      2.39 * Fe2O3 -
      3.1 * CaO -
      4.5 * MgO -
      7.49 * (Na2O + K2O) -
      2.1 * SO3 -
      0.63 * TiO2;
  } else {
    return 1395 +
      1.2 * SiO2 +
      0.9 * Al2O3 -
      2.5 * Fe2O3 -
      3.1 * CaO -
      4.5 * MgO -
      7.2 * (Na2O + K2O) -
      1.7 * SO3 -
      0.63 * TiO2;
  }
}

// function buildFeatures(values) {
//   const [
//     SiO2,
//     Al2O3,
//     Fe2O3,
//     CaO,
//     MgO,
//     Na2O,
//     K2O,
//     SO3,
//     TiO2,
//   ] = values;

//   const SiAl = SiO2 + 0.8 * Al2O3;   
//   const Flux = CaO + MgO + Fe2O3;    
//   const Alk  = Na2O + K2O;            

//   return [
//     SiAl,
//     Flux,
//     Alk,
//     SO3,
//     TiO2,
//   ];
// }

// Build the SAME feature vector used in training:
// [SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, SO3, TiO2, baseAFT]
function buildFeaturesForModel(values) {
  // Expected order:
  // [SiO2,Al2O3,Fe2O3,CaO,MgO,Na2O,K2O,SO3,TiO2,P2O5,S]

  if (!Array.isArray(values)) return [];

  let arr = values.map(v => Number(v ?? 0));

  // ensure at least 9 oxides
  while (arr.length < 9) arr.push(0);

  // ensure P2O5 and S exist
  while (arr.length < 11) arr.push(0);

  // Hybrid models add base formula
  if (activeModel.type === "hybrid_rf" || activeModel.type === "hybrid_xgb") {
    const base = formulaAFT(arr.slice(0, 9));
    return [...arr.slice(0, 11), base];
  }

  return arr.slice(0, 11);
}


// Apply 1%/99% winsorization cutoffs saved during training.
// If preprocessing.json not present, return features unchanged.
function applyPreprocessing(features) {
  if (!Array.isArray(features)) return [];

  if (!preprocessing || !Array.isArray(preprocessing.lower)) {
    return features.slice();
  }

  const out = features.slice();

  for (let i = 0; i < out.length; i++) {
    const lo = preprocessing.lower[i];
    const hi = preprocessing.upper[i];

    if (typeof lo === "number" && out[i] < lo) out[i] = lo;
    if (typeof hi === "number" && out[i] > hi) out[i] = hi;
  }

  return out;
}




// PATCH /update-coal
// body: { idType: 'coalId' | '_id', idValue: string, updates: { <field>: value, ... } }
app.patch('/update-coal', async (req, res) => {
  try {
    const { idType, idValue, updates } = req.body;
    if (!idType || !idValue || !updates || typeof updates !== 'object') {
      return res.status(400).json({ error: 'idType, idValue and updates object required' });
    }

    // build query
    let query = {};
if (idType === '_id') {
  if (!mongoose.Types.ObjectId.isValid(idValue)) {
    return res.status(400).json({ error: 'Invalid _id' });
  }
  query = { _id: new mongoose.Types.ObjectId(idValue) };
}

    // find target
    const target = await Coal.findOne(query).lean();
    if (!target) return res.status(404).json({ error: 'Coal not found' });

    // If coalId is being changed, ensure uniqueness (avoid clobbering another record)
    if (updates.coalId !== undefined && updates.coalId !== null) {
      const newCoalId = String(updates.coalId);
      if (newCoalId !== String(target.coalId)) {
        const conflict = await Coal.findOne({ coalId: newCoalId }).lean();
        if (conflict && String(conflict._id) !== String(target._id)) {
          return res.status(409).json({ error: 'coalId already in use by another document' });
        }
      }
    }

    // Remove fields we must not update
    delete updates._id;
    delete updates.__v;

    // Build $set body
    const setObj = {};
    Object.keys(updates).forEach(k => {
      // allow top-level updates only. (If you have nested 'properties' objects, send e.g. updates.properties = {...} from frontend)
      setObj[k] = updates[k];
    });

    const updated = await Coal.findOneAndUpdate(query, { $set: setObj }, { new: true }).lean();
    if (!updated) return res.status(500).json({ error: 'Update failed' });

    return res.json({ message: 'updated', updated });
  } catch (err) {
    console.error('/update-coal error', err);
    return res.status(500).json({ error: 'Internal server error', details: String(err) });
  }
});


// app.post("/calculate-aft", async (req, res) => {
//   try {
//     const values = req.body && req.body.values;
//     if (!Array.isArray(values) || values.length < 9) {
//       return res.status(400).json({ error: "Please send values: array of 9 oxides [SiO2,Al2O3,Fe2O3,CaO,MgO,Na2O,K2O,SO3,TiO2]" });
//     }

//     // build features (includes base formula as 10th feature)
//     const features = buildFeaturesForModel(values);
//     const prepped = applyPreprocessing(features);

//     console.log("Base AFT (formula):", features[9]);
//     console.log("Features for ML model:", prepped);

//     // Predict using the model (2D input)
//     let predicted = null;
//     try {
//       predicted = model.predict([prepped])[0];
//     } catch (err) {
//       console.warn('Model.predict failed:', String(err));
//       predicted = features[9]; // fallback: use base formula if model fails
//     }

//     // clamp and round
//     predicted = Math.round(Math.max(1000, Math.min(1600, predicted)));

//     console.log("Final AFT Prediction:", predicted);
//     return res.status(200).json({ prediction: predicted, method: 'ml+base' });
//   } catch (err) {
//     console.error('/calculate-aft error:', err);
//     return res.status(500).json({ error: 'Internal error calculating AFT', details: String(err) });
//   }
// });


// // ---------- AUTH ROUTES ----------

// // POST /auth/login  { email, password }
// app.post('/auth/login', async (req, res) => {
//   try {
//     const { email, password } = req.body;
//     if (!email || !password) return res.status(400).json({ error: 'Email & password required' });

//     const user = await User.findOne({ email: email.toLowerCase().trim() });
//     if (!user) return res.status(401).json({ error: 'Invalid credentials' });

//     // check locked
//     if (user.lockedUntil && user.lockedUntil > new Date()) {
//       return res.status(403).json({ error: 'Account locked until ' + user.lockedUntil.toISOString() });
//     }

//     const ok = await bcrypt.compare(password, user.passwordHash);
//     if (!ok) return res.status(401).json({ error: 'Invalid credentials' });

//     // update last IP
//     const ip = getClientIp(req);
//     user.lastIP = ip;
//     user.ipHistory = user.ipHistory || [];
//     user.ipHistory.push({ ip, when: new Date() });
//     await user.save();

//     // create session
//     req.session.userId = user._id.toString();

//     res.json({ message: 'Logged in', trialsLeft: user.trialsLeft });
//   } catch (err) {
//     console.error('/auth/login error', err);
//     res.status(500).json({ error: 'Login failed' });
//   }
// });

// // POST /auth/logout
// app.post('/auth/logout', (req, res) => {
//   req.session.destroy(err => {
//     if (err) console.error('session destroy err', err);
//     res.json({ message: 'Logged out' });
//   });
// });

// // GET /auth/status - returns basic auth info
// app.get('/auth/status', async (req, res) => {
//   try {
//     if (!req.session || !req.session.userId) return res.json({ authenticated: false });
//     const user = await User.findById(req.session.userId, 'email trialsLeft lockedUntil lastIP');
//     if (!user) return res.json({ authenticated: false });
//     return res.json({
//       authenticated: true,
//       email: user.email,
//       trialsLeft: user.trialsLeft,
//       lockedUntil: user.lockedUntil,
//       lastIP: user.lastIP
//     });
//   } catch (err) {
//     console.error('/auth/status error', err);
//     res.status(500).json({ error: 'Status check failed' });
//   }
// });


// ---------- ROUTES TO REPLACE ----------

// root route



app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'login.html'));
});


app.get('/api/get-active-model', (req, res) => {
  return res.json({
    success: true,
    model: (activeModel && activeModel.type) ? activeModel.type : 'base_formula'
  });
});

// DOWNLOAD TEMPLATE (optionally include data via ?includeData=true)
app.get("/download-template", async (req, res) => {
  try {
    const includeData = String(req.query.includeData || '').toLowerCase() === 'true';
    const workbook = new ExcelJS.Workbook();
    const worksheet = workbook.addWorksheet("Coal Upload Template");

    const instructionText = `Instructions:
1) Row 2 is headers. Required header: "Coal".
2) Optional header: "Coal ID" (or "Coal Source ID") to supply your own IDs.
3) Other example headers: SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, TiO2, SO3, P2O5, Mn3O4, Sulphur, GCV, Cost, Transport ID, Shipment date.`;
    worksheet.mergeCells('A1:R1');
    const instructionCell = worksheet.getCell('A1');
    instructionCell.value = instructionText;
    instructionCell.alignment = { vertical: 'top', horizontal: 'left', wrapText: true };
    worksheet.getRow(1).height = 80;

    // header row - include Coal ID as first header
    const headers = [
      "Coal ID", "Coal", "SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O",
      "TiO2", "SO3", "P2O5", "Mn3O4", "SulphurS", "gcv", "Cost"
    ];
    worksheet.addRow(headers);
    const headerRow = worksheet.getRow(2);
    headerRow.font = { bold: true };
    headerRow.alignment = { horizontal: 'center' };

    headers.forEach((_, i) => worksheet.getColumn(i + 1).width = 18);

    if (includeData) {
      const docs = await Coal.find({}, { __v: 0 }).lean().exec();
      function pick(o, ...keys) {
        for (const k of keys) if (o && o[k] !== undefined && o[k] !== null) return o[k];
        return '';
      }
      for (const d of docs) {
        const rowValues = [
          pick(d, 'coalId', 'Coal ID'),
          pick(d, 'coal', 'Coal', 'name'),
          pick(d, 'SiO2'), pick(d, 'Al2O3'), pick(d, 'Fe2O3'), pick(d, 'CaO'), pick(d, 'MgO'),
          pick(d, 'Na2O'), pick(d, 'K2O'), pick(d, 'TiO2'), pick(d, 'SO3'), pick(d, 'P2O5'),
          pick(d, 'Mn3O4'), pick(d, 'Sulphur', 'S'), pick(d, 'GCV'), pick(d, 'cost'),
          pick(d, 'Transport ID','transportId'),
          (d.shipmentDate instanceof Date) ? d.shipmentDate.toISOString().split('T')[0] : (d.shipmentDate || '')
        ];
        worksheet.addRow(rowValues);
      }
    }

    res.setHeader("Content-Disposition", `attachment; filename=${includeData ? 'Coal_Data_Export.xlsx' : 'Coal_Upload_Template.xlsx'}`);
    res.setHeader("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
    await workbook.xlsx.write(res);
    res.status(200).end();
  } catch (err) {
    console.error('/download-template error:', err);
    res.status(500).send('Template generation failed');
  }
});

// ---------- MULTER (file upload) SETUP ----------
const storage = multer.memoryStorage();          // keep file in memory buffer
const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }        // optional: limit to 50MB (adjust if needed)
});

// --- CONFIG: Python ML service URL

function removeCoalNameColumn(csvText) {

  const lines = csvText.trim().split('\n');

  const headers = lines[0].split(',');

  const coalIndex = headers.findIndex(h => h.trim().toLowerCase() === 'coalname');

  if (coalIndex === -1) {
    return csvText; // nothing to remove
  }

  const newHeaders = headers.filter((_, i) => i !== coalIndex);

  const newLines = [newHeaders.join(',')];

  for (let i = 1; i < lines.length; i++) {

    const cols = lines[i].split(',');

    const filtered = cols.filter((_, j) => j !== coalIndex);

    newLines.push(filtered.join(','));
  }

  return newLines.join('\n');
}

// --- updated /api/compare-models ---
app.post('/api/compare-models', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ success: false, message: "No file uploaded" });

    // --- Save the uploaded file to server path (with Excel -> CSV conversion support) ---
    const ext = path.extname(req.file.originalname || '').toLowerCase();
    let csvText;

    if (ext === '.xlsx' || ext === '.xls') {
      try {
        csvText = convertExcelToCSV(req.file.buffer);
      } catch (convErr) {
        console.error('/api/compare-models Excel -> CSV conversion failed:', convErr && convErr.message);
        return res.status(400).json({ success: false, message: 'Failed to convert Excel to CSV', details: String(convErr) });
      }
    } else {
      csvText = req.file.buffer.toString('utf8').replace(/\r\n/g, '\n').trim() + '\n';
    }

    // remove CoalName column if present (keeps training CSV columns consistent)
    csvText = removeCoalNameColumn(csvText);
    // normalize line endings and ensure trailing newline
    csvText = csvText.replace(/\r\n/g, '\n').trim() + '\n';

    const lastPath = path.join(__dirname, 'last_training_upload.csv');

    // Determine train mode (form field from multipart/form-data or query param)
    const trainMode = ((req.body && req.body.trainMode) || req.query.trainMode || 'new').toString().toLowerCase();

    if (trainMode === 'append' && fs.existsSync(lastPath)) {
      // Append: read existing file, check headers match, then append rows (no dedupe)
      try {
        const existingText = fs.readFileSync(lastPath, 'utf8').replace(/\r\n/g, '\n').trim();
        const existingLines = existingText ? existingText.split('\n') : [];
        const newLines = csvText ? csvText.split('\n') : [];

        if (existingLines.length === 0) {
          // existing file empty for some reason -> just write new
          fs.writeFileSync(lastPath, csvText, 'utf8');
          console.log('/api/compare-models: existing dataset empty, wrote uploaded CSV to', lastPath);
        } else {
          // Check header compatibility
          const existingHeader = existingLines[0].trim();
          const newHeader = newLines[0] ? newLines[0].trim() : '';
          if (existingHeader !== newHeader) {
            return res.status(400).json({
              success: false,
              message: 'Header mismatch between existing dataset and uploaded file. Ensure the same columns (order and names).',
              details: { existingHeader, newHeader }
            });
          }

          // Build merged content: header + existing rows + new rows (excluding header)
          const mergedLines = [
            existingHeader,
            ...existingLines.slice(1),
            ...newLines.slice(1)
          ].filter(l => l !== undefined && l !== null); // keep as-is (no dedupe)
          const mergedText = mergedLines.join('\n').trim() + '\n';

          fs.writeFileSync(lastPath, mergedText, 'utf8');
          console.log('/api/compare-models appended uploaded CSV to', lastPath);
        }
      } catch (appendErr) {
        console.error('/api/compare-models append error:', appendErr);
        return res.status(500).json({ success: false, message: 'Failed to append to existing dataset', details: String(appendErr) });
      }
    } else {
      // Default behaviour (train on new only) — overwrite last_training_upload.csv
      fs.writeFileSync(lastPath, csvText, 'utf8');
      console.log('/api/compare-models saved CSV to', lastPath, '(mode:', trainMode, ')');
    }

    // Collect progress events in-memory so we can return them in the final response
    const progressLog = [];
    const pushProgress = (ev) => {
      // ensure small payloads (strip large mem snapshots if present)
      const toPush = Object.assign({}, ev);
      if (toPush._mem && Object.keys(toPush._mem).length > 5) {
        // keep only rss/heapUsed if verbose mem present
        toPush._mem = { rss: toPush._mem.rss, heapUsed: toPush._mem.heapUsed };
      }
      progressLog.push(toPush);
    };

    // Options: allow callers to supply ?debug=1 for more verbosity, ?timeoutMs=...
    const verbose = req.query.debug === '1' || req.query.debug === 'true';
    const timeoutMs = req.query.timeoutMs ? Number(req.query.timeoutMs) : (process.env.DEFAULT_TRAIN_TIMEOUT_MS ? Number(process.env.DEFAULT_TRAIN_TIMEOUT_MS) : undefined);

    // Run Node models
    let nodeResults = {};
    try {
      nodeResults = await runAllModels(lastPath, {
        onProgress: pushProgress,
        verbose,
        continueOnError: true,
        debuggerOnError: false,
        timeoutMs
      });
    } catch (runErr) {
      // capture fatal error and continue to attempt python call (or return error)
      console.error('runAllModels fatal error', runErr);
      progressLog.push({ event: 'fatal', message: String(runErr) });
      // We'll attempt Python step below and then return combined error if needed.
    }

    // Try python XGBoost training/prediction (best-effort)
    let xgbResults = {};
    try {
      const pyResp = await fetch(`${ML_SERVICE_URL}/train_from_path`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: lastPath })
      });

      if (pyResp.ok) {
        const pyJson = await pyResp.json();
        xgbResults = pyJson.results || pyJson;
        progressLog.push({ event: 'python_done', message: 'Python XGBoost finished', details: xgbResults });
      } else {
        const txt = await pyResp.text();
        progressLog.push({ event: 'python_error', message: 'Python training returned non-OK', details: txt });
        console.warn('Python training failed:', txt);
      }
    } catch (pyErr) {
      progressLog.push({ event: 'python_unreachable', message: 'Python ML service not reachable', details: pyErr && pyErr.message });
      console.warn('Python service not reachable:', pyErr && pyErr.message);
    }

    // Merge results and return with progressLog
    const mergedResults = {
      ...nodeResults,
      ...xgbResults
    };

    return res.json({
      success: true,
      results: mergedResults,
      trainingPath: lastPath,
      progressLog
    });

  } catch (err) {
    console.error('Model comparison error:', err);
    return res.status(500).json({
      success: false,
      message: "Server error during model comparison",
      details: String(err)
    });
  }
});






app.get('/model-comparison', (req, res) => {
  res.sendFile(path.join(__dirname, 'model_compare.html'));
});

// UPLOAD EXCEL
// app.post("/upload-excel", upload.single("file"), async (req, res) => {
//   if (!req.file) return res.status(400).json({ error: "No file uploaded" });
//   try {
//     const workbook = xlsx.read(req.file.buffer, { type: "buffer" });
//     if (!workbook.SheetNames || workbook.SheetNames.length === 0) return res.status(400).json({ error: "No sheets found" });
//     const sheet = workbook.Sheets[workbook.SheetNames[0]];
//     const rows = xlsx.utils.sheet_to_json(sheet, { header: 1, defval: null });

//     // find header row index by common tokens
//     const headerRowIndex = rows.findIndex(r => Array.isArray(r) && r.some(c => c && /coal|sio2|gcv|sulphur/i.test(String(c))));
//     if (headerRowIndex === -1) return res.status(400).json({ error: "Could not find header row. Ensure headers like 'Coal' exist." });

//     const rawHeaders = rows[headerRowIndex].map(h => (h === null || h === undefined) ? '' : String(h).trim());
//     const dataRows = rows.slice(headerRowIndex + 1);

//     // header mapping including coalId variants
//     const headerMap = {
//       'coal': 'coal', 'coal source name': 'coal', 'name': 'coal',
//       'coalid': 'coalId', 'coal id': 'coalId', 'coal_id': 'coalId', 'coalsourceid': 'coalId', 'coal source id': 'coalId',
//       'sio2': 'SiO2','al2o3':'Al2O3','fe2o3':'Fe2O3','cao':'CaO','mgo':'MgO',
//       'na2o':'Na2O','k2o':'K2O','tio2':'TiO2','so3':'SO3','p2o5':'P2O5','mn3o4':'Mn3O4',
//       'sulphur':'Sulphur', 's': 'Sulphur', 'gcv':'GCV', 'cost':'cost', 'transport id':'Transport ID', 'shipment date':'shipmentDate'
//     };

//     function canonicalHeader(s) {
//       if (!s) return '';
//       const simple = String(s).replace(/[\s_\-\.]/g,'').replace(/[₂₃₄]/g, m => ({'₂':'2','₃':'3','₄':'4'}[m])).toLowerCase();
//       const direct = Object.keys(headerMap).find(k => k.toLowerCase() === String(s).toLowerCase());
//       if (direct) return headerMap[direct];
//       const found = Object.keys(headerMap).find(k => k.replace(/[\s_\-\.]/g,'').toLowerCase() === simple);
//       return found ? headerMap[found] : s;
//     }

//     const canonicalHeaders = rawHeaders.map(h => canonicalHeader(h));
//     const parsed = dataRows.map(row => {
//       if (!Array.isArray(row) || row.every(c => c === null || (typeof c === 'string' && c.trim() === ''))) return null;
//       const out = {};
//       for (let i = 0; i < canonicalHeaders.length; i++) {
//         const key = canonicalHeaders[i];
//         if (!key) continue;
//         let val = row[i] === undefined ? null : row[i];
//         if (key.toLowerCase().includes('date') && typeof val === 'number') val = excelSerialToDate(val);
//         if (val === '') val = null;
//         if (val !== null && typeof val !== 'number') {
//           const maybeNum = Number(String(val).replace(/,/g, '').trim());
//           if (!Number.isNaN(maybeNum)) val = Math.round(maybeNum * 100) / 100;
//         }
//         out[key] = val;
//       }
//       if (!out.coal || String(out.coal).trim() === '') return null;
//       return out;
//     }).filter(Boolean);

//     if (!parsed.length) return res.status(400).json({ error: "No valid data rows found after header" });

//     // assign sequential coalId when missing (continuing from max existing numeric coalId)
//     if (parsed.length) {
//       const existingMaxDoc = await Coal.find({ coalId: { $exists: true } }).sort({ coalId: -1 }).limit(1).lean().exec();
//       let nextCoalId = 1;
//       if (existingMaxDoc && existingMaxDoc.length) {
//         const candidate = Number(existingMaxDoc[0].coalId);
//         if (Number.isFinite(candidate)) nextCoalId = Math.max(1, Math.floor(candidate) + 1);
//       }
//       parsed.forEach(r => {
//         if (r.coalId === undefined || r.coalId === null || String(r.coalId).trim() === '') {
//           r.coalId = String(nextCoalId);
//           nextCoalId++;
//         } else {
//           r.coalId = String(r.coalId).trim();
//         }
//       });
//     }

//     const inserted = await Coal.insertMany(parsed, { ordered: false });
//     res.json({ message: "Data uploaded successfully", rowsParsed: parsed.length, rowsInserted: inserted.length, sample: inserted.slice(0,5) });
//   } catch (error) {
//     console.error("Error processing file (upload-excel):", error);
//     res.status(500).json({ error: "Failed to process file", details: String(error) });
//   }
// });
// REPLACE the existing /upload-excel route with this block
// REPLACE your existing /upload-excel route with this block
// app.post("/upload-excel", upload.single("file"), async (req, res) => {
//   if (!req.file) return res.status(400).json({ error: "No file uploaded" });

//   try {
//     const workbook = xlsx.read(req.file.buffer, { type: "buffer" });
//     if (!workbook.SheetNames || workbook.SheetNames.length === 0) {
//       return res.status(400).json({ error: "No sheets found in workbook" });
//     }

//     const sheet = workbook.Sheets[workbook.SheetNames[0]];
//     const rows = xlsx.utils.sheet_to_json(sheet, { header: 1, defval: null });

//     // find header row index (robust: looks for 'coal' token)
//     const headerRowIndex = rows.findIndex(r => Array.isArray(r) && r.some(c => c && /coal/i.test(String(c))));
//     if (headerRowIndex === -1) {
//       return res.status(400).json({ error: "Could not find header row. Ensure 'Coal' header exists." });
//     }

//     const rawHeaders = rows[headerRowIndex].map(h => (h === null || h === undefined) ? '' : String(h).trim());
//     const dataRows = rows.slice(headerRowIndex + 1);

//     // Canonical DB keys we will allow (strict: unknown headers are ignored)
//     const allowedCanonicalKeys = [
//       'coalId','coal',
//       'SiO2','Al2O3','Fe2O3','CaO','MgO','Na2O','K2O','TiO2',
//       'SO3','P2O5','Mn3O4',
//       'Sulphur','GCV','cost'
//     ];

//     // Header alias map -> canonical DB key
//     const headerMap = {
//       'coal': 'coal',
//       'coalid': 'coalId', 'coal id': 'coalId', 'coal_id': 'coalId',
//       'sio2': 'SiO2', 'si o2': 'SiO2', 'si02': 'SiO2', 'si o₂': 'SiO2',
//       'al2o3': 'Al2O3', 'al₂o₃': 'Al2O3',
//       'fe2o3': 'Fe2O3', 'fe₂o₃': 'Fe2O3',
//       'cao': 'CaO', 'mgo': 'MgO',
//       'na2o': 'Na2O', 'k2o': 'K2O',
//       'tio2': 'TiO2', 'tio₂': 'TiO2',
//       'so3': 'SO3', 'so₃': 'SO3',
//       'p2o5': 'P2O5', 'p₂o₅': 'P2O5',
//       'mn3o4': 'Mn3O4', 'mn₃o₄': 'Mn3O4',
//       'sulphur': 'Sulphur', 'sulphurs': 'Sulphur', 's': 'Sulphur',
//       'gcv': 'GCV', 'g c v': 'GCV', 'g.c.v.': 'GCV',
//       'cost': 'cost', 'price': 'cost', 'rate': 'cost'
//     };

//     function normalizeHeaderString(s) {
//       if (s === null || s === undefined) return '';
//       let t = String(s).trim();
//       t = t.replace(/₂/g, '2').replace(/₃/g, '3').replace(/₄/g, '4');
//       t = t.toLowerCase().replace(/[\(\)\[\]\.,\/\\\-\_]/g, ' ').replace(/\s+/g, ' ').trim();
//       return t;
//     }

//     function canonicalHeader(raw) {
//       if (!raw && raw !== 0) return '';
//       const norm = normalizeHeaderString(raw);

//       if (headerMap.hasOwnProperty(norm)) {
//         const mapped = headerMap[norm];
//         if (allowedCanonicalKeys.includes(mapped)) return mapped;
//         return '';
//       }

//       const compact = norm.replace(/\s+/g, '');
//       const foundKey = Object.keys(headerMap).find(k => k.replace(/\s+/g,'') === compact);
//       if (foundKey) {
//         const mapped = headerMap[foundKey];
//         if (allowedCanonicalKeys.includes(mapped)) return mapped;
//       }

//       const directAllowed = allowedCanonicalKeys.find(k => k.toLowerCase() === String(raw).trim().toLowerCase());
//       if (directAllowed) return directAllowed;
//       return '';
//     }

//     const canonicalHeaders = rawHeaders.map(h => canonicalHeader(h));

//     if (!canonicalHeaders.includes('coal')) {
//       return res.status(400).json({ error: "Required header 'Coal' not found. Ensure file has a 'Coal' column." });
//     }

//     // Parse rows into canonical-keyed objects (unknown columns skipped)
//     const parsed = dataRows.map((row) => {
//       if (!Array.isArray(row) || row.every(c => c === null || (typeof c === 'string' && c.trim() === ''))) return null;
//       const out = {};
//       for (let i = 0; i < canonicalHeaders.length; i++) {
//         const key = canonicalHeaders[i];
//         if (!key) continue;
//         let val = row[i] === undefined ? null : row[i];

//         // convert numeric-ish strings to Number (limit to 2 decimals)
//         if (val !== null && typeof val !== 'number') {
//           const maybeNum = Number(String(val).replace(/,/g, '').trim());
//           if (!Number.isNaN(maybeNum)) val = Math.round(maybeNum * 100) / 100;
//         }

//         if (val === '') val = null;
//         out[key] = val;
//       }
//       if (!out.coal || String(out.coal).trim() === '') return null;
//       return out;
//     }).filter(Boolean);

//     if (!parsed.length) {
//       return res.status(400).json({ error: "No valid data rows found after header (check your Excel rows)" });
//     }

//     // assign sequential coalId when missing (keep existing logic)
//     const docs = parsed.slice();
//     if (docs.length) {
//       const existingMaxDoc = await Coal.find({ coalId: { $exists: true } }).sort({ coalId: -1 }).limit(1).lean().exec();
//       let nextCoalId = 1;
//       if (existingMaxDoc && existingMaxDoc.length) {
//         const candidate = Number(existingMaxDoc[0].coalId);
//         if (Number.isFinite(candidate)) nextCoalId = Math.max(1, Math.floor(candidate) + 1);
//       }
//       docs.forEach(r => {
//         if (r.coalId === undefined || r.coalId === null || String(r.coalId).trim() === '') {
//           r.coalId = String(nextCoalId);
//           nextCoalId++;
//         } else {
//           r.coalId = String(r.coalId).trim();
//         }
//       });
//     }

//     // Final sanitize: keep only allowed canonical keys
//     const sanitized = docs.map(r => {
//       const o = {};
//       Object.keys(r).forEach(k => {
//         if (allowedCanonicalKeys.includes(k)) o[k] = r[k];
//       });
//       return o;
//     });

//     // --- NEW: map canonical keys to existing DB field names (so we don't create duplicates)
//     // Build mapping canonicalKey -> actual DB field name (if present)
//     const dbFieldMap = {}; // e.g. { 'Sulphur': 'SulphurS', 'GCV': 'gcv', ... }

//     const sampleDoc = await Coal.findOne().lean(); // inspect one existing doc (if any)
//     if (sampleDoc) {
//       Object.keys(sampleDoc).forEach(fieldName => {
//         // determine if this DB field corresponds to one of our canonical keys
//         const can = canonicalHeader(fieldName);
//         if (can && allowedCanonicalKeys.includes(can)) {
//           dbFieldMap[can] = fieldName; // use the exact DB field name found
//         }
//       });
//     }

//     // ensure defaults requested by you: map Sulphur->SulphurS and GCV->gcv when not inferred
//     if (!dbFieldMap['Sulphur']) dbFieldMap['Sulphur'] = 'SulphurS';
//     if (!dbFieldMap['GCV']) dbFieldMap['GCV'] = 'gcv';

//     // For all other canonical keys not mapped, use the canonical name as the DB field name
//     allowedCanonicalKeys.forEach(k => {
//       if (!dbFieldMap[k]) dbFieldMap[k] = k;
//     });

//     // transform sanitized docs to use DB field names (so we won't create 'Sulphur' or 'GCV' if mapping points to 'SulphurS'/'gcv')
//     const finalDocs = sanitized.map(rowCanonical => {
//       const obj = {};
//       Object.keys(rowCanonical).forEach(canKey => {
//         const targetField = dbFieldMap[canKey] || canKey;
//         // If both targetField and the canonical name would collide, we prefer writing to targetField.
//         obj[targetField] = rowCanonical[canKey];
//       });
//       return obj;
//     });

//     // Insert into DB
//     const inserted = await Coal.insertMany(finalDocs, { ordered: false });

//     return res.json({
//       message: "Data uploaded successfully",
//       rowsParsed: finalDocs.length,
//       rowsInserted: inserted.length,
//       fieldMappingPreview: dbFieldMap,
//       sampleInserted: inserted.slice(0,5)
//     });

//   } catch (err) {
//     console.error("Error in upload-excel (strict header handler + DB-mapping):", err);
//     return res.status(500).json({ error: "Failed to process file", details: String(err) });
//   }
// });
app.post("/upload-excel", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  try {
    const workbook = xlsx.read(req.file.buffer, { type: "buffer" });
    if (!workbook.SheetNames || workbook.SheetNames.length === 0) {
      return res.status(400).json({ error: "No sheets found in workbook" });
    }

    const sheet = workbook.Sheets[workbook.SheetNames[0]];
    const rows = xlsx.utils.sheet_to_json(sheet, { header: 1, defval: null });

    // Canonical DB keys we will allow (strict: unknown headers are ignored)
    const allowedCanonicalKeys = [
      'coalId','coal',
      'SiO2','Al2O3','Fe2O3','CaO','MgO','Na2O','K2O','TiO2',
      'SO3','P2O5','Mn3O4',
      'Sulphur','GCV','cost'
    ];

    // Header alias map -> canonical DB key
    const headerMap = {
      'coal': 'coal',
      'coalid': 'coalId', 'coal id': 'coalId', 'coal_id': 'coalId',
      'sio2': 'SiO2', 'si o2': 'SiO2', 'si02': 'SiO2', 'si o₂': 'SiO2',
      'al2o3': 'Al2O3', 'al₂o₃': 'Al2O3',
      'fe2o3': 'Fe2O3', 'fe₂o₃': 'Fe2O3',
      'cao': 'CaO', 'mgo': 'MgO',
      'na2o': 'Na2O', 'k2o': 'K2O',
      'tio2': 'TiO2', 'tio₂': 'TiO2',
      'so3': 'SO3', 'so₃': 'SO3',
      'p2o5': 'P2O5', 'p₂o₅': 'P2O5',
      'mn3o4': 'Mn3O4', 'mn₃o₄': 'Mn3O4',
      'sulphur': 'Sulphur', 'sulphurs': 'Sulphur', 's': 'Sulphur',
      'gcv': 'GCV', 'g c v': 'GCV', 'g.c.v.': 'GCV',
      'cost': 'cost', 'price': 'cost', 'rate': 'cost'
    };

    function normalizeHeaderString(s) {
      if (s === null || s === undefined) return '';
      let t = String(s).trim();
      t = t.replace(/₂/g, '2').replace(/₃/g, '3').replace(/₄/g, '4');
      t = t.toLowerCase().replace(/[\(\)\[\]\.,\/\\\-\_]/g, ' ').replace(/\s+/g, ' ').trim();
      return t;
    }

    function canonicalHeader(raw) {
      if (!raw && raw !== 0) return '';
      const norm = normalizeHeaderString(raw);

      if (headerMap.hasOwnProperty(norm)) {
        const mapped = headerMap[norm];
        if (allowedCanonicalKeys.includes(mapped)) return mapped;
        return '';
      }

      const compact = norm.replace(/\s+/g, '');
      const foundKey = Object.keys(headerMap).find(k => k.replace(/\s+/g,'') === compact);
      if (foundKey) {
        const mapped = headerMap[foundKey];
        if (allowedCanonicalKeys.includes(mapped)) return mapped;
      }

      const directAllowed = allowedCanonicalKeys.find(k => k.toLowerCase() === String(raw).trim().toLowerCase());
      if (directAllowed) return directAllowed;
      return '';
    }

    // --- NEW: only check first then second row for header row (in that order)
    let headerRowIndex = -1;
    let rawHeaders = [];

    // helper to test a candidate row index
    const testCandidate = (idx) => {
      if (idx < 0 || idx >= rows.length) return false;
      const candidateRow = rows[idx];
      if (!Array.isArray(candidateRow)) return false;
      const canonicalCandidates = candidateRow.map(h => canonicalHeader(h));
      // require presence of 'coal' header to accept this row as header
      return canonicalCandidates.includes('coal');
    };

    if (testCandidate(0)) {
      headerRowIndex = 0;
      rawHeaders = rows[0].map(h => (h === null || h === undefined) ? '' : String(h).trim());
    } else if (testCandidate(1)) {
      headerRowIndex = 1;
      rawHeaders = rows[1].map(h => (h === null || h === undefined) ? '' : String(h).trim());
    } else {
      // neither first nor second row worked -> incorrect format
      return res.status(400).json({ error: "Could not find header row in first two rows. Ensure the file's headers are on the 1st or 2nd row and that a 'Coal' column exists." });
    }

    const dataRows = rows.slice(headerRowIndex + 1);

    // Build canonical headers from chosen rawHeaders
    const canonicalHeaders = rawHeaders.map(h => canonicalHeader(h));

    // Parse rows into canonical-keyed objects (unknown columns skipped)
    const parsed = dataRows.map((row) => {
      if (!Array.isArray(row) || row.every(c => c === null || (typeof c === 'string' && c.trim() === ''))) return null;
      const out = {};
      for (let i = 0; i < canonicalHeaders.length; i++) {
        const key = canonicalHeaders[i];
        if (!key) continue;
        let val = row[i] === undefined ? null : row[i];

        // convert numeric-ish strings to Number (limit to 2 decimals)
        if (val !== null && typeof val !== 'number') {
          const maybeNum = Number(String(val).replace(/,/g, '').trim());
          if (!Number.isNaN(maybeNum)) val = Math.round(maybeNum * 100) / 100;
        }

        if (val === '') val = null;
        out[key] = val;
      }
      if (!out.coal || String(out.coal).trim() === '') return null;
      return out;
    }).filter(Boolean);

    if (!parsed.length) {
      return res.status(400).json({ error: "No valid data rows found after header (check your Excel rows)" });
    }

    // assign sequential coalId when missing (keep existing logic)
    const docs = parsed.slice();
    if (docs.length) {
      const existingMaxDoc = await Coal.find({ coalId: { $exists: true } }).sort({ coalId: -1 }).limit(1).lean().exec();
      let nextCoalId = 1;
      if (existingMaxDoc && existingMaxDoc.length) {
        const candidate = Number(existingMaxDoc[0].coalId);
        if (Number.isFinite(candidate)) nextCoalId = Math.max(1, Math.floor(candidate) + 1);
      }
      docs.forEach(r => {
        if (r.coalId === undefined || r.coalId === null || String(r.coalId).trim() === '') {
          r.coalId = String(nextCoalId);
          nextCoalId++;
        } else {
          r.coalId = String(r.coalId).trim();
        }
      });
    }

    // Final sanitize: keep only allowed canonical keys
    const sanitized = docs.map(r => {
      const o = {};
      Object.keys(r).forEach(k => {
        if (allowedCanonicalKeys.includes(k)) o[k] = r[k];
      });
      return o;
    });

    // --- NEW: map canonical keys to existing DB field names (so we don't create duplicates)
    // Build mapping canonicalKey -> actual DB field name (if present)
    const dbFieldMap = {}; // e.g. { 'Sulphur': 'SulphurS', 'GCV': 'gcv', ... }

    const sampleDoc = await Coal.findOne().lean(); // inspect one existing doc (if any)
    if (sampleDoc) {
      Object.keys(sampleDoc).forEach(fieldName => {
        // determine if this DB field corresponds to one of our canonical keys
        const can = canonicalHeader(fieldName);
        if (can && allowedCanonicalKeys.includes(can)) {
          dbFieldMap[can] = fieldName; // use the exact DB field name found
        }
      });
    }

    // ensure defaults requested by you: map Sulphur->SulphurS and GCV->gcv when not inferred
    if (!dbFieldMap['Sulphur']) dbFieldMap['Sulphur'] = 'SulphurS';
    if (!dbFieldMap['GCV']) dbFieldMap['GCV'] = 'gcv';

    // For all other canonical keys not mapped, use the canonical name as the DB field name
    allowedCanonicalKeys.forEach(k => {
      if (!dbFieldMap[k]) dbFieldMap[k] = k;
    });

    // transform sanitized docs to use DB field names (so we won't create 'Sulphur' or 'GCV' if mapping points to 'SulphurS'/'gcv')
    const finalDocs = sanitized.map(rowCanonical => {
      const obj = {};
      Object.keys(rowCanonical).forEach(canKey => {
        const targetField = dbFieldMap[canKey] || canKey;
        // If both targetField and the canonical name would collide, we prefer writing to targetField.
        obj[targetField] = rowCanonical[canKey];
      });
      return obj;
    });

    // Insert into DB
    const inserted = await Coal.insertMany(finalDocs, { ordered: false });

    return res.json({
      message: "Data uploaded successfully",
      rowsParsed: finalDocs.length,
      rowsInserted: inserted.length,
      fieldMappingPreview: dbFieldMap,
      sampleInserted: inserted.slice(0,5)
    });

  } catch (err) {
    console.error("Error in upload-excel (strict header handler + DB-mapping):", err);
    return res.status(500).json({ error: "Failed to process file", details: String(err) });
  }
});
// FETCH ALL DATA
app.get("/fetch-data", async (req, res) => {
  try {
    const data = await Coal.find({}, { __v: 0 }).lean();
    res.json(data);
  } catch (error) {
    console.error("Error fetching data:", error);
    res.status(500).json({ error: "Failed to fetch data" });
  }
});

// DELETE selected rows — supports Mongo _id (24-hex) and coalId
app.delete("/delete-data", async (req, res) => {
  try {
    const { ids } = req.body;
    if (!Array.isArray(ids) || ids.length === 0) return res.status(400).json({ error: "No IDs provided" });

    const objectIds = ids.filter(id => /^[0-9a-fA-F]{24}$/.test(String(id)));
    const coalIdValues = ids.filter(id => !/^[0-9a-fA-F]{24}$/.test(String(id)));

    let totalDeleted = 0;
    if (objectIds.length) {
      const r1 = await Coal.deleteMany({ _id: { $in: objectIds } });
      totalDeleted += (r1 && r1.deletedCount) ? r1.deletedCount : 0;
    }
    if (coalIdValues.length) {
      const r2 = await Coal.deleteMany({ coalId: { $in: coalIdValues } });
      totalDeleted += (r2 && r2.deletedCount) ? r2.deletedCount : 0;
    }

    if (totalDeleted === 0) return res.status(404).json({ error: "No data found" });
    res.json({ message: `${totalDeleted} data deleted successfully` });
  } catch (error) {
    console.error("Error deleting data:", error);
    res.status(500).json({ error: "Failed to delete data" });
  }
});

// AUTH ROUTES
app.post('/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email & password required' });
    const user = await User.findOne({ email: email.toLowerCase().trim() });
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });

    if (user.lockedUntil && user.lockedUntil > new Date()) {
      return res.status(403).json({ error: 'Account locked until ' + user.lockedUntil.toISOString() });
    }
    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) return res.status(401).json({ error: 'Invalid credentials' });

    const ip = getClientIp(req);
    user.lastIP = ip;
    user.ipHistory = user.ipHistory || [];
    user.ipHistory.push({ ip, when: new Date() });
    await user.save();

    req.session.userId = user._id.toString();
    res.json({ message: 'Logged in' });
  } catch (err) {
    console.error('/auth/login error', err);
    res.status(500).json({ error: 'Login failed' });
  }
});

app.post('/auth/logout', (req, res) => {
  req.session.destroy(err => {
    if (err) console.error('session destroy err', err);
    res.json({ message: 'Logged out' });
  });
});

app.get('/auth/status', async (req, res) => {
  try {
    if (!req.session || !req.session.userId) return res.json({ authenticated: false });
    const user = await User.findById(req.session.userId, 'email trialsLeft lockedUntil lastIP');
    if (!user) return res.json({ authenticated: false });
    return res.json({
      authenticated: true,
      email: user.email,
      trialsLeft: user.trialsLeft,
      lockedUntil: user.lockedUntil,
      lastIP: user.lastIP
    });
  } catch (err) {
    console.error('/auth/status error', err);
    res.status(500).json({ error: 'Status check failed' });
  }
});
// Activate model (frontend calls this)
app.post('/api/activate-model', async (req, res) => {
  try {
    const modelName = req.body.modelName;
    if (!modelName) return res.json({ success: false, message: "Model name missing" });

    let scriptPath = null;
    if (modelName === 'pure_rf') scriptPath = path.join(__dirname, 'rf2.js');
    else if (modelName === 'hybrid_rf') scriptPath = path.join(__dirname, 'hybrid2.js');
    else if (modelName === 'ann') scriptPath = path.join(__dirname, 'ann2.js');
    else if (modelName === 'base_formula') {
      // Switch to base formula without invoking training script
      fs.writeFileSync('active_model.json', JSON.stringify({ type: 'base_formula' }, null, 2));
      loadActiveModel();
      refreshActiveModelFromMetadata();
      return res.json({ success: true });
    }
    else if (modelName === 'pure_xgb' || modelName === 'hybrid_xgb') {
  // For XGBoost we DO NOT run JS trainer
  fs.writeFileSync('active_model.json', JSON.stringify({ type: modelName }, null, 2));
  activeModel.type = modelName;
  activeModel.model = null;
  activeModel.raw = null;
  activeModel.annScaling = null;
  return res.json({ success: true });
}
    else return res.json({ success: false, message: "Invalid model selected" });

    if (!fs.existsSync(scriptPath)) return res.json({ success: false, message: "Script not found" });

    // prefer the user-uploaded CSV if available
    const lastPath = path.join(__dirname, 'last_training_upload.csv');
    const trainingArg = fs.existsSync(lastPath) ? lastPath : (process.argv[2] || 'aft_training_data1.csv');

    console.log("Running trainer:", scriptPath, "with training CSV:", trainingArg);

    const child = spawn(process.execPath, [scriptPath, trainingArg], { env: process.env });

    child.stdout.on('data', (data) => { process.stdout.write('[trainer stdout] ' + data.toString()); });
    child.stderr.on('data', (data) => { process.stderr.write('[trainer stderr] ' + data.toString()); });

    child.on('close', (code) => {
      if (code !== 0) {
        console.error("Training failed with exit code", code);
        return res.json({ success: false, message: "Training failed" });
      }

      // Mark active model and reload in-memory model
      fs.writeFileSync('active_model.json', JSON.stringify({ type: modelName }, null, 2));
      try { loadActiveModel(); } catch (e) { console.warn('loadActiveModel after activation failed', e && e.message); }

      console.log("Model activated:", modelName);
      return res.json({ success: true });
    });

  } catch (err) {
    console.error('/api/activate-model error:', err);
    return res.json({ success: false, message: String(err) });
  }
});

// ---------- OPTIMIZE ROUTE (modified to enforce trials & log IP) ----------
// app.post("/optimize", requireAuth, async (req, res) => {
//   try {
//     const user = req.currentUser;
//     // ensure not locked (requireAuth already checks but double-check)
//     if (user.lockedUntil && user.lockedUntil > new Date()) {
//       return res.status(403).json({ error: 'Account locked until ' + user.lockedUntil.toISOString() });
//     }

//     // check trials
//     if ((user.trialsLeft || 0) <= 0) {
//       // enforce lock for 24 hours starting now
//       user.lockedUntil = new Date(Date.now() + 24 * 3600 * 1000);
//       await user.save();
//       req.session.destroy(()=>{});
//       return res.status(403).json({ error: 'Trials exhausted. Account locked for 24 hours.' });
//     }

//     // record IP for this calculation call
//     const ip = getClientIp(req);
//     user.lastIP = ip;
//     user.ipHistory = user.ipHistory || [];
//     user.ipHistory.push({ ip, when: new Date() });

//     // ---- YOUR EXISTING OPTIMIZATION LOGIC STARTS HERE ----
//     const { blends } = req.body;
//     if (!blends || !Array.isArray(blends) || blends.length === 0) {
//         return res.status(400).json({ error: "Invalid blend data" });
//     }
//     const oxideCols = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'Na2O', 'K2O', 'SO3', 'TiO2'];
//     const coalNames = blends.map(b => b.coal);
//     const oxideValues = blends.map(b => oxideCols.map(col => b.properties[col] || 0));
//     const minMaxBounds = blends.map(b => [b.min, b.max]);
//     const costsPerTon = blends.map(b => b.cost);
//     const gcvValue = blends.map(b => b.properties.Gcv);
//     const individualCoalAFTs = oxideValues.map((vals, i) => ({
//       coal: coalNames[i],
//       predicted_aft: calculateAFT(vals)
//     }));
//     function* generateCombinations(bounds, step) {
//       function* helper(index, combo) {
//         if (index === bounds.length) {
//           const sum = combo.reduce((a, b) => a + b, 0);
//           if (sum === 100) yield combo;
//           return;
//         }
//         const [min, max] = bounds[index];
//         for (let i = min; i <= max; i += step) yield* helper(index + 1, [...combo, i]);
//       }
//       yield* helper(0, []);
//     }
//     const step = 1;
//     const validBlends = [];
//     for (const blend of generateCombinations(minMaxBounds, step)) {
//       const weights = blend.map(x => x / 100);
//       const blendedOxides = oxideCols.map((_, i) =>
//         oxideValues.reduce((sum, val, idx) => sum + val[i] * weights[idx], 0)
//       );
//       const predictedAFT = calculateAFT(blendedOxides);
//       const totalgcv = blend.reduce((sum, pct, i ) => sum + pct*gcvValue[i], 0) / 100;
//       const totalCost = blend.reduce((sum, pct, i) => sum + pct * costsPerTon[i], 0) / 100;
//       validBlends.push({ blend, predicted_aft: predictedAFT, cost: totalCost, gcv: totalgcv, blended_oxides: blendedOxides });
//     }
//     if (validBlends.length === 0) return res.status(404).json({ message: "No valid blends found" });
//     const aftVals = validBlends.map(b => b.predicted_aft);
//     const costVals = validBlends.map(b => b.cost);
//     const aftMin = Math.min(...aftVals);
//     const aftMax = Math.max(...aftVals);
//     const costMin = Math.min(...costVals);
//     const costMax = Math.max(...costVals);
//     const blendScores = validBlends.map((b, i) => {
//       const aftNorm = (b.predicted_aft - aftMin) / (aftMax - aftMin);
//       const costNorm = (costMax - b.cost) / (costMax - costMin);
//       return aftNorm + costNorm;
//     });
//     const bestAftBlend = validBlends[aftVals.indexOf(Math.max(...aftVals))];
//     const cheapestBlend = validBlends[costVals.indexOf(Math.min(...costVals))];
//     const balancedBlend = validBlends[blendScores.indexOf(Math.max(...blendScores))];
//     const currentWeights = blends.map(b => b.current / 100);
//     const currentBlendedOxides = oxideCols.map((_, i) =>
//       oxideValues.reduce((sum, val, idx) => sum + val[i] * currentWeights[idx], 0)
//     );
//     const currentAFT = calculateAFT(currentBlendedOxides);
//     const currentGCV = blends.reduce((sum, b, i) => sum + (b.current * gcvValue[i]), 0) / 100;
//     const currentCost = blends.reduce((sum, b, i) => sum + (b.current * costsPerTon[i]), 0) / 100;
//     const currentBlend = { blend: blends.map(b => b.current), predicted_aft: currentAFT, gcv: currentGCV, cost: currentCost };
//     // ---- YOUR EXISTING OPTIMIZATION LOGIC ENDS HERE ----

//     // decrement trials and save user
//     user.trialsLeft = (user.trialsLeft || 1) - 1;
//     // if hits 0 we lock and destroy session
//     if (user.trialsLeft <= 0) {
//       user.lockedUntil = new Date(Date.now() + 24 * 3600 * 1000);
//       await user.save();
//       req.session.destroy(()=>{});
//       return res.status(200).json({
//          message: 'Calculation ran and this was your final trial. Account locked for 24 hours.',
//          best_aft_blend: bestAftBlend,
//          cheapest_blend: cheapestBlend,
//          balanced_blend: balancedBlend,
//          current_blend: currentBlend,
//          individual_coal_afts: individualCoalAFTs,
//          trialsLeft: 0,
//          lockedUntil: user.lockedUntil
//       });
//     } else {
//       await user.save();
//       return res.json({
//         best_aft_blend: bestAftBlend,
//         cheapest_blend: cheapestBlend,
//         balanced_blend: balancedBlend,
//         current_blend: currentBlend,
//         individual_coal_afts: individualCoalAFTs,
//         trialsLeft: user.trialsLeft
//       });
//     }
//   } catch (err) {
//     console.error("Optimization error:", err);
//     res.status(500).json({ error: "Internal server error" });
//   }
// });
// ---------- OPTIMIZE ROUTE (replace the existing handler with this) ----------
app.post("/optimize", requireAuth, async (req, res) => {
  try {
    const user = req.currentUser;
    // double-check lock
    if (user.lockedUntil && user.lockedUntil > new Date()) {
      return res.status(403).json({ error: 'Account locked until ' + user.lockedUntil.toISOString() });
    }

    // check trials
    if ((user.trialsLeft || 0) <= 0) {
      user.lockedUntil = new Date(Date.now() + 24 * 3600 * 1000);
      await user.save();
      req.session.destroy(()=>{});
      return res.status(403).json({ error: 'Trials exhausted. Account locked for 24 hours.' });
    }

    // record IP for this calculation call
    const ip = getClientIp(req);
    user.lastIP = ip;
    user.ipHistory = user.ipHistory || [];
    user.ipHistory.push({ ip, when: new Date() });

    // ---- OPTIMIZATION LOGIC ----
    const { blends } = req.body;
    if (!blends || !Array.isArray(blends) || blends.length === 0) {
      return res.status(400).json({ error: "Invalid blend data" });
    }

    // oxide columns order used by calculateAFT
    const oxideCols = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'Na2O', 'K2O', 'SO3', 'TiO2'];

    // prepare arrays
    const coalNames = blends.map(b => b.coal || '');
    const oxideValues = blends.map(b => oxideCols.map(col => {
      const v = (b.properties && (b.properties[col] ?? b.properties[col.toUpperCase()] ?? b.properties[col.toLowerCase()])) ?? 0;
      const n = Number(v);
      return Number.isFinite(n) ? n : 0;
    }));

    // sanitize min/max/current
    const minMaxBounds = blends.map(b => {
      const min = Number.isFinite(Number(b.min)) ? Number(b.min) : 0;
      const max = Number.isFinite(Number(b.max)) ? Number(b.max) : 100;
      // ensure sensible order and clamp
      const mm = Math.max(0, Math.min(100, min));
      const mx = Math.max(mm, Math.min(100, max));
      return [Math.round(mm), Math.round(mx)];
    });

    // sanitize costs: if blank/invalid -> 0
    const costsPerTon = blends.map(b => {
      if (b.cost === null || b.cost === undefined || b.cost === '') return 0;
      const n = Number(String(b.cost).replace(/,/g, '').trim());
      return Number.isFinite(n) ? n : 0;
    });

    // sanitize gcv values (if missing -> 0)
    const gcvValue = blends.map(b => {
      const gRaw = b.properties && (b.properties.GCV ?? b.properties.Gcv ?? b.properties.gcv ?? b.properties.gcv);
      const n = Number(gRaw);
      return Number.isFinite(n) ? n : 0;
    });

    // step granularity (1% step)
    const step = 1;

    // generator for combinations with bounds and integer steps that sum to 100
    function* generateCombinations(bounds, idx = 0, acc = []) {
      if (idx === bounds.length - 1) {
        // last one is determined so that sum is 100 - sum(acc)
        const sumSoFar = acc.reduce((s, v) => s + v, 0);
        const lastVal = 100 - sumSoFar;
        const [minLast, maxLast] = bounds[idx];
        if (lastVal >= minLast && lastVal <= maxLast) {
          yield [...acc, lastVal];
        }
        return;
      }
      const [min, max] = bounds[idx];
      for (let v = min; v <= max; v += step) {
        const sumSoFar = acc.reduce((s, vv) => s + vv, 0) + v;
        // quick pruning: if sumSoFar > 100, break this loop (since further v increases sum)
        if (sumSoFar > 100) break;
        // minimal possible remaining (all remaining mins)
        const remainingMin = bounds.slice(idx + 1).reduce((s, b) => s + b[0], 0);
        const remainingMax = bounds.slice(idx + 1).reduce((s, b) => s + b[1], 0);
        // if even with all max we can't reach 100, skip this v
        if (sumSoFar + remainingMax < 100) continue;
        // if even with all min we already exceed 100, skip
        if (sumSoFar + remainingMin > 100) continue;
        yield* generateCombinations(bounds, idx + 1, [...acc, v]);
      }
    }

    // Evaluate valid blends
    const validBlends = [];
    for (const blend of generateCombinations(minMaxBounds)) {
      // weights as fractions for oxide blending
      const weights = blend.map(x => x / 100);
      // blended oxides
      const blendedOxides = oxideCols.map((_, oi) =>
        oxideValues.reduce((sum, val, idx) => sum + val[oi] * weights[idx], 0)
      );

      // predicted AFT using your calculateAFT (assumed defined above)
      const predictedAFT = calculateAFT(blendedOxides);
      console.log('Evaluated blend:', blend, 'Predicted AFT:', predictedAFT);

      // total GCV and cost: blend array is percentage integers, divide by 100
      const totalGcv = blend.reduce((sum, pct, i) => sum + pct * (gcvValue[i] || 0), 0) / 100;
      const totalCost = blend.reduce((sum, pct, i) => sum + pct * (costsPerTon[i] || 0), 0) / 100;

      validBlends.push({
        blend, // array of integer percentages summing to 100
        predicted_aft: predictedAFT,
        cost: totalCost,
        gcv: totalGcv,
        blended_oxides: blendedOxides
      });
    }

    if (validBlends.length === 0) {
      return res.status(404).json({ message: "No valid blends found" });
    }

    // helper arrays
    const aftVals = validBlends.map(b => b.predicted_aft);
    const costVals = validBlends.map(b => b.cost);

    // compute mins/max safely
    const aftMin = Math.min(...aftVals);
    const aftMax = Math.max(...aftVals);
    const costMin = Math.min(...costVals);
    const costMax = Math.max(...costVals);

    // scoring to find a "balanced" blend (maximize aft normalized + cost normalized)
    const blendScores = validBlends.map(b => {
      const aftNorm = (aftMax === aftMin) ? 0.5 : (b.predicted_aft - aftMin) / (aftMax - aftMin);
      // costNorm higher when cost is lower
      const costNorm = (costMax === costMin) ? 0.5 : (costMax - b.cost) / (costMax - costMin);
      return aftNorm + costNorm;
    });

    // pick best variants
    const indexOfBestAft = aftVals.indexOf(Math.max(...aftVals));
    const indexOfCheapest = costVals.indexOf(Math.min(...costVals));
    const indexOfBalanced = blendScores.indexOf(Math.max(...blendScores));

    const bestAftBlend = validBlends[indexOfBestAft];
    const cheapestBlend = validBlends[indexOfCheapest];
    const balancedBlend = validBlends[indexOfBalanced];

    // compute current blend (from user-supplied 'current' values)
    const currentWeights = blends.map(b => {
      const n = Number(b.current);
      return Number.isFinite(n) ? n / 100 : 0;
    });

    const currentBlendedOxides = oxideCols.map((_, oi) =>
      oxideValues.reduce((sum, val, idx) => sum + val[oi] * (currentWeights[idx] || 0), 0)
    );
    const currentAFT = calculateAFT(currentBlendedOxides);
    console.log('Current blend oxides:', currentBlendedOxides, 'Predicted AFT:', currentAFT);
    const currentGCV = blends.reduce((sum, b, i) => sum + (Number(b.current) || 0) * (gcvValue[i] || 0), 0) / 100;
    const currentCost = blends.reduce((sum, b, i) => sum + (Number(b.current) || 0) * (costsPerTon[i] || 0), 0) / 100;
    const currentBlend = { blend: blends.map(b => Number(b.current) || 0), predicted_aft: currentAFT, gcv: currentGCV, cost: currentCost };

    // individual coal AFTs (based on each coal's oxide vector)
    const individualCoalAFTs = oxideValues.map((vals, i) => ({
      coal: coalNames[i] || `coal-${i}`,
      predicted_aft: calculateAFT(vals)
    }));

    console.log('Optimization complete. Best AFT blend:', individualCoalAFTs);

    // Decrement trial for user and save (if you want to count this run)
    user.trialsLeft = Math.max(0, (user.trialsLeft || 0) - 1);
    if (user.trialsLeft <= 0) {
      user.lockedUntil = new Date(Date.now() + 24 * 3600 * 1000);
    }
    await user.save();

    console.log(`best AFT: ${bestAftBlend.blend}, cheapest cost: ${cheapestBlend.blend}, balanced score: ${balancedBlend.blend}, trials left: ${user.trialsLeft}`)

    // return response shaped to client expectations
    return res.json({
      best_aft_blend: bestAftBlend,
      cheapest_blend: cheapestBlend,
      balanced_blend: balancedBlend,
      current_blend: currentBlend,
      individual_coal_afts: individualCoalAFTs
    });

  } catch (err) {
    console.error('/optimize error', err);
    return res.status(500).json({ error: 'Optimization failed', details: String(err) });
  }
});

// ---------- COMPATIBILITY API (for second website / input.html) ----------

// Return array of normalized coal docs for dropdowns
app.get(['/api/coal','/api/coals','/api/coal/list','/api/coal/all'], async (req, res) => {
  try {
    const docs = await Coal.find({}).lean().exec();
    const normalized = docs.map(d => normalizeCoalDoc(d));
    return res.json(normalized);
  } catch (err) {
    console.error('GET /api/coals error:', err);
    return res.status(500).json({ error: err.message || 'Server error' });
  }
});

// Minimal payload for names-only requests
app.get('/api/coalnames', async (req, res) => {
  try {
    const docs = await Coal.find({}, { coal: 1 }).lean().exec();
    const minimal = docs.map(d => ({ _id: d._id, coal: d.coal || d['Coal source name'] || d.name }));
    return res.json(minimal);
  } catch (err) {
    console.error('GET /api/coalnames error:', err);
    return res.status(500).json({ error: err.message || 'Server error' });
  }
});

// Return shape expected by model.html (coal_data: [...])
app.get('/get_coal_types', async (req, res) => {
  try {
    const docs = await Coal.find({}).lean().exec();
    const requiredProps = [
      "SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O", "TiO2",
      "SO3", "P2O5", "Mn3O4", "Sulphur (S)", "GCV"
    ];
    const coalData = docs.map(row => {
      const id = String(row._id || row.id || '');
      const coalType = row.coal || row.name || row['Coal source name'] || '';
      const transportId = row['Transport ID'] || row.transportId || null;
      const properties = {};
      requiredProps.forEach(prop => {
        properties[prop] = row[prop] !== undefined ? row[prop] : (row[prop.replace('2','₂')] !== undefined ? row[prop.replace('2','₂')] : null);
      });
      if ((properties['GCV'] === null || properties['GCV'] === undefined) && (row.gcv || row.GCV || row.Gcv)) {
        properties['GCV'] = row.gcv || row.GCV || row.Gcv;
      }
      if ((properties['Sulphur (S)'] === null || properties['Sulphur (S)'] === undefined)) {
        properties['Sulphur (S)'] = row['Sulphur (S)'] || row['SulphurS'] || row['Sulphur'] || row.S || null;
      }
      return { id, coalType, transportId, properties };
    });
    return res.json({ coal_data: coalData });
  } catch (error) {
    console.error('/get_coal_types error:', error);
    return res.status(500).json({ error: 'Failed to fetch coal types' });
  }
});
// POST /consume-trial
// Decrements trialsLeft by 1 for the logged-in user, locks account for 24h if it reaches 0,
// logs the user out by destroying the session, and returns current trials and lockedUntil.
app.post('/consume-trial', requireAuth, async (req, res) => {
  try {
    const user = req.currentUser; // set by requireAuth

    // If already locked (should be handled by requireAuth) return locked info
    if (user.lockedUntil && user.lockedUntil > new Date()) {
      return res.status(403).json({ error: 'Account locked', lockedUntil: user.lockedUntil, trialsLeft: user.trialsLeft });
    }

    // Decrement only if > 0
    user.trialsLeft = (user.trialsLeft || 0) - 1;

    // If trials go to 0 or negative, lock for 24 hours and destroy session
    if (user.trialsLeft <= 0) {
      user.trialsLeft = 0;
      user.lockedUntil = new Date(Date.now() + 24 * 3600 * 1000);
      await user.save();

      // destroy session (log out)
      req.session.destroy(err => {
        if (err) console.error('session destroy error during consume-trial', err);
        // respond to client (session is gone)
        return res.json({ message: 'Trials exhausted. Account locked for 24 hours.', trialsLeft: user.trialsLeft, lockedUntil: user.lockedUntil });
      });
      return;
    }

    // Otherwise, save and return trialsLeft
    await user.save();
    return res.json({ message: 'Trial consumed', trialsLeft: user.trialsLeft, lockedUntil: user.lockedUntil || null });
  } catch (err) {
    console.error('/consume-trial error', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// ---------- START SERVER ----------
// ---------- START SERVER ----------
if (!process.env.VERCEL) {
  // Only start a real HTTP server when running locally (dev)
  app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
  });
}

// Always export the Express app so @vercel/node can use it as a handler
module.exports = app;

