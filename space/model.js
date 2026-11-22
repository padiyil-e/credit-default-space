// Feature pipeline and random forest inference, mirroring training/features.py
// and scikit-learn's predict_proba. Used by index.html and by the CI tests (Node).

export const PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"];
export const BILL_AMT = [1, 2, 3, 4, 5, 6].map((i) => `BILL_AMT${i}`);
export const PAY_AMT = [1, 2, 3, 4, 5, 6].map((i) => `PAY_AMT${i}`);
export const RAW_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
  ...PAY_STATUS, ...BILL_AMT, ...PAY_AMT];
const MONEY = new Set(["LIMIT_BAL", ...BILL_AMT, ...PAY_AMT]);

const signedLog = (x) => (x === 0 ? 0 : Math.sign(x) * Math.log1p(Math.abs(x)));

/** Raw UCI-format client (object keyed by RAW_COLUMNS) -> object keyed by feature name. */
export function buildFeatures(raw) {
  const missing = RAW_COLUMNS.filter((c) => raw[c] === undefined || raw[c] === "" || Number.isNaN(Number(raw[c])));
  if (missing.length) throw new Error(`Missing or non-numeric columns: ${missing.join(", ")}`);
  const r = Object.fromEntries(RAW_COLUMNS.map((c) => [c, Number(raw[c])]));

  const sex = r.SEX === 1 ? "male" : "female";
  const education = { 1: "graduate", 2: "university", 3: "high_school" }[r.EDUCATION] ?? "other";
  const marriage = { 1: "married", 2: "single" }[r.MARRIAGE] ?? "other";

  const pay = PAY_STATUS.map((c) => r[c]);
  const limit = Math.max(r.LIMIT_BAL, 1);
  let util = 0;
  for (const c of BILL_AMT) util += r[c] / limit;

  const f = {};
  f.LIMIT_BAL = r.LIMIT_BAL;
  f.AGE = r.AGE;
  for (const c of [...PAY_STATUS, ...BILL_AMT, ...PAY_AMT]) f[c] = r[c];
  f.MONTHS_DELINQUENT = pay.filter((v) => v > 0).length;
  f.MAX_DELINQUENCY = Math.max(...pay.map((v) => Math.max(v, 0)));
  f.AVG_UTILIZATION = util / 6;
  for (const c of MONEY) f[c] = signedLog(f[c]);
  f.SEX_female = sex === "female" ? 1 : 0;
  f.EDUCATION_university = education === "university" ? 1 : 0;
  f.EDUCATION_high_school = education === "high_school" ? 1 : 0;
  f.EDUCATION_other = education === "other" ? 1 : 0;
  f.MARRIAGE_single = marriage === "single" ? 1 : 0;
  f.MARRIAGE_other = marriage === "other" ? 1 : 0;
  return f;
}

/** Probability of default: the average of each tree's leaf probability.
 *  scikit-learn compares float32 inputs with float64 thresholds, so inputs are
 *  rounded to float32 (Math.fround) to give identical splits. */
export function predictProba(model, raw) {
  const feats = buildFeatures(raw);
  const x = model.features.map((name) => Math.fround(feats[name]));
  let sum = 0;
  for (const t of model.trees) {
    let n = 0;
    while (t.f[n] !== -1) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
    sum += t.v[n];
  }
  return sum / model.trees.length;
}

/** Minimal CSV parser for numeric client files (header row + comma-separated values). */
export function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  return lines.slice(1).filter((l) => l.trim()).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(header.map((h, i) => [h, (cells[i] ?? "").trim().replace(/^"|"$/g, "")]));
  });
}
