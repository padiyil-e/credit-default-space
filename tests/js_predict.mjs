// Scores every client in a CSV with the browser model (space/model.js) and prints JSON.
// Usage: node tests/js_predict.mjs data/credit_default.csv
import { readFileSync } from "node:fs";
import { predictProba, parseCsv } from "../space/model.js";

const model = JSON.parse(readFileSync(new URL("../space/model.json", import.meta.url)));
const rows = parseCsv(readFileSync(process.argv[2], "utf8"));
process.stdout.write(JSON.stringify(rows.map((r) => predictProba(model, r))));
