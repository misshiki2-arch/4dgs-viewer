#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {
    input: null,
    time: null,
    sigmaThresh: 3.0,
    sigmaMode: 'exp',
    sampleCount: 10
  };

  const rest = [...argv];
  while (rest.length > 0) {
    const a = rest.shift();
    if (!a) continue;

    if (!args.input && !a.startsWith('--')) {
      args.input = a;
      continue;
    }
    if (a === '--time') {
      args.time = Number(rest.shift());
      continue;
    }
    if (a === '--sigma-thresh') {
      args.sigmaThresh = Number(rest.shift());
      continue;
    }
    if (a === '--sigma-mode') {
      args.sigmaMode = String(rest.shift() || 'exp');
      continue;
    }
    if (a === '--sample-count') {
      args.sampleCount = Math.max(1, parseInt(rest.shift(), 10));
      continue;
    }
    throw new Error(`Unknown argument: ${a}`);
  }

  if (!args.input) {
    throw new Error('Input file is required.');
  }
  if (!['exp', 'linear'].includes(args.sigmaMode)) {
    throw new Error(`--sigma-mode must be exp or linear, got: ${args.sigmaMode}`);
  }
  return args;
}

function finiteValues(arr) {
  if (!arr) return [];
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) out.push(v);
  }
  return out;
}

function quantileSorted(sorted, q) {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const p = (sorted.length - 1) * q;
  const i0 = Math.floor(p);
  const i1 = Math.min(sorted.length - 1, i0 + 1);
  const t = p - i0;
  return sorted[i0] * (1 - t) + sorted[i1] * t;
}

function summarizeArray(name, arr) {
  const vals = finiteValues(arr);
  vals.sort((a, b) => a - b);

  if (vals.length === 0) {
    return {
      name,
      count: 0,
      min: NaN,
      q01: NaN,
      q10: NaN,
      q50: NaN,
      q90: NaN,
      q99: NaN,
      max: NaN,
      mean: NaN
    };
  }

  let sum = 0;
  for (const v of vals) sum += v;

  return {
    name,
    count: vals.length,
    min: vals[0],
    q01: quantileSorted(vals, 0.01),
    q10: quantileSorted(vals, 0.10),
    q50: quantileSorted(vals, 0.50),
    q90: quantileSorted(vals, 0.90),
    q99: quantileSorted(vals, 0.99),
    max: vals[vals.length - 1],
    mean: sum / vals.length
  };
}

function fmtNum(v) {
  if (!Number.isFinite(v)) return 'NaN';
  return Number(v).toFixed(6);
}

function printSummary(s) {
  console.log(`[${s.name}] count=${s.count}`);
  console.log(`  min=${fmtNum(s.min)} q01=${fmtNum(s.q01)} q10=${fmtNum(s.q10)} q50=${fmtNum(s.q50)} q90=${fmtNum(s.q90)} q99=${fmtNum(s.q99)} max=${fmtNum(s.max)} mean=${fmtNum(s.mean)}`);
}

function getTemporalSigma(raw, i, mode) {
  if (!raw.scale_t) return NaN;
  const s = raw.scale_t[i];
  if (!Number.isFinite(s)) return NaN;
  if (mode === 'linear') return s;
  return Math.exp(s);
}

function buildTemporalStats(raw, time, sigmaThresh, sigmaMode, sampleCount) {
  if (!raw.t || !raw.scale_t) {
    return {
      hasTemporal: false
    };
  }

  const n = raw.N;
  let passed = 0;
  let rejected = 0;

  const absDt = new Float64Array(n);
  const sigmaT = new Float64Array(n);
  const ratio = new Float64Array(n);

  const passedSamples = [];
  const rejectedSamples = [];

  for (let i = 0; i < n; i++) {
    const t0 = raw.t[i];
    const sig = getTemporalSigma(raw, i, sigmaMode);
    const dt = Math.abs(time - t0);

    absDt[i] = dt;
    sigmaT[i] = sig;
    ratio[i] = Number.isFinite(sig) && sig !== 0 ? (dt / sig) : NaN;

    const ok = Number.isFinite(sig) ? (dt <= sigmaThresh * sig) : true;
    if (ok) {
      passed++;
      if (passedSamples.length < sampleCount) {
        passedSamples.push({ i, t: t0, sigmaT: sig, absDt: dt, ratio: ratio[i] });
      }
    } else {
      rejected++;
      if (rejectedSamples.length < sampleCount) {
        rejectedSamples.push({ i, t: t0, sigmaT: sig, absDt: dt, ratio: ratio[i] });
      }
    }
  }

  return {
    hasTemporal: true,
    passed,
    rejected,
    passRatio: n > 0 ? passed / n : NaN,
    absDtSummary: summarizeArray('abs(time - t)', absDt),
    sigmaTSummary: summarizeArray(`sigmaT(${sigmaMode})`, sigmaT),
    ratioSummary: summarizeArray('|dt|/sigmaT', ratio),
    passedSamples,
    rejectedSamples
  };
}

function printSampleBlock(title, arr) {
  console.log(title);
  if (!arr || arr.length === 0) {
    console.log('  (none)');
    return;
  }
  for (const s of arr) {
    console.log(`  i=${s.i} t=${fmtNum(s.t)} sigmaT=${fmtNum(s.sigmaT)} absDt=${fmtNum(s.absDt)} ratio=${fmtNum(s.ratio)}`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(process.cwd(), args.input);

  if (!fs.existsSync(inputPath)) {
    throw new Error(`File not found: ${inputPath}`);
  }

  const parserPath = path.resolve(projectRoot, 'demo', 'js', 'splat4d_parser_v2.js');
  if (!fs.existsSync(parserPath)) {
    throw new Error(`Parser not found: ${parserPath}`);
  }

  const parserMod = await import(pathToFileURL(parserPath).href);
  if (typeof parserMod.parseSplat4DV2 !== 'function') {
    throw new Error('parseSplat4DV2 was not found in js/splat4d_parser_v2.js');
  }

  const buf = fs.readFileSync(inputPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const raw = parserMod.parseSplat4DV2(ab);

  console.log('=== Basic ===');
  console.log(`file=${inputPath}`);
  console.log(`N=${raw.N}`);
  console.log(`activeShDegree=${raw.activeShDegree}`);
  console.log(`activeShDegreeT=${raw.activeShDegreeT}`);
  console.log(`rot4d=${raw.rot4d}`);
  console.log(`hasT=${!!raw.t}  hasScalingT=${!!raw.scale_t}`);

  if (raw.t) {
    printSummary(summarizeArray('t', raw.t));
  } else {
    console.log('[t] missing');
  }

  if (raw.scale_t) {
    printSummary(summarizeArray('scale_t(raw)', raw.scale_t));

    const sigmaExp = new Float64Array(raw.N);
    const sigmaLinear = new Float64Array(raw.N);
    for (let i = 0; i < raw.N; i++) {
      const s = raw.scale_t[i];
      sigmaExp[i] = Number.isFinite(s) ? Math.exp(s) : NaN;
      sigmaLinear[i] = s;
    }
    printSummary(summarizeArray('sigmaT(exp)', sigmaExp));
    printSummary(summarizeArray('sigmaT(linear)', sigmaLinear));
  } else {
    console.log('[scale_t] missing');
  }

  if (args.time !== null) {
    console.log('');
    console.log('=== Temporal Culling Check ===');
    console.log(`time=${fmtNum(args.time)} sigmaThresh=${fmtNum(args.sigmaThresh)} sigmaMode=${args.sigmaMode}`);

    const stats = buildTemporalStats(raw, args.time, args.sigmaThresh, args.sigmaMode, args.sampleCount);
    if (!stats.hasTemporal) {
      console.log('temporal fields are missing');
      return;
    }

    console.log(`passed=${stats.passed} rejected=${stats.rejected} passRatio=${fmtNum(stats.passRatio)}`);
    printSummary(stats.absDtSummary);
    printSummary(stats.sigmaTSummary);
    printSummary(stats.ratioSummary);
    printSampleBlock('passed samples:', stats.passedSamples);
    printSampleBlock('rejected samples:', stats.rejectedSamples);
  } else {
    console.log('');
    console.log('No --time specified, so only raw field summaries were printed.');
  }
}

main().catch(err => {
  console.error('[ERROR]', err && err.stack ? err.stack : err);
  process.exit(1);
});
