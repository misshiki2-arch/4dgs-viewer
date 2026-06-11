import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const demoRoot = path.join(projectRoot, 'demo');
const buildDemoRoot = path.join(projectRoot, 'build', 'demo');
const manifestPath = path.join(buildDemoRoot, '.demo-sync-manifest.json');
const threeSource = path.join(
  projectRoot,
  'node_modules',
  'three',
  'build',
  'three.module.js'
);
const threeDest = path.join(buildDemoRoot, 'lib', 'three.module.js');
const MTIME_TOLERANCE_MS = 1;

function toPosixPath(value) {
  return value.split(path.sep).join('/');
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function loadManifest() {
  try {
    const data = await fs.readFile(manifestPath, 'utf8');
    const parsed = JSON.parse(data);
    return Array.isArray(parsed.managedFiles) ? parsed.managedFiles : [];
  } catch {
    return [];
  }
}

async function listFiles(root, base = root) {
  const entries = await fs.readdir(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...await listFiles(fullPath, base));
      continue;
    }
    if (!entry.isFile()) continue;
    files.push(toPosixPath(path.relative(base, fullPath)));
  }
  return files;
}

async function shouldCopy(src, dest) {
  if (!await pathExists(dest)) return true;
  const [srcStat, destStat] = await Promise.all([fs.stat(src), fs.stat(dest)]);
  return (
    srcStat.size !== destStat.size ||
    Math.abs(srcStat.mtimeMs - destStat.mtimeMs) > MTIME_TOLERANCE_MS
  );
}

async function copyIfChanged(src, dest) {
  if (!await shouldCopy(src, dest)) return false;
  await fs.mkdir(path.dirname(dest), { recursive: true });
  await fs.copyFile(src, dest);
  const srcStat = await fs.stat(src);
  await fs.utimes(dest, srcStat.atime, srcStat.mtime);
  return true;
}

async function removeEmptyParents(startDir, stopDir) {
  let current = startDir;
  while (current.startsWith(stopDir) && current !== stopDir) {
    try {
      await fs.rmdir(current);
    } catch {
      return;
    }
    current = path.dirname(current);
  }
}

async function main() {
  await fs.mkdir(buildDemoRoot, { recursive: true });
  const previousManagedFiles = new Set(await loadManifest());
  const sourceFiles = await listFiles(demoRoot);
  const currentManagedFiles = new Set(sourceFiles);
  let copied = 0;
  let skipped = 0;
  let removed = 0;

  for (const relativePath of sourceFiles) {
    const src = path.join(demoRoot, relativePath);
    const dest = path.join(buildDemoRoot, relativePath);
    if (await copyIfChanged(src, dest)) {
      copied += 1;
    } else {
      skipped += 1;
    }
  }

  for (const relativePath of previousManagedFiles) {
    if (currentManagedFiles.has(relativePath)) continue;
    const dest = path.join(buildDemoRoot, relativePath);
    if (!await pathExists(dest)) continue;
    await fs.unlink(dest);
    removed += 1;
    await removeEmptyParents(path.dirname(dest), buildDemoRoot);
  }

  const threeCopied = await copyIfChanged(threeSource, threeDest);
  await fs.writeFile(
    manifestPath,
    `${JSON.stringify({
      schemaVersion: 1,
      sourceRoot: 'demo',
      destinationRoot: 'build/demo',
      managedFiles: [...currentManagedFiles].sort()
    }, null, 2)}\n`
  );

  console.log(
    JSON.stringify({
      status: 'ok',
      copied,
      skipped,
      removed,
      threeModule: threeCopied ? 'copied' : 'unchanged',
      managedFileCount: currentManagedFiles.size
    }, null, 2)
  );
}

await main();
