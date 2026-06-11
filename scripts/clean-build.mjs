import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const buildDir = path.join(projectRoot, 'build');

function assertSafeBuildDir(targetDir) {
  const relative = path.relative(projectRoot, targetDir);
  if (relative !== 'build') {
    throw new Error(`Refusing to remove unexpected build path: ${targetDir}`);
  }
}

assertSafeBuildDir(buildDir);
await fs.rm(buildDir, { recursive: true, force: true });
console.log(JSON.stringify({
  status: 'ok',
  removed: 'build'
}, null, 2));
