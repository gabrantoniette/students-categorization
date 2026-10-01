import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
const projectRoot = new URL('..', import.meta.url);

// Removes all whitespace, so the checks don't depend on how tfjs aligns the printed columns
const compact = (text) => text.replace(/\s+/g, '');

test('index.js prints the input (xs) and output (ys) tensors', async () => {
    const { stdout } = await execFileAsync(process.execPath, ['index.js'], { cwd: projectRoot });
    const output = compact(stdout);

    assert.ok(
        output.includes('[[0.33,1,0,0,1,0,0],[0,0,1,0,0,1,0],[1,0,0,1,0,0,1]]'),
        `input tensor (xs) not found in the output:\n${stdout}`
    );
    assert.ok(
        output.includes('[[1,0,0],[0,1,0],[0,0,1]]'),
        `output tensor (ys) not found in the output:\n${stdout}`
    );
});
