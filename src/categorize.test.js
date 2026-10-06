import { test } from 'node:test'
import assert from 'node:assert/strict'

import { bestCategory, pendingStudents } from './categorize.js'

test('a student who already has a category is not categorized again', () => {
    const students = [{ id: 1 }, { id: 2 }, { id: 3 }]
    const categorized = [{ id: 2, category: 'basic' }]

    assert.deepEqual(pendingStudents(students, categorized).map(s => s.id), [1, 3])
})

test('the category is the most likely one, and every probability is kept', () => {
    const result = bestCategory([0.4288, 0.5616, 0.0095], ['premium', 'medium', 'basic'])

    assert.equal(result.category, 'medium')
    assert.equal(result.probability, 0.5616)
    assert.deepEqual(result.probabilities, { premium: 0.4288, medium: 0.5616, basic: 0.0095 })
})
