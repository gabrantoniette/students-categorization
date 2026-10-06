// New students from the gold layer that do not have a category yet
export function pendingStudents(students, categorized) {
    const done = new Set(categorized.map(student => student.id))
    return students.filter(student => !done.has(student.id))
}

// The most likely category, its probability, and the probability of every category
export function bestCategory(probabilities, labels) {
    const best = probabilities.indexOf(Math.max(...probabilities))
    return {
        category: labels[best],
        probability: probabilities[best],
        probabilities: Object.fromEntries(labels.map((label, index) => [label, probabilities[index]])),
    }
}
