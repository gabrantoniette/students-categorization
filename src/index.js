import tf from '@tensorflow/tfjs-node';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import tensor_people_normalized from './gold/xs.json' with { type : 'json'};
import tensor_labels from './gold/ys.json' with { type : 'json'};
import { bestCategory, pendingStudents } from './categorize.js';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()
    model.add(tf.layers.dense({inputShape: [7], // seven inputs (age + three colors + three locations)
        units: 80,  // eighty neurons
        activation: 'relu'})) // negative values = 0 ; positve values keep as it is
                              // with relu activation the model can learn complex patterns.


    model.add(tf.layers.dense({units: 3, // category output quantities ('premium', 'medium', 'basic')
        activation: 'softmax'})) // softmax normalize the outputs into probabilities.
                                 // so the output would be saomthing like [0.7, 0.2, 0.1]
                                 // 70% premium, 20% medium, 10% basic 


    model.compile({optimizer: 'adam', // do adjustments above error's  
        loss: 'categoricalCrossentropy', // loss measure error's sizes
        metrics: ['accuracy']}) // accuracy shows correct's percentage


    await model.fit( // await makes the .fit finish the learn process before moving foward to 'return model'
        inputXs,
        outputYs,
        {
            verbose: 0, // turnoff automatic tensorflow log. we are already building our own one.
            shuffle: true, // shuffle the dataset at each recheck 
            epochs: 100 // how many times the model check the entire dataset
            //callbacks: { // follow each progress step showing at console.
            //    onEpochEnd: (epoch, log) => console.log(
            //        `Epoch: ${epoch}: loss = ${log.loss}`
            //    )
            //}
        }
    )

    return model

}


async function prediction (model, inputXs) {

    const output = model.predict(inputXs)
    const pred_array = await output.array()

    return pred_array[0].map((prob, index) => ({prob, index}))

}


// Labels of the categories to be predicted (one-hot encoded)
// [premium, medium, basic]
const label_names = ["premium", "medium", "basic"]; // Label order
//const tensorLabels

// New students come from `python -m src.upsert`, and their categories are kept in categorized.json
const new_students_file = new URL('./gold/new_students.json', import.meta.url)
const categorized_file = new URL('./gold/categorized.json', import.meta.url)
const readList = (file) => existsSync(file) ? JSON.parse(readFileSync(file, 'utf-8')) : []

const categorized = readList(categorized_file)
const pending = pendingStudents(readList(new_students_file), categorized)

if (pending.length === 0) {
    console.log('No new students to categorize. Run python -m src.upsert to add them.')
} else {
    // Create the input (xs) and output (ys) tensors used to train the model
    const inputXs = tf.tensor2d(tensor_people_normalized)
    const outputYs = tf.tensor2d(tensor_labels)

    const model = await trainModel(inputXs,outputYs)

    for (const student of pending) {
        // a student is a [1, 7] tensor: the same seven numbers the training set has
        const predictions = await prediction(model, tf.tensor2d([student.xs]))
        const result = bestCategory(predictions.map(p => p.prob), label_names)
        categorized.push({ id: student.id, name: student.name, created_at: student.created_at, ...result })
        console.log(`${student.name}: ${result.category} (${(result.probability * 100).toFixed(2)}%)`)
    }

    writeFileSync(categorized_file, JSON.stringify(categorized, null, 2))
    console.log(`${pending.length} new student${pending.length === 1 ? '' : 's'} categorized -> src/gold/categorized.json`)
}