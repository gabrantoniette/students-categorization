import tf from '@tensorflow/tfjs-node';
import tensor_people_normalized from './gold/xs.json' with { type : 'json'};
import tensor_labels from './gold/ys.json' with { type : 'json'};

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

// Create the input (xs) and output (ys) tensors used to train the model
const inputXs = tf.tensor2d(tensor_people_normalized)
const outputYs = tf.tensor2d(tensor_labels)

const model = await trainModel(inputXs,outputYs)  

const predictions = await prediction(
    model, 
    // pick up student number N of the dataset for example the following one:
    inputXs.slice([4999, 0], [1, 7])
    // studant 5000 from 0 iterating one line in seven columns
    // .slice() is mandatory to iterate tensors.
    // it does not work for arrays. 
)

const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${label_names[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')

console.log(results)