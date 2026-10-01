import tf, { Optimizer } from '@tensorflow/tfjs-node';
import tensorPeopleNormalized from '../src/gold/xs.json' with { type : 'json'};
import tensorLabels from '../src/gold/ys.json' with { type : 'json'};

async function trainModel(inputXs, outputYs) {
    // setting model to seven input (age + three colors + three locations) and eighty neurons
    // relu act as a filter 'selecting' real meaning inputs to the model.
    model.add(tf.layers.dense({inputShape: [7], units: [80], activation: 'relu'}))

    // setting three neurons outputs: 'premium', 'medium', 'basic' 
    // activation: softmax normalize the outputs into probabilities.
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}))

    // model compile
    // optimizer: Adam (adaptative moment estimation)  
    model.compile({Optimizer: 'adam', loss: 'categoricalCrossenttropy', metrics: ['accuracy']})

    // model training
    await model.fit(
        inputXs,
        outputYs,
        {
            verbos: 0,
            suffle: true,
            epochs: 100,
            callbacks: {
                onEpochEnd: (epoch, log) => console.log(
                    'Epoch: ${epoch}: loss =  ${log.loss}'
                )
            }
        }
    )

    return model 

}


// Labels of the categories to be predicted (one-hot encoded)
// [premium, medium, basic]
const labelNames = ["premium", "medium", "basic"]; // Label order
//const tensorLabels

// Create the input (xs) and output (ys) tensors used to train the model
const inputXs = tf.tensor2d(tensorPeopleNormalized)
const outputYs = tf.tensor2d(tensorLabels)

const model = trainModel(inputXs,outputYs)  