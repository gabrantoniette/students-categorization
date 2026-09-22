import tf from '@tensorflow/tfjs-node';

// Sample people for training (each person with age, color and location)
// const people = [
//     { name: "Erick", age: 30, color: "blue", location: "São Paulo" },
//     { name: "Ana", age: 25, color: "red", location: "Rio" },
//     { name: "Carlos", age: 40, color: "green", location: "Curitiba" }
// ];

// Input vectors with values already normalized and one-hot encoded
// Order: [normalized_age, blue, red, green, São Paulo, Rio, Curitiba]
// const tensorPeople = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// We only use the numeric data, since the neural network only understands numbers.
// tensorPeopleNormalized is the model's input dataset.
const tensorPeopleNormalized = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels of the categories to be predicted (one-hot encoded)
// [premium, medium, basic]
const labelNames = ["premium", "medium", "basic"]; // Label order
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Create the input (xs) and output (ys) tensors used to train the model
const inputXs = tf.tensor2d(tensorPeopleNormalized)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();
