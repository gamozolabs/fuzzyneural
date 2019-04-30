#![feature(duration_as_u128)]

pub mod rng;

use std::time::Instant;
use crate::rng::Rng;

#[derive(Debug, Clone)]
pub enum Layer {
    Forward {
        /// Number of hidden values at this layer
        size: usize,

        /// Weights for this layer
        /// Number of weights should be `size` * `previous_layer_size`
        weights: Vec<f32>,
    },

    Bias {
        /// Biases (value to add to all inputs from the previous layer)
        /// Number of biases should be equal to `size` which should be identical
        /// to the previous layer size.
        biases: Vec<f32>,
    },

    Relu,
}

#[derive(Debug, Clone)]
pub struct Network {
    /// All the layers in the neural network, in order
    layers: Vec<Layer>,

    /// Number of input values
    size: usize,

    /// Neurons for even numbered layers
    /// Inputs start in this layer
    layer_even: Vec<f32>,

    /// Neurons for odd numbered layers
    layer_odd: Vec<f32>,

    /// Random number generator
    rng: Rng,
}

impl Network {
    /// Create a new, empty, neural network. Input set size based on `size`
    pub fn new(size: usize) -> Self {
        Network {
            layers:     Vec::new(),
            size:       size,
            layer_even: vec![0.; size], // initially the input layer
            layer_odd:  Vec::new(),
            rng:        Rng::new(),
        }
    }

    /// Adds a layer to the neural network
    pub fn add_layer(&mut self, mut layer: Layer) {
        // Validate the network weights and sizes line up
        let mut cur_size = self.size;
        for layer in self.layers.iter() {
            match layer {
                Layer::Forward { size, weights } => {
                    assert!(weights.len() == (cur_size * size),
                        "Invalid number of weights for layer");
                    cur_size = *size;
                }
                Layer::Bias { biases } => {
                    assert!(biases.len() == cur_size,
                        "Invalid number of biases");
                }
                Layer::Relu => {}
            }
        }

        // Initialize the weights
        match &mut layer {
            Layer::Forward { size, weights } => {
                // Initialize weights
                weights.clear();
                for _ in 0..(*size * cur_size) {
                    weights.push(self.rng.rand_f32(-1.0, 1.0));
                }
            }
            Layer::Bias { biases } => {
                // Initialize biases
                biases.clear();
                for _ in 0..cur_size {
                    biases.push(self.rng.rand_f32(-1.0, 1.0));
                }
            }
            Layer::Relu => {}
        }

        // Add the layer!
        self.layers.push(layer);
    }

    /// Randomly mutate weights
    pub fn mutate_weights(&mut self) {
        for layer in self.layers.iter_mut() {
            match layer {
                Layer::Forward { weights, .. } => {
                    for _ in 0..self.rng.rand() & 3 {
                        let pick = self.rng.rand() as usize % weights.len();
                        weights[pick] = self.rng.rand_f32(-1.0, 1.0);
                    }
                }
                Layer::Bias { biases } => {
                    for _ in 0..self.rng.rand() & 3 {
                        let pick = self.rng.rand() as usize % biases.len();
                        biases[pick] = self.rng.rand_f32(-1.0, 1.0);
                    }
                }
                Layer::Relu => {}
            }
        }
    }

    /// Propagates the input layer through the network, returning a reference
    /// to the resulting output layer
    pub fn forward_propagate(&mut self) -> &[f32] {
        let mut cur_size = self.size;

        for (layer_depth, layer) in self.layers.iter().enumerate() {
            // Alternate the double buffers depending on which layer we are
            // processing
            let (input_layer, output_layer) = if (layer_depth & 1) == 0 {
                (&self.layer_even, &mut self.layer_odd)
            } else {
                (&self.layer_odd, &mut self.layer_even)
            };

            // Clear the output layer we're about to fill up
            output_layer.clear();

            match layer {
                Layer::Forward { size, weights } => {
                    // Initialize the current weight index to zero
                    let mut weight_id = 0;

                    // For each neuron at this layer
                    for _output_id in 0..*size {
                        // Create a new accumulator
                        let mut neuron = 0f32;

                        // Go through each input from the previous layer, and
                        // multiply it by the corresponding weight
                        for input_id in 0..cur_size {
                            // Multiply against weight and add to existing
                            // accumulator
                            neuron +=
                                input_layer[input_id] * weights[weight_id];

                            // Update weight index to the next weight
                            weight_id += 1;
                        }

                        // Save this neuron in the output layer
                        output_layer.push(neuron);
                    }

                    // Pedantic asserts
                    assert!(weight_id == weights.len(),
                        "Did not use all weights");
                    assert!(output_layer.len() == *size,
                        "Invalid output size for forward layer");

                    // Update current size
                    cur_size = *size;
                }
                Layer::Bias { biases } => {
                    // Apply a bias to the all neurons from the previous layer
                    for (input, bias) in input_layer.iter().zip(biases.iter()) {
                        output_layer.push(input + bias);
                    }

                    // Pedantic assert
                    assert!(output_layer.len() == biases.len() &&
                            input_layer.len() == biases.len(),
                        "Did not use all biases");
                }
                Layer::Relu => {
                    for &input in input_layer.iter() {
                        output_layer.push(
                            if input > 1.0 {
                                1.0
                            } else if input < -1.0 {
                                -1.0
                            } else {
                                input
                            }
                        )
                    }

                    // Pedantic assert
                    assert!(output_layer.len() == input_layer.len(),
                        "Relu layer output incorrect");
                }
            }
        }

        // Return the output layer
        if (self.layers.len() & 1) == 0 {
            &self.layer_even
        } else {
            &self.layer_odd
        }
    }
}

fn main() {
    let mut network = Network::new(2);

    network.add_layer(Layer::Forward { size: 3, weights: Vec::new() });
    network.add_layer(Layer::Bias    { biases: Vec::new() });
    network.add_layer(Layer::Relu);

    network.add_layer(Layer::Forward { size: 5, weights: Vec::new() });
    network.add_layer(Layer::Bias    { biases: Vec::new() });
    network.add_layer(Layer::Relu);

    network.add_layer(Layer::Forward { size: 1, weights: Vec::new() });
    network.add_layer(Layer::Bias    { biases: Vec::new() });
    network.add_layer(Layer::Relu);

    /// Xor inputs
    const INPUT_SETS: [[f32; 2]; 4] = [
        [-1., -1.],
        [-1.,  1.],
        [ 1., -1.],
        [ 1.,  1.],
    ];

    /// All xor results
    const EXPECTED: [f32; 4] = [-1., 1., 1., -1.];

    // Currently known best network. Tuple is (error_rate, saved network)
    let mut best_network: (f32, Network) = (std::f32::MAX, network);

    let start_time = Instant::now();
    for iter_id in 1u64.. {
        let mut error_rate = 0f32;

        // Create a copy of the best network
        let mut network = best_network.1.clone();
        network.rng.reseed();
        network.mutate_weights();

        // Go through all of our training inputs
        for (inputs, &expected) in INPUT_SETS.iter().zip(EXPECTED.iter()) {
            network.layer_even.clear();
            network.layer_even.push(inputs[0] as f32);
            network.layer_even.push(inputs[1] as f32);

            let result = network.forward_propagate()[0];

            //print!("{:8.4?} => {:8.4} | Expected {:8.4}\n", inputs, result, expected);

            let error = (expected - result) * (expected - result);
            error_rate += error;
        }

        //print!("ER {:8.4}\n", error_rate);

        // If this is the first result, or we improved the error rate, update
        // the best network
        if error_rate < best_network.0 {
            best_network = (error_rate, network.clone());

            print!("Improved error rate to {}\n", error_rate);

            if error_rate == 0. {
                panic!("Found perfect network");
            }
        }

        if (iter_id & 0xffffff) == 0 {
            let uptime = (Instant::now() - start_time).as_nanos() as f64 / 1_000_000_000.0;
            let iters_per_sec = iter_id as f64 / uptime;
            print!("Performance {:12.6} M/sec\n", iters_per_sec / 1000000.);
        }
    }
}
