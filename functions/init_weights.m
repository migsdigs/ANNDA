% Function init_weights to initialise the weights to be used for single
% layer perceptron
function weight_matrix = init_weights(number_inputs, number_outputs)

    weight_matrix = randn(number_inputs, number_outputs);
end