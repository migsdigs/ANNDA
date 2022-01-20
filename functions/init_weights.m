% Function init_weights to initialise the weights to be used for single
% layer perceptron
function weight_matrix = init_weights(number_outputs, number_inputs)

    weight_matrix = randn(number_outputs, number_inputs);
end