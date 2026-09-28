use std::str::FromStr;
use std::io::Read;
use rand::distr::Distribution;

struct Dataset {
    images: ndarray::Array2<f32>,
    labels: ndarray::Array2<f32>
}

const SIZE: usize = 28*28;

fn parse_data_line(line: &str) -> (Vec<f32>, f32) {
    let mut pixel = Vec::<f32>::new();

    for x in line.split(',') {
        match f32::from_str(x) {
            Ok(val) => {
                pixel.push(val);
            }
            _ => panic!("invalid value")
        }
    }
    (pixel[1..].iter().map(|x| x/255.0).collect(), pixel[0])
}

fn parse_data(path: &str) -> Dataset{
    let mut file = std::fs::File::open(path).unwrap();

    let mut images = ndarray::Array::zeros((0, SIZE));
    let mut labels = ndarray::Array::zeros((0, 10));
    let mut content = String::new();

    file.read_to_string(&mut content).unwrap();

    for line in content.lines().skip(1).take_while(|x| !x.is_empty()) {
        let line = parse_data_line(line);

        let pixel = line.0;
        let label = line.1 as usize;

        let label: Vec<f32> = (0..10).map(|x| if label == x {1.0} else {0.0}).collect();

        images.push_row(ndarray::ArrayView::from(&pixel)).unwrap();
        labels.push_row(ndarray::ArrayView::from(&label)).unwrap();

    }

    Dataset{images, labels}
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Layer {
    weights: ndarray::Array2<f32>,
    blases: ndarray::Array1<f32>
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Network{
    layers: Vec<Layer>
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn softmax(values: ndarray::ArrayView1<f32>)->ndarray::Array1<f32> {
    let max = values.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
    let shifted_values = values.map(|x| x - max);

    let sum: f32 = shifted_values.iter().map(|x| x.exp()).sum();

    (0..values.len()).map(|x| shifted_values[x].exp() / sum).collect()
}

fn step(x: f32) -> f32 {
    if x > 0.0 {1.0} else {0.0}
}

impl Network {
    fn new(layer_size: &[usize]) -> Self {
        Network{
            layers: layer_size.iter().zip(layer_size.iter().skip(1)).map(|(&in_size, &out_size)| {
                let mut rng = rand::rng();
                let limit = (6.0 / in_size as f32).sqrt();
                let dist = rand::distr::Uniform::new(-limit, limit).unwrap();
                Layer {
                    weights: ndarray::Array::zeros((in_size, out_size)).map(|_: &f32| dist.sample(&mut rng)),
                    blases: ndarray::Array::zeros(out_size)
                }
            } 
        ).collect(),
    }}

    fn propagate_forward(&self, input: &ndarray::ArrayView2<f32>) -> (Vec<ndarray::Array2<f32>>, Vec<ndarray::Array2<f32>>) {
        let mut activations = vec![input.to_owned()];
        let mut linear_values = vec![];

        for layer in self.layers.iter().take(self.layers.len() - 1) {
            let linear = activations.last().unwrap().dot(&layer.weights) + &layer.blases;
            let activation = linear.map(|&x| relu(x));
            activations.push(activation);
            linear_values.push(linear);
        }

        let linear = activations.last().unwrap().dot(&self.layers.last().unwrap().weights) + &self.layers.last().unwrap().blases;
        activations.push(linear.clone());
        linear_values.push(linear);

        (activations, linear_values)
    }

    fn propagate_backward(&mut self, activations: Vec<ndarray::Array2<f32>>, line_values: Vec<ndarray::Array2<f32>>, gradient: ndarray::Array2<f32>, rate: f32) {
        let mut cur_gradient = gradient;
        let batch_size = cur_gradient.nrows() as f32;
        for layer in (0..self.layers.len()).rev() {
            if layer != self.layers.len() -1 {
                let step = line_values[layer].map(|&v| step(v));
                cur_gradient = cur_gradient * step;
            }
            let weight_gradient = activations[layer].t().dot(&cur_gradient) / batch_size;
            let bias_gradient = &cur_gradient.mean_axis(ndarray::Axis(0)).unwrap();
            cur_gradient = cur_gradient.dot(&self.layers[layer].weights.t());
            self.layers[layer] = Layer {
                weights: &self.layers[layer].weights - rate * weight_gradient,
                blases: &self.layers[layer].blases - rate * bias_gradient,
            }
        }
    }

   fn learn(&mut self,dataset: &Dataset,epochs: usize,batch_size: usize,rate: f32,) {
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batches = 0;

            for (input, labels) in dataset.images.axis_chunks_iter(ndarray::Axis(0), batch_size).zip(dataset.labels.axis_chunks_iter(ndarray::Axis(0), batch_size))
            {
                let (activations, linears) = self.propagate_forward(&input);

                let output = activations.last().unwrap();

                let mut probabilities = ndarray::Array::zeros((0, output.ncols()));

                for row in output.axis_iter(ndarray::Axis(0)) {
                    probabilities.push_row(softmax(row).view()).unwrap();}

                let loss: f32 = probabilities.axis_iter(ndarray::Axis(0)).zip(labels.axis_iter(ndarray::Axis(0))).map(|(p, y)| {p.iter().zip(y.iter()).map(|(&p, &y)| {
                                if y > 0.0 {
                                    -(p + 1e-7).ln()
                                } else {
                                    0.0
                                }
                            }).sum::<f32>()
                    }).sum::<f32>()
                    / labels.nrows() as f32;

                epoch_loss += loss;
                batches += 1;

                let gradient = probabilities - labels;

                self.propagate_backward(activations,linears,gradient,rate,);
            }

            println!("Epoch {}/{} loss: {}",epoch + 1,epochs,epoch_loss / batches as f32);

            check(self, dataset);
        }
    }

    pub fn evaluate(&self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        let (activation, _) = self.propagate_forward(input);
        let output = activation.last().unwrap();

        let mut probabilities = ndarray::Array::zeros((0, output.ncols()));

        for digit_estimates in output.axis_iter(ndarray::Axis(0)) {
            probabilities.push_row(softmax(digit_estimates).view()).unwrap()
        }
        probabilities
    }

    fn save(&self, path: &str) {
        let data = bincode::serialize(self).unwrap();
        std::fs::write(path, data).unwrap();
    }

    #[warn(dead_code)]
    fn load(path: &str) -> Self {
        let data = std::fs::read(path).unwrap();
        bincode::deserialize(&data).unwrap()
    }

}

fn check(network: &Network, dataset: &Dataset) {
    let mut num_mistakes = 0;
    let mut num_total = 0;

    let predict = network.evaluate(&dataset.images.view());

    for (predict, target_row) in predict.axis_iter(ndarray::Axis(0)).zip(dataset.labels.axis_iter(ndarray::Axis(0))) {
        let predict_digit = predict.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;
        let actial_digit = target_row.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;

        if predict_digit != actial_digit {
            num_mistakes += 1;
        }
        num_total +=1;
    }
    println!("{} mistakes out of {} ({}%)", num_mistakes, num_total, 100.0 * num_mistakes as f32 / num_total as f32);
}

fn main() {
    let test_data = parse_data("../data/mnist_test.csv");
    let train_data = parse_data("../data/mnist_train.csv");

    let mut network = Network::new(&[28*28, 100, 10]);
    check(&network, &test_data);

    network.learn(&train_data, 3000, 64, 0.1);
    network.save("test.network");

    check(&network, &train_data);
    check(&network, &test_data);
}