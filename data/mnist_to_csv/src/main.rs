use csv::Writer;
use mnist::{Mnist, MnistBuilder};
use std::error::Error;

const IMAGE_SIZE: usize = 28 * 28;

fn write_csv(
    path: &str,
    images: &[u8],
    labels: &[u8],
) -> Result<(), Box<dyn Error>> {
    assert_eq!(
        images.len(),
        labels.len() * IMAGE_SIZE,
        "Количество пикселей не соответствует количеству labels"
    );

    let mut writer = Writer::from_path(path)?;

    let mut header = Vec::with_capacity(IMAGE_SIZE + 1);
    header.push("label".to_string());

    for i in 0..IMAGE_SIZE {
        header.push(format!("pixel_{i}"));
    }

    writer.write_record(&header)?;

    for (label, image) in labels.iter().zip(images.chunks_exact(IMAGE_SIZE)) {
        let mut row = Vec::with_capacity(IMAGE_SIZE + 1);
        row.push(label.to_string());

        for &pixel in image {
            let normalized = pixel as f32 / 255.0;
            row.push(normalized.to_string());
        }

        writer.write_record(&row)?;
    }

    writer.flush()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("../")
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    write_csv("mnist_train.csv", &trn_img, &trn_lbl)?;
    write_csv("mnist_test.csv", &tst_img, &tst_lbl)?;

    println!("Создан файл mnist_train.csv");
    println!("Создан файл mnist_test.csv");

    Ok(())
}