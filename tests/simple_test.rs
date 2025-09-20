use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray2, new_ndarray2};

#[test]
fn simple_lambda_format() {
    let format = format_new_ndarray2!(for i => tensor[i] + i);

    assert_eq!(
        format.to_string(),
        "{ \
            let i_dimension = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor); \
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn(\
                i_dimension, \
                | i | { \
                    (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget(& tensor, i) }) + i \
                }\
            ) \
        } "
    );

    let format = format_new_ndarray2!(for i j => tensor[i, j]);

    assert_eq!(
        format.to_string(),
        "{ \
            let i_dimension = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor).0; \
            let j_dimension = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor).1; \
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 2usize] >> :: from_shape_fn(\
                (i_dimension, j_dimension,), \
                | (i, j,) | { \
                    (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget(& tensor, (i, j,)) }) \
                }\
            ) \
        } "
    );
}

#[test]
fn simple_lambda_generation() {
    let tensor = Array1::<f64>::from_shape_fn(13, |i| 20.0 - i as f64 * 1.5);
    let modified_tensor = new_ndarray2!(for i => tensor[i] + i as f64);
    let mut i = 0;
    let i = &mut i;
    let expected_tensor = tensor.map(|x| {
        let v = x + *i as f64;
        *i += 1;
        v
    });
    assert_eq!(expected_tensor, modified_tensor);

    let tensor = Array2::<f64>::from_shape_fn((10, 9), |(i, j)| i as f64 * 3.0 + j as f64 * 7.5);
    let transpose = new_ndarray2!(for i j => tensor[j, i]);
    assert_eq!(tensor.reversed_axes(), transpose);
}
