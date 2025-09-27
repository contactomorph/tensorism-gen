use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray2, new_ndarray2};

#[test]
fn simple_lambda_format() {
    let format = format_new_ndarray2!(for i => tensor[i] + i);

    assert_eq!(
        format.to_string(),
        "{ \
            let global_dim_for_i = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor); \
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn(\
                global_dim_for_i, \
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
            let global_dim_for_i = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor).0; \
            let global_dim_for_j = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor).1; \
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 2usize] >> :: from_shape_fn(\
                (global_dim_for_i, global_dim_for_j,), \
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

#[test]
fn layered_lambda_format() {
    let format = format_new_ndarray2!(for i => (for j => tensor1[i, j]).sum() + (for j => tensor2[j, i]).sum() * tensor3[i]);
    assert_eq!(
        format.to_string(),
        "{ \
            let local_dim_for_00_j = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor1).1; \
            let local_dim_for_20_j = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor2).0; \
            let global_dim_for_i = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor1).0; \
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn(\
                global_dim_for_i, \
                | i | { \
                    (\
                        (0usize .. local_dim_for_00_j).map(| j | { (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget(& tensor1, (i, j,)) }) })\
                    ).sum() + (\
                        (0usize .. local_dim_for_20_j).map(| j | { (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget(& tensor2, (j, i,)) }) })\
                    ).sum() * (* unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget(& tensor3, i) }) \
                }\
            ) \
        } "
    );
}

#[test]
fn layered_lambda_generation() {
    let tensor1 = Array2::<i32>::from_shape_fn((7, 2), |(i, j)| {
        ((i * j) as i32 - 3 * (i as i32) + 5 * (j as i32) - 6) % 11
    });
    let tensor2 = Array2::<i32>::from_shape_fn((2, 7), |(i, j)| {
        ((i * j) as i32 + 3 * (i as i32) - 5 * (j as i32) - 6) % 11
    });
    let tensor3 = Array1::<i32>::from_shape_fn(7, |i| ((3 * i + 4) % 7) as i32);
    let result = new_ndarray2!(for i => (for j => tensor1[i, j]).sum::<i32>() + (for j => tensor2[j, i]).sum::<i32>() * tensor3[i]);
    let expected = Array1::from_vec(vec![-43i32, -12, -21, -95, -40, -60, -12]);
    assert_eq!(expected, result)
}

#[test]
fn tensor_references_are_accepted() {
    let mut tensor1 = Array1::<i32>::from_shape_fn(7, |i| i as i32);
    let tensor2 = Array1::<i32>::from_shape_fn(7, |i| -(i as i32));

    let tensor1_ref = &mut tensor1;
    let tensor2_ref = &tensor2;
    
    let result = new_ndarray2!(for i => tensor1_ref[i] + tensor2_ref[i]);
    let expected = Array1::from_vec(vec![0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(expected, result)
}