use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray, new_ndarray};

#[test]
fn filter_lambda_format() {
    let format = format_new_ndarray!(for i => (for j if tensor2[j] < j as i32 => tensor1[i, j]).sum::<i32>() + i as i32);

    asserts::equivalent!(
        format,
        r"{
            let dim_number_1 = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor1).1;
            let dim_number_0 = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor1).0;
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn(
                dim_number_0,
                | i | {
                    (
                        ( 0usize .. dim_number_1 ).filter( | & j | {
                            ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor2, j ) } ) < j as i32
                        } ).map( | j | {
                            ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor1, ( i, j, ) ) } )
                        } )
                    ).sum ::< i32 > ( ) + i as i32
                }
            )
        } "
    );

    let format = format_new_ndarray!(for i => (for j k if tensor3[j] < tensor4[k] => tensor1[i, j] * tensor2[i, k]).sum::<f64>());

    asserts::equivalent!(
        format,
        r##"{
            let dim_number_1 = :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor1 ).1;
            let dim_number_2 = :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor2 ).1;
            let dim_number_0 = :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor1 ).0;
            if dim_number_0 != :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor2 ).0 {
                panic! ("Dimensions are not matching between tensor1[ i, _ ] and tensor2[ i, _ ]");
            }
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn(
                dim_number_0,
                | i | {
                    (
                        ( 0usize .. dim_number_2 ).flat_map( move | k | {
                            ( 0usize .. dim_number_1 ).map( move | j | { ( j, k, ) } )
                        } ).filter( | & ( j, k, ) | {
                            ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor3, j ) } ) < ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor4, k ) } )
                        } ).map( | ( j, k, ) | {
                            ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor1, ( i, j, ) ) } ) * ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor2, ( i, k, ) ) } )
                        } )
                    ).sum ::< f64 > ( )
                }
            )
        } "##
    );
}

#[test]
fn filter_lambda_generation() {
    let tensor1 = Array2::<i32>::from_shape_fn((3, 5), |(i, j)| (i as i32) * (j as i32) - 7);
    let tensor2 = Array1::<i32>::from_shape_fn(5, |i| 1 - (i as i32));
    let result = new_ndarray!(for i => (for j if tensor2[j] < j as i32 => tensor1[i, j]).sum::<i32>() + i as i32);
    let expected = Array1::<i32>::from_shape_fn(3, |i| {
        (0..5)
            .filter(|&j| tensor2[j] < j as i32)
            .map(|j| tensor1[(i, j)])
            .sum::<i32>()
            + i as i32
    });
    assert_eq!(expected, result);

    let tensor1 = Array2::<f64>::from_shape_fn((3, 5), |(i, j)| (i as f64) * (j as f64) - 1.5);
    let tensor2 = Array2::<f64>::from_shape_fn((3, 2), |(i, j)| 0.5 * (j as f64) - (i as f64));
    let tensor3 = Array1::<f64>::from_shape_fn(5, |i| i as f64 - 2.0);
    let tensor4 = Array1::<f64>::from_shape_fn(2, |i| 1.0 - (i as f64));
    let result = new_ndarray!(for i => (for j k if tensor3[j] < tensor4[k] => tensor1[i, j] * tensor2[i, k]).sum::<f64>());
    let expected = Array1::<f64>::from_shape_fn(3, |i| {
        (0..2)
            .flat_map(|k| (0..5).map(move |j| (j, k)))
            .filter(|&(j, k)| tensor3[j] < tensor4[k])
            .map(|(j, k)| tensor1[(i, j)] * tensor2[(i, k)])
            .sum::<f64>()
    });
    assert_eq!(expected, result);
}
