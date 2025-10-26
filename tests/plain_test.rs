use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray, new_ndarray};

#[test]
fn plain_value_lambda_format() {
    let format = format_new_ndarray!(for i => tensor[i, plain: 1]);

    asserts::equivalent!(
        format,
        r#"{
            let dim_number_0 = :: ndarray :: ArrayBase :: < _, _ > :: dim(& tensor).0;
            let plain_value_0 : usize = 1;
            if plain_value_0 >= :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor ).1  {
                panic! ( "Plain value is out of bounds in tensor[ _, plain ]" );
            }
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 1usize] >> :: from_shape_fn(
                dim_number_0,
                | i | { ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor, ( i, plain_value_0, ) ) } ) }
            )
        } "#
    );

    let format =
        format_new_ndarray!(for i j => tensor1[indexer[plain: 3 * n - 2, j], i, plain: 2 * n + 1]);

    asserts::equivalent!(
        format,
        r#"{
            let dim_number_2 = :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor1 ).0;
            if dim_number_2 != crate :: tensorism :: Reindexing2 :: get_output_bound( & indexer ) {
                panic! ( "Dimensions are not matching between tensor1[ indexer, _, _ ] and indexer = indexer[ _, _ ]" );
            }
            let dim_number_0 = :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor1 ).1;
            let dim_number_1 = crate :: tensorism :: Reindexing2 :: get_input1_bound( & indexer );
            let plain_value_0 : usize = 3 * n - 2;
            if plain_value_0 >= crate :: tensorism :: Reindexing2 :: get_input0_bound( & indexer ) {
                panic! ( "Plain value is out of bounds in indexer[ plain, _ ]" );
            }
            let plain_value_1 : usize = 2 * n + 1;
            if plain_value_1 >= :: ndarray :: ArrayBase :: < _, _ > :: dim( & tensor1 ).2 {
                panic! ( "Plain value is out of bounds in tensor1[ _, _, plain ]" );
            }
            :: ndarray :: Array :: < _, :: ndarray :: Dim < [:: ndarray :: Ix; 2usize] >> :: from_shape_fn(
                (dim_number_0, dim_number_1, ),
                | ( i, j, ) | {
                    ( * unsafe { :: ndarray :: ArrayBase :: < _, _ > :: uget( & tensor1,
                        (
                            crate :: tensorism :: Reindexing2 :: get_unchecked( & indexer, plain_value_0, j ),
                            i,
                            plain_value_1,
                        )
                    ) } )
                }
            )
        } "#
    );
}

#[test]
fn plain_value_lambda_generation() {
    let tensor = Array2::<i32>::from_shape_fn((5, 3), |(i, j)| {
        ((i * j) as i32 - 3 * (i as i32) + 5 * (j as i32) - 6) % 11
    });
    let result = new_ndarray!(for i => tensor[i, plain: 1]);
    let expected = Array1::from_vec(vec![-1i32, -3, -5, -7, -9]);
    assert_eq!(expected, result);
}
