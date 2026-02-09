use ndarray::{Array1, Array2};
use tensorism_gen::{format_new_ndarray, new_ndarray};

#[test]
fn indexing_alias_format() {
    let format = format_new_ndarray!(for i let j = sort[i] => tensor1[j]);

    asserts::equivalent!(
        format,
        r#"{
            let dim_number_0 = ::tensorism::Reindexing1::get_input0_bound( &sort );
            let dim_number_1 = ::tensorism::Reindexing1::get_output_bound( &sort );
            if dim_number_1 != ::ndarray::ArrayBase::<_, _>::dim( &tensor1 ) {
                panic!( "Dimensions are not matching between j = sort[ _ ] and tensor1[ j ]" );
            }
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; 1usize]>>::from_shape_fn(
                dim_number_0,
                | i | {
                    let j = unsafe { ::tensorism::Reindexing1::get_unchecked( &sort, i ) };
                    (* unsafe { ::ndarray::ArrayRef::<_, _>::uget(&tensor1, j) })
                }
            )
        } "#
    );

    let format = format_new_ndarray!(for i j let k = combine[i, j] => tensor2[i, k]);
    asserts::equivalent!(
        format,
        r#"{
            let dim_number_0 = ::tensorism::Reindexing2::get_input0_bound( &combine );
            if dim_number_0 != ::ndarray::ArrayBase::<_, _>::dim( &tensor2 ).0 {
                panic!( "Dimensions are not matching between combine[ i, _ ] and tensor2[ i, _ ]" );
            }
            let dim_number_1 = ::tensorism::Reindexing2::get_input1_bound( &combine );
            let dim_number_2 = ::tensorism::Reindexing2::get_output_bound( &combine );
            if dim_number_2 != ::ndarray::ArrayBase::<_, _>::dim( &tensor2 ).1 {
                panic!( "Dimensions are not matching between k = combine[ _, _ ] and tensor2[ _, k ]" );
            }
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; 2usize]>>::from_shape_fn(
                ( dim_number_0, dim_number_1, ),
                | ( i, j, ) | {
                    let k = unsafe { ::tensorism::Reindexing2::get_unchecked( &combine, i, j ) };
                    (* unsafe { ::ndarray::ArrayRef::<_, _>::uget(&tensor2, (i, k, )) })
                }
            )
        } "#
    );

    let format = format_new_ndarray!(for i j let k = sup[middle[sub[i, j], j]] => tensor3[k, j, i]);
    asserts::equivalent!(
        format,
        r#"{
            let dim_number_2 = ::tensorism::Reindexing1::get_input0_bound( &sup );
            if dim_number_2 != ::tensorism::Reindexing2::get_output_bound( &middle ) {
                panic!( "Dimensions are not matching between sup[ middle ] and middle = middle[ _, _ ]" );
            }
            let dim_number_3 = ::tensorism::Reindexing2::get_input0_bound( &middle );
            if dim_number_3 != ::tensorism::Reindexing2::get_output_bound( &sub ) {
                panic!( "Dimensions are not matching between middle[ sub, _ ] and sub = sub[ _, _ ]" );
            }
            let dim_number_0 = ::tensorism::Reindexing2::get_input0_bound( &sub );
            if dim_number_0 != ::ndarray::ArrayBase::<_, _>::dim( &tensor3 ).2 {
                panic!( "Dimensions are not matching between sub[ i, _ ] and tensor3[ _, _, i ]" );
            }
            let dim_number_1 = ::tensorism::Reindexing2::get_input1_bound( &sub );
            if dim_number_1 != ::tensorism::Reindexing2::get_input1_bound( &middle ) {
                panic!( "Dimensions are not matching between sub[ _, j ] and middle[ _, j ]" );
            }
            if dim_number_1 != ::ndarray::ArrayBase::<_, _>::dim( &tensor3 ).1 {
                panic!( "Dimensions are not matching between sub[ _, j ] and tensor3[ _, j, _ ]" );
            }
            let dim_number_4 = ::tensorism::Reindexing1::get_output_bound( &sup );
            if dim_number_4 != ::ndarray::ArrayBase::<_, _>::dim( &tensor3 ).0 {
                panic!( "Dimensions are not matching between k = sup[ _ ] and tensor3[ k, _, _ ]" );
            }
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; 2usize]>>::from_shape_fn(
                ( dim_number_0, dim_number_1, ),
                | ( i, j, ) | {
                    let k = unsafe { ::tensorism::Reindexing1::get_unchecked(
                        &sup,
                        ::tensorism::Reindexing2::get_unchecked(
                            &middle,
                            ::tensorism::Reindexing2::get_unchecked( &sub, i, j ),
                            j
                        )
                    ) };
                    (* unsafe { ::ndarray::ArrayRef::<_, _>::uget( &tensor3, (k, j, i, )) })
                }
            )
        } "#
    );

    let format = format_new_ndarray!(for i j => tensor3[sup[middle[sub[i, j], j]], j, i]);
    asserts::equivalent!(
        format,
        r#"{
            let dim_number_2 = ::ndarray::ArrayBase::<_, _>::dim( &tensor3 ).0;
            if dim_number_2 != ::tensorism::Reindexing1::get_output_bound( &sup ) {
                panic!( "Dimensions are not matching between tensor3[ sup, _, _ ] and sup = sup[ _ ]" );
            }
            let dim_number_3 = ::tensorism::Reindexing1::get_input0_bound( &sup );
            if dim_number_3 != ::tensorism::Reindexing2::get_output_bound( &middle ) {
                panic!( "Dimensions are not matching between sup[ middle ] and middle = middle[ _, _ ]" );
            }
            let dim_number_4 = ::tensorism::Reindexing2::get_input0_bound( &middle );
            if dim_number_4 != ::tensorism::Reindexing2::get_output_bound( &sub ) {
                panic!( "Dimensions are not matching between middle[ sub, _ ] and sub = sub[ _, _ ]" );
            }
            let dim_number_0 = ::tensorism::Reindexing2::get_input0_bound( &sub );
            if dim_number_0 != ::ndarray::ArrayBase::<_, _>::dim( &tensor3 ).2 {
                panic!( "Dimensions are not matching between sub[ i, _ ] and tensor3[ _, _, i ]" );
            }
            let dim_number_1 = ::tensorism::Reindexing2::get_input1_bound( &sub );
            if dim_number_1 != ::tensorism::Reindexing2::get_input1_bound( &middle ) {
                panic!( "Dimensions are not matching between sub[ _, j ] and middle[ _, j ]" );
            }
            if dim_number_1 != ::ndarray::ArrayBase::<_, _>::dim( &tensor3 ).1 {
                panic!( "Dimensions are not matching between sub[ _, j ] and tensor3[ _, j, _ ]" );
            }
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; 2usize]>>::from_shape_fn(
                ( dim_number_0, dim_number_1, ),
                | ( i, j, ) | {
                    (* unsafe { ::ndarray::ArrayRef::<_, _>::uget( &tensor3,
                        (
                            ::tensorism::Reindexing1::get_unchecked(
                                &sup,
                                ::tensorism::Reindexing2::get_unchecked(
                                    &middle,
                                    ::tensorism::Reindexing2::get_unchecked( &sub, i, j ),
                                    j
                                )
                            ),
                            j,
                            i,
                        )
                    ) })
                }
            )
        } "#
    );
}

#[test]
fn reverse_alias_format() {
    let format = format_new_ndarray!(for i let j = rev: i => tensor1[j, i]);

    asserts::equivalent!(
        format,
        r#"{
            let dim_number_0 = ::ndarray::ArrayBase::<_, _>::dim( &tensor1 ).1;
            let dim_number_1 = dim_number_0;
            if dim_number_1 != ::ndarray::ArrayBase::<_, _>::dim( &tensor1 ).0 {
                panic!( "Dimensions are not matching between j = _ and tensor1[ j, _ ]" );
            }
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; 1usize]>>::from_shape_fn(
                dim_number_0,
                | i | {
                    let j = dim_number_0 - 1 - i;
                    (* unsafe { ::ndarray::ArrayRef::<_, _>::uget(&tensor1, (j, i,)) })
                }
            )
        } "#
    );
}

#[test]
fn reverse_alias_generation() {
    let tensor1 = Array2::<i32>::from_shape_fn((5, 5), |(j, i)| (j as i32) * (i as i32 + 6));
    let expected = Array1::<i32>::from_vec(vec![24, 21, 16, 9, 0]);

    let result = new_ndarray!(for i let j = rev: i => tensor1[j, i]);
    assert_eq!(expected, result);

    let result = new_ndarray!(for i => tensor1[rev: i, i]);
    assert_eq!(expected, result);
}

#[test]
fn direct_alias_format() {
    let format = format_new_ndarray!(for i let j = i => tensor1[j, i]);

    asserts::equivalent!(
        format,
        r#"{
            let dim_number_0 = ::ndarray::ArrayBase::<_, _>::dim( &tensor1 ).1;
            let dim_number_1 = dim_number_0;
            if dim_number_1 != ::ndarray::ArrayBase::<_, _>::dim( &tensor1 ).0 {
                panic!( "Dimensions are not matching between j = _ and tensor1[ j, _ ]" );
            }
            ::ndarray::Array::<_, ::ndarray::Dim<[::ndarray::Ix; 1usize]>>::from_shape_fn(
                dim_number_0,
                | i | {
                    let j = i;
                    (* unsafe { ::ndarray::ArrayRef::<_, _>::uget(&tensor1, (j, i,)) })
                }
            )
        } "#
    );
}
